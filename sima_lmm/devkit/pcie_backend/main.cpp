// pcie-genai-backend — the card side of PCIe GenAI.
//
// Loads a text model over PCIe exactly like `llima run --pcie` (the
// PcieFileProvider, one ELF on disk at a time), then serves prompts from the
// host over simaai_svc notifications and streams the tokens back.
// Launched by the host's RemoteRuntime as:
//   /usr/bin/pcie-genai-backend --model <serve-root subfolder> --queue <N>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "backend_loop.hpp"
#include "dl_svc_client.hpp"
#include "lifecycle.hpp"
#include "pcie_file_provider.hpp"
#include "recv_sweep.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"
#include "vlm_generator.hpp"

namespace {
using namespace simaai::llima;
using namespace simaai::llima::pcie_backend;

constexpr const char* kProgramName = "pcie-genai-backend";

std::atomic<bool> g_stop{false};
static_assert(std::atomic<bool>::is_always_lock_free, "the signal handler needs a lock-free flag");

void on_signal(int) { g_stop.store(true); }

void install_signal_handlers() {
    struct sigaction sa{};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
}

struct Args {
    std::string model;                         // subfolder under the host serve root
    int queue = -1;
    std::string serve_root = "models";
    std::string image_serve_root = "data";     // host [serve] root a VLM image is pulled from
    std::optional<std::filesystem::path> recv_root;
};

std::string usage() {
    return "usage: pcie-genai-backend --model <serve-root subfolder> --queue <N> "
           "[--serve-root models] [--image-serve-root data] [--recv-root <dir>]";
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(flag + " needs a value");
            return argv[++i];
        };
        if (flag == "--model") a.model = value();
        else if (flag == "--queue") a.queue = std::stoi(value());
        else if (flag == "--serve-root") a.serve_root = value();
        else if (flag == "--image-serve-root") a.image_serve_root = value();
        else if (flag == "--recv-root") a.recv_root = value();
        else if (flag == "--model-options") (void)value();   // tensor-path flag; unused here
        else throw std::invalid_argument("unknown argument: " + flag);
    }
    if (a.model.empty() || a.queue < 0) throw std::invalid_argument("--model and --queue are required");
    return a;
}

std::filesystem::path env_dir(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return (value != nullptr && *value != '\0') ? value : fallback;
}

// A non-negative integer from the environment, or `fallback` (with a warning
// when the variable is set but not usable). The host always launches us as
// "--model <m> --queue <n>", so tuning knobs come from the environment, like
// SIMA_NEAT_PCIE_RUN_DIR above; the /usr/bin shim can export them.
long long env_non_negative(const char* name, long long fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    if (const auto parsed = parse_non_negative(value)) return *parsed;
    spdlog::warn("pcie-genai-backend: ignoring {}='{}' (not a non-negative integer); using {}",
                 name, value, fallback);
    return fallback;
}

// Delete leftovers in the recv root and log every decision. Never throws.
void clean_recv_root(const std::filesystem::path& root, std::chrono::seconds max_age,
                     const char* when) {
    try {
        const OpenFiles open = open_files_under(root);
        if (open.unreadable > 0) {
            spdlog::warn("recv sweep ({}): could not inspect {} processes (not running as "
                         "root?); a file only they hold open looks unused, so only the age "
                         "rule protects it", when, open.unreadable);
        }
        const SweepResult r = sweep_recv_root(root, open.paths, max_age);
        for (const auto& e : r.deleted) {
            spdlog::info("recv sweep ({}): deleted {} ({} B, {:.0f} s old)", when,
                         e.path.string(), e.bytes, e.age_s);
        }
        for (const auto& e : r.kept) {
            spdlog::info("recv sweep ({}): kept {} ({} B, {:.0f} s old): {}", when,
                         e.path.string(), e.bytes, e.age_s, e.reason);
        }
        spdlog::info("recv sweep ({}): {} deleted, {} B freed, {} kept", when, r.deleted.size(),
                     r.bytes_deleted, r.kept.size());
    } catch (const std::exception& e) {
        spdlog::warn("recv sweep ({}): skipped: {}", when, e.what());
    }
}

void set_status(const StatusWriter& writer, BackendStatus& status, std::string state,
                std::string message, std::optional<std::string> error_code = std::nullopt) {
    status.state = std::move(state);
    status.message = std::move(message);
    status.error_code = std::move(error_code);
    writer.write(status);
}

// Start-up order, and why:
//  1. claim the queue (pid file) - a second backend on the same queue stops
//     here, before it can touch the log or the status file of the owner;
//  2. send output to the queue log - the host started us with nohup;
//  3. status "starting" - the host now sees that we are alive;
//  4. open the two svc handles - fail fast if the pep daemon is not running,
//     before the slow LLiMa connect and model load;
//  5. connect() and load the model over PCIe (this takes minutes);
//  6. subscribe BEFORE "ready" - the host sends its first prompt as soon as
//     it sees "ready", and a note with no listener is dropped;
//  7. signal handlers only after the load, then status "ready", then serve.
// The model is destroyed before disconnect(), because it uses the runtime.
int run(const Args& args) {
    // Must be set before connect(): LLiMa reads it once into a static
    // (setup.cpp:80). Serial load is what keeps only ONE ELF in the recv root.
    ::setenv("SIMA_LLIMA_RUN_DISABLE_PARALLEL_LOAD", "1", 1);

    const auto run_dir = env_dir("SIMA_NEAT_PCIE_RUN_DIR", "/run/sima-neat/pcie");
    const auto log_dir = env_dir("SIMA_NEAT_PCIE_LOG_DIR", "/var/log/sima-neat/pcie");
    require_directory(run_dir);
    require_directory(log_dir);
    const std::string q = "q" + std::to_string(args.queue);

    QueueOwnership ownership(run_dir / (q + ".pid"), run_dir / (q + ".status"), kProgramName);
    // Throws QueueBusyError if a live pcie-genai-backend owns this queue.
    ownership.acquire();
    redirect_to_log(log_dir / (q + ".log"));

    const StatusWriter writer(run_dir / (q + ".status"));
    BackendStatus status;
    status.queue = args.queue;
    status.pid = static_cast<long long>(::getpid());
    status.model = args.model;
    status.started_at = now_utc_iso8601();
    set_status(writer, status, "starting", "loading model over PCIe");

    bool connected = false;
    // Leftovers in the recv root use RAM (it is a tmpfs) and once OOM-killed a
    // load. Clean at start (before our first pull, so none of OUR files can be
    // between pull and load) and on every exit we control. The age rule also
    // removes our own files that are still there at exit (GGUF, lora .npy).
    const std::chrono::seconds recv_max_age{env_non_negative("SIMA_NEAT_PCIE_RECV_MAX_AGE_S", 30)};
    const long long min_mem_mib = env_non_negative("SIMA_NEAT_PCIE_MIN_MEM_MIB", 2048);
    std::optional<std::filesystem::path> recv_root;
    try {
        recv_root = args.recv_root ? args.recv_root : recv_root_from_pep_conf();
        if (!recv_root) {
            throw std::runtime_error("cannot find the pep daemon recv root; pass --recv-root");
        }
        clean_recv_root(*recv_root, recv_max_age, "start");
        // Measured on mla-hl83 (Llama-3.2-3B): a load takes ~1.7 GiB of
        // MemAvailable. Fail now with a clear message instead of an OOM kill
        // a minute into the load.
        if (min_mem_mib > 0) {
            if (const auto avail = mem_available_mib(); avail && *avail < min_mem_mib) {
                throw std::runtime_error(fmt::format(
                    "not enough free memory to load the model: {} MiB available, need {} MiB "
                    "(see the recv sweep lines in the log for files kept in {})",
                    *avail, min_mem_mib, recv_root->string()));
            }
        }
        DlSvcClient in;    // subscribe + recv: this thread only
        DlSvcClient out;   // notify only: worker + LLiMa streamer threads, serialized by EventBridge

        // Qualified on purpose: a bare connect() could resolve to the POSIX
        // socket ::connect() through the using-directive above.
        simaai::llima::connect({}, log_dir / (q + ".llima.log"), spdlog::level::info);
        connected = true;

        LoopExit exit = LoopExit::Stopped;
        {
            // model_path == provider root == recv root: the rule that
            // makes the MLA family selector match the reserved ELF paths.
            auto provider = std::make_shared<PcieFileProvider>(*recv_root, args.serve_root, args.model);
            VisionLanguageModel vlm(*recv_root, std::nullopt, std::nullopt, provider);
            VlmGenerator generator(vlm, *recv_root, args.image_serve_root);
            BackendLoop loop(in, out, generator);
            loop.subscribe();            // before READY: a prompt with no listener is dropped
            install_signal_handlers();   // after the load: a SIGTERM during the load just ends us
            set_status(writer, status, "ready", "model loaded; waiting for genai.prompt");
            exit = loop.run(g_stop);
            set_status(writer, status, "stopping",
                       exit == LoopExit::DaemonLost ? "lost the pep daemon" : "signal received");
        }   // model and loop are destroyed before disconnect()

        // Sweep BEFORE disconnect(): disconnect() calls spdlog::shutdown(),
        // which drops the default logger, and logging after that crashes.
        // The model is already destroyed here, so its files are closed.
        clean_recv_root(*recv_root, recv_max_age, "exit");
        simaai::llima::disconnect();
        connected = false;
        if (exit == LoopExit::DaemonLost) {
            set_status(writer, status, "failed", "lost the pep daemon", "svc_disconnected");
            return 1;
        }
        set_status(writer, status, "exited", "backend stopped");
        return 0;
    } catch (const std::exception& e) {
        spdlog::error("pcie-genai-backend: {}", e.what());
        set_status(writer, status, "failed", e.what(), "genai_backend");
        // Before disconnect(), for the same reason as above.
        if (recv_root) clean_recv_root(*recv_root, recv_max_age, "exit");
        if (connected) {
            try { simaai::llima::disconnect(); } catch (...) {}
        }
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << '\n' << usage() << '\n';
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "pcie-genai-backend: " << e.what() << '\n';
        return 1;
    }
}
