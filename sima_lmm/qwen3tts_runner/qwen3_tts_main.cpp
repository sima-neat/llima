#include "qwen3tts_runner.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

using simaai::llima::connect_mla_rt;
using simaai::llima::disconnect_mla_rt;
using simaai::llima::qwen3tts::Qwen3TtsRunner;
using simaai::llima::qwen3tts::RequestOptions;

namespace {
struct Args {
    std::filesystem::path model_dir, components_dir;
    std::filesystem::path out_wav{"qwen3tts.wav"};
    std::filesystem::path report{"qwen3tts.json"};
    RequestOptions request;
    uint32_t warmup_runs{1}, timed_runs{1};
    bool preload_models{};
};
void usage() {
    std::cout << "qwen3tts --model-dir DIR --components-dir DIR [options]\n"
              << "  --prompt TEXT --speaker NAME --language NAME --max-frames N --seed N\n"
              << "  --no-sample | --sample --subtalker-sample | --subtalker-no-sample --prefill-mode prefix_kv|n1\n"
              << "  --codec-n128 | --codec-n1\n"
              << "  --endpoint-disable | --endpoint-silence-rms F --endpoint-silence-frames N --endpoint-end-pad-frames N\n"
              << "  --warmup-runs N --timed-runs N --preload-model (lazy-compatible) --out-wav FILE --report FILE\n";
}
Args parse(int argc, char** argv) {
    Args args; args.request.prompt = "Hello from Qwen3-TTS on the devkit.";
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string { if (++i >= argc) throw std::runtime_error("Missing value for " + key); return argv[i]; };
        if (key == "--model-dir") args.model_dir = value();
        else if (key == "--components-dir") args.components_dir = value();
        else if (key == "--prompt") args.request.prompt = value();
        else if (key == "--speaker") args.request.speaker = value();
        else if (key == "--language") args.request.language = value();
        else if (key == "--max-frames") args.request.max_frames = std::stoul(value());
        else if (key == "--seed") args.request.seed = std::stoul(value());
        else if (key == "--top-k") args.request.top_k = std::stoul(value());
        else if (key == "--top-p") args.request.top_p = std::stof(value());
        else if (key == "--temperature") args.request.temperature = std::stof(value());
        else if (key == "--repetition-penalty") args.request.repetition_penalty = std::stof(value());
        else if (key == "--subtalker-top-k") args.request.subtalker_top_k = std::stoul(value());
        else if (key == "--subtalker-top-p") args.request.subtalker_top_p = std::stof(value());
        else if (key == "--subtalker-temperature") args.request.subtalker_temperature = std::stof(value());
        else if (key == "--prefill-mode") args.request.prefill_mode = value();
        else if (key == "--codec-n128") args.request.codec_n128_hybrid = true;
        else if (key == "--codec-n1") args.request.codec_n128_hybrid = false;
        else if (key == "--endpoint-disable") args.request.streaming_endpoint = false;
        else if (key == "--endpoint-silence-rms") args.request.endpoint_silence_rms = std::stof(value());
        else if (key == "--endpoint-silence-frames") args.request.endpoint_silence_frames = std::stoul(value());
        else if (key == "--endpoint-end-pad-frames") args.request.endpoint_end_pad_frames = std::stoul(value());
        else if (key == "--warmup-runs") args.warmup_runs = std::stoul(value());
        else if (key == "--timed-runs") args.timed_runs = std::stoul(value());
        else if (key == "--out-wav") args.out_wav = value();
        else if (key == "--report") args.report = value();
        else if (key == "--preload-model") args.preload_models = false; // Raw ELF models remain lazily bound; warmup retains them.
        else if (key == "--no-sample") args.request.do_sample = false;
        else if (key == "--sample") args.request.do_sample = true;
        else if (key == "--subtalker-no-sample") args.request.subtalker_do_sample = false;
        else if (key == "--subtalker-sample") args.request.subtalker_do_sample = true;
        else if (key == "--help") { usage(); std::exit(0); }
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (args.model_dir.empty() || args.components_dir.empty()) throw std::runtime_error("--model-dir and --components-dir are required");
    args.request.output_wav = args.out_wav;
    if (args.request.prefill_mode != "prefix_kv" && args.request.prefill_mode != "n1") throw std::runtime_error("--prefill-mode must be prefix_kv or n1");
    return args;
}
nlohmann::json metrics_json(const simaai::llima::qwen3tts::RunMetrics& m) {
    return {{"prompt_tokens",m.prompt_tokens},{"frames",m.frames},{"prompt_time",m.prompt_time},{"generation_time",m.generation_time},{"code_predictor_time",m.code_predictor_time},{"code_predictor_mla_time",m.code_predictor_mla_time},{"backbone_feedback_time",m.backbone_feedback_time},{"backbone_decode_time",m.backbone_decode_time},{"backbone_decode_mla_time",m.backbone_decode_mla_time},{"codec_to_wav_time",m.codec_to_wav_time},{"wav_write_time",m.wav_write_time},{"ttft",m.ttft},{"ttf_frame",m.ttf_frame},{"ttfa",m.ttfa},{"e2e_time",m.e2e_time},{"codec_tail_uploads",m.codec_tail_uploads},{"codec_tail_downloads",m.codec_tail_downloads},{"codec_tail_chunks",m.codec_tail_chunks},{"prefix_kv_reused",m.prefix_kv_reused},{"prefix_kv_device_resident",m.prefix_kv_device_resident},{"codec_n128_hybrid",m.codec_n128_hybrid},{"prefix_kv_static_tokens",m.prefix_kv_static_tokens},{"generated_frames_before_endpoint",m.generated_frames_before_endpoint},{"endpoint_enabled",m.endpoint_enabled},{"endpoint_triggered",m.endpoint_triggered},{"endpoint_trigger_frame",m.endpoint_trigger_frame},{"endpoint_retained_pad_frames",m.endpoint_retained_pad_frames},{"endpoint_discarded_confirmation_frames",m.endpoint_discarded_confirmation_frames},{"endpoint_silence_rms_threshold",m.endpoint_silence_rms_threshold},{"endpoint_prefix_rms",m.endpoint_prefix_rms},{"frames_sha256",m.frames_sha256},{"input_ids_sha256",m.input_ids_sha256},{"prefill_sha256",m.prefill_sha256},{"backbone_prefill_hidden_sha256",m.backbone_prefill_hidden_sha256},{"cp_initial_input_sha256",m.cp_initial_input_sha256},{"cp_codebook0_input_sha256",m.cp_codebook0_input_sha256},{"cp_codebook0_logits_sha256",m.cp_codebook0_logits_sha256},{"codec_prefix_sha256",m.codec_prefix_sha256},{"codec_tail_input_sha256",m.codec_tail_input_sha256},{"wav_path",m.wav_path.string()}};
}
nlohmann::json timing_summary(const nlohmann::json& runs) {
    nlohmann::json summary = nlohmann::json::object();
    for (const char* field : {"prompt_time", "generation_time", "code_predictor_time", "code_predictor_mla_time", "backbone_feedback_time", "backbone_decode_time", "backbone_decode_mla_time", "codec_to_wav_time", "ttft", "ttf_frame", "ttfa", "e2e_time"}) {
        std::vector<double> values;
        for (const auto& run : runs) values.push_back(run.at(field).get<double>());
        std::sort(values.begin(), values.end());
        const auto percentile = [&values](double p) {
            const auto index = static_cast<size_t>(std::ceil(p * values.size())) - 1;
            return values.at(std::min(index, values.size() - 1));
        };
        summary[field] = {{"min", values.front()}, {"median", percentile(0.5)}, {"p95", percentile(0.95)}};
    }
    return summary;
}
}
int main(int argc, char** argv) {
    try {
        const Args args = parse(argc, argv);
        connect_mla_rt({});
        Qwen3TtsRunner engine(args.model_dir, args.components_dir, args.preload_models);
        engine.initialize();
        // Match the working Python launcher: seed once, then let warmup advance the
        // code-predictor PCG stream before the timed request is rendered.
        engine.set_seed(args.request.seed);
        for (uint32_t i=0; i<args.warmup_runs; ++i) { auto request=args.request; request.output_wav.clear(); engine.run(request); }
        nlohmann::json report; report["paths"]={{"model_dir",std::filesystem::weakly_canonical(args.model_dir).string()},{"components_dir",std::filesystem::weakly_canonical(args.components_dir).string()},{"raw_elf_dir",std::filesystem::weakly_canonical(args.model_dir / "mpk").string()},{"raw_tail_contract",std::filesystem::weakly_canonical(args.model_dir / "devkit" / "codec_tail_raw_mla_contract.json").string()}}; report["request"]={{"prompt",args.request.prompt},{"speaker",args.request.speaker},{"language",args.request.language},{"seed",args.request.seed},{"max_frames",args.request.max_frames},{"do_sample",args.request.do_sample},{"subtalker_do_sample",args.request.subtalker_do_sample},{"prefill_mode",args.request.prefill_mode},{"codec_n128_hybrid",args.request.codec_n128_hybrid},{"streaming_endpoint",args.request.streaming_endpoint},{"endpoint_silence_rms",args.request.endpoint_silence_rms},{"endpoint_silence_frames",args.request.endpoint_silence_frames},{"endpoint_end_pad_frames",args.request.endpoint_end_pad_frames}}; report["warmup_runs"]=args.warmup_runs; report["timed_runs"]=nlohmann::json::array();
        for (uint32_t i=0; i<args.timed_runs; ++i) {
            auto request=args.request;
            if (args.timed_runs>1) request.output_wav=args.out_wav.parent_path()/(args.out_wav.stem().string()+".timed"+std::to_string(i+1)+args.out_wav.extension().string());
            auto result=engine.run(request);
            auto timed = metrics_json(result.metrics);
            timed["frames"] = nlohmann::json::array();
            for (const auto& frame : result.frames) timed["frames"].push_back(frame);
            report["timed_runs"].push_back(std::move(timed));
        }
        report["timing_summary"] = timing_summary(report["timed_runs"]);
        std::ofstream out(args.report); out << report.dump(2) << '\n'; std::cout << report.dump(2) << std::endl;
        engine.finalize(); disconnect_mla_rt(); return 0;
    } catch (const std::exception& e) { std::cerr << "qwen3tts: " << e.what() << std::endl; try { disconnect_mla_rt(); } catch (...) {} return 1; }
}
