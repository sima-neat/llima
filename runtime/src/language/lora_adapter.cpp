#include "language/language_model_state.hpp"
#include "mla/runtime_context.hpp"

#include <array>
#include <cnpy.h>
#include <regex>
#include <stdexcept>

namespace simaai::llima {

namespace {

auto adapter_model_maps(LanguageWorkspace& workspace) {
    return std::array{
        &workspace.pre_model_map, &workspace.post_model_map, &workspace.router_model_map,
        &workspace.expert_model_map, &workspace.conv_model_map, &workspace.conv_final_model_map,
        &workspace.linear_model_map
    };
}

} // namespace

void LanguageModel::_bind_adapter(const std::shared_ptr<LoRAAdapter>& adapter) {
    if (_workspace->bound_adapter == adapter) return;
    std::lock_guard lock(RuntimeContext::instance().execution_mutex);
    using Buffers = std::map<std::string, MLABuffer*>;
    struct Change {
        MLAModelWithBuffer& model;
        Buffers buffers;
        std::shared_ptr<const RuntimeRelocation> next;
        std::shared_ptr<const RuntimeRelocation> previous;
    };
    std::vector<Change> changes;
    const Buffers empty;
    std::map<size_t, std::shared_ptr<const RuntimeRelocation>> prepared;
    for (auto* models : adapter_model_maps(*_workspace)) {
        for (auto& [key, model] : *models) {
            const Buffers* desired = &empty;
            if (adapter) {
                const auto found = adapter->bindings.find(model._model_idx);
                if (found != adapter->bindings.end()) desired = &found->second;
            }
            if (model._reloc_buffers == *desired) continue;
            // A grouped final post can share its ELF with the single-token path.
            auto [slot, first] = prepared.try_emplace(model._model_idx);
            if (first) slot->second = model._prepare_reloc(*desired, true);
            changes.push_back({model, *desired, slot->second, model._program->active_reloc});
        }
    }

    // All ELF keys and ports are valid. Publish wrapper bindings only after every
    // native update succeeds. A failed driver call leaves existing requests retryable.
    size_t applied = 0;
    try {
        for (auto& change : changes) {
            change.model._program->activate_relocation(change.next);
            ++applied;
        }
    } catch (...) {
        while (applied > 0) {
            auto& change = changes[--applied];
            try {
                change.model._program->activate_relocation(change.previous);
            } catch (const std::exception& error) {
                // The wrapper still owns its original binding and will retry on submission.
                _logger->error("Could not restore MLA relocation: {}", error.what());
            }
        }
        throw;
    }
    for (auto& change : changes) {
        change.model._reloc_buffers.swap(change.buffers);
        change.model._relocation = std::move(change.next);
    }
    _workspace->bound_adapter = adapter;
}

void LanguageModel::set_reloc(const std::string& reloc_name) {
    if (_workspace->bound_adapter && _workspace->bound_adapter->name == reloc_name) return;
    if (_files->pulls_files())
        throw std::runtime_error("LoRA requires a local model directory; the asset provider cannot list npy_files");
    std::lock_guard execution_lock(RuntimeContext::instance().execution_mutex);
    std::erase_if(_resources->adapters, [](const auto& entry) { return entry.second.expired(); });
    const auto found = _resources->adapters.find(reloc_name);
    auto adapter = found == _resources->adapters.end() ? nullptr : found->second.lock();
    if (!adapter) {
        adapter = std::make_shared<LoRAAdapter>();
        adapter->name = reloc_name;
        const auto directory = _devkit_dir.parent_path() / "npy_files" / reloc_name;
        std::map<std::string, size_t> active_graphs;
        const auto& paths = RuntimeContext::instance().paths;
        for (auto* models : adapter_model_maps(*_workspace)) {
            for (const auto& [key, model] : *models)
                active_graphs.emplace(paths.at(model._model_idx).filename().string(), model._model_idx);
        }
        // Compiler weight-map names identify the ELF before the relocation section.
        static const std::regex graph_pattern(
            R"((_n(\d+)_(?:pre_layer\d+|post_layer\d+(?:_expert\d+|_conv_final)?|router_layer\d+|layer\d+_(?:conv|linear)))(?!_expert|_conv_final)(?=[_.]))");
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.path().extension() != ".npy") continue;
            std::smatch match;
            const auto filename = entry.path().filename().string();
            if (!std::regex_search(filename, match, graph_pattern))
                throw std::invalid_argument("Invalid file name for relocation: " + filename);
            const auto elf = _cfg.language_model_name + match[1].str() + "_stage1_mla.elf";
            const auto graph = active_graphs.find(elf);
            if (graph == active_graphs.end()) {
                const auto width = std::stoul(match[2]);
                // Packages may retain compiled group graphs which this configuration skips.
                if (width > 1 && width != _cfg.lm_cfg.get_single_num_tokens()
                    && (!_use_group_token_models || width == _cfg.pipeline_cfg.input_token_group_size)
                    && _files->exists("elf_files/" + elf)) continue;
                throw std::invalid_argument("LoRA names a graph not present in the model: " + filename);
            }
            const auto tensor = cnpy::npy_load(_files->get_path("npy_files/" + reloc_name + "/" + filename));
            if (tensor.word_size != 1 || tensor.fortran_order || tensor.num_vals == 0)
                throw std::invalid_argument("LoRA tensor must contain contiguous row-major bytes: " + filename);
            const auto key = entry.path().stem().string();
            // Compiler-produced arrays already contain the packed MLA section bytes.
            auto [buffer, inserted] = adapter->buffers.try_emplace(
                key, key, std::vector<size_t>{tensor.num_bytes()}, "int8", false);
            buffer->second.allocate();
            buffer->second.upload(tensor.data<void>());
            adapter->bindings[graph->second].emplace(key, &buffer->second);
        }
        if (adapter->bindings.empty())
            throw std::runtime_error("No LoRA relocation tensors found in " + directory.string());
    }
    auto& cached = _resources->adapters[reloc_name];
    _bind_adapter(adapter);
    cached = adapter;
    invalidate_prompt_reuse();
}

void LanguageModel::unset_reloc() {
    if (!_workspace->bound_adapter) return;
    std::lock_guard execution_lock(RuntimeContext::instance().execution_mutex);
    _bind_adapter(nullptr);
    invalidate_prompt_reuse();
}

} // namespace simaai::llima
