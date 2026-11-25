#pragma once

#include "generation_types.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace simaai::llima {

// Assigns bounded physical slots to named contexts and the unnamed legacy context.
// The serialized model owner keeps storage and execution separate from this policy.
class CacheSlots {
public:
    struct Selection { size_t index; bool created; };
    explicit CacheSlots(size_t capacity) : used_(checked_capacity(capacity), false) {}

    void check_available(const std::optional<std::string>& id) const {
        validate(id);
        if (!assigned_.contains(id) && assigned_.size() == used_.size())
            throw KVCacheCapacityError("All " + std::to_string(used_.size()) +
                                       " KV-cache slots are assigned; remove a cache before using a new cache_id");
    }
    Selection acquire(const std::optional<std::string>& id) {
        check_available(id);
        if (const auto found = assigned_.find(id); found != assigned_.end())
            return {found->second, false};
        const auto free = std::find(used_.begin(), used_.end(), false);
        const auto index = static_cast<size_t>(free - used_.begin());
        assigned_.emplace(id, index);
        used_[index] = true;
        return {index, true};
    }
    std::optional<size_t> remove(const std::string& id) {
        validate(id);
        const auto found = assigned_.find(id);
        if (found == assigned_.end()) return std::nullopt;
        const auto index = found->second;
        assigned_.erase(found);
        used_[index] = false;
        return index;
    }
    void clear() {
        assigned_.clear();
        std::fill(used_.begin(), used_.end(), false);
    }
    size_t count() const { return assigned_.size(); }
    size_t capacity() const { return used_.size(); }

private:
    static size_t checked_capacity(size_t capacity) {
        if (!capacity) throw std::invalid_argument("max_kv_cache_slots must be positive");
        return capacity;
    }
    static void validate(const std::optional<std::string>& id) {
        if (id && id->empty()) throw std::invalid_argument("cache_id must be non-empty or omitted");
    }
    std::vector<bool> used_;
    std::map<std::optional<std::string>, size_t> assigned_;
};

} // namespace simaai::llima
