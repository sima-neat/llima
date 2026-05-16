#ifndef _SIMA_LLIMA_NLOHMANN_OPTIONAL_
#define _SIMA_LLIMA_NLOHMANN_OPTIONAL_

#include <optional>

#include <nlohmann/json.hpp>

namespace nlohmann {

template <typename T>
struct adl_serializer<std::optional<T>> {
    static void to_json(json& j, const std::optional<T>& value) {
        if (value.has_value()) {
            j = *value;
        } else {
            j = nullptr;
        }
    }

    static void from_json(const json& j, std::optional<T>& value) {
        if (j.is_null()) {
            value = std::nullopt;
        } else {
            value = j.get<T>();
        }
    }
};

} // namespace nlohmann

#endif
