// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 SiMa.ai

#pragma once

#include <cstddef>
#include <memory>

namespace simaai::llima {

class MlaExecutionSession;
class MLAModelWithBuffer;
class MlaExecutionSegment;
class LanguageModel;

/*
 * LLiMa-private, transactionally materialized ordinary-decode plan.
 *
 * This type deliberately lives outside installed mla_model.hpp: ProcessMLA,
 * Neat and general LLiMa callers must see only the generic segment API. The
 * implementation stores all jobs in one flat arena and only two integers per
 * position. seal() binds every entry while the session is exclusively idle,
 * then publishes the arena with package/adapter and exact parent-allocation
 * generations. A stale plan is rejected before the first submit.
 */
class MlaExecutionPlan {
    friend class MLAModelWithBuffer;
    friend class MlaExecutionSegment;
    friend class LanguageModel;

  public:
    explicit MlaExecutionPlan(
        std::shared_ptr<MlaExecutionSession> session
    );
    ~MlaExecutionPlan();
    MlaExecutionPlan(MlaExecutionPlan&&) noexcept;
    MlaExecutionPlan& operator=(MlaExecutionPlan&&) noexcept;
    MlaExecutionPlan(const MlaExecutionPlan&) = delete;
    MlaExecutionPlan& operator=(const MlaExecutionPlan&) = delete;

    void begin_position();
    void end_position();
    void seal();
    [[nodiscard]] bool valid(std::size_t position) const noexcept;
    [[nodiscard]] std::size_t position_count() const noexcept;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}  // namespace simaai::llima
