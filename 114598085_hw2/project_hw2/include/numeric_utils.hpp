#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rik_cv
{

namespace detail
{

[[nodiscard]] inline std::uint8_t clamp_to_u8(double value) noexcept {
    const double rounded = std::round(value);
    const double clamped = std::clamp(rounded, 0.0, 255.0);
    return static_cast<std::uint8_t>(clamped);
}

}  // namespace detail

}  // namespace rik_cv
