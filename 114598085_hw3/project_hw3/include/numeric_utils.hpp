#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rik_cv
{

namespace detail
{

inline constexpr double numeric_pi = 3.14159265358979323846;

[[nodiscard]] inline std::uint8_t clamp_to_u8(double value) noexcept {
    const double rounded = std::round(value);
    const double clamped = std::clamp(rounded, 0.0, 255.0);
    return static_cast<std::uint8_t>(clamped);
}

[[nodiscard]] inline double cyclic_distance(double a, double b,
                                            double period) noexcept {
    double diff = std::fmod(std::abs(a - b), period);
    return std::min(diff, period - diff);
}

[[nodiscard]] inline double angle_distance_pi(double a, double b) noexcept {
    return cyclic_distance(a, b, numeric_pi);
}

}  // namespace detail

}  // namespace rik_cv
