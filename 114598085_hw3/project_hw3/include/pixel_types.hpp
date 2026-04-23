#pragma once

#include <compare>
#include <cstdint>

namespace rik_cv
{

struct Gray8
{
    std::uint8_t v{};

    // Gray8 is a scalar luminance value — total ordering is natural.
    // clang-format off
    friend constexpr std::strong_ordering operator<=>(Gray8, Gray8) noexcept = default;
    // clang-format on
};

struct Bgr8
{
    std::uint8_t b{};
    std::uint8_t g{};
    std::uint8_t r{};
};

struct Rgb8
{
    std::uint8_t r{};
    std::uint8_t g{};
    std::uint8_t b{};
};

static_assert(sizeof(Gray8) == 1);
static_assert(sizeof(Bgr8) == 3);
static_assert(sizeof(Rgb8) == 3);

}  // namespace rik_cv
