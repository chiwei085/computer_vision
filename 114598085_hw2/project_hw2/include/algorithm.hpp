#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <iterator>

namespace rik_cv
{

namespace detail
{

template <class T>
concept byte_pixel =
    std::is_trivially_copyable_v<T> && sizeof(T) == 1 && requires(T t) {
        { t.v } -> std::convertible_to<std::uint8_t>;
    };

}  // namespace detail

template <std::forward_iterator Iter>
    requires detail::byte_pixel<std::iter_value_t<Iter>>
[[nodiscard]] constexpr std::iter_value_t<Iter> median_value(
    Iter first, Iter last) noexcept {
    assert(first != last);

    using T = std::iter_value_t<Iter>;

    std::array<std::uint16_t, 256> hist{};
    std::size_t n = 0;
    for (auto it = first; it != last; ++it, ++n) {
        ++hist[static_cast<std::uint8_t>(it->v)];
    }

    const std::size_t mid = n / 2;
    std::size_t cumulative = 0;
    for (int i = 0; i < 256; ++i) {
        cumulative += hist[i];
        if (cumulative > mid) {
            return T{static_cast<std::uint8_t>(i)};
        }
    }
    return T{static_cast<std::uint8_t>(255)};  // unreachable for valid range
}

template <detail::byte_pixel T>
[[nodiscard]] T median_value(const T* first, const T* last) noexcept {
    assert(first != last);

    std::array<std::uint16_t, 256> hist{};
    std::size_t n = 0;
    for (const T* it = first; it != last; ++it, ++n) {
        ++hist[static_cast<std::uint8_t>(it->v)];
    }

    const std::size_t mid = n / 2;
    std::size_t cumulative = 0;
    for (int i = 0; i < 256; ++i) {
        cumulative += hist[i];
        if (cumulative > mid) {
            return T{static_cast<std::uint8_t>(i)};
        }
    }
    return T{static_cast<std::uint8_t>(255)};  // unreachable for valid range
}

}  // namespace rik_cv
