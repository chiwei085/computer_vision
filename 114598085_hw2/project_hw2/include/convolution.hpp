#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "image.hpp"
#include "numeric_utils.hpp"
#include "padding.hpp"
#include "pixel_types.hpp"

namespace rik_cv
{

struct convolution_error : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

namespace detail
{

template <class Kernel>
concept kernel_container =
    std::ranges::sized_range<Kernel> && requires(const Kernel& kernel) {
    std::data(kernel);
};

template <class Kernel>
using kernel_value_t =
    std::remove_cvref_t<decltype(*std::data(std::declval<const Kernel&>()))>;

[[nodiscard]] constexpr int conv_output_extent(int input_extent,
                                               int stride) noexcept {
    return input_extent <= 0 ? 0 : (input_extent + stride - 1) / stride;
}

}  // namespace detail

template <detail::kernel_container Kernel, class Padding = reflect_101_padding>
requires std::convertible_to<detail::kernel_value_t<Kernel>, double>&&
    std::regular_invocable<Padding&, ImageView<const Gray8>, int, int>&&
        std::convertible_to<
            std::invoke_result_t<Padding&, ImageView<const Gray8>, int, int>,
            Gray8> inline void
        conv(ImageView<const Gray8> src, ImageView<Gray8> dst,
             const Kernel& kernel, int kernel_width, int kernel_height,
             int stride_x = 1, int stride_y = 1, Padding padding = {}) {
    if (kernel_width <= 0 || kernel_height <= 0) {
        throw convolution_error("[conv] kernel dimensions must be positive");
    }
    if (stride_x <= 0 || stride_y <= 0) {
        throw convolution_error("[conv] stride must be positive");
    }
    if (static_cast<std::size_t>(kernel_width) *
            static_cast<std::size_t>(kernel_height) !=
        std::size(kernel)) {
        throw convolution_error("[conv] kernel size does not match dimensions");
    }
    if (dst.width() != detail::conv_output_extent(src.width(), stride_x) ||
        dst.height() != detail::conv_output_extent(src.height(), stride_y)) {
        throw convolution_error(
            "[conv] destination extent does not match stride-adjusted output");
    }

    const auto* kernel_data = std::data(kernel);
    const int kernel_anchor_x = (kernel_width - 1) / 2;
    const int kernel_anchor_y = (kernel_height - 1) / 2;

    // Padding is handled by a strategy object so callers can change border
    // behavior without changing the convolution logic itself.
    // The default strategy follows OpenCV-style reflect-101 border handling.
    for (int y = 0; y < dst.height(); ++y) {
        auto dst_row = dst.row_span(y);

        for (int x = 0; x < dst.width(); ++x) {
            double sum = 0.0;
            const int src_origin_y = y * stride_y;
            const int src_origin_x = x * stride_x;

            for (int ky = 0; ky < kernel_height; ++ky) {
                for (int kx = 0; kx < kernel_width; ++kx) {
                    const int src_y = src_origin_y + ky - kernel_anchor_y;
                    const int src_x = src_origin_x + kx - kernel_anchor_x;
                    const Gray8 sample =
                        std::invoke(padding, src, src_y, src_x);
                    const std::size_t kernel_index =
                        static_cast<std::size_t>(ky) *
                            static_cast<std::size_t>(kernel_width) +
                        static_cast<std::size_t>(kx);

                    sum += static_cast<double>(sample.v) *
                           static_cast<double>(kernel_data[kernel_index]);
                }
            }

            dst_row[x] = Gray8{detail::clamp_to_u8(sum)};
        }
    }
}

template <detail::kernel_container Kernel, class Padding = reflect_101_padding>
requires std::convertible_to<detail::kernel_value_t<Kernel>, double>&&
    std::regular_invocable<Padding&, ImageView<const Gray8>, int, int>&&
        std::convertible_to<
            std::invoke_result_t<Padding&, ImageView<const Gray8>, int, int>,
            Gray8> [[nodiscard]] inline Image<Gray8>
        conv(ImageView<const Gray8> src, const Kernel& kernel, int kernel_width,
             int kernel_height, int stride_x = 1, int stride_y = 1,
             Padding padding = {}) {
    if (stride_x <= 0 || stride_y <= 0) {
        throw convolution_error("[conv] stride must be positive");
    }

    auto dst =
        make_image<Gray8>(detail::conv_output_extent(src.width(), stride_x),
                          detail::conv_output_extent(src.height(), stride_y));
    conv(src, dst.view(), kernel, kernel_width, kernel_height, stride_x,
         stride_y, padding);
    return dst;
}

}  // namespace rik_cv
