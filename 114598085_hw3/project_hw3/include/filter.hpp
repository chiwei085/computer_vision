#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>

#include "algorithm.hpp"
#include "image.hpp"
#include "numeric_utils.hpp"
#include "padding.hpp"
#include "pixel_types.hpp"

namespace rik_cv
{

struct filter_error : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

template <class Padding>
concept GrayPadding =
    std::regular_invocable<Padding&, ImageView<const Gray8>, int, int> &&
    std::convertible_to<
        std::invoke_result_t<Padding&, ImageView<const Gray8>, int, int>,
        Gray8>;

namespace detail
{

inline constexpr int median_small_radius_threshold = 2;

[[nodiscard]] constexpr int strided_extent(int input_extent,
                                           int stride) noexcept {
    return input_extent <= 0 ? 0 : (input_extent + stride - 1) / stride;
}

[[nodiscard]] constexpr int padded_extent(int input_extent, int kernel_size,
                                          int pad_size, int stride) noexcept {
    if (input_extent <= 0 || kernel_size <= 0 || pad_size < 0 || stride <= 0) {
        return 0;
    }

    const int numerator = input_extent + 2 * pad_size - kernel_size;
    if (numerator < 0) {
        return 0;
    }

    return numerator / stride + 1;
}

template <int Radius>
    requires(Radius >= 1)
[[nodiscard]] constexpr double default_gaussian_sigma() noexcept {
    return static_cast<double>(Radius) / 2.0 + 0.5;
}

template <int Radius>
    requires(Radius >= 1)
[[nodiscard]] inline std::array<double, 2 * Radius + 1> make_gaussian_kernel(
    double sigma) {
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        throw filter_error("[gaussian_filter] sigma must be finite and > 0");
    }

    constexpr int ksize = 2 * Radius + 1;
    std::array<double, ksize> kernel{};
    const double inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);

    double sum = 0.0;
    for (int i = -Radius; i <= Radius; ++i) {
        const double x = static_cast<double>(i);
        const double w = std::exp(-(x * x) * inv_2sigma2);
        kernel[static_cast<std::size_t>(i + Radius)] = w;
        sum += w;
    }

    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw filter_error("[gaussian_filter] invalid kernel normalization");
    }

    for (double& w : kernel) {
        w /= sum;
    }

    return kernel;
}

template <class Histogram, class Padding>
    requires GrayPadding<Padding>
inline void update_histogram_region(Histogram& hist, ImageView<const Gray8> src,
                                    int row_begin, int row_end, int col_begin,
                                    int col_end, Padding& padding, int delta) {
    for (int y = row_begin; y < row_end; ++y) {
        for (int x = col_begin; x < col_end; ++x) {
            const Gray8 sample = std::invoke(padding, src, y, x);
            hist[static_cast<std::size_t>(sample.v)] += delta;
        }
    }
}

template <class Histogram>
[[nodiscard]] inline Gray8 histogram_median_value(const Histogram& hist,
                                                  int sample_count) {
    const int target = sample_count / 2;
    int cumulative = 0;
    for (std::size_t i = 0; i < hist.size(); ++i) {
        cumulative += hist[i];
        if (cumulative > target) {
            return Gray8{static_cast<std::uint8_t>(i)};
        }
    }
    return Gray8{255};
}

template <int Radius, class Padding>
    requires GrayPadding<Padding>
inline void median_filter_rebuild_window(ImageView<const Gray8> src,
                                         ImageView<Gray8> dst, int pad_x,
                                         int pad_y, int stride_x, int stride_y,
                                         Padding& padding) {
    constexpr int ksize = 2 * Radius + 1;
    constexpr auto window_n =
        static_cast<std::size_t>(ksize) * static_cast<std::size_t>(ksize);
    std::array<Gray8, window_n> scratch;

    for (int y = 0; y < dst.height(); ++y) {
        for (int x = 0; x < dst.width(); ++x) {
            std::size_t idx = 0;
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    const int src_y = y * stride_y + ky - pad_y;
                    const int src_x = x * stride_x + kx - pad_x;
                    scratch[idx++] = std::invoke(padding, src, src_y, src_x);
                }
            }
            dst(y, x) = median_value(scratch.data(), scratch.data() + window_n);
        }
    }
}

template <int Radius, class Padding>
    requires GrayPadding<Padding>
inline void median_filter_sliding_histogram(ImageView<const Gray8> src,
                                            ImageView<Gray8> dst, int pad_x,
                                            int pad_y, int stride_x,
                                            int stride_y, Padding& padding) {
    constexpr int ksize = 2 * Radius + 1;
    constexpr int sample_count = ksize * ksize;

    using histogram_type = std::array<int, 256>;
    histogram_type row_hist{};

    const int first_col_begin = -pad_x;
    const int first_col_end = first_col_begin + ksize;

    if (dst.height() > 0) {
        const int first_row_begin = -pad_y;
        const int first_row_end = first_row_begin + ksize;
        update_histogram_region(row_hist, src, first_row_begin, first_row_end,
                                first_col_begin, first_col_end, padding, +1);
    }

    for (int y = 0; y < dst.height(); ++y) {
        if (y > 0) {
            const int old_row_begin = (y - 1) * stride_y - pad_y;
            const int old_row_end = old_row_begin + ksize;
            const int new_row_begin = y * stride_y - pad_y;
            const int new_row_end = new_row_begin + ksize;

            update_histogram_region(row_hist, src, old_row_begin,
                                    std::min(new_row_begin, old_row_end),
                                    first_col_begin, first_col_end, padding,
                                    -1);
            update_histogram_region(
                row_hist, src, std::max(old_row_end, new_row_begin),
                new_row_end, first_col_begin, first_col_end, padding, +1);
        }

        histogram_type hist = row_hist;
        auto dst_row = dst.row_span(y);
        const int row_begin = y * stride_y - pad_y;
        const int row_end = row_begin + ksize;

        for (int x = 0; x < dst.width(); ++x) {
            dst_row[x] = histogram_median_value(hist, sample_count);
            if (x + 1 >= dst.width()) {
                continue;
            }

            const int old_col_begin = x * stride_x - pad_x;
            const int old_col_end = old_col_begin + ksize;
            const int new_col_begin = (x + 1) * stride_x - pad_x;
            const int new_col_end = new_col_begin + ksize;

            update_histogram_region(
                hist, src, row_begin, row_end, old_col_begin,
                std::min(new_col_begin, old_col_end), padding, -1);
            update_histogram_region(hist, src, row_begin, row_end,
                                    std::max(old_col_end, new_col_begin),
                                    new_col_end, padding, +1);
        }
    }
}

}  // namespace detail

template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
inline void median_filter(ImageView<const Gray8> src, ImageView<Gray8> dst,
                          int pad_x, int pad_y, int stride_x = 1,
                          int stride_y = 1, Padding padding = {}) {
    if (pad_x < 0 || pad_y < 0) {
        throw filter_error("[median_filter] padding size must be non-negative");
    }
    if (stride_x <= 0 || stride_y <= 0) {
        throw filter_error("[median_filter] stride must be positive");
    }

    constexpr int ksize = 2 * Radius + 1;
    if (dst.width() !=
            detail::padded_extent(src.width(), ksize, pad_x, stride_x) ||
        dst.height() !=
            detail::padded_extent(src.height(), ksize, pad_y, stride_y)) {
        throw filter_error(
            "[median_filter] dst extent does not match padded output");
    }

    if constexpr (Radius <= detail::median_small_radius_threshold) {
        detail::median_filter_rebuild_window<Radius>(
            src, dst, pad_x, pad_y, stride_x, stride_y, padding);
    }
    else {
        detail::median_filter_sliding_histogram<Radius>(
            src, dst, pad_x, pad_y, stride_x, stride_y, padding);
    }
}

template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
inline void median_filter(ImageView<const Gray8> src, ImageView<Gray8> dst,
                          int stride_x = 1, int stride_y = 1,
                          Padding padding = {}) {
    median_filter<Radius>(src, dst, Radius, Radius, stride_x, stride_y,
                          padding);
}

template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
[[nodiscard]] inline Image<Gray8> median_filter(ImageView<const Gray8> src,
                                                int pad_x, int pad_y,
                                                int stride_x = 1,
                                                int stride_y = 1,
                                                Padding padding = {}) {
    constexpr int ksize = 2 * Radius + 1;
    auto dst = make_image<Gray8>(
        detail::padded_extent(src.width(), ksize, pad_x, stride_x),
        detail::padded_extent(src.height(), ksize, pad_y, stride_y));
    median_filter<Radius>(src, dst.view(), pad_x, pad_y, stride_x, stride_y,
                          padding);
    return dst;
}

template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
[[nodiscard]] inline Image<Gray8> median_filter(ImageView<const Gray8> src,
                                                int stride_x = 1,
                                                int stride_y = 1,
                                                Padding padding = {}) {
    return median_filter<Radius>(src, Radius, Radius, stride_x, stride_y,
                                 padding);
}

// gaussian_filter — in-place overload with explicit sigma.
//
// Applies a separable Gaussian blur using a compile-time radius and runtime
// sigma. The implementation performs a horizontal 1D pass into a temporary
// buffer, then a vertical 1D pass into the destination image.
template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
inline void gaussian_filter(ImageView<const Gray8> src, ImageView<Gray8> dst,
                            double sigma, int pad_x, int pad_y,
                            int stride_x = 1, int stride_y = 1,
                            Padding padding = {}) {
    if (pad_x < 0 || pad_y < 0) {
        throw filter_error(
            "[gaussian_filter] padding size must be non-negative");
    }
    if (stride_x <= 0 || stride_y <= 0) {
        throw filter_error("[gaussian_filter] stride must be positive");
    }

    constexpr int ksize = 2 * Radius + 1;
    if (dst.width() !=
            detail::padded_extent(src.width(), ksize, pad_x, stride_x) ||
        dst.height() !=
            detail::padded_extent(src.height(), ksize, pad_y, stride_y)) {
        throw filter_error(
            "[gaussian_filter] dst extent does not match padded output");
    }

    const auto kernel = detail::make_gaussian_kernel<Radius>(sigma);
    const int required_rows =
        dst.height() == 0 ? 0 : (dst.height() - 1) * stride_y + ksize;
    auto horizontal = make_image<double>(dst.width(), required_rows);

    for (int py = 0; py < required_rows; ++py) {
        const int src_y = py - pad_y;
        auto horizontal_row = horizontal.view().row_span(py);
        for (int x = 0; x < dst.width(); ++x) {
            double sum = 0.0;
            const int src_origin_x = x * stride_x - pad_x;
            for (int kx = 0; kx < ksize; ++kx) {
                const int src_x = src_origin_x + kx;
                const Gray8 sample = std::invoke(padding, src, src_y, src_x);
                sum += static_cast<double>(sample.v) *
                       kernel[static_cast<std::size_t>(kx)];
            }
            horizontal_row[x] = sum;
        }
    }

    for (int y = 0; y < dst.height(); ++y) {
        auto dst_row = dst.row_span(y);
        const int padded_origin_y = y * stride_y;
        for (int x = 0; x < dst.width(); ++x) {
            double sum = 0.0;
            for (int ky = 0; ky < ksize; ++ky) {
                sum += horizontal(padded_origin_y + ky, x) *
                       kernel[static_cast<std::size_t>(ky)];
            }
            dst_row[x] = Gray8{detail::clamp_to_u8(sum)};
        }
    }
}

// gaussian_filter — returning overload with explicit sigma.
template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
inline void gaussian_filter(ImageView<const Gray8> src, ImageView<Gray8> dst,
                            double sigma, int stride_x = 1, int stride_y = 1,
                            Padding padding = {}) {
    gaussian_filter<Radius>(src, dst, sigma, Radius, Radius, stride_x, stride_y,
                            padding);
}

template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
[[nodiscard]] inline Image<Gray8> gaussian_filter(ImageView<const Gray8> src,
                                                  double sigma, int pad_x,
                                                  int pad_y, int stride_x = 1,
                                                  int stride_y = 1,
                                                  Padding padding = {}) {
    constexpr int ksize = 2 * Radius + 1;
    auto dst = make_image<Gray8>(
        detail::padded_extent(src.width(), ksize, pad_x, stride_x),
        detail::padded_extent(src.height(), ksize, pad_y, stride_y));
    gaussian_filter<Radius>(src, dst.view(), sigma, pad_x, pad_y, stride_x,
                            stride_y, padding);
    return dst;
}

template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
[[nodiscard]] inline Image<Gray8> gaussian_filter(ImageView<const Gray8> src,
                                                  double sigma,
                                                  int stride_x = 1,
                                                  int stride_y = 1,
                                                  Padding padding = {}) {
    return gaussian_filter<Radius>(src, sigma, Radius, Radius, stride_x,
                                   stride_y, padding);
}

// gaussian_filter — in-place overload using a radius-based default sigma.
template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
inline void gaussian_filter(ImageView<const Gray8> src, ImageView<Gray8> dst,
                            int pad_x, int pad_y, int stride_x = 1,
                            int stride_y = 1, Padding padding = {}) {
    gaussian_filter<Radius>(src, dst, detail::default_gaussian_sigma<Radius>(),
                            pad_x, pad_y, stride_x, stride_y, padding);
}

template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
inline void gaussian_filter(ImageView<const Gray8> src, ImageView<Gray8> dst,
                            int stride_x = 1, int stride_y = 1,
                            Padding padding = {}) {
    gaussian_filter<Radius>(src, dst, detail::default_gaussian_sigma<Radius>(),
                            Radius, Radius, stride_x, stride_y, padding);
}

template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
[[nodiscard]] inline Image<Gray8> gaussian_filter(ImageView<const Gray8> src,
                                                  int pad_x, int pad_y,
                                                  int stride_x = 1,
                                                  int stride_y = 1,
                                                  Padding padding = {}) {
    return gaussian_filter<Radius>(src,
                                   detail::default_gaussian_sigma<Radius>(),
                                   pad_x, pad_y, stride_x, stride_y, padding);
}

template <int Radius, class Padding = reflect_101_padding>
    requires(Radius >= 1) && GrayPadding<Padding>
[[nodiscard]] inline Image<Gray8> gaussian_filter(ImageView<const Gray8> src,
                                                  int stride_x = 1,
                                                  int stride_y = 1,
                                                  Padding padding = {}) {
    return gaussian_filter<Radius>(src, Radius, Radius, stride_x, stride_y,
                                   padding);
}

[[nodiscard]] inline Image<Gray8> gaussian_filter(ImageView<const Gray8> src,
                                                  int radius, double sigma) {
    switch (radius) {
        case 1:
            return gaussian_filter<1>(src, sigma);
        case 2:
            return gaussian_filter<2>(src, sigma);
        case 3:
            return gaussian_filter<3>(src, sigma);
        default:
            throw filter_error("[gaussian_filter] radius must be 1, 2, or 3");
    }
}

template <int MedianRadius, int GaussianRadius,
          class Padding = reflect_101_padding>
    requires(MedianRadius >= 1) && (GaussianRadius >= 1) && GrayPadding<Padding>
[[nodiscard]] inline Image<Gray8> median_then_gaussian_filter(
    ImageView<const Gray8> src, int median_pad_x, int median_pad_y,
    int gaussian_pad_x, int gaussian_pad_y, Padding padding = {}) {
    auto median_result = median_filter<MedianRadius, Padding>(
        src, median_pad_x, median_pad_y, 1, 1, padding);
    return gaussian_filter<GaussianRadius, Padding>(
        median_result.as_const_view(), gaussian_pad_x, gaussian_pad_y, 1, 1,
        padding);
}

template <int GaussianRadius, int MedianRadius,
          class Padding = reflect_101_padding>
    requires(GaussianRadius >= 1) && (MedianRadius >= 1) && GrayPadding<Padding>
[[nodiscard]] inline Image<Gray8> gaussian_then_median_filter(
    ImageView<const Gray8> src, int gaussian_pad_x, int gaussian_pad_y,
    int median_pad_x, int median_pad_y, Padding padding = {}) {
    auto gaussian_result = gaussian_filter<GaussianRadius, Padding>(
        src, gaussian_pad_x, gaussian_pad_y, 1, 1, padding);
    return median_filter<MedianRadius, Padding>(gaussian_result.as_const_view(),
                                                median_pad_x, median_pad_y, 1,
                                                1, padding);
}

}  // namespace rik_cv
