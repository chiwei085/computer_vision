#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "image.hpp"
#include "pixel_transform.hpp"
#include "pixel_types.hpp"

namespace rik_cv
{

// This header defines framework-level color conversion rules for rik_cv.
// The conversion formulas and channel reordering behavior follow common image
// processing conventions and are chosen to match widely used behavior such as
// OpenCV's color conversions, but this module itself does not depend on
// OpenCV. All logic below is implemented in terms of rik_cv pixel types,
// Image, and ImageView only.

struct color_trans_error : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

enum class ColorConversion
{
    bgr_to_gray,
    rgb_to_gray,
    gray_to_bgr,
    gray_to_rgb,
    bgr_to_rgb,
    rgb_to_bgr,
};

namespace detail
{

template <class>
inline constexpr bool always_false_v = false;

template <ColorConversion Code>
struct color_conversion_traits;

template <>
struct color_conversion_traits<ColorConversion::bgr_to_gray>
{
    using src_pixel = Bgr8;
    using dst_pixel = Gray8;
};

template <>
struct color_conversion_traits<ColorConversion::rgb_to_gray>
{
    using src_pixel = Rgb8;
    using dst_pixel = Gray8;
};

template <>
struct color_conversion_traits<ColorConversion::gray_to_bgr>
{
    using src_pixel = Gray8;
    using dst_pixel = Bgr8;
};

template <>
struct color_conversion_traits<ColorConversion::gray_to_rgb>
{
    using src_pixel = Gray8;
    using dst_pixel = Rgb8;
};

template <>
struct color_conversion_traits<ColorConversion::bgr_to_rgb>
{
    using src_pixel = Bgr8;
    using dst_pixel = Rgb8;
};

template <>
struct color_conversion_traits<ColorConversion::rgb_to_bgr>
{
    using src_pixel = Rgb8;
    using dst_pixel = Bgr8;
};

}  // namespace detail

[[nodiscard]] inline std::uint8_t rgb_to_gray_value(std::uint8_t r,
                                                    std::uint8_t g,
                                                    std::uint8_t b) {
    // BT.601 luminosity method: Y ~= 0.299 R + 0.587 G + 0.114 B
    const std::uint32_t weighted_sum = 77u * r + 150u * g + 29u * b + 128u;
    return static_cast<std::uint8_t>(weighted_sum >> 8);
}

template <ColorConversion Code, class SrcPixel>
[[nodiscard]] inline auto cvt_color(const SrcPixel& pixel) {
    using expected_src =
        typename detail::color_conversion_traits<Code>::src_pixel;
    static_assert(std::is_same_v<SrcPixel, expected_src>,
                  "source pixel type does not match conversion code");

    if constexpr (Code == ColorConversion::bgr_to_gray) {
        return Gray8{rgb_to_gray_value(pixel.r, pixel.g, pixel.b)};
    }
    else if constexpr (Code == ColorConversion::rgb_to_gray) {
        return Gray8{rgb_to_gray_value(pixel.r, pixel.g, pixel.b)};
    }
    else if constexpr (Code == ColorConversion::gray_to_bgr) {
        return Bgr8{pixel.v, pixel.v, pixel.v};
    }
    else if constexpr (Code == ColorConversion::gray_to_rgb) {
        return Rgb8{pixel.v, pixel.v, pixel.v};
    }
    else if constexpr (Code == ColorConversion::bgr_to_rgb) {
        return Rgb8{pixel.r, pixel.g, pixel.b};
    }
    else if constexpr (Code == ColorConversion::rgb_to_bgr) {
        return Bgr8{pixel.b, pixel.g, pixel.r};
    }
    else {
        static_assert(detail::always_false_v<SrcPixel>,
                      "unsupported pixel conversion code");
    }
}

template <ColorConversion Code, class SrcPixel, class DstPixel>
inline void cvt_color(ImageView<const SrcPixel> src, ImageView<DstPixel> dst) {
    using expected_src =
        typename detail::color_conversion_traits<Code>::src_pixel;
    using expected_dst =
        typename detail::color_conversion_traits<Code>::dst_pixel;

    static_assert(std::is_same_v<SrcPixel, expected_src>,
                  "source pixel type does not match conversion code");
    static_assert(std::is_same_v<DstPixel, expected_dst>,
                  "destination pixel type does not match conversion code");

    transform_pixels(
        src, dst, [](const SrcPixel& pixel) { return cvt_color<Code>(pixel); });
}

template <class SrcPixel, class DstPixel>
inline void cvt_color(ImageView<const SrcPixel> src, ImageView<DstPixel> dst,
                      ColorConversion code) {
    switch (code) {
        case ColorConversion::bgr_to_gray:
            if constexpr (std::is_same_v<SrcPixel, Bgr8> &&
                          std::is_same_v<DstPixel, Gray8>) {
                return cvt_color<ColorConversion::bgr_to_gray>(src, dst);
            }
            break;
        case ColorConversion::rgb_to_gray:
            if constexpr (std::is_same_v<SrcPixel, Rgb8> &&
                          std::is_same_v<DstPixel, Gray8>) {
                return cvt_color<ColorConversion::rgb_to_gray>(src, dst);
            }
            break;
        case ColorConversion::gray_to_bgr:
            if constexpr (std::is_same_v<SrcPixel, Gray8> &&
                          std::is_same_v<DstPixel, Bgr8>) {
                return cvt_color<ColorConversion::gray_to_bgr>(src, dst);
            }
            break;
        case ColorConversion::gray_to_rgb:
            if constexpr (std::is_same_v<SrcPixel, Gray8> &&
                          std::is_same_v<DstPixel, Rgb8>) {
                return cvt_color<ColorConversion::gray_to_rgb>(src, dst);
            }
            break;
        case ColorConversion::bgr_to_rgb:
            if constexpr (std::is_same_v<SrcPixel, Bgr8> &&
                          std::is_same_v<DstPixel, Rgb8>) {
                return cvt_color<ColorConversion::bgr_to_rgb>(src, dst);
            }
            break;
        case ColorConversion::rgb_to_bgr:
            if constexpr (std::is_same_v<SrcPixel, Rgb8> &&
                          std::is_same_v<DstPixel, Bgr8>) {
                return cvt_color<ColorConversion::rgb_to_bgr>(src, dst);
            }
            break;
    }

    throw color_trans_error(
        "[cvt_color] conversion code does not match source/destination pixel "
        "types");
}

}  // namespace rik_cv
