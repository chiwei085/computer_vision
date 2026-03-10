#pragma once

#include <cstdint>

#include "image.hpp"
#include "pixel_transform.hpp"
#include "pixel_types.hpp"

namespace rik_cv
{

[[nodiscard]] inline std::uint8_t quantize_to_4_levels(std::uint8_t value) {
    if (value < 64u) {
        return 31u;
    }
    if (value < 128u) {
        return 95u;
    }
    if (value < 192u) {
        return 159u;
    }
    return 223u;
}

[[nodiscard]] inline Gray8 quantize_to_4_levels(const Gray8& pixel) {
    return Gray8{quantize_to_4_levels(pixel.v)};
}

inline void quantize_4_levels(ImageView<const Gray8> src,
                              ImageView<Gray8> dst) {
    transform_pixels(src, dst, [](const Gray8& pixel) {
        return quantize_to_4_levels(pixel);
    });
}

}  // namespace rik_cv
