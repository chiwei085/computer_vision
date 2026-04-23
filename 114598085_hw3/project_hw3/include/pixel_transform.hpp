#pragma once

#include <concepts>
#include <functional>
#include <type_traits>

#include "image.hpp"

namespace rik_cv
{

template <class SrcPixel, class DstPixel, class Fn>
requires std::regular_invocable<Fn&, const SrcPixel&>&& std::convertible_to<
    std::invoke_result_t<Fn&, const SrcPixel&>, DstPixel> inline void
transform_pixels(ImageView<const SrcPixel> src, ImageView<DstPixel> dst,
                 Fn&& fn) {
    if (!same_extent(src, dst)) {
        throw image_view_error(
            "[transform_pixels] source and destination extents differ");
    }

    for (int y = 0; y < src.height(); ++y) {
        const auto src_row = src.row_span(y);
        auto dst_row = dst.row_span(y);

        for (int x = 0; x < src.width(); ++x) {
            dst_row[x] = std::invoke(fn, src_row[x]);
        }
    }
}

}  // namespace rik_cv
