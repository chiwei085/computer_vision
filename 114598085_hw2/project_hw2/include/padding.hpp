#pragma once

#include "image.hpp"
#include "pixel_types.hpp"

namespace rik_cv
{

struct reflect_101_padding
{
    [[nodiscard]] Gray8 operator()(ImageView<const Gray8> src, int y,
                                   int x) const noexcept {
        const int cy = reflect_101_index(y, src.height());
        const int cx = reflect_101_index(x, src.width());
        return src(cy, cx);
    }

private:
    [[nodiscard]] static int reflect_101_index(int i, int extent) noexcept {
        if (extent <= 1) {
            return 0;
        }

        while (i < 0 || i >= extent) {
            if (i < 0) {
                i = -i;
            }
            else {
                i = 2 * extent - i - 2;
            }
        }
        return i;
    }
};

}  // namespace rik_cv
