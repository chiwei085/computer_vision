#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "filter.hpp"
#include "image.hpp"
#include "padding.hpp"
#include "pixel_types.hpp"

namespace rik_cv
{

struct edge_detection_error : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

enum class GradientNorm
{
    l1,
    l2,
    // Uses dx^2 + dy^2. Thresholds must use the same squared scale.
    l2_squared,
};

enum class ThresholdMode
{
    absolute,
    relative_to_max,
};

enum class EdgeConnectivity
{
    four,
    eight,
};

struct GaussianSmoothingConfig
{
    // Currently dispatches to compile-time Gaussian kernels with radius 1..3.
    int radius{2};
    double sigma{1.4};
};

struct EdgeThresholdConfig
{
    double low{50.0};
    double high{100.0};
    ThresholdMode mode{ThresholdMode::absolute};
};

struct HysteresisConfig
{
    EdgeThresholdConfig threshold{};
    EdgeConnectivity connectivity{EdgeConnectivity::eight};
};

struct CannyConfig
{
    std::optional<GaussianSmoothingConfig> smoothing{GaussianSmoothingConfig{}};
    GradientNorm gradient_norm{GradientNorm::l2};
    HysteresisConfig hysteresis{};
};

struct Gradient2d
{
    std::int16_t dx{};
    std::int16_t dy{};
};

static_assert(sizeof(Gradient2d) == 4);

namespace detail
{

enum class QuantizedGradientDirection
{
    horizontal,
    diagonal_down,
    vertical,
    diagonal_up,
};

[[nodiscard]] inline Gradient2d make_gradient(int dx, int dy) {
    return Gradient2d{
        .dx = static_cast<std::int16_t>(dx),
        .dy = static_cast<std::int16_t>(dy),
    };
}

[[nodiscard]] inline Gradient2d sobel_at_padded(ImageView<const Gray8> src,
                                                int y, int x) {
    const reflect_101_padding padding{};
    const auto sample = [&](int sy, int sx) -> int {
        return static_cast<int>(padding(src, sy, sx).v);
    };

    const int top_left = sample(y - 1, x - 1);
    const int top = sample(y - 1, x);
    const int top_right = sample(y - 1, x + 1);
    const int left = sample(y, x - 1);
    const int right = sample(y, x + 1);
    const int bottom_left = sample(y + 1, x - 1);
    const int bottom = sample(y + 1, x);
    const int bottom_right = sample(y + 1, x + 1);

    return make_gradient(-top_left + top_right - 2 * left + 2 * right -
                             bottom_left + bottom_right,
                         -top_left - 2 * top - top_right + bottom_left +
                             2 * bottom + bottom_right);
}

[[nodiscard]] inline Gradient2d sobel_at_interior(ImageView<const Gray8> src,
                                                  int y, int x) noexcept {
    const auto row_above = src.row_span(y - 1);
    const auto row = src.row_span(y);
    const auto row_below = src.row_span(y + 1);

    const int top_left = row_above[x - 1].v;
    const int top = row_above[x].v;
    const int top_right = row_above[x + 1].v;
    const int left = row[x - 1].v;
    const int right = row[x + 1].v;
    const int bottom_left = row_below[x - 1].v;
    const int bottom = row_below[x].v;
    const int bottom_right = row_below[x + 1].v;

    return make_gradient(-top_left + top_right - 2 * left + 2 * right -
                             bottom_left + bottom_right,
                         -top_left - 2 * top - top_right + bottom_left +
                             2 * bottom + bottom_right);
}

[[nodiscard]] inline float magnitude_value(Gradient2d gradient,
                                           GradientNorm norm) {
    const float dx = static_cast<float>(gradient.dx);
    const float dy = static_cast<float>(gradient.dy);

    switch (norm) {
        case GradientNorm::l1:
            return std::abs(dx) + std::abs(dy);
        case GradientNorm::l2:
            return std::sqrt(dx * dx + dy * dy);
        case GradientNorm::l2_squared:
            return dx * dx + dy * dy;
    }

    throw edge_detection_error("[gradient_magnitude] unsupported norm");
}

[[nodiscard]] inline QuantizedGradientDirection quantize_direction(
    Gradient2d gradient) noexcept {
    constexpr float tan_22_5 = 0.414213562F;
    constexpr float tan_67_5 = 2.41421356F;

    const float ax = std::abs(static_cast<float>(gradient.dx));
    const float ay = std::abs(static_cast<float>(gradient.dy));

    if (ay <= ax * tan_22_5) {
        return QuantizedGradientDirection::horizontal;
    }
    if (ay >= ax * tan_67_5) {
        return QuantizedGradientDirection::vertical;
    }
    if ((gradient.dx >= 0 && gradient.dy >= 0) ||
        (gradient.dx < 0 && gradient.dy < 0)) {
        return QuantizedGradientDirection::diagonal_down;
    }
    return QuantizedGradientDirection::diagonal_up;
}

template <class MagnitudeAt>
[[nodiscard]] inline std::pair<float, float> adjacent_magnitudes(
    int y, int x, QuantizedGradientDirection direction,
    MagnitudeAt&& magnitude_at) {
    switch (direction) {
        case QuantizedGradientDirection::horizontal:
            return {magnitude_at(y, x - 1), magnitude_at(y, x + 1)};
        case QuantizedGradientDirection::diagonal_down:
            return {magnitude_at(y - 1, x - 1), magnitude_at(y + 1, x + 1)};
        case QuantizedGradientDirection::vertical:
            return {magnitude_at(y - 1, x), magnitude_at(y + 1, x)};
        case QuantizedGradientDirection::diagonal_up:
            return {magnitude_at(y - 1, x + 1), magnitude_at(y + 1, x - 1)};
    }

    return {0.0F, 0.0F};
}

[[nodiscard]] inline std::pair<float, float> resolved_thresholds(
    ImageView<const float> values, const EdgeThresholdConfig& config) {
    if (config.low < 0.0 || config.high < 0.0 || config.low > config.high) {
        throw edge_detection_error(
            "[hysteresis_edges] thresholds must satisfy 0 <= low <= high");
    }

    if (config.mode == ThresholdMode::absolute) {
        return {static_cast<float>(config.low),
                static_cast<float>(config.high)};
    }

    if (config.low > 1.0 || config.high > 1.0) {
        throw edge_detection_error(
            "[hysteresis_edges] relative thresholds must be in [0, 1]");
    }

    float max_value = 0.0F;
    for (int y = 0; y < values.height(); ++y) {
        for (int x = 0; x < values.width(); ++x) {
            max_value = std::max(max_value, values(y, x));
        }
    }
    return {static_cast<float>(config.low) * max_value,
            static_cast<float>(config.high) * max_value};
}

[[nodiscard]] inline bool connected_neighbor(EdgeConnectivity connectivity,
                                             int dy, int dx) noexcept {
    if (dy == 0 && dx == 0) {
        return false;
    }
    if (connectivity == EdgeConnectivity::eight) {
        return true;
    }
    return std::abs(dy) + std::abs(dx) == 1;
}

template <class MagnitudeAt>
inline void suppress_at(ImageView<const Gradient2d> gradients,
                        ImageView<float> suppressed, int y, int x,
                        MagnitudeAt&& magnitude_at) {
    const float magnitude = magnitude_at(y, x);
    if (magnitude <= 0.0F) {
        suppressed(y, x) = 0.0F;
        return;
    }

    const auto direction = quantize_direction(gradients(y, x));
    const auto [before, after] =
        adjacent_magnitudes(y, x, direction, magnitude_at);

    suppressed(y, x) =
        magnitude >= before && magnitude >= after ? magnitude : 0.0F;
}

[[nodiscard]] inline Image<Gray8> smoothed_image(
    ImageView<const Gray8> src, const GaussianSmoothingConfig& config) {
    return gaussian_filter(src, config.radius, config.sigma);
}

[[nodiscard]] inline Image<float> non_maximum_suppression_from_gradients(
    ImageView<const Gradient2d> gradients, GradientNorm norm) {
    Image<float> suppressed(gradients.width(), gradients.height());
    const auto magnitude_at = [&](int y, int x) -> float {
        if (!gradients.in_bounds(y, x)) {
            return 0.0F;
        }
        return magnitude_value(gradients(y, x), norm);
    };

    if (gradients.width() <= 2 || gradients.height() <= 2) {
        for (int y = 0; y < gradients.height(); ++y) {
            for (int x = 0; x < gradients.width(); ++x) {
                suppress_at(gradients, suppressed.view(), y, x, magnitude_at);
            }
        }
        return suppressed;
    }

    auto suppressed_view = suppressed.view();
    const auto magnitude_at_interior = [&](int y, int x) -> float {
        return magnitude_value(gradients(y, x), norm);
    };
    for (int y = 1; y + 1 < gradients.height(); ++y) {
        for (int x = 1; x + 1 < gradients.width(); ++x) {
            suppress_at(gradients, suppressed_view, y, x,
                        magnitude_at_interior);
        }
    }
    for (int x = 0; x < gradients.width(); ++x) {
        suppress_at(gradients, suppressed_view, 0, x, magnitude_at);
        suppress_at(gradients, suppressed_view, gradients.height() - 1, x,
                    magnitude_at);
    }
    for (int y = 1; y + 1 < gradients.height(); ++y) {
        suppress_at(gradients, suppressed_view, y, 0, magnitude_at);
        suppress_at(gradients, suppressed_view, y, gradients.width() - 1,
                    magnitude_at);
    }

    return suppressed;
}

[[nodiscard]] inline Image<Gray8> canny_edges_impl(ImageView<const Gray8> src,
                                                   const CannyConfig& config);

}  // namespace detail

[[nodiscard]] inline Image<Gradient2d> sobel_gradients(
    ImageView<const Gray8> src) {
    Image<Gradient2d> gradients(src.width(), src.height());

    if (src.width() <= 2 || src.height() <= 2) {
        for (int y = 0; y < src.height(); ++y) {
            for (int x = 0; x < src.width(); ++x) {
                gradients(y, x) = detail::sobel_at_padded(src, y, x);
            }
        }
        return gradients;
    }

    for (int y = 1; y + 1 < src.height(); ++y) {
        for (int x = 1; x + 1 < src.width(); ++x) {
            gradients(y, x) = detail::sobel_at_interior(src, y, x);
        }
    }

    for (int x = 0; x < src.width(); ++x) {
        gradients(0, x) = detail::sobel_at_padded(src, 0, x);
        gradients(src.height() - 1, x) =
            detail::sobel_at_padded(src, src.height() - 1, x);
    }
    for (int y = 1; y + 1 < src.height(); ++y) {
        gradients(y, 0) = detail::sobel_at_padded(src, y, 0);
        gradients(y, src.width() - 1) =
            detail::sobel_at_padded(src, y, src.width() - 1);
    }

    return gradients;
}

[[nodiscard]] inline Image<float> gradient_magnitude(
    ImageView<const Gradient2d> gradients,
    GradientNorm norm = GradientNorm::l2) {
    Image<float> magnitudes(gradients.width(), gradients.height());

    for (int y = 0; y < gradients.height(); ++y) {
        for (int x = 0; x < gradients.width(); ++x) {
            magnitudes(y, x) = detail::magnitude_value(gradients(y, x), norm);
        }
    }

    return magnitudes;
}

[[nodiscard]] inline Image<float> non_maximum_suppression(
    ImageView<const Gradient2d> gradients, ImageView<const float> magnitudes) {
    if (!same_extent(gradients, magnitudes)) {
        throw edge_detection_error(
            "[non_maximum_suppression] gradient and magnitude extents differ");
    }

    Image<float> suppressed(gradients.width(), gradients.height());
    const auto magnitude_at = [&](int y, int x) -> float {
        if (!magnitudes.in_bounds(y, x)) {
            return 0.0F;
        }
        return magnitudes(y, x);
    };

    if (gradients.width() <= 2 || gradients.height() <= 2) {
        for (int y = 0; y < gradients.height(); ++y) {
            for (int x = 0; x < gradients.width(); ++x) {
                detail::suppress_at(gradients, suppressed.view(), y, x,
                                    magnitude_at);
            }
        }
        return suppressed;
    }

    auto suppressed_view = suppressed.view();
    const auto magnitude_at_interior = [&](int y, int x) -> float {
        return magnitudes(y, x);
    };
    for (int y = 1; y + 1 < gradients.height(); ++y) {
        for (int x = 1; x + 1 < gradients.width(); ++x) {
            detail::suppress_at(gradients, suppressed_view, y, x,
                                magnitude_at_interior);
        }
    }
    for (int x = 0; x < gradients.width(); ++x) {
        detail::suppress_at(gradients, suppressed_view, 0, x, magnitude_at);
        detail::suppress_at(gradients, suppressed_view, gradients.height() - 1,
                            x, magnitude_at);
    }
    for (int y = 1; y + 1 < gradients.height(); ++y) {
        detail::suppress_at(gradients, suppressed_view, y, 0, magnitude_at);
        detail::suppress_at(gradients, suppressed_view, y,
                            gradients.width() - 1, magnitude_at);
    }

    return suppressed;
}

[[nodiscard]] inline Image<Gray8> hysteresis_edges(
    ImageView<const float> src, const HysteresisConfig& config = {}) {
    const auto [low_threshold, high_threshold] =
        detail::resolved_thresholds(src, config.threshold);

    constexpr std::uint8_t weak_edge = 1;
    constexpr std::uint8_t strong_edge = 255;

    Image<Gray8> edges(src.width(), src.height());
    std::vector<std::pair<int, int>> stack;
    // Heuristic: assume edges cover roughly 1/16 of pixels to avoid early
    // reallocations without reserving a full image-sized stack.
    stack.reserve(static_cast<std::size_t>(src.width()) *
                  static_cast<std::size_t>(src.height()) / 16);

    for (int y = 0; y < src.height(); ++y) {
        for (int x = 0; x < src.width(); ++x) {
            const float value = src(y, x);
            if (value >= high_threshold) {
                edges(y, x) = Gray8{strong_edge};
                stack.emplace_back(y, x);
            }
            else if (value >= low_threshold) {
                edges(y, x) = Gray8{weak_edge};
            }
            else {
                edges(y, x) = Gray8{0};
            }
        }
    }

    auto edges_view = edges.view();
    while (!stack.empty()) {
        const auto [y, x] = stack.back();
        stack.pop_back();

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (!detail::connected_neighbor(config.connectivity, dy, dx)) {
                    continue;
                }

                const int ny = y + dy;
                const int nx = x + dx;
                if (!edges_view.in_bounds(ny, nx) ||
                    edges_view(ny, nx).v != weak_edge) {
                    continue;
                }

                edges_view(ny, nx) = Gray8{strong_edge};
                stack.emplace_back(ny, nx);
            }
        }
    }

    for (int y = 0; y < edges.height(); ++y) {
        for (int x = 0; x < edges.width(); ++x) {
            if (edges(y, x).v != strong_edge) {
                edges(y, x) = Gray8{0};
            }
        }
    }

    return edges;
}

namespace detail
{

[[nodiscard]] inline Image<Gray8> canny_edges_impl(ImageView<const Gray8> src,
                                                   const CannyConfig& config) {
    const auto gradients = sobel_gradients(src);
    const auto suppressed = [&]() -> Image<float> {
        // l2: sqrt is expensive, cache magnitude once and look it up in NMS.
        // l1 / l2_squared: cheap enough to compute on the fly and skip the
        // buffer.
        if (config.gradient_norm == GradientNorm::l2) {
            const auto magnitudes = gradient_magnitude(
                gradients.as_const_view(), config.gradient_norm);
            return non_maximum_suppression(gradients.as_const_view(),
                                           magnitudes.as_const_view());
        }
        return non_maximum_suppression_from_gradients(gradients.as_const_view(),
                                                      config.gradient_norm);
    }();
    return hysteresis_edges(suppressed.as_const_view(), config.hysteresis);
}

}  // namespace detail

[[nodiscard]] inline Image<Gray8> canny_edges(ImageView<const Gray8> src,
                                              CannyConfig config = {}) {
    if (config.smoothing.has_value()) {
        const auto smoothed = detail::smoothed_image(src, *config.smoothing);
        return detail::canny_edges_impl(smoothed.as_const_view(), config);
    }

    return detail::canny_edges_impl(src, config);
}

}  // namespace rik_cv
