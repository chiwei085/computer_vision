#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "image.hpp"
#include "pixel_types.hpp"

namespace rik_cv
{

struct hough_transform_error : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct HoughLineConfig
{
    double rho_resolution{1.0};
    double theta_resolution_degrees{1.0};
    int vote_threshold{100};
    int max_lines{20};
    int suppression_radius{8};
};

struct HoughLine
{
    double rho{};
    double theta_radians{};
    int votes{};
};

namespace detail
{

inline constexpr double pi = 3.14159265358979323846;

struct Point2i
{
    int x{};
    int y{};
};

[[nodiscard]] inline int accumulator_index(int rho_index, int theta_index,
                                           int theta_bins) noexcept {
    return rho_index * theta_bins + theta_index;
}

inline void validate_hough_config(const HoughLineConfig& config) {
    if (!(config.rho_resolution > 0.0) ||
        !(config.theta_resolution_degrees > 0.0)) {
        throw hough_transform_error(
            "[hough_lines] rho/theta resolutions must be positive");
    }
    if (config.theta_resolution_degrees > 180.0) {
        throw hough_transform_error(
            "[hough_lines] theta resolution must be <= 180 degrees");
    }
    if (config.vote_threshold < 1) {
        throw hough_transform_error(
            "[hough_lines] vote threshold must be positive");
    }
    if (config.max_lines < 0) {
        throw hough_transform_error("[hough_lines] max_lines must be >= 0");
    }
    if (config.suppression_radius < 0) {
        throw hough_transform_error(
            "[hough_lines] suppression radius must be >= 0");
    }
}

[[nodiscard]] inline bool is_local_accumulator_maximum(
    const std::vector<int>& accumulator, int rho_bins, int theta_bins,
    int rho_index, int theta_index, int radius) {
    const int center_votes =
        accumulator[accumulator_index(rho_index, theta_index, theta_bins)];

    for (int dr = -radius; dr <= radius; ++dr) {
        const int r = rho_index + dr;
        if (r < 0 || r >= rho_bins) {
            continue;
        }

        for (int dt = -radius; dt <= radius; ++dt) {
            if (dr == 0 && dt == 0) {
                continue;
            }

            int t = theta_index + dt;
            if (t < 0) {
                t += theta_bins;
            }
            if (t >= theta_bins) {
                t -= theta_bins;
            }

            if (accumulator[accumulator_index(r, t, theta_bins)] >
                center_votes) {
                return false;
            }
        }
    }

    return true;
}

[[nodiscard]] inline Image<Rgb8> copy_rgb_image(ImageView<const Rgb8> src) {
    Image<Rgb8> dst(src.width(), src.height());
    auto dst_view = dst.view();
    for (int y = 0; y < src.height(); ++y) {
        const auto src_row = src.row_span(y);
        auto dst_row = dst_view.row_span(y);
        std::copy(src_row.begin(), src_row.end(), dst_row.begin());
    }
    return dst;
}

inline void set_pixel_if_in_bounds(ImageView<Rgb8> image, int x, int y,
                                   Rgb8 color) {
    if (image.in_bounds(y, x)) {
        image(y, x) = color;
    }
}

inline void draw_line_segment(ImageView<Rgb8> image, Point2i p0, Point2i p1,
                              Rgb8 color) {
    int x0 = p0.x;
    int y0 = p0.y;
    const int x1 = p1.x;
    const int y1 = p1.y;

    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;

    while (true) {
        set_pixel_if_in_bounds(image, x0, y0, color);
        if (x0 == x1 && y0 == y1) {
            break;
        }

        const int doubled_error = 2 * error;
        if (doubled_error >= dy) {
            error += dy;
            x0 += sx;
        }
        if (doubled_error <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

inline void add_intersection(std::vector<Point2i>& points, double x, double y,
                             int width, int height) {
    constexpr double eps = 1.0e-6;
    if (x < -eps || y < -eps || x > static_cast<double>(width - 1) + eps ||
        y > static_cast<double>(height - 1) + eps) {
        return;
    }

    const auto point = Point2i{
        .x = static_cast<int>(
            std::lround(std::clamp(x, 0.0, static_cast<double>(width - 1)))),
        .y = static_cast<int>(
            std::lround(std::clamp(y, 0.0, static_cast<double>(height - 1)))),
    };

    for (const Point2i& existing : points) {
        if (existing.x == point.x && existing.y == point.y) {
            return;
        }
    }
    points.push_back(point);
}

[[nodiscard]] inline std::vector<Point2i> clipped_line_points(
    const HoughLine& line, int width, int height) {
    std::vector<Point2i> points;
    points.reserve(4);

    const double cos_theta = std::cos(line.theta_radians);
    const double sin_theta = std::sin(line.theta_radians);

    if (std::abs(sin_theta) > 1.0e-9) {
        add_intersection(points, 0.0, line.rho / sin_theta, width, height);
        add_intersection(
            points, static_cast<double>(width - 1),
            (line.rho - static_cast<double>(width - 1) * cos_theta) / sin_theta,
            width, height);
    }

    if (std::abs(cos_theta) > 1.0e-9) {
        add_intersection(points, line.rho / cos_theta, 0.0, width, height);
        add_intersection(
            points,
            (line.rho - static_cast<double>(height - 1) * sin_theta) /
                cos_theta,
            static_cast<double>(height - 1), width, height);
    }

    return points;
}

}  // namespace detail

[[nodiscard]] inline std::vector<HoughLine> hough_lines(
    ImageView<const Gray8> edges, const HoughLineConfig& config = {}) {
    detail::validate_hough_config(config);

    if (edges.empty() || config.max_lines == 0) {
        return {};
    }

    const double max_rho = std::hypot(static_cast<double>(edges.width() - 1),
                                      static_cast<double>(edges.height() - 1));
    const int rho_bins =
        static_cast<int>(std::floor((2.0 * max_rho) / config.rho_resolution)) +
        1;
    const int theta_bins =
        static_cast<int>(std::ceil(180.0 / config.theta_resolution_degrees));

    std::vector<double> cos_table(static_cast<std::size_t>(theta_bins));
    std::vector<double> sin_table(static_cast<std::size_t>(theta_bins));
    for (int theta_index = 0; theta_index < theta_bins; ++theta_index) {
        const double theta = static_cast<double>(theta_index) *
                             config.theta_resolution_degrees * detail::pi /
                             180.0;
        cos_table[static_cast<std::size_t>(theta_index)] = std::cos(theta);
        sin_table[static_cast<std::size_t>(theta_index)] = std::sin(theta);
    }

    std::vector<int> accumulator(static_cast<std::size_t>(rho_bins) *
                                 static_cast<std::size_t>(theta_bins));

    for (int y = 0; y < edges.height(); ++y) {
        for (int x = 0; x < edges.width(); ++x) {
            if (edges(y, x).v == 0) {
                continue;
            }

            for (int theta_index = 0; theta_index < theta_bins; ++theta_index) {
                const double rho =
                    static_cast<double>(x) *
                        cos_table[static_cast<std::size_t>(theta_index)] +
                    static_cast<double>(y) *
                        sin_table[static_cast<std::size_t>(theta_index)];
                const int rho_index = static_cast<int>(
                    std::lround((rho + max_rho) / config.rho_resolution));

                if (rho_index >= 0 && rho_index < rho_bins) {
                    ++accumulator[detail::accumulator_index(
                        rho_index, theta_index, theta_bins)];
                }
            }
        }
    }

    std::vector<HoughLine> lines;
    for (int rho_index = 0; rho_index < rho_bins; ++rho_index) {
        for (int theta_index = 0; theta_index < theta_bins; ++theta_index) {
            const int votes = accumulator[detail::accumulator_index(
                rho_index, theta_index, theta_bins)];
            if (votes < config.vote_threshold) {
                continue;
            }
            if (!detail::is_local_accumulator_maximum(
                    accumulator, rho_bins, theta_bins, rho_index, theta_index,
                    config.suppression_radius)) {
                continue;
            }

            lines.push_back(HoughLine{
                .rho = static_cast<double>(rho_index) * config.rho_resolution -
                       max_rho,
                .theta_radians = static_cast<double>(theta_index) *
                                 config.theta_resolution_degrees * detail::pi /
                                 180.0,
                .votes = votes,
            });
        }
    }

    std::sort(lines.begin(), lines.end(),
              [](const HoughLine& lhs, const HoughLine& rhs) {
                  return lhs.votes > rhs.votes;
              });
    if (static_cast<int>(lines.size()) > config.max_lines) {
        lines.resize(static_cast<std::size_t>(config.max_lines));
    }
    return lines;
}

[[nodiscard]] inline Image<Rgb8> draw_hough_lines(
    ImageView<const Rgb8> image, const std::vector<HoughLine>& lines,
    Rgb8 color = Rgb8{255, 0, 0}) {
    Image<Rgb8> result = detail::copy_rgb_image(image);
    auto result_view = result.view();

    for (const HoughLine& line : lines) {
        const auto points =
            detail::clipped_line_points(line, image.width(), image.height());
        if (points.size() < 2) {
            continue;
        }
        detail::draw_line_segment(result_view, points[0], points[1], color);
    }

    return result;
}

[[nodiscard]] inline Image<Rgb8> hough_lines_image(
    ImageView<const Rgb8> image, ImageView<const Gray8> edges,
    const HoughLineConfig& config = {}, Rgb8 color = Rgb8{255, 0, 0}) {
    if (!same_extent(image, edges)) {
        throw hough_transform_error(
            "[hough_lines_image] image and edge extents differ");
    }

    const auto lines = hough_lines(edges, config);
    return draw_hough_lines(image, lines, color);
}

}  // namespace rik_cv
