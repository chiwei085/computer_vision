#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "edge_detection.hpp"
#include "line_detection.hpp"
#include "numeric_utils.hpp"

namespace hw3
{

struct Point2d
{
    double x{};
    double y{};
};

struct LineSegment2i
{
    int x0{};
    int y0{};
    int x1{};
    int y1{};
};

struct QuadBorderConfig
{
    double angle_bin_degrees{10.0};
    double min_cluster_angle_degrees{15.0};
    double min_rho_span{50.0};
    double vertex_tolerance{80.0};
    double min_area_ratio{0.02};
};

struct ImagePipelineConfig
{
    std::string_view stem;
    rik_cv::GaussianSmoothingConfig q1;
    rik_cv::CannyConfig q2;
    rik_cv::HoughLineConfig q3;
    QuadBorderConfig q3_border{};
};

namespace detail
{

[[nodiscard]] inline double normalized_theta(double theta) {
    double wrapped = std::fmod(theta, rik_cv::detail::numeric_pi);
    if (wrapped < 0.0) {
        wrapped += rik_cv::detail::numeric_pi;
    }
    return wrapped;
}

[[nodiscard]] inline double unwrap_theta_after(double theta, double cut_angle) {
    const double wrapped = normalized_theta(theta);
    return wrapped < cut_angle ? wrapped + rik_cv::detail::numeric_pi : wrapped;
}

[[nodiscard]] inline std::pair<std::vector<rik_cv::HoughLine>,
                               std::vector<rik_cv::HoughLine>>
cluster_lines_by_angle(const std::vector<rik_cv::HoughLine>& lines,
                       const QuadBorderConfig& config) {
    const double bin_size = std::max(config.angle_bin_degrees, 1.0) *
                            rik_cv::detail::numeric_pi / 180.0;
    const int bin_count = std::max(
        2, static_cast<int>(std::ceil(rik_cv::detail::numeric_pi / bin_size)));

    std::vector<int> histogram(static_cast<std::size_t>(bin_count));
    for (const rik_cv::HoughLine& line : lines) {
        const int bin =
            std::min(bin_count - 1,
                     static_cast<int>(normalized_theta(line.theta_radians) /
                                      rik_cv::detail::numeric_pi *
                                      static_cast<double>(bin_count)));
        ++histogram[static_cast<std::size_t>(bin)];
    }

    const auto cut_it = std::min_element(histogram.begin(), histogram.end());
    const int cut_bin = static_cast<int>(cut_it - histogram.begin());
    const double cut_angle = (static_cast<double>(cut_bin) + 0.5) *
                             rik_cv::detail::numeric_pi /
                             static_cast<double>(bin_count);

    struct OrderedLine
    {
        double theta{};
        rik_cv::HoughLine line{};
    };

    std::vector<OrderedLine> ordered;
    ordered.reserve(lines.size());
    for (const rik_cv::HoughLine& line : lines) {
        ordered.push_back(OrderedLine{
            .theta = unwrap_theta_after(line.theta_radians, cut_angle),
            .line = line,
        });
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const OrderedLine& lhs, const OrderedLine& rhs) {
                  return lhs.theta < rhs.theta;
              });

    int split_after = -1;
    double largest_gap = -1.0;
    for (int i = 0; i + 1 < static_cast<int>(ordered.size()); ++i) {
        const double gap = ordered[static_cast<std::size_t>(i + 1)].theta -
                           ordered[static_cast<std::size_t>(i)].theta;
        if (gap > largest_gap) {
            largest_gap = gap;
            split_after = i;
        }
    }

    std::vector<rik_cv::HoughLine> group_a;
    std::vector<rik_cv::HoughLine> group_b;
    if (split_after < 0) {
        return {group_a, group_b};
    }

    for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
        auto& group = i <= split_after ? group_a : group_b;
        group.push_back(ordered[static_cast<std::size_t>(i)].line);
    }
    return {group_a, group_b};
}

[[nodiscard]] inline std::vector<rik_cv::HoughLine> outer_lines_by_rho(
    const std::vector<rik_cv::HoughLine>& group, double min_span) {
    if (group.size() < 2) {
        return {};
    }

    const auto [min_it, max_it] = std::minmax_element(
        group.begin(), group.end(),
        [](const rik_cv::HoughLine& lhs, const rik_cv::HoughLine& rhs) {
            return lhs.rho < rhs.rho;
        });
    if (std::abs(max_it->rho - min_it->rho) < min_span) {
        return {};
    }
    return {*min_it, *max_it};
}

[[nodiscard]] inline std::optional<Point2d> line_intersection(
    const rik_cv::HoughLine& a, const rik_cv::HoughLine& b) {
    const double ca = std::cos(a.theta_radians);
    const double sa = std::sin(a.theta_radians);
    const double cb = std::cos(b.theta_radians);
    const double sb = std::sin(b.theta_radians);
    const double det = ca * sb - sa * cb;
    if (std::abs(det) < 1.0e-9) {
        return std::nullopt;
    }

    return Point2d{
        .x = (a.rho * sb - b.rho * sa) / det,
        .y = (ca * b.rho - cb * a.rho) / det,
    };
}

[[nodiscard]] inline double cross(Point2d a, Point2d b, Point2d c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

[[nodiscard]] inline double polygon_area(const std::vector<Point2d>& points) {
    double area = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Point2d a = points[i];
        const Point2d b = points[(i + 1) % points.size()];
        area += a.x * b.y - b.x * a.y;
    }
    return std::abs(area) * 0.5;
}

[[nodiscard]] inline int clamp_to_image_coord(double value, int max_value) {
    return static_cast<int>(
        std::lround(std::clamp(value, 0.0, static_cast<double>(max_value))));
}

[[nodiscard]] inline LineSegment2i segment_from_points(
    Point2d a, Point2d b, int width, int height) {
    return LineSegment2i{
        .x0 = clamp_to_image_coord(a.x, width - 1),
        .y0 = clamp_to_image_coord(a.y, height - 1),
        .x1 = clamp_to_image_coord(b.x, width - 1),
        .y1 = clamp_to_image_coord(b.y, height - 1),
    };
}

[[nodiscard]] inline bool validate_quad_lines(
    const std::vector<rik_cv::HoughLine>& a,
    const std::vector<rik_cv::HoughLine>& b, int width, int height,
    const QuadBorderConfig& config) {
    if (a.size() != 2 || b.size() != 2) {
        return false;
    }
    const double min_group_angle_gap =
        config.min_cluster_angle_degrees * rik_cv::detail::numeric_pi / 180.0;
    if (rik_cv::detail::angle_distance_pi(
            a[0].theta_radians, b[0].theta_radians) < min_group_angle_gap) {
        return false;
    }

    std::vector<Point2d> points;
    points.reserve(4);
    for (const rik_cv::HoughLine& line_a : a) {
        for (const rik_cv::HoughLine& line_b : b) {
            const auto point = line_intersection(line_a, line_b);
            if (!point.has_value()) {
                return false;
            }
            if (point->x < -config.vertex_tolerance ||
                point->y < -config.vertex_tolerance ||
                point->x >
                    static_cast<double>(width - 1) + config.vertex_tolerance ||
                point->y >
                    static_cast<double>(height - 1) + config.vertex_tolerance) {
                return false;
            }
            points.push_back(*point);
        }
    }

    const Point2d center{
        .x = (points[0].x + points[1].x + points[2].x + points[3].x) / 4.0,
        .y = (points[0].y + points[1].y + points[2].y + points[3].y) / 4.0,
    };
    std::sort(points.begin(), points.end(), [center](Point2d lhs, Point2d rhs) {
        return std::atan2(lhs.y - center.y, lhs.x - center.x) <
               std::atan2(rhs.y - center.y, rhs.x - center.x);
    });

    double sign = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double turn = cross(points[i], points[(i + 1) % points.size()],
                                  points[(i + 2) % points.size()]);
        if (std::abs(turn) < 1.0e-6) {
            return false;
        }
        if (sign == 0.0) {
            sign = turn;
        }
        else if ((turn > 0.0) != (sign > 0.0)) {
            return false;
        }
    }

    const double min_area = static_cast<double>(width) *
                            static_cast<double>(height) * config.min_area_ratio;
    return polygon_area(points) >= min_area;
}

[[nodiscard]] inline std::vector<rik_cv::HoughLine> top_lines(
    const std::vector<rik_cv::HoughLine>& lines, int count) {
    std::vector<rik_cv::HoughLine> result;
    const int n = std::min(count, static_cast<int>(lines.size()));
    result.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        result.push_back(lines[static_cast<std::size_t>(i)]);
    }
    return result;
}

[[nodiscard]] inline std::vector<LineSegment2i>
top_line_image_segments(const std::vector<rik_cv::HoughLine>& lines,
                        int count, int width, int height) {
    std::vector<LineSegment2i> result;
    const int n = std::min(count, static_cast<int>(lines.size()));
    result.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto points = rik_cv::detail::clipped_line_points(
            lines[static_cast<std::size_t>(i)], width, height);
        if (points.size() >= 2) {
            result.push_back(LineSegment2i{
                .x0 = points[0].x,
                .y0 = points[0].y,
                .x1 = points[1].x,
                .y1 = points[1].y,
            });
        }
    }
    return result;
}

[[nodiscard]] inline std::vector<LineSegment2i>
quad_line_segments_from_groups(const std::vector<rik_cv::HoughLine>& a,
                               const std::vector<rik_cv::HoughLine>& b,
                               int width, int height) {
    if (a.size() != 2 || b.size() != 2) {
        return {};
    }

    std::vector<LineSegment2i> segments;
    segments.reserve(4);

    for (const rik_cv::HoughLine& line_a : a) {
        const auto p0 = line_intersection(line_a, b[0]);
        const auto p1 = line_intersection(line_a, b[1]);
        if (!p0.has_value() || !p1.has_value()) {
            return {};
        }
        segments.push_back(segment_from_points(*p0, *p1, width, height));
    }

    for (const rik_cv::HoughLine& line_b : b) {
        const auto p0 = line_intersection(line_b, a[0]);
        const auto p1 = line_intersection(line_b, a[1]);
        if (!p0.has_value() || !p1.has_value()) {
            return {};
        }
        segments.push_back(segment_from_points(*p0, *p1, width, height));
    }

    return segments;
}

}  // namespace detail

[[nodiscard]] inline std::vector<rik_cv::HoughLine> select_border_quad_lines(
    const std::vector<rik_cv::HoughLine>& lines, int width, int height,
    const QuadBorderConfig& config) {
    if (lines.size() < 4) {
        return detail::top_lines(lines, 4);
    }

    const auto [group_a, group_b] =
        detail::cluster_lines_by_angle(lines, config);
    for (const double span_scale : {1.0, 0.5, 0.25}) {
        auto selected_a = detail::outer_lines_by_rho(
            group_a, config.min_rho_span * span_scale);
        auto selected_b = detail::outer_lines_by_rho(
            group_b, config.min_rho_span * span_scale);
        if (!detail::validate_quad_lines(selected_a, selected_b, width, height,
                                         config)) {
            continue;
        }

        selected_a.insert(selected_a.end(), selected_b.begin(),
                          selected_b.end());
        return selected_a;
    }

    return detail::top_lines(lines, 4);
}

[[nodiscard]] inline std::vector<LineSegment2i>
select_border_quad_segments(const std::vector<rik_cv::HoughLine>& lines,
                            int width, int height,
                            const QuadBorderConfig& config) {
    if (lines.size() < 4) {
        return detail::top_line_image_segments(lines, 4, width, height);
    }

    const auto [group_a, group_b] =
        detail::cluster_lines_by_angle(lines, config);
    for (const double span_scale : {1.0, 0.5, 0.25}) {
        auto selected_a = detail::outer_lines_by_rho(
            group_a, config.min_rho_span * span_scale);
        auto selected_b = detail::outer_lines_by_rho(
            group_b, config.min_rho_span * span_scale);
        if (!detail::validate_quad_lines(selected_a, selected_b, width, height,
                                         config)) {
            continue;
        }

        return detail::quad_line_segments_from_groups(selected_a, selected_b,
                                                      width, height);
    }

    return detail::top_line_image_segments(lines, 4, width, height);
}

[[nodiscard]] inline rik_cv::Image<rik_cv::Rgb8> draw_line_segments(
    rik_cv::ImageView<const rik_cv::Rgb8> image,
    const std::vector<LineSegment2i>& segments,
    rik_cv::Rgb8 color = rik_cv::Rgb8{0, 255, 0}) {
    auto result = rik_cv::detail::copy_rgb_image(image);
    auto result_view = result.view();

    for (const LineSegment2i& segment : segments) {
        rik_cv::detail::draw_line_segment(
            result_view,
            rik_cv::detail::Point2i{.x = segment.x0, .y = segment.y0},
            rik_cv::detail::Point2i{.x = segment.x1, .y = segment.y1}, color);
    }

    return result;
}

inline constexpr std::array<ImagePipelineConfig, 3> image_configs{
    ImagePipelineConfig{
        .stem = "img1",
        .q1 = {.radius = 2, .sigma = 1.4},
        .q2 =
            {
                .smoothing = std::nullopt,
                .hysteresis =
                    {
                        .threshold =
                            {
                                .low = 55.0,
                                .high = 110.0,
                            },
                    },
            },
        .q3 = {.vote_threshold = 105, .max_lines = 30},
    },
    ImagePipelineConfig{
        .stem = "img2",
        .q1 = {.radius = 2, .sigma = 1.4},
        .q2 =
            {
                .smoothing = std::nullopt,
                .hysteresis =
                    {
                        .threshold =
                            {
                                .low = 55.0,
                                .high = 110.0,
                            },
                    },
            },
        .q3 = {.vote_threshold = 105, .max_lines = 30, .suppression_radius = 4},
    },
    ImagePipelineConfig{
        .stem = "img3",
        .q1 = {.radius = 2, .sigma = 1.4},
        .q2 =
            {
                .smoothing = std::nullopt,
                .hysteresis =
                    {
                        .threshold =
                            {
                                .low = 50.0,
                                .high = 100.0,
                            },
                    },
            },
        .q3 = {.vote_threshold = 100, .max_lines = 30},
    },
};

}  // namespace hw3
