#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "color_trans.hpp"
#include "image.hpp"
#include "opencv_bridge.hpp"

namespace rik_cv
{

using gray_histogram = std::array<std::uint32_t, 256>;

struct HistogramRenderOptions
{
    int width{1600};
    int height{1000};
    int margin_left{140};
    int margin_right{50};
    int margin_top{80};
    int margin_bottom{140};
    std::string title{"Image Histogram"};
    std::string x_label{"Pixel Value"};
    std::string y_label{"Count"};
};

[[nodiscard]] inline gray_histogram accumulate_histogram(
    ImageView<const Gray8> src) {
    gray_histogram hist{};
    if (src.empty()) {
        return hist;
    }

    if (src.is_contiguous()) {
        const Gray8* ptr = src.data();
        const Gray8* end =
            ptr + static_cast<std::ptrdiff_t>(src.width()) * src.height();
        for (; ptr != end; ++ptr) {
            ++hist[ptr->v];
        }
        return hist;
    }

    for (int y = 0; y < src.height(); ++y) {
        const Gray8* row = src.row_ptr(y);
        for (int x = 0; x < src.width(); ++x) {
            ++hist[row[x].v];
        }
    }
    return hist;
}

namespace detail
{

[[nodiscard]] inline cv::Scalar bgr(std::uint8_t b, std::uint8_t g,
                                    std::uint8_t r) {
    return cv::Scalar{static_cast<double>(b), static_cast<double>(g),
                      static_cast<double>(r)};
}

inline void draw_centered_text(cv::Mat& canvas, std::string_view text,
                               cv::Point center, int font_face,
                               double font_scale, cv::Scalar color,
                               int thickness) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(std::string{text}, font_face,
                                          font_scale, thickness, &baseline);
    const cv::Point origin{center.x - size.width / 2,
                           center.y + size.height / 2};
    cv::putText(canvas, std::string{text}, origin, font_face, font_scale, color,
                thickness, cv::LINE_AA);
}

inline void draw_rotated_text(cv::Mat& canvas, std::string_view text,
                              cv::Point center, int font_face,
                              double font_scale, cv::Scalar color,
                              int thickness) {
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        std::string{text}, font_face, font_scale, thickness, &baseline);
    cv::Mat text_img(text_size.height + baseline + 8, text_size.width + 8,
                     CV_8UC3, bgr(255, 255, 255));
    cv::putText(text_img, std::string{text}, cv::Point{4, text_size.height + 4},
                font_face, font_scale, color, thickness, cv::LINE_AA);

    cv::Mat rotated;
    cv::rotate(text_img, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);

    const int x0 = std::max(0, center.x - rotated.cols / 2);
    const int y0 = std::max(0, center.y - rotated.rows / 2);
    const int x1 = std::min(canvas.cols, x0 + rotated.cols);
    const int y1 = std::min(canvas.rows, y0 + rotated.rows);
    const cv::Rect dst_roi{x0, y0, x1 - x0, y1 - y0};
    const cv::Rect src_roi{0, 0, dst_roi.width, dst_roi.height};
    rotated(src_roi).copyTo(canvas(dst_roi));
}

}  // namespace detail

[[nodiscard]] inline Image<Rgb8> render_histogram_image(
    const gray_histogram& hist, const HistogramRenderOptions& options = {}) {
    const int plot_left = options.margin_left;
    const int plot_top = options.margin_top;
    const int plot_right = options.width - options.margin_right;
    const int plot_bottom = options.height - options.margin_bottom;
    const int plot_width = plot_right - plot_left;
    const int plot_height = plot_bottom - plot_top;

    cv::Mat canvas(options.height, options.width, CV_8UC3,
                   detail::bgr(255, 255, 255));

    const auto max_it = std::max_element(hist.begin(), hist.end());
    const std::uint32_t max_count = max_it == hist.end() ? 0u : *max_it;
    const std::uint32_t y_max = max_count == 0 ? 1u : max_count;

    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const auto axis_color = detail::bgr(40, 40, 40);
    const auto grid_color = detail::bgr(225, 225, 225);
    const auto bar_fill = detail::bgr(185, 137, 95);
    const auto bar_edge = detail::bgr(120, 82, 53);
    const auto text_color = detail::bgr(30, 30, 30);

    cv::rectangle(canvas,
                  cv::Rect{plot_left, plot_top, plot_width, plot_height},
                  detail::bgr(250, 250, 250), cv::FILLED);

    for (int tick = 0; tick <= 5; ++tick) {
        const int y = plot_bottom - (tick * plot_height) / 5;
        cv::line(canvas, cv::Point{plot_left, y}, cv::Point{plot_right, y},
                 grid_color, 1, cv::LINE_AA);

        const std::uint32_t value = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(tick) * y_max) / 5u);
        int baseline = 0;
        const std::string label = std::to_string(value);
        const cv::Size label_size =
            cv::getTextSize(label, font_face, 0.75, 2, &baseline);
        cv::putText(canvas, label,
                    cv::Point{plot_left - 16 - label_size.width,
                              y + label_size.height / 2},
                    font_face, 0.75, text_color, 2, cv::LINE_AA);
    }

    for (int tick = 0; tick <= 8; ++tick) {
        const int bin = (tick * 255) / 8;
        const int x = plot_left + (bin * plot_width) / 255;
        cv::line(canvas, cv::Point{x, plot_top}, cv::Point{x, plot_bottom},
                 grid_color, 1, cv::LINE_AA);
        cv::putText(canvas, std::to_string(bin),
                    cv::Point{x - 18, plot_bottom + 35}, font_face, 0.65,
                    text_color, 2, cv::LINE_AA);
    }

    cv::line(canvas, cv::Point{plot_left, plot_bottom},
             cv::Point{plot_right, plot_bottom}, axis_color, 2, cv::LINE_AA);
    cv::line(canvas, cv::Point{plot_left, plot_top},
             cv::Point{plot_left, plot_bottom}, axis_color, 2, cv::LINE_AA);

    for (int bin = 0; bin < 256; ++bin) {
        const int x0 = plot_left + (bin * plot_width) / 256;
        const int x1 = plot_left + ((bin + 1) * plot_width) / 256;
        const int bin_width = std::max(1, x1 - x0);
        const int bar_height = static_cast<int>(
            (static_cast<std::uint64_t>(hist[bin]) * plot_height) / y_max);
        const int y0 = plot_bottom - bar_height;

        cv::rectangle(canvas,
                      cv::Rect{x0, y0, bin_width, std::max(1, bar_height)},
                      bar_fill, cv::FILLED);
        if (bin_width >= 3) {
            cv::rectangle(canvas,
                          cv::Rect{x0, y0, bin_width, std::max(1, bar_height)},
                          bar_edge, 1, cv::LINE_8);
        }
    }

    detail::draw_centered_text(
        canvas, options.title,
        cv::Point{options.width / 2, options.margin_top / 2 + 10}, font_face,
        1.1, text_color, 2);
    detail::draw_centered_text(
        canvas, options.x_label,
        cv::Point{(plot_left + plot_right) / 2, options.height - 45}, font_face,
        0.95, text_color, 2);
    detail::draw_rotated_text(canvas, options.y_label,
                              cv::Point{55, (plot_top + plot_bottom) / 2},
                              font_face, 0.95, text_color, 2);

    Image<Rgb8> output(canvas.cols, canvas.rows);
    const auto bgr_view = to_image_view<const Bgr8>(canvas);
    cvt_color(bgr_view, output.view(), ColorConversion::bgr_to_rgb);
    return output;
}

[[nodiscard]] inline Image<Rgb8> make_histogram_image(
    ImageView<const Gray8> src, const HistogramRenderOptions& options = {}) {
    return render_histogram_image(accumulate_histogram(src), options);
}

}  // namespace rik_cv
