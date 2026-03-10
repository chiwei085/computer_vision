#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <type_traits>

#include "color_trans.hpp"
#include "image.hpp"
#include "pixel_types.hpp"

namespace rik_cv
{

struct opencv_bridge_error : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct opencv_backend
{
};

namespace detail
{

template <class T>
struct cv_mat_traits;

template <>
struct cv_mat_traits<std::uint8_t>
{
    static constexpr int type = CV_8UC1;
};

template <>
struct cv_mat_traits<std::int8_t>
{
    static constexpr int type = CV_8SC1;
};

template <>
struct cv_mat_traits<std::uint16_t>
{
    static constexpr int type = CV_16UC1;
};

template <>
struct cv_mat_traits<std::int16_t>
{
    static constexpr int type = CV_16SC1;
};

template <>
struct cv_mat_traits<std::int32_t>
{
    static constexpr int type = CV_32SC1;
};

template <>
struct cv_mat_traits<float>
{
    static constexpr int type = CV_32FC1;
};

template <>
struct cv_mat_traits<double>
{
    static constexpr int type = CV_64FC1;
};

template <>
struct cv_mat_traits<Gray8>
{
    static constexpr int type = CV_8UC1;
};

template <>
struct cv_mat_traits<Bgr8>
{
    static constexpr int type = CV_8UC3;
};

template <>
struct cv_mat_traits<Rgb8>
{
    static constexpr int type = CV_8UC3;
};

template <class T>
concept cv_bridgeable = requires {
    cv_mat_traits<std::remove_cv_t<T>>::type;
};

template <class T>
constexpr int cv_mat_type_v = cv_mat_traits<std::remove_cv_t<T>>::type;

template <class T>
inline void validate_mat_compatibility(const cv::Mat& mat) {
    if (mat.empty()) {
        throw opencv_bridge_error("[opencv_bridge] cv::Mat is empty");
    }
    if (mat.dims != 2) {
        throw opencv_bridge_error(
            "[opencv_bridge] only 2D cv::Mat is supported");
    }
    if (mat.type() != cv_mat_type_v<T>) {
        throw opencv_bridge_error("[opencv_bridge] cv::Mat type mismatch");
    }
}

template <class T>
[[nodiscard]] inline cv::Mat wrap_const_view(ImageView<const T> view) {
    return cv::Mat(view.height(), view.width(), cv_mat_type_v<T>,
                   const_cast<void*>(static_cast<const void*>(view.data())),
                   static_cast<std::size_t>(view.stride_bytes()));
}

template <class T>
[[nodiscard]] inline Image<T> copy_from_view(ImageView<const T> view) {
    Image<T> image(view.width(), view.height());
    for (int y = 0; y < view.height(); ++y) {
        const std::span<const T> src_row = view.row_span(y);
        std::span<T> dst_row = image.view().row_span(y);
        for (int x = 0; x < view.width(); ++x) {
            dst_row[x] = src_row[x];
        }
    }
    return image;
}

}  // namespace detail

template <class T>
requires detail::cv_bridgeable<T> [[nodiscard]] inline ImageView<T>
to_image_view(cv::Mat& mat) {
    detail::validate_mat_compatibility<T>(mat);

    return ImageView<T>{reinterpret_cast<T*>(mat.data), mat.cols, mat.rows,
                        static_cast<std::ptrdiff_t>(mat.step)};
}

template <class T>
requires detail::cv_bridgeable<T> [[nodiscard]] inline ImageView<
    const std::remove_cv_t<T>>
to_image_view(const cv::Mat& mat) {
    using value_type = std::remove_cv_t<T>;

    detail::validate_mat_compatibility<value_type>(mat);

    return ImageView<const value_type>{
        reinterpret_cast<const value_type*>(mat.data), mat.cols, mat.rows,
        static_cast<std::ptrdiff_t>(mat.step)};
}

template <class T>
requires detail::cv_bridgeable<T> [[nodiscard]] inline cv::Mat to_cv_mat(
    ImageView<T> view) {
    view.validate();

    return cv::Mat(view.height(), view.width(), detail::cv_mat_type_v<T>,
                   static_cast<void*>(view.data()),
                   static_cast<std::size_t>(view.stride_bytes()));
}

template <class T>
requires detail::cv_bridgeable<T> [[nodiscard]] inline cv::Mat make_cv_mat_like(
    const ImageView<T>& view) {
    return cv::Mat(view.height(), view.width(), detail::cv_mat_type_v<T>);
}

template <>
struct image_io<opencv_backend>
{
    [[nodiscard]] static Image<Rgb8> load(const std::filesystem::path& path) {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            throw opencv_bridge_error("[opencv_bridge] failed to load image: " +
                                      path.string());
        }

        Image<Rgb8> rgb_image(image.cols, image.rows);
        const ImageView<const Bgr8> src_view =
            to_image_view<Bgr8>(image).as_const();
        cvt_color(src_view, rgb_image.view(), ColorConversion::bgr_to_rgb);
        return rgb_image;
    }

    template <class T>
    static void save(const std::filesystem::path& path,
                     ImageView<const T> view) {
        if constexpr (std::is_same_v<T, Rgb8>) {
            Image<Bgr8> bgr_image(view.width(), view.height());
            cvt_color(view, bgr_image.view(), ColorConversion::rgb_to_bgr);
            return save(path, bgr_image.view().as_const());
        }

        const cv::Mat mat = detail::wrap_const_view(view);
        if (!cv::imwrite(path.string(), mat)) {
            throw opencv_bridge_error("[opencv_bridge] failed to save image: " +
                                      path.string());
        }
    }
};

}  // namespace rik_cv
