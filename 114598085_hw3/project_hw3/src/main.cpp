#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "color_trans.hpp"
#include "edge_detection.hpp"
#include "filter.hpp"
#include "hw3_pipeline.hpp"
#include "image.hpp"
#include "line_detection.hpp"
#include "opencv_bridge.hpp"
#include "pixel_types.hpp"

namespace
{

template <class T>
void write_image(const std::filesystem::path& output_path,
                 rik_cv::ImageView<const T> image) {
    std::cout << "Saving " << output_path.filename().string() << " -> "
              << output_path << '\n';
    rik_cv::save_image<rik_cv::opencv_backend>(output_path, image);
}

[[nodiscard]] rik_cv::Image<rik_cv::Gray8> make_q1_image(
    rik_cv::ImageView<const rik_cv::Rgb8> rgb_image,
    const rik_cv::GaussianSmoothingConfig& config) {
    auto gray_image = rik_cv::make_image<rik_cv::Gray8>(rgb_image.width(),
                                                        rgb_image.height());
    rik_cv::cvt_color<rik_cv::ColorConversion::rgb_to_gray>(rgb_image,
                                                            gray_image.view());

    return rik_cv::gaussian_filter(gray_image.as_const_view(), config.radius,
                                   config.sigma);
}

void process_image(const std::filesystem::path& test_dir,
                   const std::filesystem::path& result_dir,
                   const hw3::ImagePipelineConfig& config) {
    const std::string stem{config.stem};
    const std::filesystem::path input_path = test_dir / (stem + ".png");
    const std::filesystem::path q1_output_path =
        result_dir / (stem + "_q1.png");
    const std::filesystem::path q2_output_path =
        result_dir / (stem + "_q2.png");
    const std::filesystem::path q3_output_path =
        result_dir / (stem + "_q3.png");

    const auto rgb_image =
        rik_cv::load_image<rik_cv::opencv_backend>(input_path);
    const auto q1_image = make_q1_image(rgb_image.as_const_view(), config.q1);
    // HW3 defines Q2 as Canny over Q1's grayscale + Gaussian blur output.
    const auto q2_image =
        rik_cv::canny_edges(q1_image.as_const_view(), config.q2);
    const auto q3_candidates =
        rik_cv::hough_lines(q2_image.as_const_view(), config.q3);
    const auto q3_segments = hw3::select_border_quad_segments(
        q3_candidates, rgb_image.width(), rgb_image.height(), config.q3_border);
    const auto q3_image =
        hw3::draw_line_segments(rgb_image.as_const_view(), q3_segments);

    write_image(q1_output_path, q1_image.as_const_view());
    write_image(q2_output_path, q2_image.as_const_view());
    write_image(q3_output_path, q3_image.as_const_view());
}

}  // namespace

int main() {
    namespace fs = std::filesystem;

    try {
        const fs::path project_root = PROJECT_HW3_SOURCE_DIR;
        const fs::path test_dir = project_root / "test_imgs";
        const fs::path result_dir = project_root / "result_imgs";

        fs::create_directories(result_dir);

        for (const hw3::ImagePipelineConfig& config : hw3::image_configs) {
            process_image(test_dir, result_dir, config);
        }
    }
    catch (const std::exception& error) {
        std::cerr << "HW3 failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
