#include <array>
#include <filesystem>
#include <iostream>

#include "color_trans.hpp"
#include "convolution.hpp"
#include "opencv_bridge.hpp"
#include "quantization.hpp"

int main() {
    namespace fs = std::filesystem;

    try {
        const fs::path project_root = PROJECT_HW1_SOURCE_DIR;
        const fs::path input_path = project_root / "test_imgs" / "CKS.png";
        const fs::path gray_output_path =
            project_root / "result_imgs" / "CKS_Q1.png";
        const fs::path quantized_output_path =
            project_root / "result_imgs" / "CKS_Q2.png";
        const fs::path convolved_output_path =
            project_root / "result_imgs" / "CKS_Q3.png";
        const fs::path downsampled_gray_output_path =
            project_root / "result_imgs" / "CKS_Q4a.png";
        const fs::path downsampled_blur_output_path =
            project_root / "result_imgs" / "CKS_Q4b.png";

        auto input_rgb = rik_cv::load_image<rik_cv::opencv_backend>(input_path);

        auto gray_image = rik_cv::make_image<rik_cv::Gray8>(input_rgb.width(),
                                                            input_rgb.height());
        auto gray_view = gray_image.view();
        rik_cv::cvt_color(input_rgb.as_const_view(), gray_view,
                          rik_cv::ColorConversion::rgb_to_gray);

        auto quantized_image = rik_cv::make_image<rik_cv::Gray8>(
            gray_image.width(), gray_image.height());
        rik_cv::quantize_4_levels(gray_view.as_const(), quantized_image.view());

        constexpr std::array<double, 9> blur_kernel{
            1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
            1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        };
        auto convolved_image =
            rik_cv::conv(gray_view.as_const(), blur_kernel, 3, 3, 1, 1);

        constexpr std::array<double, 4> downsample_kernel{
            1.0,
            0.0,
            0.0,
            0.0,
        };
        auto downsampled_gray_image =
            rik_cv::conv(gray_view.as_const(), downsample_kernel, 2, 2, 2, 2);
        auto downsampled_blur_image = rik_cv::conv(
            convolved_image.as_const_view(), downsample_kernel, 2, 2, 2, 2);

        rik_cv::save_image<rik_cv::opencv_backend>(gray_output_path,
                                                   gray_image);
        rik_cv::save_image<rik_cv::opencv_backend>(quantized_output_path,
                                                   quantized_image);
        rik_cv::save_image<rik_cv::opencv_backend>(convolved_output_path,
                                                   convolved_image);
        rik_cv::save_image<rik_cv::opencv_backend>(downsampled_gray_output_path,
                                                   downsampled_gray_image);
        rik_cv::save_image<rik_cv::opencv_backend>(downsampled_blur_output_path,
                                                   downsampled_blur_image);
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
