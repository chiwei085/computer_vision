#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "benchmark_utils.hpp"
#include "color_trans.hpp"
#include "filter.hpp"
#include "histogram.hpp"
#include "image.hpp"
#include "opencv_bridge.hpp"
#include "pixel_types.hpp"

int main() {
    namespace fs = std::filesystem;

    try {
        const fs::path project_root = PROJECT_HW2_SOURCE_DIR;
        const fs::path noise_path =
            project_root / "test_imgs" / "CKS_noise.png";
#ifndef NDEBUG
        const fs::path ref_path =
            project_root / "test_imgs" / "CKS_grayscale.png";
#endif
        const fs::path result_dir = project_root / "result_imgs";
        const fs::path report_dir = project_root / "result_reports";
        const fs::path q1_output = result_dir / "CKS_Q1.png";
        const fs::path q2_output = result_dir / "CKS_Q2.png";
        const fs::path q3_output = result_dir / "CKS_Q3.png";
        const fs::path noise_hist_output = result_dir / "CKS_noise-his.png";
        const fs::path q1_hist_output = result_dir / "CKS_Q1-his.png";
        const fs::path q2_hist_output = result_dir / "CKS_Q2-his.png";
        const fs::path q3_hist_output = result_dir / "CKS_Q3-his.png";
#ifndef NDEBUG
        const fs::path q1_psnr_csv = report_dir / "CKS_Q1_psnr.csv";
        const fs::path q2_psnr_csv = report_dir / "CKS_Q2_psnr.csv";
        const fs::path q3_psnr_csv = report_dir / "CKS_Q3_psnr.csv";
        const fs::path summary_log = report_dir / "report_summary.log";

        const std::vector<int> benchmark_radii{1, 2, 3, 4, 5};
        const std::vector<int> benchmark_pad_sizes{0, 1, 2, 3, 4, 5};
        const std::vector<int> benchmark_strides{1, 2, 3};
        const auto benchmark_config_set = hw2_bench::make_benchmark_configs(
            benchmark_radii, benchmark_pad_sizes, benchmark_strides);

        const std::vector<int> q3_coarse_radii{1, 3, 5};
        const std::vector<int> q3_coarse_pad_sizes{0, 2, 4};
        const auto q3_coarse_config_set =
            hw2_bench::make_combined_benchmark_configs(q3_coarse_radii,
                                                       q3_coarse_pad_sizes);
#endif

        fs::create_directories(result_dir);
        fs::create_directories(report_dir);

        const auto noise_rgb =
            rik_cv::load_image<rik_cv::opencv_backend>(noise_path);
        auto noise = rik_cv::make_image<rik_cv::Gray8>(noise_rgb.width(),
                                                       noise_rgb.height());
        rik_cv::cvt_color<rik_cv::ColorConversion::rgb_to_gray>(
            noise_rgb.as_const_view(), noise.view());

#ifndef NDEBUG
        const auto ref_rgb =
            rik_cv::load_image<rik_cv::opencv_backend>(ref_path);
        auto ref_gray = rik_cv::make_image<rik_cv::Gray8>(ref_rgb.width(),
                                                          ref_rgb.height());
        rik_cv::cvt_color<rik_cv::ColorConversion::rgb_to_gray>(
            ref_rgb.as_const_view(), ref_gray.view());
#endif

        auto noise_hist = rik_cv::make_histogram_image(noise.as_const_view());
        std::cout << "Saving CKS_noise-his.png -> " << noise_hist_output
                  << '\n';
        rik_cv::save_image<rik_cv::opencv_backend>(noise_hist_output,
                                                   noise_hist);

#ifdef NDEBUG
        constexpr int q1_best_radius = 1;
        constexpr int q1_best_pad = 5;
        constexpr int q1_best_stride = 3;
        auto q1_image = rik_cv::median_filter<q1_best_radius>(
            noise.as_const_view(), q1_best_pad, q1_best_pad, q1_best_stride,
            q1_best_stride);
        auto q1_hist = rik_cv::make_histogram_image(q1_image.as_const_view());
        std::cout << "Release mode: using benchmark best config for Q1 "
                     "(median r=1 pad=5 stride=3x3)\n";
        std::cout << "Saving CKS_Q1.png -> " << q1_output << '\n';
        std::cout << "Saving CKS_Q1-his.png -> " << q1_hist_output << '\n';
        rik_cv::save_image<rik_cv::opencv_backend>(q1_output, q1_image);
        rik_cv::save_image<rik_cv::opencv_backend>(q1_hist_output, q1_hist);

        constexpr int q2_best_radius = 3;
        constexpr int q2_best_pad = 5;
        constexpr int q2_best_stride = 3;
        auto q2_image = rik_cv::gaussian_filter<q2_best_radius>(
            noise.as_const_view(), q2_best_pad, q2_best_pad, q2_best_stride,
            q2_best_stride);
        auto q2_hist = rik_cv::make_histogram_image(q2_image.as_const_view());
        std::cout << "Release mode: using benchmark best config for Q2 "
                     "(gaussian r=3 pad=5 stride=3x3)\n";
        std::cout << "Saving CKS_Q2.png -> " << q2_output << '\n';
        std::cout << "Saving CKS_Q2-his.png -> " << q2_hist_output << '\n';
        rik_cv::save_image<rik_cv::opencv_backend>(q2_output, q2_image);
        rik_cv::save_image<rik_cv::opencv_backend>(q2_hist_output, q2_hist);

        constexpr int q3_median_best_radius = 1;
        constexpr int q3_median_best_pad = 5;
        constexpr int q3_gaussian_best_radius = 1;
        constexpr int q3_gaussian_best_pad = 5;
        auto q3_image = rik_cv::median_then_gaussian_filter<
            q3_median_best_radius, q3_gaussian_best_radius>(
            noise.as_const_view(), q3_median_best_pad, q3_median_best_pad,
            q3_gaussian_best_pad, q3_gaussian_best_pad);
        auto q3_hist = rik_cv::make_histogram_image(q3_image.as_const_view());
        std::cout << "Release mode: using benchmark best config for Q3 "
                     "(median->gaussian, median r=1 pad=5, gaussian r=1 "
                     "pad=5)\n";
        std::cout << "Saving CKS_Q3.png -> " << q3_output << '\n';
        std::cout << "Saving CKS_Q3-his.png -> " << q3_hist_output << '\n';
        rik_cv::save_image<rik_cv::opencv_backend>(q3_output, q3_image);
        rik_cv::save_image<rik_cv::opencv_backend>(q3_hist_output, q3_hist);
#else
        auto median_summary = hw2_bench::benchmark_configs(
            "Median filter benchmark (kernel, pad size, stride sweep):",
            benchmark_config_set, noise.as_const_view(),
            ref_gray.as_const_view(),
            [&]<int Radius, class Padding>(
                rik_cv::ImageView<const rik_cv::Gray8> input, int pad_x,
                int pad_y, int stride_x, int stride_y, Padding padding) {
                return rik_cv::median_filter<Radius, Padding>(
                    input, pad_x, pad_y, stride_x, stride_y, padding);
            });
        hw2_bench::print_benchmark_summary(median_summary);
        hw2_bench::print_top_k_runs(median_summary);
        hw2_bench::print_local_sensitivity(median_summary);
        hw2_bench::print_close_candidate_tradeoffs(median_summary);
        std::cout << "Saving CKS_Q1.png -> " << q1_output << '\n';
        auto q1_hist = rik_cv::make_histogram_image(
            median_summary.best().image.as_const_view());
        std::cout << "Saving CKS_Q1-his.png -> " << q1_hist_output << '\n';
        std::cout << "Saving Q1 PSNR CSV -> " << q1_psnr_csv << '\n';
        rik_cv::save_image<rik_cv::opencv_backend>(q1_output,
                                                   median_summary.best().image);
        rik_cv::save_image<rik_cv::opencv_backend>(q1_hist_output, q1_hist);
        hw2_bench::save_benchmark_csv(median_summary, q1_psnr_csv);

        std::cout << '\n';

        auto gaussian_summary = hw2_bench::benchmark_configs(
            "Gaussian filter benchmark (kernel, pad size, stride sweep):",
            benchmark_config_set, noise.as_const_view(),
            ref_gray.as_const_view(),
            [&]<int Radius, class Padding>(
                rik_cv::ImageView<const rik_cv::Gray8> input, int pad_x,
                int pad_y, int stride_x, int stride_y, Padding padding) {
                return rik_cv::gaussian_filter<Radius, Padding>(
                    input, pad_x, pad_y, stride_x, stride_y, padding);
            });
        hw2_bench::print_benchmark_summary(gaussian_summary);
        hw2_bench::print_top_k_runs(gaussian_summary);
        hw2_bench::print_local_sensitivity(gaussian_summary);
        hw2_bench::print_close_candidate_tradeoffs(gaussian_summary);
        std::cout << "Saving CKS_Q2.png -> " << q2_output << '\n';
        auto q2_hist = rik_cv::make_histogram_image(
            gaussian_summary.best().image.as_const_view());
        std::cout << "Saving CKS_Q2-his.png -> " << q2_hist_output << '\n';
        std::cout << "Saving Q2 PSNR CSV -> " << q2_psnr_csv << '\n';
        rik_cv::save_image<rik_cv::opencv_backend>(
            q2_output, gaussian_summary.best().image);
        rik_cv::save_image<rik_cv::opencv_backend>(q2_hist_output, q2_hist);
        hw2_bench::save_benchmark_csv(gaussian_summary, q2_psnr_csv);

        std::cout << '\n';

        auto q3_coarse_summary = hw2_bench::benchmark_combined_configs(
            "Q3 coarse benchmark (median + gaussian, both orders):",
            q3_coarse_config_set, noise.as_const_view(),
            ref_gray.as_const_view());
        const auto& q3_coarse_best = q3_coarse_summary.best();

        std::vector<hw2_bench::CombinedBenchmarkConfig> q3_fine_config_set;
        for (const auto order : {q3_coarse_best.order}) {
            for (const int median_radius : hw2_bench::neighbor_values(
                     q3_coarse_best.median_radius, 1, 5)) {
                for (const int median_pad : hw2_bench::neighbor_values(
                         q3_coarse_best.median_pad, 0, 5)) {
                    for (const int gaussian_radius : hw2_bench::neighbor_values(
                             q3_coarse_best.gaussian_radius, 1, 5)) {
                        for (const int gaussian_pad :
                             hw2_bench::neighbor_values(
                                 q3_coarse_best.gaussian_pad, 0, 5)) {
                            q3_fine_config_set.push_back(
                                hw2_bench::CombinedBenchmarkConfig{
                                    .order = order,
                                    .median_radius = median_radius,
                                    .median_pad = median_pad,
                                    .gaussian_radius = gaussian_radius,
                                    .gaussian_pad = gaussian_pad,
                                });
                        }
                    }
                }
            }
        }

        auto q3_fine_summary = hw2_bench::benchmark_combined_configs(
            "Q3 fine benchmark (local refinement around coarse winner):",
            q3_fine_config_set, noise.as_const_view(),
            ref_gray.as_const_view());
        auto q3_summary = hw2_bench::merge_combined_summaries(
            "Q3 combined filter benchmark (coarse-to-fine):",
            std::move(q3_coarse_summary), std::move(q3_fine_summary));
        hw2_bench::print_combined_summary(q3_summary);
        hw2_bench::print_combined_top_k(q3_summary);
        hw2_bench::print_combined_local_sensitivity(q3_summary);
        hw2_bench::print_combined_close_candidate_tradeoffs(q3_summary);
        std::cout << "Saving CKS_Q3.png -> " << q3_output << '\n';
        auto q3_hist = rik_cv::make_histogram_image(
            q3_summary.best().image.as_const_view());
        std::cout << "Saving CKS_Q3-his.png -> " << q3_hist_output << '\n';
        std::cout << "Saving Q3 PSNR CSV -> " << q3_psnr_csv << '\n';
        rik_cv::save_image<rik_cv::opencv_backend>(q3_output,
                                                   q3_summary.best().image);
        rik_cv::save_image<rik_cv::opencv_backend>(q3_hist_output, q3_hist);
        hw2_bench::save_combined_benchmark_csv(q3_summary, q3_psnr_csv);

        std::ofstream report_out(summary_log);
        if (!report_out) {
            throw std::runtime_error("failed to open report summary log: " +
                                     summary_log.string());
        }
        hw2_bench::append_benchmark_summary_log(report_out, median_summary);
        hw2_bench::append_benchmark_summary_log(report_out, gaussian_summary);
        hw2_bench::append_combined_benchmark_summary_log(report_out,
                                                         q3_summary);
        std::cout << "Saving summary log -> " << summary_log << '\n';
#endif
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
