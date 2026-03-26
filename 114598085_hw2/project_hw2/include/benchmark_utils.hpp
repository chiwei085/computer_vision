#pragma once

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "filter.hpp"
#include "image.hpp"

namespace hw2_bench
{

namespace ansi
{

inline constexpr std::string_view reset = "\033[0m";
inline constexpr std::string_view bold = "\033[1m";
inline constexpr std::string_view dim = "\033[2m";
inline constexpr std::string_view cyan = "\033[36m";
inline constexpr std::string_view green = "\033[32m";
inline constexpr std::string_view yellow = "\033[33m";

}  // namespace ansi

inline constexpr double kMseEpsilon = 1e-10;
inline constexpr double kMaxPixelSquared = 255.0 * 255.0;

[[nodiscard]] inline double compute_psnr(
    rik_cv::ImageView<const rik_cv::Gray8> ref,
    rik_cv::ImageView<const rik_cv::Gray8> test) {
    double sum_sq = 0.0;
    for (int y = 0; y < ref.height(); ++y) {
        for (int x = 0; x < ref.width(); ++x) {
            const double d = static_cast<double>(ref(y, x).v) -
                             static_cast<double>(test(y, x).v);
            sum_sq += d * d;
        }
    }
    const double n =
        static_cast<double>(ref.width()) * static_cast<double>(ref.height());
    const double mse = sum_sq / n;
    if (mse < kMseEpsilon) {
        return std::numeric_limits<double>::infinity();
    }
    return 10.0 * std::log10(kMaxPixelSquared / mse);
}

struct BenchmarkRun
{
    int radius{};
    int kernel_size{};
    int pad_size{};
    int stride_x{};
    int stride_y{};
    int output_width{};
    int output_height{};
    double psnr{};
    rik_cv::Image<rik_cv::Gray8> image{};
};

struct BenchmarkConfig
{
    int radius{};
    int pad_size{};
    int stride_x{1};
    int stride_y{1};
};

template <class Run>
struct SummaryRunsBase
{
    std::string title;
    std::vector<Run> runs;
    std::size_t best_index{};

    [[nodiscard]] const Run& best() const {
        if (runs.empty()) {
            throw std::runtime_error("[benchmark] no runs were recorded");
        }
        return runs[best_index];
    }
};

struct BenchmarkSummary : SummaryRunsBase<BenchmarkRun>
{
};

enum class FilterOrder
{
    median_then_gaussian,
    gaussian_then_median,
};

struct CombinedBenchmarkRun
{
    FilterOrder order{};
    int median_radius{};
    int median_pad{};
    int gaussian_radius{};
    int gaussian_pad{};
    int output_width{};
    int output_height{};
    double psnr{};
    rik_cv::Image<rik_cv::Gray8> image{};
};

struct CombinedBenchmarkConfig
{
    FilterOrder order{};
    int median_radius{};
    int median_pad{};
    int gaussian_radius{};
    int gaussian_pad{};
};

struct CombinedBenchmarkSummary : SummaryRunsBase<CombinedBenchmarkRun>
{
};

[[nodiscard]] inline const char* filter_order_name(FilterOrder order) noexcept {
    switch (order) {
        case FilterOrder::median_then_gaussian:
            return "median->gaussian";
        case FilterOrder::gaussian_then_median:
            return "gaussian->median";
    }
#if defined(__clang__) || defined(__GNUC__)
    __builtin_unreachable();
#else
    std::abort();
#endif
}

[[nodiscard]] inline std::size_t estimated_work_units(const BenchmarkRun& run) {
    return static_cast<std::size_t>(run.kernel_size) *
           static_cast<std::size_t>(run.kernel_size) *
           static_cast<std::size_t>(run.output_width) *
           static_cast<std::size_t>(run.output_height);
}

[[nodiscard]] inline bool simpler_than(const BenchmarkRun& a,
                                       const BenchmarkRun& b) {
    if (a.radius != b.radius) {
        return a.radius < b.radius;
    }
    if (a.pad_size != b.pad_size) {
        return a.pad_size < b.pad_size;
    }
    if (a.stride_x != b.stride_x) {
        return a.stride_x > b.stride_x;
    }
    return a.stride_y > b.stride_y;
}

[[nodiscard]] inline std::size_t estimated_work_units(
    const CombinedBenchmarkRun& run) {
    const std::size_t median_kernel =
        static_cast<std::size_t>(2 * run.median_radius + 1);
    const std::size_t gaussian_kernel =
        static_cast<std::size_t>(2 * run.gaussian_radius + 1);
    return (median_kernel * median_kernel + gaussian_kernel * gaussian_kernel) *
           static_cast<std::size_t>(run.output_width) *
           static_cast<std::size_t>(run.output_height);
}

[[nodiscard]] inline bool better_candidate(double psnr, std::size_t index,
                                           double best_psnr,
                                           std::size_t best_index) {
    return psnr > best_psnr || (psnr == best_psnr && index < best_index);
}

[[nodiscard]] inline bool simpler_than(const CombinedBenchmarkRun& a,
                                       const CombinedBenchmarkRun& b) {
    if (a.median_radius != b.median_radius) {
        return a.median_radius < b.median_radius;
    }
    if (a.gaussian_radius != b.gaussian_radius) {
        return a.gaussian_radius < b.gaussian_radius;
    }
    if (a.median_pad != b.median_pad) {
        return a.median_pad < b.median_pad;
    }
    if (a.gaussian_pad != b.gaussian_pad) {
        return a.gaussian_pad < b.gaussian_pad;
    }
    return static_cast<int>(a.order) < static_cast<int>(b.order);
}

template <class Config>
class ProgressPrinter
{
public:
    ProgressPrinter(std::string title, std::size_t total)
        : title_(std::move(title)),
          total_(total),
          start_(std::chrono::steady_clock::now()),
          interactive_(::isatty(fileno(stdout)) != 0) {}

    template <class DescribeFn>
    void update(std::size_t completed, const Config& config,
                DescribeFn describe_fn) const {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_sec =
            std::chrono::duration<double>(now - start_).count();
        const double percent = total_ == 0
                                   ? 100.0
                                   : 100.0 * static_cast<double>(completed) /
                                         static_cast<double>(total_);
        const double eta_sec =
            completed == 0
                ? 0.0
                : elapsed_sec * (static_cast<double>(total_ - completed) /
                                 static_cast<double>(completed));

        std::cout << (interactive_ ? "\r" : "") << ansi::dim << "[progress] "
                  << ansi::reset << title_ << " " << completed << '/' << total_
                  << " (" << std::fixed << std::setprecision(1) << percent
                  << "%)  " << describe_fn(config) << " elapsed=" << ansi::cyan
                  << std::setprecision(2) << elapsed_sec << "s" << ansi::reset
                  << " eta="
                  << (completed == 0
                          ? std::string("--")
                          : std::to_string(static_cast<int>(eta_sec)))
                  << "s" << (interactive_ ? "   " : "\n") << std::flush;
    }

    void finish() const {
        if (interactive_) {
            std::cout << '\n';
        }
    }

private:
    std::string title_;
    std::size_t total_{};
    std::chrono::steady_clock::time_point start_;
    bool interactive_{};
};

template <class Config, class Result, class EvalFn, class ProgressFn>
[[nodiscard]] inline std::vector<Result> parallel_evaluate_configs(
    const std::vector<Config>& configs, EvalFn eval_fn,
    ProgressFn progress_fn) {
    if (configs.empty()) {
        return {};
    }

    const std::size_t total = configs.size();
    std::vector<std::optional<Result>> slots(total);
    std::atomic<std::size_t> next_index{0};
    std::atomic<std::size_t> completed{0};
    std::mutex error_mutex;
    std::mutex progress_mutex;
    std::condition_variable progress_cv;
    std::exception_ptr first_error;
    std::stop_source stop_source;
    std::optional<Config> last_completed_config;

    const unsigned hw = std::thread::hardware_concurrency();
    const std::size_t worker_count = std::max<std::size_t>(
        1, std::min<std::size_t>(total, hw == 0 ? 4 : hw));

    std::vector<std::jthread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&](std::stop_token st) {
            while (!st.stop_requested()) {
                const std::size_t index = next_index.fetch_add(1);
                if (index >= total) {
                    break;
                }

                try {
                    slots[index].emplace(eval_fn(configs[index], index));
                    {
                        std::scoped_lock lock(progress_mutex);
                        last_completed_config = configs[index];
                    }
                    completed.fetch_add(1);
                    progress_cv.notify_one();
                }
                catch (...) {
                    {
                        std::scoped_lock lock(error_mutex);
                        if (!first_error) {
                            first_error = std::current_exception();
                        }
                    }
                    stop_source.request_stop();
                    progress_cv.notify_all();
                    break;
                }
            }
        });
    }

    std::size_t last_completed = static_cast<std::size_t>(-1);
    while (true) {
        std::optional<Config> config_snapshot;
        {
            std::unique_lock lock(progress_mutex);
            progress_cv.wait(lock, [&] {
                return stop_source.stop_requested() ||
                       completed.load() != last_completed;
            });
            config_snapshot = last_completed_config;
        }

        const std::size_t done = completed.load();
        if (done != last_completed) {
            progress_fn(done, config_snapshot);
            last_completed = done;
        }
        if (stop_source.stop_requested() || done >= total) {
            break;
        }
    }

    workers.clear();

    if (first_error) {
        std::rethrow_exception(first_error);
    }

    std::vector<Result> results;
    results.reserve(total);
    for (auto& slot : slots) {
        if (!slot.has_value()) {
            throw std::runtime_error(
                "[benchmark] missing result slot after parallel evaluation");
        }
        results.push_back(std::move(*slot));
    }
    return results;
}

template <class Run>
[[nodiscard]] inline std::vector<std::size_t> make_ranked_order(
    const std::vector<Run>& runs) {
    std::vector<std::size_t> order(runs.size());
    std::vector<std::size_t> costs(runs.size());
    for (std::size_t i = 0; i < runs.size(); ++i) {
        order[i] = i;
        costs[i] = estimated_work_units(runs[i]);
    }

    std::sort(order.begin(), order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                  const auto& a = runs[lhs];
                  const auto& b = runs[rhs];
                  if (a.psnr != b.psnr) {
                      return a.psnr > b.psnr;
                  }
                  if (costs[lhs] != costs[rhs]) {
                      return costs[lhs] < costs[rhs];
                  }
                  return simpler_than(a, b);
              });
    return order;
}

template <class Summary, class Config, class Run, class EvalFn,
          class DescribeFn>
[[nodiscard]] inline Summary run_parallel_benchmark_summary(
    std::string title, const std::vector<Config>& configs, EvalFn eval_fn,
    DescribeFn describe_fn) {
    Summary summary;
    summary.title = std::move(title);
    ProgressPrinter<Config> progress(summary.title, configs.size());
    std::mutex best_mutex;
    std::size_t best_index = std::numeric_limits<std::size_t>::max();
    double best_psnr = -std::numeric_limits<double>::infinity();
    rik_cv::Image<rik_cv::Gray8> best_image;

    std::cout << summary.title << '\n';

    summary.runs = parallel_evaluate_configs<Config, Run>(
        configs,
        [&](const Config& config, std::size_t index) {
            auto [run, image] = eval_fn(config);
            {
                std::scoped_lock lock(best_mutex);
                if (better_candidate(run.psnr, index, best_psnr, best_index)) {
                    best_psnr = run.psnr;
                    best_index = index;
                    best_image = image;
                }
            }
            return run;
        },
        [&](std::size_t done, const std::optional<Config>& last_config) {
            progress.update(done, last_config.value_or(Config{}), describe_fn);
        });

    if (!summary.runs.empty()) {
        summary.best_index = best_index;
        summary.runs[summary.best_index].image = std::move(best_image);
    }
    progress.finish();
    return summary;
}

[[nodiscard]] inline rik_cv::Image<rik_cv::Gray8> make_reference_target(
    rik_cv::ImageView<const rik_cv::Gray8> src, int radius, int pad_size,
    int stride_x, int stride_y) {
    if (radius < 1 || pad_size < 0 || stride_x <= 0 || stride_y <= 0) {
        throw std::runtime_error("[benchmark] invalid benchmark geometry");
    }

    const int kernel_size = 2 * radius + 1;
    const int out_w = rik_cv::detail::padded_extent(src.width(), kernel_size,
                                                    pad_size, stride_x);
    const int out_h = rik_cv::detail::padded_extent(src.height(), kernel_size,
                                                    pad_size, stride_y);
    auto dst = rik_cv::make_image<rik_cv::Gray8>(out_w, out_h);

    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            const int src_y = y * stride_y + radius - pad_size;
            const int src_x = x * stride_x + radius - pad_size;
            dst(y, x) = rik_cv::reflect_101_padding{}(src, src_y, src_x);
        }
    }
    return dst;
}

template <int Radius, int MaxRadius, class Fn>
decltype(auto) dispatch_radius(int target_radius, Fn&& fn) {
    if (target_radius == Radius) {
        return std::forward<Fn>(fn).template operator()<Radius>();
    }
    if constexpr (Radius < MaxRadius) {
        return dispatch_radius<Radius + 1, MaxRadius>(target_radius,
                                                      std::forward<Fn>(fn));
    }
    else {
        throw std::runtime_error("[benchmark] unsupported radius: " +
                                 std::to_string(target_radius));
    }
}

template <int MaxRadius = 5, class Fn>
decltype(auto) dispatch_two_radii(int first_radius, int second_radius,
                                  Fn&& fn) {
    return dispatch_radius<1, MaxRadius>(first_radius, [&]<int FirstRadius>() {
        return dispatch_radius<1, MaxRadius>(
            second_radius, [&]<int SecondRadius>() -> decltype(auto) {
                return std::forward<Fn>(fn)
                    .template operator()<FirstRadius, SecondRadius>();
            });
    });
}

template <int MaxRadius = 5, class FilterFn>
[[nodiscard]] inline rik_cv::Image<rik_cv::Gray8> run_filter_config(
    const BenchmarkConfig& config, rik_cv::ImageView<const rik_cv::Gray8> noise,
    FilterFn& filter_fn) {
    return dispatch_radius<1, MaxRadius>(config.radius, [&]<int Radius>() {
        return filter_fn
            .template operator()<Radius, rik_cv::reflect_101_padding>(
                noise, config.pad_size, config.pad_size, config.stride_x,
                config.stride_y, rik_cv::reflect_101_padding{});
    });
}

template <class FilterFn>
[[nodiscard]] inline BenchmarkSummary benchmark_configs(
    std::string title, const std::vector<BenchmarkConfig>& configs,
    rik_cv::ImageView<const rik_cv::Gray8> noise,
    rik_cv::ImageView<const rik_cv::Gray8> ref, FilterFn filter_fn) {
    return run_parallel_benchmark_summary<BenchmarkSummary, BenchmarkConfig,
                                          BenchmarkRun>(
        std::move(title), configs,
        [&](const BenchmarkConfig& config) {
            auto result = run_filter_config(config, noise, filter_fn);
            const auto ref_target =
                make_reference_target(ref, config.radius, config.pad_size,
                                      config.stride_x, config.stride_y);
            const double psnr = compute_psnr(ref_target.as_const_view(),
                                             result.as_const_view());
            return std::pair{BenchmarkRun{
                                 .radius = config.radius,
                                 .kernel_size = 2 * config.radius + 1,
                                 .pad_size = config.pad_size,
                                 .stride_x = config.stride_x,
                                 .stride_y = config.stride_y,
                                 .output_width = result.width(),
                                 .output_height = result.height(),
                                 .psnr = psnr,
                             },
                             std::move(result)};
        },
        [](const BenchmarkConfig& cfg) {
            std::ostringstream oss;
            oss << "radius=" << ansi::yellow << cfg.radius << ansi::reset
                << " pad=" << ansi::yellow << cfg.pad_size << ansi::reset
                << " stride=" << cfg.stride_x << 'x' << cfg.stride_y;
            return oss.str();
        });
}

[[nodiscard]] inline std::vector<BenchmarkConfig> make_benchmark_configs(
    const std::vector<int>& radii, const std::vector<int>& pad_sizes,
    const std::vector<int>& strides) {
    std::vector<BenchmarkConfig> configs;
    configs.reserve(radii.size() * pad_sizes.size() * strides.size());

    for (const int radius : radii) {
        for (const int pad_size : pad_sizes) {
            for (const int stride : strides) {
                configs.push_back(BenchmarkConfig{
                    .radius = radius,
                    .pad_size = pad_size,
                    .stride_x = stride,
                    .stride_y = stride,
                });
            }
        }
    }
    return configs;
}

[[nodiscard]] inline std::vector<CombinedBenchmarkConfig>
make_combined_benchmark_configs(const std::vector<int>& radii,
                                const std::vector<int>& pad_sizes) {
    struct FilterSpec
    {
        int radius{};
        int pad{};
    };

    std::vector<CombinedBenchmarkConfig> configs;
    configs.reserve(2 * radii.size() * radii.size() * pad_sizes.size() *
                    pad_sizes.size());

    std::vector<FilterSpec> specs;
    specs.reserve(radii.size() * pad_sizes.size());
    for (const int radius : radii) {
        for (const int pad_size : pad_sizes) {
            specs.push_back(FilterSpec{.radius = radius, .pad = pad_size});
        }
    }

    for (const auto order : {FilterOrder::median_then_gaussian,
                             FilterOrder::gaussian_then_median}) {
        for (const auto& median : specs) {
            for (const auto& gaussian : specs) {
                configs.push_back(CombinedBenchmarkConfig{
                    .order = order,
                    .median_radius = median.radius,
                    .median_pad = median.pad,
                    .gaussian_radius = gaussian.radius,
                    .gaussian_pad = gaussian.pad,
                });
            }
        }
    }
    return configs;
}

inline void print_benchmark_summary(const BenchmarkSummary& summary) {
    std::cout << ansi::bold << ansi::cyan << summary.title << ansi::reset
              << '\n';
    for (const auto& run : summary.runs) {
        std::cout << "  radius=" << ansi::yellow << run.radius << ansi::reset
                  << "  kernel=" << run.kernel_size << 'x' << run.kernel_size
                  << "  pad=" << run.pad_size << "  stride=" << run.stride_x
                  << 'x' << run.stride_y << "  PSNR=" << ansi::green
                  << std::fixed << std::setprecision(3) << run.psnr << " dB"
                  << ansi::reset << '\n';
    }

    const auto& best = summary.best();
    std::cout << '\n'
              << ansi::bold << ansi::green << "Best: " << ansi::reset
              << "radius=" << best.radius << "  kernel=" << best.kernel_size
              << 'x' << best.kernel_size << "  pad=" << best.pad_size
              << "  stride=" << best.stride_x << 'x' << best.stride_y
              << "  PSNR=" << ansi::bold << ansi::green << std::fixed
              << std::setprecision(3) << best.psnr << " dB" << ansi::reset
              << '\n';
}

inline void save_benchmark_csv(const BenchmarkSummary& summary,
                               const std::filesystem::path& output_path) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("[benchmark] failed to open output file: " +
                                 output_path.string());
    }

    out << "radius,kernel_size,pad_size,stride_x,stride_y,psnr_db\n";
    for (const auto& run : summary.runs) {
        out << run.radius << ',' << run.kernel_size << ',' << run.pad_size
            << ',' << run.stride_x << ',' << run.stride_y << ',' << std::fixed
            << std::setprecision(6) << run.psnr << '\n';
    }
}

inline void print_top_k_runs(const BenchmarkSummary& summary,
                             std::size_t k = 5) {
    const auto order = make_ranked_order(summary.runs);

    const std::size_t count = std::min(k, order.size());
    std::cout << ansi::bold << "Top-" << count << " Candidates" << ansi::reset
              << '\n';
    for (std::size_t rank = 0; rank < count; ++rank) {
        const auto& run = summary.runs[order[rank]];
        std::cout << "  #" << (rank + 1) << "  radius=" << run.radius
                  << " kernel=" << run.kernel_size << 'x' << run.kernel_size
                  << " pad=" << run.pad_size << " stride=" << run.stride_x
                  << 'x' << run.stride_y << " PSNR=" << std::fixed
                  << std::setprecision(3) << run.psnr << " dB"
                  << " cost=" << estimated_work_units(run) << '\n';
    }
}

inline void print_local_sensitivity(const BenchmarkSummary& summary) {
    const auto& best = summary.best();
    std::cout << ansi::bold << "Local Sensitivity Around Best" << ansi::reset
              << '\n';

    bool found = false;
    for (const auto& run : summary.runs) {
        const bool radius_neighbor = run.pad_size == best.pad_size &&
                                     run.stride_x == best.stride_x &&
                                     run.stride_y == best.stride_y &&
                                     std::abs(run.radius - best.radius) == 1;
        const bool pad_neighbor = run.radius == best.radius &&
                                  run.stride_x == best.stride_x &&
                                  run.stride_y == best.stride_y &&
                                  std::abs(run.pad_size - best.pad_size) == 1;
        const bool stride_neighbor =
            run.radius == best.radius && run.pad_size == best.pad_size &&
            run.stride_x == run.stride_y && best.stride_x == best.stride_y &&
            std::abs(run.stride_x - best.stride_x) == 1;

        if (!(radius_neighbor || pad_neighbor || stride_neighbor)) {
            continue;
        }

        found = true;
        std::cout << "  radius=" << run.radius << " kernel=" << run.kernel_size
                  << 'x' << run.kernel_size << " pad=" << run.pad_size
                  << " stride=" << run.stride_x << 'x' << run.stride_y
                  << " delta_psnr=" << std::showpos << std::fixed
                  << std::setprecision(3) << (run.psnr - best.psnr) << " dB"
                  << std::noshowpos << '\n';
    }

    if (!found) {
        std::cout
            << "  No immediate neighbors were present in the sampled grid.\n";
    }
}

inline void print_close_candidate_tradeoffs(const BenchmarkSummary& summary,
                                            double max_psnr_gap_db = 0.1) {
    const auto& best = summary.best();
    std::cout << ansi::bold << "Close Candidate Tradeoffs" << ansi::reset
              << '\n';
    const auto best_cost = estimated_work_units(best);

    bool found = false;
    for (const auto& run : summary.runs) {
        if (&run == &best) {
            continue;
        }

        const double gap = best.psnr - run.psnr;
        if (gap < 0.0 || gap > max_psnr_gap_db) {
            continue;
        }

        found = true;
        const auto run_cost = estimated_work_units(run);
        std::cout << "  radius=" << run.radius << " kernel=" << run.kernel_size
                  << 'x' << run.kernel_size << " pad=" << run.pad_size
                  << " stride=" << run.stride_x << 'x' << run.stride_y
                  << " gap=" << std::fixed << std::setprecision(3) << gap
                  << " dB" << " cost_ratio=" << std::setprecision(2)
                  << (best_cost == 0 ? 0.0
                                     : static_cast<double>(run_cost) /
                                           static_cast<double>(best_cost))
                  << " simpler=" << (simpler_than(run, best) ? "yes" : "no")
                  << '\n';
    }

    if (!found) {
        std::cout << "  No candidates were within " << std::fixed
                  << std::setprecision(3) << max_psnr_gap_db
                  << " dB of the best run.\n";
    }
}

[[nodiscard]] inline CombinedBenchmarkSummary benchmark_combined_configs(
    std::string title, const std::vector<CombinedBenchmarkConfig>& configs,
    rik_cv::ImageView<const rik_cv::Gray8> noise,
    rik_cv::ImageView<const rik_cv::Gray8> ref) {
    return run_parallel_benchmark_summary<CombinedBenchmarkSummary,
                                          CombinedBenchmarkConfig,
                                          CombinedBenchmarkRun>(
        std::move(title), configs,
        [&](const CombinedBenchmarkConfig& config) {
            auto result = dispatch_two_radii(
                config.median_radius, config.gaussian_radius,
                [&]<int MedianRadius, int GaussianRadius>() {
                    if (config.order == FilterOrder::median_then_gaussian) {
                        return rik_cv::median_then_gaussian_filter<
                            MedianRadius, GaussianRadius>(
                            noise, config.median_pad, config.median_pad,
                            config.gaussian_pad, config.gaussian_pad);
                    }

                    return rik_cv::gaussian_then_median_filter<GaussianRadius,
                                                               MedianRadius>(
                        noise, config.gaussian_pad, config.gaussian_pad,
                        config.median_pad, config.median_pad);
                });

            const auto ref_stage1 =
                config.order == FilterOrder::median_then_gaussian
                    ? make_reference_target(ref, config.median_radius,
                                            config.median_pad, 1, 1)
                    : make_reference_target(ref, config.gaussian_radius,
                                            config.gaussian_pad, 1, 1);
            const auto ref_target =
                config.order == FilterOrder::median_then_gaussian
                    ? make_reference_target(ref_stage1.as_const_view(),
                                            config.gaussian_radius,
                                            config.gaussian_pad, 1, 1)
                    : make_reference_target(ref_stage1.as_const_view(),
                                            config.median_radius,
                                            config.median_pad, 1, 1);
            const double psnr = compute_psnr(ref_target.as_const_view(),
                                             result.as_const_view());
            return std::pair{CombinedBenchmarkRun{
                                 .order = config.order,
                                 .median_radius = config.median_radius,
                                 .median_pad = config.median_pad,
                                 .gaussian_radius = config.gaussian_radius,
                                 .gaussian_pad = config.gaussian_pad,
                                 .output_width = result.width(),
                                 .output_height = result.height(),
                                 .psnr = psnr,
                             },
                             std::move(result)};
        },
        [](const CombinedBenchmarkConfig& cfg) {
            std::ostringstream oss;
            oss << "order=" << filter_order_name(cfg.order)
                << " mr=" << ansi::yellow << cfg.median_radius << ansi::reset
                << " mp=" << cfg.median_pad << " gr=" << ansi::yellow
                << cfg.gaussian_radius << ansi::reset
                << " gp=" << cfg.gaussian_pad;
            return oss.str();
        });
}

inline void print_combined_summary(const CombinedBenchmarkSummary& summary) {
    std::cout << ansi::bold << ansi::cyan << summary.title << ansi::reset
              << '\n';
    const auto& best = summary.best();
    std::cout << '\n'
              << ansi::bold << ansi::green << "Best: " << ansi::reset
              << "order=" << filter_order_name(best.order)
              << " median(r=" << best.median_radius
              << ", pad=" << best.median_pad << ")"
              << " gaussian(r=" << best.gaussian_radius
              << ", pad=" << best.gaussian_pad << ")" << " PSNR=" << ansi::bold
              << ansi::green << std::fixed << std::setprecision(3) << best.psnr
              << " dB" << ansi::reset << '\n';
}

inline void print_combined_top_k(const CombinedBenchmarkSummary& summary,
                                 std::size_t k = 5) {
    const auto order = make_ranked_order(summary.runs);

    const std::size_t count = std::min(k, order.size());
    std::cout << ansi::bold << "Top-" << count << " Q3 Candidates"
              << ansi::reset << '\n';
    for (std::size_t rank = 0; rank < count; ++rank) {
        const auto& run = summary.runs[order[rank]];
        std::cout << "  #" << (rank + 1)
                  << " order=" << filter_order_name(run.order)
                  << " median(r=" << run.median_radius
                  << ",p=" << run.median_pad
                  << ") gaussian(r=" << run.gaussian_radius
                  << ",p=" << run.gaussian_pad << ") PSNR=" << std::fixed
                  << std::setprecision(3) << run.psnr << " dB"
                  << " cost=" << estimated_work_units(run) << '\n';
    }
}

inline void print_combined_local_sensitivity(
    const CombinedBenchmarkSummary& summary) {
    const auto& best = summary.best();
    std::cout << ansi::bold << "Q3 Local Sensitivity Around Best" << ansi::reset
              << '\n';

    bool found = false;
    for (const auto& run : summary.runs) {
        const bool same_order = run.order == best.order;
        const bool median_radius_neighbor =
            same_order && run.median_pad == best.median_pad &&
            run.gaussian_radius == best.gaussian_radius &&
            run.gaussian_pad == best.gaussian_pad &&
            std::abs(run.median_radius - best.median_radius) == 1;
        const bool median_pad_neighbor =
            same_order && run.median_radius == best.median_radius &&
            run.gaussian_radius == best.gaussian_radius &&
            run.gaussian_pad == best.gaussian_pad &&
            std::abs(run.median_pad - best.median_pad) == 1;
        const bool gaussian_radius_neighbor =
            same_order && run.median_radius == best.median_radius &&
            run.median_pad == best.median_pad &&
            run.gaussian_pad == best.gaussian_pad &&
            std::abs(run.gaussian_radius - best.gaussian_radius) == 1;
        const bool gaussian_pad_neighbor =
            same_order && run.median_radius == best.median_radius &&
            run.median_pad == best.median_pad &&
            run.gaussian_radius == best.gaussian_radius &&
            std::abs(run.gaussian_pad - best.gaussian_pad) == 1;

        if (!(median_radius_neighbor || median_pad_neighbor ||
              gaussian_radius_neighbor || gaussian_pad_neighbor)) {
            continue;
        }

        found = true;
        std::cout << "  order=" << filter_order_name(run.order)
                  << " median(r=" << run.median_radius
                  << ",p=" << run.median_pad
                  << ") gaussian(r=" << run.gaussian_radius
                  << ",p=" << run.gaussian_pad
                  << ") delta_psnr=" << std::showpos << std::fixed
                  << std::setprecision(3) << (run.psnr - best.psnr) << " dB"
                  << std::noshowpos << '\n';
    }

    if (!found) {
        std::cout
            << "  No immediate neighbors were present in the sampled grid.\n";
    }
}

inline void print_combined_close_candidate_tradeoffs(
    const CombinedBenchmarkSummary& summary, double max_psnr_gap_db = 0.1) {
    const auto& best = summary.best();
    std::cout << ansi::bold << "Q3 Close Candidate Tradeoffs" << ansi::reset
              << '\n';
    const auto best_cost = estimated_work_units(best);

    bool found = false;
    for (const auto& run : summary.runs) {
        if (&run == &best) {
            continue;
        }

        const double gap = best.psnr - run.psnr;
        if (gap < 0.0 || gap > max_psnr_gap_db) {
            continue;
        }

        found = true;
        const auto run_cost = estimated_work_units(run);
        std::cout << "  order=" << filter_order_name(run.order)
                  << " median(r=" << run.median_radius
                  << ",p=" << run.median_pad
                  << ") gaussian(r=" << run.gaussian_radius
                  << ",p=" << run.gaussian_pad << ") gap=" << std::fixed
                  << std::setprecision(3) << gap << " dB"
                  << " cost_ratio=" << std::setprecision(2)
                  << (best_cost == 0 ? 0.0
                                     : static_cast<double>(run_cost) /
                                           static_cast<double>(best_cost))
                  << " simpler=" << (simpler_than(run, best) ? "yes" : "no")
                  << '\n';
    }

    if (!found) {
        std::cout << "  No candidates were within " << std::fixed
                  << std::setprecision(3) << max_psnr_gap_db
                  << " dB of the best run.\n";
    }
}

inline void save_combined_benchmark_csv(
    const CombinedBenchmarkSummary& summary,
    const std::filesystem::path& output_path) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error(
            "[combined_benchmark] failed to open output file: " +
            output_path.string());
    }

    out << "order,median_radius,median_pad,gaussian_radius,gaussian_pad,psnr_"
           "db\n";
    for (const auto& run : summary.runs) {
        out << filter_order_name(run.order) << ',' << run.median_radius << ','
            << run.median_pad << ',' << run.gaussian_radius << ','
            << run.gaussian_pad << ',' << std::fixed << std::setprecision(6)
            << run.psnr << '\n';
    }
}

[[nodiscard]] inline CombinedBenchmarkSummary merge_combined_summaries(
    std::string title, CombinedBenchmarkSummary coarse,
    CombinedBenchmarkSummary fine) {
    CombinedBenchmarkSummary merged;
    merged.title = std::move(title);
    merged.runs = std::move(coarse.runs);
    for (auto& run : fine.runs) {
        merged.runs.push_back(std::move(run));
    }

    if (!merged.runs.empty()) {
        double best_psnr = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < merged.runs.size(); ++i) {
            if (better_candidate(merged.runs[i].psnr, i, best_psnr,
                                 merged.best_index)) {
                merged.best_index = i;
                best_psnr = merged.runs[i].psnr;
            }
        }
    }
    return merged;
}

[[nodiscard]] inline std::vector<int> neighbor_values(int center, int low,
                                                      int high) {
    std::vector<int> values;
    for (int v = std::max(low, center - 1); v <= std::min(high, center + 1);
         ++v) {
        values.push_back(v);
    }
    return values;
}

inline void append_benchmark_summary_log(std::ostream& os,
                                         const BenchmarkSummary& summary) {
    const auto& best = summary.best();
    os << summary.title << '\n';
    os << "Best: radius=" << best.radius << " kernel=" << best.kernel_size
       << 'x' << best.kernel_size << " pad=" << best.pad_size
       << " stride=" << best.stride_x << 'x' << best.stride_y
       << " psnr_db=" << std::fixed << std::setprecision(6) << best.psnr
       << '\n';
    os << "Top candidates:\n";

    const auto order = make_ranked_order(summary.runs);

    const std::size_t count = std::min<std::size_t>(5, order.size());
    for (std::size_t rank = 0; rank < count; ++rank) {
        const auto& run = summary.runs[order[rank]];
        os << "  #" << (rank + 1) << " radius=" << run.radius
           << " kernel=" << run.kernel_size << 'x' << run.kernel_size
           << " pad=" << run.pad_size << " stride=" << run.stride_x << 'x'
           << run.stride_y << " psnr_db=" << std::fixed << std::setprecision(6)
           << run.psnr << " cost=" << estimated_work_units(run) << '\n';
    }
    os << '\n';
}

inline void append_combined_benchmark_summary_log(
    std::ostream& os, const CombinedBenchmarkSummary& summary) {
    const auto& best = summary.best();
    os << summary.title << '\n';
    os << "Best: order=" << filter_order_name(best.order)
       << " median_radius=" << best.median_radius
       << " median_pad=" << best.median_pad
       << " gaussian_radius=" << best.gaussian_radius
       << " gaussian_pad=" << best.gaussian_pad << " psnr_db=" << std::fixed
       << std::setprecision(6) << best.psnr << '\n';
    os << "Top candidates:\n";

    const auto order = make_ranked_order(summary.runs);

    const std::size_t count = std::min<std::size_t>(5, order.size());
    for (std::size_t rank = 0; rank < count; ++rank) {
        const auto& run = summary.runs[order[rank]];
        os << "  #" << (rank + 1) << " order=" << filter_order_name(run.order)
           << " median_radius=" << run.median_radius
           << " median_pad=" << run.median_pad
           << " gaussian_radius=" << run.gaussian_radius
           << " gaussian_pad=" << run.gaussian_pad << " psnr_db=" << std::fixed
           << std::setprecision(6) << run.psnr
           << " cost=" << estimated_work_units(run) << '\n';
    }
    os << '\n';
}

}  // namespace hw2_bench
