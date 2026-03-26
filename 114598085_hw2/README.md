# 114598085 HW2

## Build

This project is configured with CMake presets in [`project_hw2/CMakePresets.json`](./project_hw2/CMakePresets.json).  
Our preferred setup is `Clang + Ninja`, because the project already provides presets for that combination.

Recommended Requirements:
- CMake 3.20 or newer
- Ninja
- Clang 10 or newer for baseline C++20 support
- OpenCV development package installed and discoverable by CMake

Recommended commands:

```bash
cd project_hw2
cmake --preset clang-ninja
cmake --build --preset build-clang-ninja
```

The executable will be generated in `project_hw2/build/clang-ninja/`.

To configure a `Release` build with the same preset, override the build type at
configure time and then build with the matching build preset:

```bash
cd project_hw2
rm -rf build/clang-ninja
cmake --preset clang-ninja -DCMAKE_BUILD_TYPE=Release
cmake --build --preset build-clang-ninja
```

Other preset options:

| Preset        | Toolchain     | Build preset        |
| ------------- | ------------- | ------------------- |
| `clang-ninja` | Clang + Ninja | `build-clang-ninja` |
| `clang-make`  | Clang + Make  | `build-clang-make`  |
| `gcc-ninja`   | GCC + Ninja   | `build-gcc-ninja`   |
| `gcc-make`    | GCC + Make    | `build-gcc-make`    |
| `default`     | Default + Ninja | `build-default`   |

## Run

After building, run the executable from the `project_hw2` directory:

```bash
./build/clang-ninja/project_hw2
```

The program loads:
- `project_hw2/test_imgs/CKS_noise.png`
- `project_hw2/test_imgs/CKS_grayscale.png`

In `Debug` builds, it benchmarks the following tasks:
- Q1: median filter parameter sweep
- Q2: gaussian filter parameter sweep
- Q3: combined median and gaussian filtering with coarse-to-fine search

In `Release` builds, it skips the benchmark sweep and directly uses the best
configurations recorded in `project_hw2/result_reports/report_summary.log`:
- Q1: median filter with `radius=1`, `pad=5`, `stride=3x3`
- Q2: gaussian filter with `radius=3`, `pad=5`, `stride=3x3`
- Q3: `median -> gaussian` with `median_radius=1`, `median_pad=5`,
  `gaussian_radius=1`, `gaussian_pad=5`

## Generated Outputs

The program writes result images to `project_hw2/result_imgs/`:
- `CKS_Q1.png`
- `CKS_Q2.png`
- `CKS_Q3.png`

`Debug` builds also write benchmark reports to `project_hw2/result_reports/`:
- `CKS_Q1_psnr.csv`
- `CKS_Q2_psnr.csv`
- `CKS_Q3_psnr.csv`
- `report_summary.log`

`Release` builds still generate the output images and histograms, but they do
not rerun the sweep or regenerate the benchmark CSV/summary files.

## Project Layout

- `project_hw2/include/`: header files
- `project_hw2/src/`: source files
- `project_hw2/test_imgs/`: input images for testing
- `project_hw2/result_imgs/`: generated output images
- `project_hw2/result_reports/`: generated benchmark CSV files and summary log

## Notes

- The project uses C++20 in [`project_hw2/CMakeLists.txt`](./project_hw2/CMakeLists.txt).
- OpenCV is required for image loading and saving through [`project_hw2/include/opencv_bridge.hpp`](./project_hw2/include/opencv_bridge.hpp).
- Warnings are enabled for Clang and GCC with `-Wall -Wextra -Wpedantic`.
