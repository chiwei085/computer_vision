# 114598085 HW3

## Build

This project is configured with CMake presets in [`project_hw3/CMakePresets.json`](./project_hw3/CMakePresets.json).  
Our preferred setup is `Clang + Ninja`, because the project already provides presets for that combination.

Recommended Requirements:
- CMake 3.20 or newer
- Ninja
- Clang 10 or newer for baseline C++20 support
- OpenCV development package installed and discoverable by CMake

Recommended commands:

```bash
cd project_hw3
cmake --preset clang-ninja
cmake --build --preset build-clang-ninja
```

The executable will be generated in `project_hw3/build/clang-ninja/`.

Other preset options:

| Preset        | Toolchain       | Build preset        |
| ------------- | --------------- | ------------------- |
| `clang-ninja` | Clang + Ninja   | `build-clang-ninja` |
| `clang-make`  | Clang + Make    | `build-clang-make`  |
| `gcc-ninja`   | GCC + Ninja     | `build-gcc-ninja`   |
| `gcc-make`    | GCC + Make      | `build-gcc-make`    |
| `default`     | Default + Ninja | `build-default`     |

## Run

After building, run the executable from the `project_hw3` directory:

```bash
./build/clang-ninja/project_hw3
```

The program loads:
- `project_hw3/test_imgs/img1.png`
- `project_hw3/test_imgs/img2.png`
- `project_hw3/test_imgs/img3.png`

For each image, it generates:
- Q1: grayscale + Gaussian blur
- Q2: Canny edge detection
- Q3: Hough line detection with final card-border drawing

## Generated Outputs

The program writes result images to `project_hw3/result_imgs/`:
- `img1_q1.png`
- `img1_q2.png`
- `img1_q3.png`
- `img2_q1.png`
- `img2_q2.png`
- `img2_q3.png`
- `img3_q1.png`
- `img3_q2.png`
- `img3_q3.png`

The report source and compiled PDF are stored at the homework root:
- `main.typ`
- `114598085_hw3.pdf`

## Project Layout

- `project_hw3/include/`: header files
- `project_hw3/src/`: source files
- `project_hw3/test_imgs/`: input images for testing
- `project_hw3/result_imgs/`: generated output images
- `main.typ`: Typst report source
- `114598085_hw3.pdf`: compiled report

## Notes

- The project uses C++20 in [`project_hw3/CMakeLists.txt`](./project_hw3/CMakeLists.txt).
- OpenCV is required for image loading and saving through [`project_hw3/include/opencv_bridge.hpp`](./project_hw3/include/opencv_bridge.hpp).
- Warnings are enabled for Clang and GCC with `-Wall -Wextra -Wpedantic`.
