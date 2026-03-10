# 114598085 HW1

## Build

This project is configured with CMake presets in [`project_hw1/CMakePresets.json`](./project_hw1/CMakePresets.json).  
Our preferred setup is `Clang + Ninja`, because the project already provides presets for that combination.

Recommended Requirements:
- CMake 3.20 or newer
- Ninja
- Clang 10 or newer for baseline C++20 support

Recommended commands:

```bash
cd project_hw1
cmake --preset clang-ninja
cmake --build --preset build-clang-ninja
```

The executable will be generated in `project_hw1/build/clang-ninja/`.

Other preset options:

| Preset        | Toolchain     | Build preset        |
| ------------- | ------------- | ------------------- |
| `clang-ninja` | Clang + Ninja | `build-clang-ninja` |
| `clang-make`  | Clang + Make  | `build-clang-make`  |
| `gcc-ninja`   | GCC + Ninja   | `build-gcc-ninja`   |
| `gcc-make`    | GCC + Make    | `build-gcc-make`    |

## Project Layout

- `project_hw1/include/`: header files
- `project_hw1/src/`: source files
- `project_hw1/test_imgs/`: input images for testing
- `project_hw1/result_imgs/`: generated output images

## Notes

- The project uses C++20 in [`project_hw1/CMakeLists.txt`](./project_hw1/CMakeLists.txt).
- Warnings are enabled for Clang and GCC with `-Wall -Wextra -Wpedantic`.

