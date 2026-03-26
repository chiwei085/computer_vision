cmake_minimum_required(VERSION 3.20)

project(project_hw2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

set(PROJECT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
)

set(PROJECT_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/algorithm.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/benchmark_utils.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/color_trans.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/convolution.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/filter.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/histogram.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/image.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/numeric_utils.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv_bridge.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/padding.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/pixel_transform.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/pixel_types.hpp"
)

if(PROJECT_SOURCES)
    add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS})
    target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            ${OpenCV_INCLUDE_DIRS}
    )

    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
    target_compile_definitions(
        ${PROJECT_NAME}
        PRIVATE
            PROJECT_HW2_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(
            ${PROJECT_NAME}
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()
else()
    add_custom_target(
        ${PROJECT_NAME}
        COMMAND
            ${CMAKE_COMMAND} -E echo
            "No C++ source files were found under ${CMAKE_CURRENT_SOURCE_DIR}/src."
    )

    message(WARNING "No C++ source files were found under ${CMAKE_CURRENT_SOURCE_DIR}/src.")
endif()
