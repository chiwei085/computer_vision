#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "pixel_types.hpp"

namespace rik_cv
{

struct image_error : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

using image_view_error = image_error;

template <class T>
requires std::is_object_v<T> class ImageView
{
public:
    using value_type = std::remove_cv_t<T>;
    using element_type = T;
    using pointer = T*;
    using reference = T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using byte_type =
        std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>;

    constexpr ImageView() noexcept = default;

    constexpr ImageView(pointer data, int width, int height)
        : ImageView(data, width, height, default_stride_bytes(width)) {}

    constexpr ImageView(pointer data, int width, int height,
                        difference_type stride_bytes)
        : data_(data),
          width_(width),
          height_(height),
          stride_bytes_(stride_bytes) {
        validate();
    }

    [[nodiscard]] constexpr pointer data() const noexcept { return data_; }
    [[nodiscard]] constexpr int width() const noexcept { return width_; }
    [[nodiscard]] constexpr int height() const noexcept { return height_; }

    [[nodiscard]] constexpr difference_type stride_bytes() const noexcept {
        return stride_bytes_;
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return data_ == nullptr || width_ <= 0 || height_ <= 0;
    }

    [[nodiscard]] constexpr bool is_contiguous() const noexcept {
        return stride_bytes_ == default_stride_bytes(width_);
    }

    [[nodiscard]] constexpr size_type row_bytes() const noexcept {
        return static_cast<size_type>(default_stride_bytes(width_));
    }

    [[nodiscard]] constexpr pointer row_ptr(int y) const noexcept {
        return reinterpret_cast<pointer>(reinterpret_cast<byte_type*>(data_) +
                                         static_cast<difference_type>(y) *
                                             stride_bytes_);
    }

    [[nodiscard]] constexpr reference operator()(int y, int x) const noexcept {
        return row_ptr(y)[x];
    }

    [[nodiscard]] std::span<T> row_span(int y) const noexcept {
        return {row_ptr(y), static_cast<size_type>(width_)};
    }

    [[nodiscard]] constexpr bool in_bounds(int y, int x) const noexcept {
        return y >= 0 && y < height_ && x >= 0 && x < width_;
    }

    [[nodiscard]] ImageView subview(int x, int y, int sub_width,
                                    int sub_height) const {
        if (x < 0 || y < 0 || sub_width < 0 || sub_height < 0) {
            throw image_view_error("subview: negative region");
        }
        if (x + sub_width > width_ || y + sub_height > height_) {
            throw image_view_error("subview: region out of bounds");
        }

        return ImageView{row_ptr(y) + x, sub_width, sub_height, stride_bytes_};
    }

    [[nodiscard]] ImageView<const value_type> as_const() const noexcept
        requires(!std::is_const_v<T>) {
        return ImageView<const value_type>{data_, width_, height_,
                                           stride_bytes_};
    }

    constexpr void validate() const {
        if (width_ < 0 || height_ < 0) {
            throw image_view_error("[ImageView] negative extent");
        }
        if ((width_ == 0 || height_ == 0) && data_ != nullptr) {
            return;
        }
        if ((width_ > 0 && height_ > 0) && data_ == nullptr) {
            throw image_view_error("[ImageView] null data for non-empty image");
        }
        if (stride_bytes_ < default_stride_bytes(width_)) {
            throw image_view_error(
                "[ImageView] stride is smaller than row size");
        }
    }

private:
    static constexpr difference_type default_stride_bytes(int width) noexcept {
        return static_cast<difference_type>(sizeof(element_type)) * width;
    }

    pointer data_{};
    int width_{};
    int height_{};
    difference_type stride_bytes_{};
};

template <class T>
class Image
{
public:
    using value_type = T;
    using size_type = std::size_t;

    Image() = default;

    Image(int width, int height) { resize(width, height); }

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] bool empty() const noexcept { return pixels_.empty(); }
    [[nodiscard]] size_type size() const noexcept { return pixels_.size(); }

    [[nodiscard]] T* data() noexcept { return pixels_.data(); }
    [[nodiscard]] const T* data() const noexcept { return pixels_.data(); }

    [[nodiscard]] ImageView<T> view() noexcept {
        return ImageView<T>{data(), width_, height_};
    }

    [[nodiscard]] ImageView<const T> view() const noexcept {
        return ImageView<const T>{data(), width_, height_};
    }

    [[nodiscard]] ImageView<const T> as_const_view() const noexcept {
        return view();
    }

    [[nodiscard]] T& operator()(int y, int x) noexcept { return view()(y, x); }

    [[nodiscard]] const T& operator()(int y, int x) const noexcept {
        return view()(y, x);
    }

    void resize(int width, int height) {
        if (width < 0 || height < 0) {
            throw image_error("[Image] negative extent");
        }

        width_ = width;
        height_ = height;
        pixels_.resize(static_cast<size_type>(width) *
                       static_cast<size_type>(height));
    }

private:
    std::vector<T> pixels_{};
    int width_{};
    int height_{};
};

template <class T>
[[nodiscard]] constexpr bool same_shape(const ImageView<T>& a,
                                        const ImageView<T>& b) noexcept {
    return a.width() == b.width() && a.height() == b.height();
}

template <class A, class B>
[[nodiscard]] constexpr bool same_extent(const ImageView<A>& a,
                                         const ImageView<B>& b) noexcept {
    return a.width() == b.width() && a.height() == b.height();
}

template <class T>
[[nodiscard]] constexpr auto make_image_view(
    T* data, int width, int height,
    std::ptrdiff_t stride_bytes) noexcept -> ImageView<T> {
    return ImageView<T>{data, width, height, stride_bytes};
}

template <class T>
[[nodiscard]] constexpr auto make_contiguous_image_view(
    T* data, int width, int height) noexcept -> ImageView<T> {
    return ImageView<T>{data, width, height,
                        static_cast<std::ptrdiff_t>(sizeof(T) * width)};
}

template <class T>
[[nodiscard]] inline Image<T> make_image(int width, int height) {
    return Image<T>{width, height};
}

template <class Backend>
struct image_io;

template <class Backend>
[[nodiscard]] inline Image<Rgb8> load_image(const std::filesystem::path& path) {
    return image_io<Backend>::load(path);
}

template <class Backend, class T>
inline void save_image(const std::filesystem::path& path,
                       ImageView<const T> view) {
    image_io<Backend>::template save<T>(path, view);
}

template <class Backend, class T>
inline void save_image(const std::filesystem::path& path, ImageView<T> view) {
    save_image<Backend>(path, view.as_const());
}

template <class Backend, class T>
inline void save_image(const std::filesystem::path& path,
                       const Image<T>& image) {
    save_image<Backend>(path, image.view());
}

template <class Backend, class T>
inline void save_image(const std::filesystem::path& path, Image<T>& image) {
    save_image<Backend>(path, image.view());
}

}  // namespace rik_cv
