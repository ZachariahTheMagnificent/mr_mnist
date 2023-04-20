#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <span>
#include <vector>

namespace mr_mnist {
class Labels {
public:
  std::size_t size;
  std::vector<unsigned char> values;

  Labels(const std::string &path) {
    auto file = std::ifstream{path, std::ifstream::binary};

    auto magic_number = std::array<char, sizeof(std::int32_t)>{};
    file.read(magic_number.data(), std::size(magic_number));
    std::reverse(magic_number.begin(), magic_number.end());

    assert(reinterpret_cast<std::int32_t &>(magic_number) == 2049);

    auto size = std::array<char, sizeof(std::int32_t)>{};
    file.read(size.data(), std::size(size));
    std::reverse(size.begin(), size.end());
    this->size = reinterpret_cast<std::int32_t &>(size);

    std::copy(std::istreambuf_iterator<char>{file},
              std::istreambuf_iterator<char>{}, std::back_inserter(values));

    assert(values.size() == this->size);
  }
};

class Images {
public:
  int width;
  int height;
  std::size_t size;
  std::vector<unsigned char> pixels;

  Images(const std::string &path) {
    auto file = std::ifstream{path, std::ifstream::binary};

    auto magic_number = std::array<char, sizeof(std::int32_t)>{};
    file.read(magic_number.data(), std::size(magic_number));
    std::reverse(magic_number.begin(), magic_number.end());

    assert(reinterpret_cast<std::int32_t &>(magic_number) == 2051);

    auto size = std::array<char, sizeof(std::int32_t)>{};
    file.read(size.data(), std::size(size));
    std::reverse(size.begin(), size.end());
    this->size = reinterpret_cast<std::int32_t &>(size);

    auto width = std::array<char, sizeof(std::int32_t)>{};
    file.read(width.data(), std::size(width));
    std::reverse(width.begin(), width.end());
    this->width = reinterpret_cast<std::int32_t &>(width);

    auto height = std::array<char, sizeof(std::int32_t)>{};
    file.read(height.data(), std::size(height));
    std::reverse(height.begin(), height.end());
    this->height = reinterpret_cast<std::int32_t &>(height);

    std::copy(std::istreambuf_iterator<char>{file},
              std::istreambuf_iterator<char>{},
              std::back_inserter(this->pixels));

    assert(this->pixels.size() == this->width * this->height * this->size);
  }
};
} // namespace mr_mnist
