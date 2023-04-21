/****************************************************************************
 * Mr MNIST
 * Copyright (C) 2023 Zachariah The Magnificent <zachariah@zachariahs.world>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/
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

namespace world::zachariahs::mr_mnist {
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
} // namespace world::zachariahs::mr_mnist
