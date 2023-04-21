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
#include <cassert>
#include <iostream>
#include <world/zachariahs/mr_mnist/math.hpp>
#include <world/zachariahs/mr_mnist/resources.hpp>

auto main(int argc, char **argv) -> int {
  std::cout << "Hello, I am Mr Mnist!\n";

  auto test_images =
      mr_mnist::Images{"share/world/zachariahs/mr_mnist/Training Data/t10k-images-idx3-ubyte"};
  auto test_labels =
      mr_mnist::Labels{"share/world/zachariahs/mr_mnist/Training Data/t10k-labels-idx1-ubyte"};

  assert(test_images.size == test_labels.size);

  auto training_images =
      mr_mnist::Images{"share/world/zachariahs/mr_mnist/Training Data/train-images-idx3-ubyte"};
  auto training_labels =
      mr_mnist::Labels{"share/world/zachariahs/mr_mnist/Training Data/train-labels-idx1-ubyte"};

  assert(training_images.size == training_labels.size);

  auto image_index = std::stoi(argv[1]);
  std::cout << static_cast<int>(test_labels.values[image_index]) << '\n';
  for (auto row = std::size_t{}; row < test_images.width; ++row) {
    for (auto column = std::size_t{}; column < test_images.height; ++column) {
      if (test_images
              .pixels[column + row * test_images.width +
                      image_index * test_images.width * test_images.height] ==
          0) {
        std::cout << '-';
      } else {
        std::cout << '*';
      }
    }
    std::cout << '\n';
  }

  return EXIT_SUCCESS;
}