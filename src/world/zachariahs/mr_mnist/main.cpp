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
#include <algorithm>
#include <cassert>
#include <iostream>
#include <world/zachariahs/mr_mnist/deep_neural_network.hpp>
#include <world/zachariahs/mr_mnist/math.hpp>
#include <world/zachariahs/mr_mnist/resources.hpp>

namespace mr_mnist = world::zachariahs::mr_mnist;

auto main(int argc, char **argv) -> int {
  std::cout << "Hello, I am Mr Mnist!\n";

  auto test_images = mr_mnist::Images{
      "share/world/zachariahs/mr_mnist/Training Data/t10k-images-idx3-ubyte"};
  auto test_labels = mr_mnist::Labels{
      "share/world/zachariahs/mr_mnist/Training Data/t10k-labels-idx1-ubyte"};

  assert(test_images.size == test_labels.size);

  auto training_images = mr_mnist::Images{
      "share/world/zachariahs/mr_mnist/Training Data/train-images-idx3-ubyte"};
  auto training_labels = mr_mnist::Labels{
      "share/world/zachariahs/mr_mnist/Training Data/train-labels-idx1-ubyte"};

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

  const auto layer_sizes = std::array{
      static_cast<std::size_t>(test_images.width * test_images.height),
      std::size_t{16}, std::size_t{16}, std::size_t{10}};
  auto random_device = std::random_device{};
  auto generator = std::mt19937_64{random_device()};
  auto digit_recognizer = mr_mnist::DeepNeuralNetwork{"savefile", generator,
                                                      layer_sizes};
  std::cout << "Neural network created!\n";
  const auto input_size = test_images.width + test_images.height;
  auto input = std::vector<float>{};
  input.resize(input_size);
  std::transform(test_images.pixels.begin(), test_images.pixels.begin() + input_size,
                 input.begin(),
                 [](const unsigned char value) { return value / 255.0f; });
  digit_recognizer(input);
  std::cout << "Bye!\n";

  return EXIT_SUCCESS;
}
