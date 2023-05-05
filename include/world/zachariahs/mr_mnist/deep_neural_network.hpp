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
#include <cmath>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <vector>
#include <world/zachariahs/mr_mnist/math.hpp>

namespace world::zachariahs::mr_mnist {
class DeepNeuralNetwork {
public:
  std::vector<float> weights;
  std::vector<float> input;
  std::vector<float> result;
  std::vector<std::size_t> layer_sizes;

  DeepNeuralNetwork(const std::string &save_file, std::mt19937_64 &generator,
                    const std::span<const std::size_t> layer_sizes) {
    this->layer_sizes.resize(layer_sizes.front());
    std::copy(layer_sizes.begin(), layer_sizes.end(),
              this->layer_sizes.begin());

    auto input_layer_size = layer_sizes.front();
    auto biggest_size = input_layer_size;
    auto num_weights = std::size_t{};
    for (const auto size : layer_sizes.subspan(1)) {
      if (size > biggest_size) {
        biggest_size = size;
      }

      num_weights += (input_layer_size + 1) * size;
      input_layer_size = size;
    }

    weights.resize(num_weights);
    input.resize(biggest_size);
    result.resize(biggest_size);

    input_layer_size = layer_sizes.front();
    auto displacement = std::size_t{};
    for (const auto size : layer_sizes.subspan(1)) {
      const auto standard_deviation = std::sqrt(2.0f / input_layer_size);
      auto random_weight =
          std::normal_distribution<float>{0.f, standard_deviation};

      for (auto node_index = std::size_t{}; node_index < size; ++node_index) {
        for (auto weight_index = std::size_t{}; weight_index < input_layer_size;
             ++weight_index) {
          weights[weight_index + displacement] = random_weight(generator);
        }
        displacement += input_layer_size + 1;
      }
      input_layer_size = size;
    }
  }

  std::vector<float> operator()(const std::span<const float> initial_input) {
    const auto layer_sizes = std::span<const std::size_t>{this->layer_sizes};

    std::copy(initial_input.begin(), initial_input.end(), input.begin());
    auto input_layer_size = layer_sizes.front();
    auto displacement = std::size_t{};
    for (const auto layer_size : layer_sizes.subspan(1)) {
      for (auto node_index = std::size_t{}; node_index < layer_size;
           ++node_index) {
        result[node_index] = weights[input_layer_size + displacement];
        for (auto weight_index = std::size_t{}; weight_index < input_layer_size;
             ++weight_index) {
          result[node_index] +=
              input[weight_index] * weights[weight_index + displacement];
        }
        result[node_index] = sigmoid(result[node_index]);
        displacement += input_layer_size + 1;
      }
      std::swap(input, result);
      input_layer_size = layer_size;
    }

    auto return_value = std::vector<float>{};
    return_value.resize(input_layer_size);
    std::copy(input.begin(), input.begin() + input_layer_size,
              return_value.begin());
    return return_value;
  }
};
} // namespace world::zachariahs::mr_mnist
