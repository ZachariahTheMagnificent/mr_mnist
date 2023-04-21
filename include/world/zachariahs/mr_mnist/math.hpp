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

namespace mr_mnist {
template <typename Type> auto sigmoid(const Type value) {
  auto x = std::exp(value);
  return x / (x + 1);
}

template <typename Type> auto sigmoid_derivative(const Type value) {
  const auto sigmoid_result = sigmoid(value);
  return sigmoid_result * (1 - sigmoid_result);
}
} // namespace mr_mnist
