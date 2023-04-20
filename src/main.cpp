#include <cassert>
#include <iostream>
#include <mr_mnist/mr_mnist.hpp>

auto main(int argc, char** argv) -> int {
  std::cout << "Hello, I am Mr Mnist!\n";

  auto test_images =
      mr_mnist::Images{"share/Training Data/t10k-images-idx3-ubyte"};
  auto test_labels =
      mr_mnist::Labels{"share/Training Data/t10k-labels-idx1-ubyte"};

  assert(test_images.size == test_labels.size);

  auto training_images =
      mr_mnist::Images{"share/Training Data/train-images-idx3-ubyte"};
  auto training_labels =
      mr_mnist::Labels{"share/Training Data/train-labels-idx1-ubyte"};

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
