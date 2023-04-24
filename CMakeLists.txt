#[===========================================================================[
 # Mr MNIST
 # Copyright (C) 2023 Zachariah The Magnificent <zachariah@zachariahs.world>
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program.  If not, see <https://www.gnu.org/licenses/>.
 ]===========================================================================]
cmake_minimum_required(VERSION 3.26)
PROJECT(mr_mnist VERSION 1.0.0.0 DESCRIPTION "My first attempt at creating artificial life" LANGUAGES CXX HOMEPAGE_URL https://www.github.com/ZachariahTheMagnificent/mr_mnist)

include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG        3.3.8
)
FetchContent_MakeAvailable(glfw)

add_executable(${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
target_sources(${PROJECT_NAME} PRIVATE FILE_SET main_modules TYPE HEADERS BASE_DIRS include PRIVATE
  include/world/zachariahs/mr_mnist/math.hpp
  include/world/zachariahs/mr_mnist/deep_neural_network.hpp
  include/world/zachariahs/mr_mnist/resources.hpp
)
target_sources(${PROJECT_NAME} PRIVATE src/world/zachariahs/mr_mnist/main.cpp)
