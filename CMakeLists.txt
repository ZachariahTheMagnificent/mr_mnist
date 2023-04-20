cmake_minimum_required(VERSION 3.26)
PROJECT(mr_mnist VERSION 1.0.0.0 DESCRIPTION "My first attempt at creating artificial life" LANGUAGES CXX HOMEPAGE_URL https://www.github.com/ZachariahTheMagnificent/mr_mnist)

include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG        3.3.7
)
FetchContent_MakeAvailable(glfw)

add_executable(${PROJECT_NAME})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
target_sources(${PROJECT_NAME} PRIVATE FILE_SET main_modules TYPE HEADERS BASE_DIRS include PRIVATE include/mr_mnist/mr_mnist.hpp)
target_sources(${PROJECT_NAME} PRIVATE src/main.cpp)
