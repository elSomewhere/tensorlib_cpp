cmake_minimum_required(VERSION 3.27)
project(tensorEigen)

set(CMAKE_CXX_STANDARD 17)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(tensorEigen main.cpp
        tensor.h)

# Link Eigen
target_link_libraries(tensorEigen Eigen3::Eigen)