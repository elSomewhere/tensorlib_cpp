cmake_minimum_required(VERSION 3.12)
project(GRUDGradientTest LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add executable
add_executable(gradient_test main.cpp)

# Set include directories
# Assuming the header files are in the same directory as main.cpp
# or adjust the path accordingly
target_include_directories(gradient_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link with Eigen3
target_link_libraries(gradient_test Eigen3::Eigen)

# Compiler-specific optimizations and warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gradient_test PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -mtune=native>
            $<$<CONFIG:Debug>:-O0 -g -Wall -Wextra -Wpedantic>
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(gradient_test PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Debug>:/Od /Zi /W4>
    )
endif()

# Compile definitions
target_compile_definitions(gradient_test PRIVATE
        EIGEN_NO_DEBUG
        EIGEN_STRONG_INLINE=inline
        $<$<CONFIG:Release>:NDEBUG>
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(gradient_test PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            _USE_MATH_DEFINES
    )
endif()

# Set output directory
set_target_properties(gradient_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print some information
message(STATUS "Building GRU-D Gradient Test")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")