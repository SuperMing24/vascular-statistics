cmake_minimum_required(VERSION 3.10)
project(vascular_statistics VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vessel_stats
    src/main.cpp
    src/data_io.cpp
)

target_include_directories(vessel_stats PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# 单行编译备选（无需 CMake）：
# g++ -O2 src/main.cpp src/data_io.cpp -I include -o build/vessel_stats.exe
