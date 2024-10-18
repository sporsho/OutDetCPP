cmake_minimum_required(VERSION 3.5)
project(outdet)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

add_executable(outdet main.cpp src/outdet.cpp src/outdet.h)
target_compile_features(outdet PUBLIC cxx_range_for)
target_link_libraries(outdet ${TORCH_LIBRARIES})
