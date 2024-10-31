cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#
#
#include(/var/local/home/aburai/outdet_cpp/cmake/thirdparty/fetch_rapids.cmake)
#include(rapids-cmake)
#include(rapids-cpm)
#include(rapids-cuda)
#include(rapids-export)
#include(rapids-find)


#find_package(cuvs REQUIRED)
#find_package(raft)
#find_package(rmm)
#find_package(stdlib)
#find_package(spdlog)
#find_package(cuvs)


#rapids_cuda_init_architectures(outdet)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(outdet LANGUAGES CXX CUDA)
find_package(Torch REQUIRED)
find_package(CUDA 11.7 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ------------- configure cuvs -----------------#

#rapids_cpm_init()
#set(BUILD_CUVS_C_LIBRARY OFF)
#include(/var/local/home/aburai/outdet_cpp/cmake/thirdparty/get_cuvs.cmake)

#set(CMAKE_CUDA_COMPILER /var/local/home/aburai/envs24/cpp/bin/nvcc)
#set(CMAKE_CUDA_STANDARD 11)
set(COMMON_DIR Common)
#set(SOURCES main.cpp src/outdet.cpp src/outdet.h ${COMMON_DIR}/helper_cuda.h ${COMMON_DIR}/helper_functions.h)
set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)
INCLUDE_DIRECTORIES(${COMMON_DIR})
#link_directories(/usr/local/cuda-11.7/lib64)
#link_directories(/var/local/home/aburai/envs24/cpp/lib)
#include_directories(/var/local/home/aburai/envs24/cpp/include)
#include_directories(/var/local/home/aburai/envs24/cpp/include/raft)
#include_directories(/var/local/home/aburai/envs24/cuml_dev/include)
#link_directories(/var/local/home/aburai/anaconda3/envs/wads/lib/python3.10/site-packages/cuml)
#INCLUDE_DIRECTORIES(/var/local/home/aburai/anaconda3/envs/wads/include)
#INCLUDE_DIRECTORIES(/usr/local/cuda-11.7/include)
#INCLUDE_DIRECTORIES(/usr/local/cuda-11.7/targets/x86_64-linux/lib)


ADD_LIBRARY(cuda_comp src/knncuda.h src/knncuda.cu)
SET_TARGET_PROPERTIES(cuda_comp PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

add_executable(outdet main.cpp src/outdet.cpp src/outdet.h)

target_compile_features(outdet PUBLIC cxx_std_17)
target_link_libraries(outdet PUBLIC ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(outdet PUBLIC cudart)
target_link_libraries(outdet PUBLIC cuda)
#target_link_libraries(outdet PUBLIC stdlib)
#target_include_directories(outdet PUBLIC /var/local/home/aburai/cuml/cpp/include)
#target_include_directories(outdet PUBLIC /var/local/home/aburai/raft/cpp/include)
#target_include_directories(outdet PUBLIC /var/local/home/aburai/cuvs/include)
#target_include_directories(outdet PUBLIC /var/local/home/aburai/envs24/rmm_dev/include)
#target_include_directories(outdet PUBLIC /usr/local/cuda-11.7/targets/x86_64-linux/include)
#set_target_properties(outdet PROPERTIES LINKER_LANGUAGE "CUDA")
#target_link_libraries(outdet PUBLIC cuvs::cuvs)
target_link_libraries(outdet PUBLIC cuda_comp)
#target_link_libraries(outdet raft)
#target_link_libraries(outdet PUBLIC rmm::rmm)
#target_link_libraries(outdet cuvs)
#target_link_libraries(outdet spdlog::spdlog)
#target_compile_definitions(outdet PUBLIC SPDLOG_FMT_EXTERNAL)

