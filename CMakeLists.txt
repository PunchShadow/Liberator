cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(Liberator CUDA)

set(CMAKE_CUDA_STANDARD 14)
SET(CMAKE_BUILD_TYPE "Release")
option(CACHE_STAT "Enable cache admission decision accuracy instrumentation" OFF)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -G -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_executable(ptGraph main.cu gpu_kernels.cu gpu_kernels.cuh range.cuh bfs-sync.cuh tools.cuh common.cuh common.cu GraphMeta.cu GraphMeta.cuh NewCalculateOpt.cu CalculateOpt.cu CalculateOpt.cuh constants.cuh TimeRecord.cu TimeRecord.cuh ArgumentParser.cu ArgumentParser.cuh cache_density.cu cache_density.cuh edge_path.cu edge_path.cuh cache_stat.cu cache_stat.cuh)

set_target_properties(
        ptGraph
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "86")

target_compile_options(ptGraph PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
if(CACHE_STAT)
    target_compile_definitions(ptGraph PRIVATE CACHE_STAT)
endif()
target_link_libraries(ptGraph -lcublas -lcurand -lgomp)
