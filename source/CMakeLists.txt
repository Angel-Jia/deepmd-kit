cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(DeePMD)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(USE_CUDA_TOOLKIT ON)

set(PYTORCH_ROOT "/opt/miniconda3/envs/pytorch1.7/lib/python3.7/site-packages/torch")
list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_ROOT}/share/cmake/Torch/")
find_package(Torch REQUIRED)

set(Python3_ROOT_DIR "/opt/miniconda3/envs/pytorch1.7/")
list(APPEND CMAKE_PREFIX_PATH "${Python3_ROOT_DIR}")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# find threads
find_package(Threads)

set(CUDA_LIBRARIES "/usr/local/cuda/lib64")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")

# find openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# define USE_CUDA_TOOLKIT
# if (USE_CUDA_TOOLKIT)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
  find_package(CUDA REQUIRED)
  add_definitions("-DGOOGLE_CUDA")
  message(STATUS "Found CUDA in ${CUDA_TOOLKIT_ROOT_DIR}, build nv GPU support")
# else()
#   message(STATUS "Will not build nv GPU support")
# endif(USE_CUDA_TOOLKIT)

set(LIB_DEEPMD		"deepmd")

# Enable C++14
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(lib/)
add_subdirectory(op/)
