cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(temp)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置编译输出目录（可选，整理编译产物）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ========================================
# 2. CUDA 编译选项配置（关键）
# ========================================
# 指定 CUDA 架构（根据你的显卡调整，比如 75=RTX 20/30 系，80=RTX 40 系，86=RTX 30 系移动端）
# 查看显卡架构：nvcc --help | grep -i arch  或 nvidia-smi 查看算力
set(CMAKE_CUDA_ARCHITECTURES 70)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)  # 暂时使用旧的FindCUDA模块
endif()
# CUDA 编译参数（可选，优化/调试）
# -O3：优化编译；-G：调试模式（可加 -g 保留调试信息）；-Xcompiler "-fPIC"：生成位置无关代码
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC")

# 调试模式下的 CUDA 编译参数（可选）
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()

find_package(CUDA REQUIRED)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(/usr/local/cuda-11.7/targets/x86_64-linux/include/)
# link_directories(/usr/local/cuda-11.7/targets/x86_64-linux/lib/)

file(GLOB SRC_CU *cu)
file(GLOB SRCS *cc)

cuda_add_library(my_cu_lib ${SRC_CU})
target_link_libraries(my_cu_lib
    ${CUDA_LIBRARIES}
    pthread
)

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME}
    my_cu_lib
    
)



# install(TARGETS sgemm_naive
#     RUNTIME DESTINATION /usr/local/bin  # 可执行文件安装路径
#     LIBRARY DESTINATION /usr/local/lib  # 动态库安装路径
#     ARCHIVE DESTINATION /usr/local/lib  # 静态库安装路径
# )