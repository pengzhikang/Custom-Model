cmake_minimum_required(VERSION 3.10)
#设置c++的版本并添加多线程的使用
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

#设置输出文件的路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/release)


#添加依赖库路径
link_directories(/usr/local/lib)

#添加头文件路径
include_directories(/usr/local/include)
include_directories(/home/pengzhikang/project/custom-model/include)
# 查找当前目录的所有源文件

# add_definitions("-D__GNUC__")
#指定生成目标
set(REQUIRE_CPP  src/json11.cpp src/create_model_sample.cpp)
add_executable(first_model ${REQUIRE_CPP})
#添加链接库

target_link_libraries(first_model libflatbuffers.a)
target_link_libraries(first_model stdc++fs)

