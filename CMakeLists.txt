
#<-------------------------------------------------->
cmake_minimum_required(VERSION 3.17)


project(OCL-Engine)
#增加omp的使用
#find_package(OpenMP REQUIRED)
#if(OPENMP_FOUND)
#    message("Found OpenMP")
#    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
#    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
#    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS})
#endif()
#设置c++的版本并添加多线程的使用
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

#设置输出文件的路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/)

set(PROTO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/flatbuffers)

#添加依赖库路径
link_directories(/usr/lib)
link_directories(${PROJECT_BINARY_DIR}/../)
#set(OPENCV_dep_LIBS opencv_imgproc opencv_imgcodecs opencv_core opencv_dnn opencv_highgui opencv_videoio)
if (WIN32)
    link_directories("C:/Program Files (x86)/Common Files/Intel/Shared Libraries")
else ()
    link_directories(/opt/intel/opencl_compilers_and_libraries_18.1.0.015/linux/compiler/lib/intel64_lin)
endif ()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    link_directories(${PROTO_PATH}/lib/release)
else()
    link_directories(${PROTO_PATH}/lib/debug)
endif()


set(OPENCL_dep_LIBS OpenCL)
set(FLATBUFFERS_dep_LIBS flatbuffers)
#添加头文件路径
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${PROTO_PATH}/include")

OPTION (Build_CreateModel             "Build CreateModel"                           OFF)
# 查找当前目录的所有源文件
if (Build_CreateModel)
    set(REQUIRE_CPP
            example/create_model_sample.cpp src/model/json11.cpp)
else ()
    set(REQUIRE_CPP
            example/main.cpp src/model/json11.cpp)
endif()
#指定生成目标
add_executable(test ${REQUIRE_CPP})
target_link_libraries(test ${OPENCL_dep_LIBS})
target_link_libraries(test ${FLATBUFFERS_dep_LIBS})
#target_link_libraries(test ${OPENCV_dep_LIBS})
#添加链接库
if (WIN32)
    target_compile_definitions(test PRIVATE -DWIN32_LEAN_AND_MEAN -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
else ()
    target_link_libraries(test stdc++fs)
endif ()

