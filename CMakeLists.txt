cmake_minimum_required(VERSION 3.10)
project(YourProjectName)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
pkg_check_modules(GMODULE REQUIRED gmodule-export-2.0)
pkg_check_modules(OPENCV REQUIRED opencv4)

include_directories(${GTK3_INCLUDE_DIRS} ${GMODULE_INCLUDE_DIRS}
                    ${OPENCV_INCLUDE_DIRS})

add_executable(main main.cpp)

target_link_libraries(main ${GTK3_LIBRARIES} ${GMODULE_LIBRARIES}
                      ${OPENCV_LIBRARIES})

set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "-rdynamic")
