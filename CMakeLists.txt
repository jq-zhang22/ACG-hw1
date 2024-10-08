cmake_minimum_required(VERSION 3.22)

project(assignment-1-mesh)

add_subdirectory(assets/meshes)
include_directories(glm)
add_executable(mesh_main src/main.cpp src/mesh.cpp src/mesh.h)
target_include_directories(mesh_main PRIVATE external/glm external/tiny_obj_loader)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/results)
