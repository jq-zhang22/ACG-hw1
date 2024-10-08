
file(GLOB mesh_files *.obj)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/meshes)
foreach(mesh_file ${mesh_files})
    file(COPY ${mesh_file} DESTINATION ${CMAKE_BINARY_DIR}/meshes)
endforeach()
