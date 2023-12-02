project(expt2)

find_package(OpenCV 4 REQUIRED)
find_package(GDAL CONFIG REQUIRED)

message(STATUS "--- expt2 ---")
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME}_img_enhancement src/img_enhancement.cpp)
add_executable(${PROJECT_NAME}_PCA_transform src/PCA_transform.cpp)

target_link_libraries(${PROJECT_NAME}_img_enhancement PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}_img_enhancement PRIVATE GDAL::GDAL)
target_link_libraries(${PROJECT_NAME}_PCA_transform PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}_PCA_transform PRIVATE GDAL::GDAL)
