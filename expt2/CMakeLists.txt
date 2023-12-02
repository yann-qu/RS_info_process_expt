project(expt2)

find_package(OpenCV 4 REQUIRED)

message(STATUS "--- expt2 ---")
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME}_img_enhancement src/img_enhancement.cpp)
add_executable(${PROJECT_NAME}_PCA_transform src/PCA_transform.cpp)

