project(expt1)

find_package(OpenCV 4 REQUIRED)

message(STATUS "--- expt1 ---")
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(${PROJECT_NAME}_cv_img_registration src/cv_img_registration.cpp)

target_link_libraries(${PROJECT_NAME}_cv_img_registration PRIVATE ${OpenCV_LIBS})

