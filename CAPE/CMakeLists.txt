cmake_minimum_required(VERSION 3.12)
project(CAPE VERSION 0.1 LANGUAGES C CXX)

option(BUILD_TOOLS "Compile tools" ON)

#########################################
######			General config	   ######
#########################################
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
       "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
       FORCE )
endif()

file(GLOB_RECURSE CAPE_HEADERS "./include/*.h" "./include/*.inl")
file(GLOB_RECURSE CAPE_SOURCES "./src/*.cpp")

add_library(${PROJECT_NAME} SHARED ${CAPE_SOURCES} ${CAPE_HEADERS})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE C)

#########################################
###### Loading 3rd party libraries ######
#########################################
## Load OpenCV
find_package(OpenCV)
if(${OpenCV_FOUND})
	MESSAGE( STATUS "OpenCV FOUND")
	target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${OpenCV_LIBRARIES})
endif(${OpenCV_FOUND})

## Load Eigen
find_package(Eigen3 REQUIRED)
if(${Eigen3_FOUND})
	MESSAGE( STATUS "Eigen FOUND")
	target_include_directories(${PROJECT_NAME} PUBLIC "/usr/include/eigen3")
endif(${Eigen3_FOUND})

find_package(realsense2 REQUIRED)
if(${realsense2_FOUND})
  MESSAGE( STATUS "realsense2 FOUND")
  target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${realsense2_LIBRARY})
  target_compile_definitions(${PROJECT_NAME} PUBLIC HAS_LIBREALSENSE2)
endif()

#########################################
######			Tools   	       ######
#########################################
if(${BUILD_TOOLS})
	add_subdirectory(tools)
endif()

