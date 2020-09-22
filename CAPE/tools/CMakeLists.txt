
add_executable(cape_offline run_cape_offline.cpp)
target_include_directories(cape_offline PUBLIC ../include)
target_link_libraries(cape_offline LINK_PUBLIC CAPE)

add_executable(cape_online run_cape_online.cpp)
target_include_directories(cape_online PUBLIC ../include)
target_link_libraries(cape_online LINK_PUBLIC CAPE)