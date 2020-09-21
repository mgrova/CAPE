add_executable(cape_offline run_cape_offline.cpp)
target_include_directories(cape_offline PUBLIC ../include)
target_link_libraries(cape_offline LINK_PUBLIC CAPE)