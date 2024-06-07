cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(TicTacToe)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(SQLite3 REQUIRED)

find_package(toml11 REQUIRED)

configure_file("${CMAKE_SOURCE_DIR}/config.toml" "${CMAKE_BINARY_DIR}/config.toml" COPYONLY)

include_directories("${CMAKE_SOURCE_DIR}/include")
file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
add_executable(TicTacToe "${SOURCES}")

target_link_libraries(TicTacToe "${TORCH_LIBRARIES}")
target_link_libraries(TicTacToe "${SQLite3_LIBRARIES}")

set_property(TARGET TicTacToe PROPERTY CXX_STANDARD 17)