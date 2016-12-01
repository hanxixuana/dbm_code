cmake_minimum_required(VERSION 3.5)
project(dbm_code)

#set(CMAKE_CXX_COMPILER /usr/local/bin/g++-6)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -D__OMP__")

# directory manipulations
set(directory
        "${CMAKE_HOME_DIRECTORY}/test"
        "${CMAKE_HOME_DIRECTORY}/lib"
        "${CMAKE_HOME_DIRECTORY}/api"
        "${CMAKE_HOME_DIRECTORY}/api/python"
        "${CMAKE_HOME_DIRECTORY}/api/r")
file(MAKE_DIRECTORY ${directory})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/test")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/lib")

include_directories(${CMAKE_HOME_DIRECTORY}/include)

# build files
set(MODEL_LIB_FILES
        lib_src/model.cpp
        lib_src/data_set.cpp
        lib_src/matrix.cpp
        lib_src/loss_function.cpp
        lib_src/base_learner.cpp
        lib_src/base_learner_trainer.cpp
        lib_src/tools.cpp)
add_library(_dbm_model SHARED ${MODEL_LIB_FILES})

set(TEST_SOURCE_FILES
        test_src/main.cpp)
add_executable(main_test ${TEST_SOURCE_FILES})

TARGET_LINK_LIBRARIES(main_test _dbm_model)