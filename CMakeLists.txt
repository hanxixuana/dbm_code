cmake_minimum_required(VERSION 3.6)
project(dbm_code)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(directory
        "${CMAKE_HOME_DIRECTORY}/output"
        "${CMAKE_HOME_DIRECTORY}/output/lib")
file(MAKE_DIRECTORY ${directory})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${directory})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/output/lib")

set(DATA_LIB_FILES
        matrix.cpp
        loss_function.cpp
        data_set.cpp)
add_library(_dbm_data SHARED ${DATA_LIB_FILES})

set(MODEL_LIB_FILES
        matrix.cpp
        loss_function.cpp
        base_learner.cpp
        base_learner_trainer.cpp
        model.cpp)
add_library(_dbm_model SHARED ${MODEL_LIB_FILES})

set(TOOL_LIB_FILES
        tools.cpp)
add_library(_dbm_tool SHARED ${TOOL_LIB_FILES})

set(SOURCE_FILES
        main.cpp)
add_executable(dbm_main ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(dbm_main _dbm_data _dbm_model _dbm_tool)