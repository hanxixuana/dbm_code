cmake_minimum_required(VERSION 3.5)
project(dbm_code)

SET(CMAKE_BUILD_TYPE DEBUG)
#SET(CMAKE_BUILD_TYPE RELEASE)

# version

set(DBM_CODE_MAJOR_VERSION 1)
set(DBM_CODE_MINOR_VERSION 0)
set(DBM_CODE_PATCH_VERSION 0)
set(DBM_CODE_VERSION
        ${DBM_CODE_MAJOR_VERSION}.${DBM_CODE_MINOR_VERSION}.${DBM_CODE_PATCH_VERSION})


################
# Preparations #
################

#set(CMAKE_CXX_COMPILER /usr/local/bin/g++-6)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D_OMP")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG_BASE_LEARNER           \
#                                        -D_DEBUG_BASE_LEARNER_TRAINER   \
#                                        -D_DEBUG_DATA_SET               \
#                                        -D_DEBUG_LOSS_FUNCTION          \
#                                        -D_DEBUG_MATRIX                 \
#                                        -D_DEBUG_MODEL                  \
#                                        -D_CD_INDICATOR                  \
#                                        -D_DEBUG_TOOLS")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DEBUG_BASE_LEARNER           \
                                        -D_DEBUG_BASE_LEARNER_TRAINER   \
                                        -D_DEBUG_DATA_SET               \
                                        -D_DEBUG_LOSS_FUNCTION          \
                                        -D_DEBUG_MATRIX                 \
                                        -D_DEBUG_MODEL                  \
                                        -D_DEBUG_TOOLS")

# directory manipulations

set(directory
        ${CMAKE_HOME_DIRECTORY}/build
        ${CMAKE_HOME_DIRECTORY}/lib
        ${CMAKE_HOME_DIRECTORY}/api
        ${CMAKE_HOME_DIRECTORY}/api/dbm_py
        ${CMAKE_HOME_DIRECTORY}/api/dbm_r)
file(MAKE_DIRECTORY ${directory})

# directories for binary shared objects and executables

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/lib)

# find header files

include_directories(${CMAKE_HOME_DIRECTORY}/include)


################
#    Library   #
################

set(MODEL_LIB_FILES
        src_lib/model.cpp
        src_lib/data_set.cpp
        src_lib/matrix.cpp
        src_lib/loss_function.cpp
        src_lib/base_learner.cpp
        src_lib/base_learner_trainer.cpp
        src_lib/tools.cpp)
add_library(_dbm_code SHARED ${MODEL_LIB_FILES})


################
#  Python API  #
################

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/api/dbm_py)

set(PY_API_LIB_FILES
        src_lib/py_api.cpp)

find_package(Boost ${Boost_VERSION})

if(Boost_FOUND)

    message( "-- Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
    message( "-- Boost_LIBRARIES; " ${Boost_LIBRARIES})
    include_directories(${Boost_INCLUDE_DIRS})

    set(Boost_USE_STATIC_LIBS OFF)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_STATIC_RUNTIME OFF)

    find_package(Boost COMPONENTS python)

elseif(not Boost_FOUND)

    message(FATAL_ERROR "Unable to find Boost Python!")

endif()

find_package(PythonLibs 2.7 REQUIRED)

if(PythonLibs_FOUND)

    message( "-- PYTHON_INCLUDE_DIRS: " ${PYTHON_INCLUDE_DIRS})
    message( "-- PYTHON_LIBRARY: " ${PYTHON_LIBRARY})
    include_directories(${PYTHON_INCLUDE_DIRS})

elseif(not PythonLibs_FOUND)

    message(FATAL_ERROR "Unable to find Python libraries!")

endif()

#set(PYTHON_LIBRARY /usr/local/Cellar/dbm_py/2.7.12_2/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib)
#set(PYTHON_INCLUDE_DIR /usr/local/Cellar/dbm_py/2.7.12_2/Frameworks/Python.framework/Versions/2.7/include/python2.7)
#
#find_package(PythonLibs 2.7 REQUIRED)
#
#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

add_library(_dbm_code_py SHARED ${PY_API_LIB_FILES})
target_link_libraries(_dbm_code_py _dbm_code ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} )


################
#     Test     #
################

set(TEST_SOURCE_FILES
        src_test/main_test.cpp)
add_executable(main_test ${TEST_SOURCE_FILES})

TARGET_LINK_LIBRARIES(main_test _dbm_code _dbm_code_py)


################
#     Copy     #
################

configure_file("${CMAKE_HOME_DIRECTORY}/src_api/dbm_code_py.py"
        "${CMAKE_HOME_DIRECTORY}/api/dbm_py/dbm_py.py"
        COPYONLY)

configure_file("${CMAKE_HOME_DIRECTORY}/src_api/__init__.py"
        "${CMAKE_HOME_DIRECTORY}/api/dbm_py/__init__.py"
        COPYONLY)

configure_file("${CMAKE_HOME_DIRECTORY}/src_test/main_test.py"
        "${CMAKE_HOME_DIRECTORY}/api/main_test.py"
        COPYONLY)

configure_file("${CMAKE_HOME_DIRECTORY}/src_lib/r_api.cpp"
        "${CMAKE_HOME_DIRECTORY}/api/dbm_r/_dbm_code_r.cpp"
        COPYONLY)