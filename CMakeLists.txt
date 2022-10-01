cmake_minimum_required( VERSION 3.20)


list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_LIST_DIR}/cmake
        "${CMAKE_CURRENT_LIST_DIR}/cmake/third_party"
        )
include(DownloadProject)

project(loop_CD_project)

#----------------------------[ global configuration ]--------------------------#

set(CMAKE_CXX_STANDARD 11)


if(UNIX )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


#----------------------------------[ options ]---------------------------------#


set(SYMPILER_BLAS_BACKEND "OpenBLAS" CACHE STRING "BLAS implementation for SYMPILER to use")
set_property(CACHE SYMPILER_BLAS_BACKEND PROPERTY STRINGS MKL OpenBLAS)

if(${SYMPILER_BLAS_BACKEND} STREQUAL "MKL")
    set(SYMPILER_USE_BLAS_MKL      ON)
    set(SYMPILER_USE_BLAS_OpenBLAS OFF)
elseif(${SYMPILER_BLAS_BACKEND} STREQUAL "OpenBLAS")
    set(SYMPILER_USE_BLAS_MKL      OFF)
    set(SYMPILER_USE_BLAS_OpenBLAS ON)
else()
    message(FATAL_ERROR "unrecognized value for `SYMPILER_BLAS_BACKEND` option: '${SYMPILER_BLAS_BACKEND}'")
endif()

add_subdirectory(sympiler)

#--------------------------------[ third party ]-------------------------------#
#if(SYMPILER_USE_BLAS_MKL)
#    find_package(MKL)
#    if(MKL_FOUND)
#        add_definitions(-DMKL) # add openblas
#    else()
#        message(STATUS "Intel MKL not found, please use OpenBLAS")
#    endif()
#elseif(SYMPILER_USE_BLAS_OpenBLAS)
#    #set(openblas_WITHOUT_LAPACK ON) # for win support
#    include(openblas)
#    add_definitions(-DOPENBLAS)
#endif()


# Find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    if(APPLE) #TODO: set path for brew install libomp
        set(OpenMP_CXX_INCLUDE_DIRS  "/usr/local/include/")
        include_directories(
                ${OpenMP_CXX_INCLUDE_DIRS}
        )
        link_libraries(
                OpenMP::OpenMP_CXX
        )
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.6")
    endif()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DOPENMP)
endif()


include_directories(
        ${MKL_INCLUDE_DIR}
        ${BLAS_INC}
        ${LBC_INC}
        sympiler/include/
        sympiler/aggregation/include/
)


set(ALL_LIBS
        ${MKL_LIBRARIES}
        metis::metis
        tbb_static
        lbc
        )



set(ALL_SOURCE_FILES
        ${BLAS_SRC_FILES}
        ${METIS_SOURCE_FILES})


if(UNIX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}   -std=c++11 -w -m64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64   -mavx2")
endif()

Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")


find_package(METIS)
if(METIS_FOUND)
    add_definitions(-DMETIS)
endif()



project(loop_CD LANGUAGES C )

find_package(MKL CONFIG OPTIONAL_COMPONENTS)


add_executable(loop_CD
        main.cpp
        )
target_link_libraries(loop_CD PUBLIC
        sym_sparse_blas
        aggregation
        )
if(MKL_FOUND)
    add_definitions(-DMKL)
    target_include_directories(loop_CD PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(loop_CD PUBLIC $<LINK_ONLY:MKL::MKL>)
endif()

#target_link_libraries(loop_CD PRIVATE OpenBLAS::OpenBLAS)
#target_compile_definitions(loop_CD PRIVATE "OPENBLAS")
