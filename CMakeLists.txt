cmake_minimum_required(VERSION 3.10)
project( holo LANGUAGES CXX)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
else()
    message(STATUS "No CUDA support")
endif()

list(APPEND CMAKE_PREFIX_PATH "~/fftw/3.3.10/lib/cmake" "~/opencv4/lib/cmake/opencv4")
message(STATUS ${CMAKE_PREFIX_PATH})
find_package(FFTW3)
if (FFTW3_FOUND)
    set(FFTW3_PATH "/home/fujimori/fftw/3.3.10")

    message(STATUS FFTW3::fftw3)
    set(FFTW3LIB "${FFTW3_PATH}/lib/libfftw3.a")
    set(FFTW3LIBF "${FFTW3_PATH}/lib/libfftw3f.a")
    set(FFTW3THREADLIB "${FFTW3_PATH}/lib/libfftw3_threads.a")
    set(FFTW3THREADLIBF "${FFTW3_PATH}/lib/libfftw3f_threads.a")
    set(FFTW3OMPLIB "${FFTW3_PATH}/lib/libfftw3_omp.a")
    set(FFTW3OMPLIBF "${FFTW3_PATH}/lib/libfftw3f_omp.a")
    set(FFTW3LIBALL ${FFTW3LIB} ${FFTW3LIBF} ${FFTW3OMPLIB} ${FFTW3OMPLIBF})
    set(FFTW3INCLUDE "${FFTW3_PATH}/include")
else()
    message(STATUS "No FFTW3 support")
endif()


find_package(OpenMP)
if (OPENMP_FOUND)
    message(STATUS "OpenMP support")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp" )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  message(STATUS "No OpenMP support")
endif()

find_package(OpenCV)


set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})
# For OpenMP
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


include_directories("${PROJECT_SOURCE_DIR}/include" ${FFTW3INCLUDE} ${OpenCV_INCLUDE_DIRS})
add_subdirectory(tests)
