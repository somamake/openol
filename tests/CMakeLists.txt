add_executable(Prop_test Prop_test.cpp)
target_compile_features(Prop_test PUBLIC cxx_std_17)
target_link_libraries(Prop_test ${FFTW3LIBALL} ${OpenCV_LIBRARIES})
target_compile_options(Prop_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

add_executable(tiltedProp_test tiltedProp_test.cpp)
target_compile_features(tiltedProp_test PUBLIC cxx_std_17)
target_link_libraries(tiltedProp_test ${FFTW3LIBALL} ${OpenCV_LIBRARIES})
target_compile_options(tiltedProp_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

# add_executable(diffraction_test diffraction_test.cpp)
# target_compile_features(diffraction_test PUBLIC cxx_std_17)
# target_link_libraries(diffraction_test ${FFTW3LIB})
# target_compile_options(diffraction_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

# add_executable(olcgh cgh.cpp)
# target_compile_features(olcgh PUBLIC cxx_std_17)
# target_compile_options(olcgh PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

add_executable(Object_test Object_test.cpp)
target_compile_features(Object_test PUBLIC cxx_std_17)
target_link_libraries(Object_test ${FFTW3LIBALL} ${OpenCV_LIBRARIES})
target_compile_options(Object_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

# add_executable(opencv_test opencv_test.cpp)
# target_compile_features(opencv_test PUBLIC cxx_std_17)
# target_link_libraries(opencv_test ${FFTW3LIBALL}  ${OpenCV_LIBRARIES})
# target_compile_options(opencv_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

if(CMAKE_CUDA_COMPILER)

add_executable(gObject_test gObject_test.cu)
target_compile_features(gObject_test PUBLIC cxx_std_17)
target_link_libraries(gObject_test CUDA::cufft ${OpenCV_LIBRARIES})
target_compile_options(gObject_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

add_executable(gProp_test gProp_test.cu)
target_compile_features(gProp_test PUBLIC cxx_std_17)
target_link_libraries(gProp_test CUDA::cufft ${OpenCV_LIBRARIES})
target_compile_options(gProp_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

add_executable(mProp_test mProp_test.cu)
target_compile_features(mProp_test PUBLIC cxx_std_17)
target_link_libraries(mProp_test CUDA::cufft ${OpenCV_LIBRARIES}  ${FFTW3LIBALL})
target_compile_options(mProp_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

# add_executable(partialProp_test partialProp_test.cu)
# target_compile_features(partialProp_test PUBLIC cxx_std_17)
# target_link_libraries(partialProp_test CUDA::cufft ${OpenCV_LIBRARIES})
# target_compile_options(partialProp_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

# add_executable(WRP_test WRP_test.cu)
# target_compile_features(WRP_test PUBLIC cxx_std_17)
# target_link_libraries(WRP_test CUDA::cufft ${OpenCV_LIBRARIES})
# target_compile_options(WRP_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

# add_executable(CghProp_test CghProp_test.cu)
# target_compile_features(CghProp_test PUBLIC cxx_std_17)
# target_link_libraries(CghProp_test CUDA::cufft ${OpenCV_LIBRARIES})
# target_compile_options(CghProp_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)

# add_executable(curand_test curand_test.cu)
# target_compile_features(curand_test PUBLIC cxx_std_17)
# target_link_libraries(curand_test CUDA::curand ${OpenCV_LIBRARIES})
# target_compile_options(curand_test PUBLIC -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}" -O3)


endif()