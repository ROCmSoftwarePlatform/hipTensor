###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################

 # find_package( hip REQUIRED )
find_package( composable_kernel 1.0.0 REQUIRED PATHS $ENV{CK_DIR}/lib/cmake /opt/rocm /opt/rocm/ck COMPONENTS device_contraction_operations device_reduction_operations device_other_operations)
rocm_package_add_dependencies("composable_kernel >= 1.0.0" COMPONENT tests)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Create test executables and deploy
function(add_hiptensor_component COMPONENT_NAME COMPONENT_SOURCES)

    message( STATUS "adding hiptensor component: ${COMPONENT_NAME}")

    # Ensure that all the sources are captured
    list(APPEND COMPONENT_SOURCES ${ARGN})

    # Build components as object files.
    # Make sure they have -fPIC and that they inherit the hip::device environment
    # so that they build for all archs in AMDGPU_TARGETS
    add_library(${COMPONENT_NAME} OBJECT ${COMPONENT_SOURCES})
    target_compile_options(${COMPONENT_NAME} PRIVATE ${CLANG_DRIVER_MODE})
    target_link_options(${COMPONENT_NAME} PRIVATE ${CLANG_DRIVER_MODE})
    set_target_properties(${COMPONENT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${COMPONENT_NAME} PRIVATE hip::device)
    if(HIPTENSOR_BUILD_COMPRESSED_DBG)
        target_compile_options(${COMPONENT_NAME} PRIVATE
            "$<$<CONFIG:Debug>:-gz>"
            "$<$<CONFIG:RelWithDebInfo>:-gz>"
            )
    endif()

endfunction()


# Generates hiptensor_contraction and hiptensor_contraction_instances
# add_subdirectory(contraction)
# Generates hiptensor_permutation and hiptensor_permutation_instances
add_subdirectory(permutation)
# Generates hiptensor_reduction and hiptensor_reduction_instances
# add_subdirectory(reduction)

