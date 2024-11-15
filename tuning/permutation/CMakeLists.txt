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

# Don't link to consume ck instances, however will customize and create our own.
# Make the ck includes visible so we can build instances.
get_target_property(composable_kernel_INCLUDES composable_kernel::device_other_operations INTERFACE_INCLUDE_DIRECTORIES)

set(HIPTENSOR_PERMUTATION_TUNING "permutation_tuning")

set(HIPTENSOR_PERMUTATION_TUNING_SOURCES
   ${CMAKE_CURRENT_SOURCE_DIR}/permutation_tuning.cpp
)

add_executable(${HIPTENSOR_PERMUTATION_TUNING} ${HIPTENSOR_PERMUTATION_TUNING_SOURCES})

set_target_properties(${HIPTENSOR_PERMUTATION_TUNING} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(${HIPTENSOR_PERMUTATION_TUNING} PRIVATE ${CMAKE_CXX_FLAGS} ${CLANG_DRIVER_MODE})
target_link_options(${HIPTENSOR_PERMUTATION_TUNING} PRIVATE ${CLANG_DRIVER_MODE})
target_include_directories(${HIPTENSOR_PERMUTATION_TUNING} PRIVATE ${composable_kernel_INCLUDES})
target_link_libraries(${HIPTENSOR_PERMUTATION_TUNING} PRIVATE hip::device hip::host)

