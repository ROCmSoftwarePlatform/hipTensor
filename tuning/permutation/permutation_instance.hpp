/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef PERMUTATION_INSTANCE_HPP
#define PERMUTATION_INSTANCE_HPP
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp>
#include <ck/tensor_operation/gpu/element/combined_element_wise_operation.hpp>

#include <ck/library/reference_tensor_operation/cpu/reference_elementwise.hpp>

#include <ck/library/utility/algorithm.hpp>
#include <ck/library/utility/check_err.hpp>
#include <ck/library/utility/device_memory.hpp>
#include <ck/library/utility/host_tensor.hpp>
#include <ck/library/utility/host_tensor_generator.hpp>

#include "util.hpp"
#include <hiptensor/internal/hiptensor_utility.hpp>

namespace hiptensor
{
    namespace tuning
    {
        using F16 = ck::half_t;
        using F32 = float;

        using PassThrough = ck::tensor_operation::element_wise::PassThrough;

        template <typename InDataTypeTuple,
                  typename OutDataTypeTuple,
                  typename ElementwiseOperation,
                  ck::index_t NumDim>
        using DeviceElementwise = ck::tensor_operation::device::
            DeviceElementwise<InDataTypeTuple, OutDataTypeTuple, ElementwiseOperation, NumDim>;

        namespace permutation
        {
            template <typename InDataTypeTuple,
                      typename OutDataTypeTuple,
                      typename ElementwiseOperation,
                      index_t NumDim,
                      index_t BlockSize,
                      index_t M0PerBlock,
                      index_t M1PerBlock,
                      index_t M0PerThread,
                      index_t M1PerThread,
                      typename ThreadClusterArrangeOrder,
                      typename InScalarPerVectorSeq,
                      typename OutScalarPerVectorSeq>
            struct HiptensorDeviceElementwiseImpl
                : public ck::tensor_operation::device::DeviceElementwiseImpl<
                      InDataTypeTuple,
                      OutDataTypeTuple,
                      ElementwiseOperation,
                      NumDim,
                      BlockSize,
                      M0PerBlock,
                      M1PerBlock,
                      M0PerThread,
                      M1PerThread,
                      ThreadClusterArrangeOrder,
                      InScalarPerVectorSeq,
                      OutScalarPerVectorSeq>
            {

                std::string GetTypeString() const override
                {
                    auto str = std::stringstream();
                    // clang-format off
                    str << NumDim << "_";
                    str << BlockSize << "_";
                    str << M0PerBlock << "_";
                    str << M1PerBlock << "_";
                    str << M0PerThread << "_";
                    str << M1PerThread << "_";
                    str << ThreadClusterArrangeOrder::At(0) << "_";
                    str << ThreadClusterArrangeOrder::At(1) << "_";
                    str << InScalarPerVectorSeq::At(0) << "_";
                    str << OutScalarPerVectorSeq::At(0);
                    // clang-format on
                    return str.str();
                }
            };

            template <typename DataType, ck::index_t RANK>
            void genInstances_256_128_128_16_16(
                std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<DataType>,
                                                              ck::Tuple<DataType>,
                                                              PassThrough,
                                                              RANK>>>& instances)
            {
                // clang-format off
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<0, 1>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<0, 1>, ck::Sequence<16>, ck::Sequence<16>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>>());
                // clang-format on
            }

            template <typename DataType, ck::index_t RANK>
            void genInstances_256_128_128_32_32(
                std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<DataType>,
                                                              ck::Tuple<DataType>,
                                                              PassThrough,
                                                              RANK>>>& instances)
            {
                // clang-format off
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<0, 1>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<0, 1>, ck::Sequence<16>, ck::Sequence<16>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<1, 0>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 32, 32, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>>());
                // clang-format on
            }

            template <typename DataType, ck::index_t RANK>
            void genInstances_256_64_64_4_4(
                std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<DataType>,
                                                              ck::Tuple<DataType>,
                                                              PassThrough,
                                                              RANK>>>& instances)
            {
                // clang-format off
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 4, 4, ck::Sequence<0, 1>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>>());
                // clang-format on
            }

            template <typename DataType, ck::index_t RANK>
            void genInstances_256_64_64_16_16(
                std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<DataType>,
                                                              ck::Tuple<DataType>,
                                                              PassThrough,
                                                              RANK>>>& instances)
            {
                // clang-format off
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<0, 1>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<0, 1>, ck::Sequence<16>, ck::Sequence<16>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 64, 64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>>());
                // clang-format on
            }

            template <typename DataType, ck::index_t RANK>
            void genInstances_256_128_128_8_8(
                std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<DataType>,
                                                              ck::Tuple<DataType>,
                                                              PassThrough,
                                                              RANK>>>& instances)
            {
                // clang-format off
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 8, 8, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 8, 8, ck::Sequence<0, 1>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 8, 8, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<2>, ck::Sequence<2>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>>());
                instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough, RANK, 256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>>());
                // clang-format on
            }

            template <typename DataType, ck::index_t RANK>
            void genInstances_miscellaneous(
                std::vector<std::unique_ptr<DeviceElementwise<ck::Tuple<DataType>,
                                                              ck::Tuple<DataType>,
                                                              PassThrough,
                                                              RANK>>>& instances)
            {
                // clang-format off
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 256, 64,  64,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 256, 128, 32,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 256, 32,  128, 4, 4,  ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128, 64,  32,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128, 32,  64,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128, 16,  128, 4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128, 128, 16,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 64,  32,  32,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 64,  16,  64,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 64,  64,  16,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 32,  32,  16,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 32,  16,  32,  4, 4, ck::Sequence<0, 1>, ck::Sequence<4>, ck::Sequence<4>>>());

				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 256, 128, 128, 8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 256, 256, 64,  8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 256,  64, 256, 8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128, 128, 64,  8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128,  64, 128, 8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128,  32, 256, 8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128, 256, 32,  8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 64,   64, 64,  8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 64,   32, 128, 8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 64,  128, 32,  8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 32,   64, 32,  8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 32,   32, 64,  8, 8, ck::Sequence<0, 1>, ck::Sequence<8>, ck::Sequence<8>>>());


				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 256,  64,  64, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 256, 128,  32, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 256,  32, 128, 4, 4,  ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128,  64,  32, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128,  32,  64, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128,  16, 128, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 128, 128,  16, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 64,   32,  32, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 64,   16,  64, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 64,   64,  16, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 32,   32,  16, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
				instances.push_back( std::make_unique<HiptensorDeviceElementwiseImpl<ck::Tuple<DataType>, ck::Tuple<DataType>, PassThrough,  RANK, 32,   16,  32, 4, 4, ck::Sequence<0, 1>, ck::Sequence<1>, ck::Sequence<1>>>());
                // clang-format on
            }

#define GEN_INSTANCES(rank)                                                                        \
    void genInstances_F16_##rank##_256_128_128_16_16(                                              \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, rank>>>& instances);    \
    void genInstances_F16_##rank##_256_128_128_8_8(                                                \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, rank>>>& instances);    \
    void genInstances_F16_##rank##_256_64_64_16_16(                                                \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, rank>>>& instances);    \
    void genInstances_F16_##rank##_256_64_64_4_4(                                                  \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, rank>>>& instances);    \
    void genInstances_F16_##rank##_miscellaneous(                                                  \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, rank>>>& instances);    \
    void genInstances_F32_##rank##_256_128_128_16_16(                                              \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, rank>>>& instances);    \
    void genInstances_F32_##rank##_256_128_128_8_8(                                                \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, rank>>>& instances);    \
    void genInstances_F32_##rank##_256_64_64_16_16(                                                \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, rank>>>& instances);    \
    void genInstances_F32_##rank##_256_64_64_4_4(                                                  \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, rank>>>& instances);    \
    void genInstances_F32_##rank##_miscellaneous(                                                  \
        std::vector<std::unique_ptr<                                                               \
            DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, rank>>>& instances);    \
                                                                                                   \
    std::vector<                                                                                   \
        std::unique_ptr<DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, rank>>>     \
        genInstances_F16_##rank()                                                                  \
    {                                                                                              \
        std::vector<                                                                               \
            std::unique_ptr<DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, rank>>> \
            instances;                                                                             \
        genInstances_F16_##rank##_256_64_64_4_4(instances);                                        \
        genInstances_F16_##rank##_256_64_64_16_16(instances);                                      \
        genInstances_F16_##rank##_256_128_128_8_8(instances);                                      \
        genInstances_F16_##rank##_256_128_128_16_16(instances);                                    \
        genInstances_F16_##rank##_miscellaneous(instances);                                        \
        return instances;                                                                          \
    }                                                                                              \
                                                                                                   \
    std::vector<                                                                                   \
        std::unique_ptr<DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, rank>>>     \
        genInstances_F32_##rank()                                                                  \
    {                                                                                              \
        std::vector<                                                                               \
            std::unique_ptr<DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, rank>>> \
            instances;                                                                             \
        genInstances_F32_##rank##_256_64_64_4_4(instances);                                        \
        genInstances_F32_##rank##_256_64_64_16_16(instances);                                      \
        genInstances_F32_##rank##_256_128_128_8_8(instances);                                      \
        genInstances_F32_##rank##_256_128_128_16_16(instances);                                    \
        genInstances_F32_##rank##_miscellaneous(instances);                                        \
        return instances;                                                                          \
    }

        }
    }
}
#endif //  PERMUTATION_INSTANCE_HPP
