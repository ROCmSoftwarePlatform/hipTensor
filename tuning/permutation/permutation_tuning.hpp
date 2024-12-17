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

#ifndef PERMUTATION_TUNING_HPP
#define PERMUTATION_TUNING_HPP
#include "permutation_instance.hpp"

namespace hiptensor
{
    namespace tuning
    {
        namespace permutation
        {
            template <int RANK>
            auto make_permutation_params(int argc, char* argv[])
            {
                std::vector<std::size_t> inputLengths(RANK);
                for(int i = 0; i < RANK; i++)
                {
                    inputLengths[i] = atoi(argv[i + 2]);
                }
                std::vector<ck::index_t> outputDims(RANK);
                for(int i = 0; i < RANK; i++)
                {
                    outputDims[i] = atoi(argv[i + RANK + 2]);
                }
                std::unordered_map<ck::index_t, std::size_t> outputLengthMap;
                for(ck::index_t i = 0; i < RANK; i++)
                {
                    outputLengthMap[i] = inputLengths[i];
                }
                std::vector<std::size_t> outputLengths(RANK);
                for(ck::index_t i = 0; i < RANK; i++)
                {
                    outputLengths[i] = outputLengthMap[outputDims[i]];
                }

                auto toCKArr
                    = [](std::vector<std::size_t> const& v, std::array<ck::index_t, RANK>& a) {
                          std::copy_n(v.begin(), RANK, a.begin());
                      };

                std::array<ck::index_t, RANK> a_strides;
                toCKArr(
                    hiptensor::stridesFromLengths(inputLengths, HIPTENSOR_DATA_LAYOUT_COL_MAJOR),
                    a_strides);
                std::array<ck::index_t, RANK> b_strides_unordered;
                toCKArr(
                    hiptensor::stridesFromLengths(outputLengths, HIPTENSOR_DATA_LAYOUT_COL_MAJOR),
                    b_strides_unordered);
                std::array<ck::index_t, RANK> b_strides;
                for(int i = 0; i < RANK; i++)
                {
                    b_strides[outputDims[i]] = b_strides_unordered[i];
                }

                return std::make_tuple(inputLengths, outputLengths, a_strides, b_strides);
            }

            template <typename InstanceGenerator,
                      typename InputDataType,
                      typename OutputDataType,
                      int RANK>
            int run(InstanceGenerator                    generator,
                    std::vector<std::size_t> const&      inputLengths,
                    std::vector<std::size_t> const&      outputLengths,
                    std::array<ck::index_t, RANK> const& a_strides,
                    std::array<ck::index_t, RANK> const& b_strides)
            {
                bool time_kernel = true;

                std::array<ck::index_t, RANK> ab_lengths;
                ck::ranges::copy(inputLengths, ab_lengths.begin());

                size_t elementsA = std::accumulate(
                    inputLengths.cbegin(), inputLengths.cend(), 1, std::multiplies<ck::index_t>{});
                size_t elementsB = std::accumulate(outputLengths.cbegin(),
                                                   outputLengths.cend(),
                                                   1,
                                                   std::multiplies<ck::index_t>{});
                size_t sizeA     = sizeof(InputDataType) * elementsA;
                size_t sizeB     = sizeof(OutputDataType) * elementsB;

                void* A_d;
                void* B_d;
                CHECK_HIP_ERROR(hipMalloc((void**)&A_d, sizeA));
                CHECK_HIP_ERROR(hipMalloc((void**)&B_d, sizeB));

                InputDataType*  A;
                OutputDataType* B;
                CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA));
                CHECK_HIP_ERROR(hipHostMalloc((void**)&B, sizeB));

                std::iota(A, A + elementsA, static_cast<InputDataType>(1.0f));

                CHECK_HIP_ERROR(hipMemcpy(A_d, A, sizeA, hipMemcpyDefault));

                auto broadcastPermutes = generator();
                std::cout << "There are " << broadcastPermutes.size() << " instances.\n";
                for(auto& broadcastPermute : broadcastPermutes)
                {
                    auto argument = broadcastPermute->MakeArgumentPointer(
                        ab_lengths, {a_strides}, {b_strides}, {A_d}, {B_d}, PassThrough{});

                    if(!broadcastPermute->IsSupportedArgument(argument.get()))
                    {
                        std::cout << broadcastPermute->GetTypeString()
                                  << " does not support this input tensor:\n";
                        continue;
                    };

                    auto  broadcastPermute_invoker_ptr = broadcastPermute->MakeInvokerPointer();
                    float ave_time                     = broadcastPermute_invoker_ptr->Run(
                        argument.get(), StreamConfig{nullptr, time_kernel, 0, 5, 10});
                    std::size_t flop = elementsA + elementsB;

                    std::size_t num_btype
                        = sizeof(InputDataType) * elementsA + sizeof(OutputDataType) * elementsB;
                    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                    float gb_per_sec = num_btype / 1.E6 / ave_time;

                    std::cout << broadcastPermute->GetTypeString() << ", ";
                    std::cout << gb_per_sec << std::endl;
                }

                CHECK_HIP_ERROR(hipHostFree(A));
                CHECK_HIP_ERROR(hipHostFree(B));
                CHECK_HIP_ERROR(hipFree(A_d));
                CHECK_HIP_ERROR(hipFree(B_d));
                return 0;
            }
        }
    }
}
#endif //  PERMUTATION_TUNING_HPP
