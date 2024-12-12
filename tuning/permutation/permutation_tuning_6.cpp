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

#include "permutation_tuning.hpp"
#include <numeric>
#include <unordered_map>

namespace hiptensor
{
    namespace tuning
    {
        namespace permutation
        {

            std::vector<
                std::unique_ptr<DeviceElementwise<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 6>>>
                genInstances_F32_6();
            std::vector<
                std::unique_ptr<DeviceElementwise<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 6>>>
                genInstances_F16_6();
        }
    }
}

int main(int argc, char* argv[])
{
    constexpr int RANK = 6;

    if(argc != (2 * RANK + 2))
    {
        std::cerr
            << "Usage: " << argv[0] << " <data_type> "
            << " <length0> <length1> <length2> <length3> <length4> <length5>"
            << " <outputDim0> <outputDim1> <outputDim2> <outputDim3> <outputDim4> <outputDim5>"
            << std::endl;
        return 1;
    }

    auto params = hiptensor::tuning::permutation::make_permutation_params<RANK>(argc, argv);
    if(std::string(argv[1]) == "F16")
    {
        hiptensor::tuning::permutation::run<
            decltype(hiptensor::tuning::permutation::genInstances_F16_6),
            hiptensor::tuning::F16,
            hiptensor::tuning::F16,
            6>(hiptensor::tuning::permutation::genInstances_F16_6,
               std::get<0>(params),
               std::get<1>(params),
               std::get<2>(params),
               std::get<3>(params));
    }
    else
    {
        hiptensor::tuning::permutation::run<
            decltype(hiptensor::tuning::permutation::genInstances_F32_6),
            hiptensor::tuning::F32,
            hiptensor::tuning::F32,
            6>(hiptensor::tuning::permutation::genInstances_F32_6,
               std::get<0>(params),
               std::get<1>(params),
               std::get<2>(params),
               std::get<3>(params));
    }
    return 0;
}
