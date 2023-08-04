/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>

#include "contraction_test.hpp"
// #include "detail/dlrm_dot.hpp"
#include "contraction_default_test_params.hpp"

namespace hiptensor
{
    struct TestParams : public ContractionDefaultTestParams
    {
    };

} // namespace hiptensor

class ContractionTestBasic : public hiptensor::ContractionTest
{
};

TEST_P(ContractionTestBasic, RunKernel)
{
    static bool ranWarmup = false;
    if(!ranWarmup)
    {
        this->Warmup();
        ranWarmup = true;
    }
    this->RunKernel();
}

INSTANTIATE_TEST_SUITE_P(
    ContractionTests,
    ContractionTestBasic,
    ::testing::Combine(::testing::ValuesIn(hiptensor::TestParams::dataTypes()),
                       ::testing::ValuesIn(hiptensor::TestParams::computeTypes()),
                       ::testing::ValuesIn(hiptensor::TestParams::algorithms()),
                       ::testing::ValuesIn(hiptensor::TestParams::operators()),
                       ::testing::ValuesIn(hiptensor::TestParams::workSizePrefrences()),
                       ::testing::ValuesIn(hiptensor::TestParams::logLevels()),
                       ::testing::ValuesIn(hiptensor::TestParams::problemLengths()),
                       ::testing::ValuesIn(hiptensor::TestParams::problemStrides()),
                       ::testing::ValuesIn(hiptensor::TestParams::alphas()),
                       ::testing::ValuesIn(hiptensor::TestParams::betas())));