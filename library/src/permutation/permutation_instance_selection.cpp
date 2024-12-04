/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include "permutation_instance_selection.hpp"

namespace hiptensor
{
    std::tuple<ck::index_t,
               ck::index_t,
               ck::index_t,
               ck::index_t,
               ck::index_t,
               std::pair<ck::index_t, ck::index_t>>
        selectInstanceParams(std::vector<Uid> const&      lengths,
                             hipDataType                  typeIn,
                             hipDataType                  typeOut,
                             hiptensorOperator_t          aOp,
                             hiptensorOperator_t          bOp,
                             hiptensor::PermutationOpId_t scale,
                             ck::index_t                  numDim)
    {
        // TODO lengths, outDims => blockSize ...
        return make_tuple(256, 64, 64, 4, 4, std::make_pair(0, 1));
    }
} // namespace hiptensor