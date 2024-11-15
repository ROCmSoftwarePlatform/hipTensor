// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include <hiptensor/internal/hiptensor_utility.hpp>

using F16 = ck::half_t;
using F32 = float;

using ADataType = F32;
using BDataType = F32;

using UnaryScale  = ck::tensor_operation::element_wise::Scale;
using UnarySquare = ck::tensor_operation::element_wise::UnarySquare;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using UnaryScaleSquare
    = ck::tensor_operation::element_wise::UnaryCombinedOp<PassThrough, PassThrough>;
using DeviceElementwisePermuteInstance = ck::tensor_operation::device::DeviceElementwiseImpl<
    ck::Tuple<ADataType>, // InDataTypeTuple
    ck::Tuple<BDataType>, // OutDataTypeTuple
    // PassThrough, // Elementwise
    UnaryScaleSquare, // UnaryScaleSquare
    3, // NumDim
    256, // BlockSize
    128, // M0PerBlock
    128, // M1PerBlock
    16, // M0PerThread
    16, // M1PerThread
    ck::Sequence<0, 1>, // ThreadClusterArrangeOrder
    ck::Sequence<4>, // InScalarPerVectorSeq
    ck::Sequence<4>>; // OutScalarPerVectorSeq

int main()
{
    bool time_kernel = true;

    std::vector<std::size_t>   nchw = {512, 512, 512};
    std::vector<std::size_t>   nhwc = {512, 512, 512};
    std::array<ck::index_t, 3> ab_lengths;
    std::array<ck::index_t, 3> a_strides
        = {1, static_cast<int>(nchw[0]), static_cast<int>(nchw[0] * nchw[1])};

    std::array<ck::index_t, 3> b_strides
        = {static_cast<int>(nhwc[2]), static_cast<int>(nhwc[0] * nhwc[2]), 1};
    ck::ranges::copy(nchw, ab_lengths.begin());

    size_t elementsA = std::accumulate(nchw.cbegin(), nchw.cend(), 1, std::multiplies<ADataType>{});
    size_t elementsB = std::accumulate(nhwc.cbegin(), nhwc.cend(), 1, std::multiplies<BDataType>{});
    size_t sizeA     = sizeof(ADataType) * elementsA;
    size_t sizeB     = sizeof(BDataType) * elementsB;

    void* A_d;
    void* B_d;
    CHECK_HIP_ERROR(hipMalloc((void**)&A_d, sizeA));
    CHECK_HIP_ERROR(hipMalloc((void**)&B_d, sizeB));

    ADataType* A;
    BDataType* B;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&B, sizeB));

    std::iota(A, A + elementsA, static_cast<ADataType>(1.0f));

    CHECK_HIP_ERROR(hipMemcpy(A_d, A, sizeA, hipMemcpyDefault));

    auto broadcastPermute = DeviceElementwisePermuteInstance{};
    auto argument
        = broadcastPermute.MakeArgumentPointer(ab_lengths,
                                               {a_strides},
                                               {b_strides},
                                               {A_d},
                                               {B_d},
                                               UnaryScaleSquare{PassThrough{}, PassThrough{}});

    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    auto  broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time
        = broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});
    std::size_t flop = std::size_t(2) * nchw[0] * nchw[1] * nchw[2];

    std::size_t num_btype = (sizeof(ADataType) + sizeof(BDataType)) * (nchw[0] * nchw[1] * nchw[2]);
    float       tflops    = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << broadcastPermute.GetTypeString() << ":\t";
    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    CHECK_HIP_ERROR(hipHostFree(A));
    CHECK_HIP_ERROR(hipHostFree(B));
    CHECK_HIP_ERROR(hipFree(A_d));
    CHECK_HIP_ERROR(hipFree(B_d));
    return 0;
}
