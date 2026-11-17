/*
 * Fuzzer for the recursive bilateral filter (SimdRecursiveBilateralFilter*).
 *
 * Threat model: the filter processes 8-bit images (attacker-supplied content)
 * with width/height/channels, sigma values and implementation flags that come
 * from untrusted configuration or model data in downstream consumers.
 * Memory-safety violations in the recursive row/column kernels under
 * adversarial shapes, strides and flags are real bugs.
 *
 * Design (project convention, cf. simd_gaussian_blur_fuzzer):
 *  - budgeted image geometry decoded from fuzz bytes;
 *  - sigma values kept in a sane numeric range (0.05..50) so the harness
 *    targets memory safety rather than the library's unvalidated-sigma
 *    float->int conversion behaviour;
 *  - differential oracle: Simd::Base::RecursiveBilateralFilterInit reference
 *    vs dispatched SimdRecursiveBilateralFilterInit, byte tolerance 2;
 *  - all four flag dimensions exercised (fast/precise, avg/max/sum colour
 *    difference, FMA-avoid), strides carrying fuzz-controlled slack.
 */
#include "Simd/SimdArray.h"
#include "Simd/SimdLib.h"
#include "Simd/SimdRecursiveBilateralFilter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace
{
    unsigned Decode(uint8_t value)
    {
        return static_cast<unsigned>(uint8_t(value - 32));
    }

    size_t Div(size_t value, size_t divisor)
    {
        return divisor ? value % divisor : 0;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    const size_t header = 8;
    if (size <= header)
        return 0;

    const size_t width = 1 + Div(Decode(data[0]), 64);
    const size_t height = 1 + Div(Decode(data[1]), 64);
    const size_t channels = 1 + Div(Decode(data[2]), 4);
    const SimdRecursiveBilateralFilterFlags flags =
        (SimdRecursiveBilateralFilterFlags)(Decode(data[3]) % 16);
    const float sigmaSpatial = 0.05f + (Decode(data[4]) % 200) * 0.25f;
    const float sigmaRange = 0.05f + (Decode(data[5]) % 200) * 0.25f;
    const size_t srcStride = width * channels + Div(Decode(data[6]), 17);
    const size_t dstStride = width * channels + Div(Decode(data[7]), 17);

    const size_t sizeI = srcStride * height;
    const size_t sizeO = dstStride * height;
    const size_t budget = size_t(1024) * size_t(1024);
    if (sizeI == 0 || sizeO == 0 || sizeI > budget || sizeO > budget)
        return 0;

    void* referenceFilter = Simd::Base::RecursiveBilateralFilterInit(width, height,
        channels, &sigmaSpatial, &sigmaRange, flags);
    void* actualFilter = SimdRecursiveBilateralFilterInit(width, height,
        channels, &sigmaSpatial, &sigmaRange, flags);
    if ((referenceFilter == NULL) != (actualFilter == NULL))
    {
        SimdRelease(referenceFilter);
        SimdRelease(actualFilter);
        std::abort();
    }
    if (referenceFilter == NULL)
        return 0;

    Simd::Array8u src(sizeI);
    Simd::Array8u dstRef(sizeO);
    Simd::Array8u dstAct(sizeO);
    if (!src.data || !dstRef.data || !dstAct.data)
    {
        SimdRelease(referenceFilter);
        SimdRelease(actualFilter);
        return 0;
    }

    const uint8_t* payload = data + header;
    const size_t payloadSize = std::min(size - header, size_t(4096));
    for (size_t index = 0; index < sizeI; ++index)
        src[index] = payload[index % payloadSize];
    for (size_t index = 0; index < sizeO; ++index)
    {
        dstRef[index] = 0x80;
        dstAct[index] = 0x80;
    }

    SimdRecursiveBilateralFilterRun(referenceFilter, src.data, srcStride, dstRef.data, dstStride);
    SimdRecursiveBilateralFilterRun(actualFilter, src.data, srcStride, dstAct.data, dstStride);

    for (size_t index = 0; index < sizeO; ++index)
    {
        const int reference = dstRef[index];
        const int actual = dstAct[index];
        if (std::abs(reference - actual) > 2)
            std::abort();
    }

    SimdRelease(referenceFilter);
    SimdRelease(actualFilter);
    return 0;
}