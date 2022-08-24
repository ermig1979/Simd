/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar,
*               2014-2018 Antonenka Mikhail,
*               2018-2018 Radchenko Andrey,
*               2019-2019 Facundo Galan.
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
*/
#include "Simd/SimdConfig.h"

#ifndef SIMD_LIB_CPP
#define SIMD_LIB_CPP
#endif

#if defined(WIN32) && !defined(SIMD_STATIC)

#define SIMD_EXPORTS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD dwReasonForCall, LPVOID lpReserved)
{
    switch (dwReasonForCall)
    {
    case DLL_PROCESS_DETACH:
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        return TRUE;
    }
    return TRUE;
}
#endif//WIN32

#include "Simd/SimdLib.h"

#include "Simd/SimdMemory.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdAlignment.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdEmpty.h"

#include "Simd/SimdGaussianBlur.h"
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdImageSave.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetDeconvolution32f.h"
#include "Simd/SimdSynetInnerProduct32f.h"
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetScale8i.h"

#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdAvx512vnni.h"
#include "Simd/SimdAvx512bf16.h"
#include "Simd/SimdAmx.h"
#include "Simd/SimdVmx.h"
#include "Simd/SimdVsx.h"
#include "Simd/SimdNeon.h"

#if !defined(SIMD_VERSION)
#include "Simd/SimdVersion.h"
#endif

namespace Simd
{
    const size_t ALIGNMENT = GetAlignment();
}

SIMD_API const char * SimdVersion()
{
    return SIMD_VERSION;
}

using namespace Simd;

SIMD_API size_t SimdCpuInfo(SimdCpuInfoType type)
{
    switch (type)
    {
    case SimdCpuInfoSockets: return Cpu::SOCKET_NUMBER;
    case SimdCpuInfoCores: return Cpu::CORE_NUMBER;
    case SimdCpuInfoThreads: return Cpu::THREAD_NUMBER;
    case SimdCpuInfoCacheL1: return Cpu::L1_CACHE_SIZE;
    case SimdCpuInfoCacheL2: return Cpu::L2_CACHE_SIZE;
    case SimdCpuInfoCacheL3: return Cpu::L3_CACHE_SIZE;
#ifdef SIMD_SSE41_ENABLE
    case SimdCpuInfoSse41: return Sse41::Enable ? 1 : 0;
#endif
#ifdef SIMD_AVX_ENABLE
    case SimdCpuInfoAvx: return Avx::Enable ? 1 : 0;
#endif
#ifdef SIMD_AVX2_ENABLE
    case SimdCpuInfoAvx2: return Avx2::Enable ? 1 : 0;
#endif
#ifdef SIMD_AVX512BW_ENABLE
    case SimdCpuInfoAvx512bw: return Avx512bw::Enable ? 1 : 0;
#endif
#ifdef SIMD_AVX512VNNI_ENABLE
    case SimdCpuInfoAvx512vnni: return Avx512vnni::Enable ? 1 : 0;
#endif
#ifdef SIMD_AVX512BF16_ENABLE
    case SimdCpuInfoAvx512bf16: return Avx512bf16::Enable ? 1 : 0;
#endif
#ifdef SIMD_AMX_ENABLE
    case SimdCpuInfoAmx: return Amx::Enable ? 1 : 0;
#endif
#ifdef SIMD_VMX_ENABLE
    case SimdCpuInfoVmx: return Vmx::Enable ? 1 : 0;
#endif
#ifdef SIMD_VSX_ENABLE
    case SimdCpuInfoVsx: return Vsx::Enable ? 1 : 0;
#endif
#ifdef SIMD_NEON_ENABLE
    case SimdCpuInfoNeon: return Neon::Enable ? 1 : 0;
#endif
    default:
        return 0;
    }
}

SIMD_API const char * SimdPerformanceStatistic()
{
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    return Base::PerformanceMeasurerStorage::s_storage.PerformanceStatistic();
#else
    return "";
#endif
}

SIMD_API void * SimdAllocate(size_t size, size_t align)
{
    return Allocate(size, align);
}

SIMD_API void SimdFree(void * ptr)
{
    Free(ptr);
}

SIMD_API size_t SimdAlign(size_t size, size_t align)
{
    return AlignHi(size, align);
}

SIMD_API size_t SimdAlignment()
{
    return Simd::ALIGNMENT;
}

SIMD_API void SimdRelease(void * context)
{
    delete (Deletable*)context;
}

SIMD_API size_t SimdGetThreadNumber()
{
    return Base::GetThreadNumber();
}

SIMD_API void SimdSetThreadNumber(size_t threadNumber)
{
    Base::SetThreadNumber(threadNumber);
}

SIMD_API SimdBool SimdGetFastMode()
{
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        return Sse41::GetFastMode();
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        return Neon::GetFastMode();
    else
#endif
        return SimdFalse;
}

SIMD_API void SimdSetFastMode(SimdBool value)
{
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::SetFastMode(value);
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        Neon::SetFastMode(value);
#endif
}

SIMD_API void SimdEmpty()
{
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::Empty();
#endif
}

SIMD_API uint32_t SimdCrc32(const void* src, size_t size)
{
    return Base::Crc32(src, size);
}

SIMD_API uint32_t SimdCrc32c(const void * src, size_t size)
{
    SIMD_EMPTY();
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable)
        return Sse41::Crc32c(src, size);
    else
#endif
        return Base::Crc32c(src, size);
}

SIMD_API void SimdAbsDifference(const uint8_t *a, size_t aStride, const uint8_t * b, size_t bStride, uint8_t *c, size_t cStride,
    size_t width, size_t height)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AbsDifference(a, aStride, b, bStride, c, cStride, width, height);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::AbsDifference(a, aStride, b, bStride, c, cStride, width, height);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::AbsDifference(a, aStride, b, bStride, c, cStride, width, height);
    else
#endif 
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::AbsDifference(a, aStride, b, bStride, c, cStride, width, height);
    else
#endif
        Base::AbsDifference(a, aStride, b, bStride, c, cStride, width, height);
}

SIMD_API void SimdAbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t * b, size_t bStride,
                                   size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
        Base::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
}

SIMD_API void SimdAbsDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
                                         const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
        Base::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
}

SIMD_API void SimdAbsDifferenceSums3x3(const uint8_t *current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
                                       size_t width, size_t height, uint64_t * sums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2)
        Avx2::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A + 2)
        Sse41::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 2)
        Vmx::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 2)
        Neon::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
    else
#endif
        Base::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
}

SIMD_API void SimdAbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
                                             const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
{
    SIMD_EMPTY();
#if defined(SIMD_AVX512BW_ENABLE) && !defined(SIMD_MASKZ_LOAD_ERROR)
    if (Avx512bw::Enable)
        Avx512bw::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2)
        Avx2::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A + 2)
        Sse41::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 2)
        Vmx::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 2)
        Neon::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
        Base::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
}

SIMD_API void SimdAbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                          uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Simd::Avx512bw::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Simd::Avx2::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdAddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                       const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
                                       uint16_t weight, uint8_t * difference, size_t differenceStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif
        Base::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
}

SIMD_API void SimdAlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
                   const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif
        Base::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
}

SIMD_API void SimdAlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
    const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride,
    size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::AlphaBlending2x(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::AlphaBlending2x(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::AlphaBlending2x(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdAlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t alpha, uint8_t* dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AlphaBlendingUniform(src, srcStride, width, height, channelCount, alpha, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::AlphaBlendingUniform(src, srcStride, width, height, channelCount, alpha, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::AlphaBlendingUniform(src, srcStride, width, height, channelCount, alpha, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::AlphaBlendingUniform(src, srcStride, width, height, channelCount, alpha, dst, dstStride);
    else
#endif
        Base::AlphaBlendingUniform(src, srcStride, width, height, channelCount, alpha, dst, dstStride);
}

SIMD_API void SimdAlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AlphaFilling(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::AlphaFilling(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::AlphaFilling(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::AlphaFilling(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
    else
#endif
        Base::AlphaFilling(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
}

SIMD_API void SimdAlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AlphaPremultiply(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable)
        Avx2::AlphaPremultiply(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::AlphaPremultiply(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        Neon::AlphaPremultiply(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::AlphaPremultiply(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdAlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AlphaUnpremultiply(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable)
        Avx2::AlphaUnpremultiply(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::AlphaUnpremultiply(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        Neon::AlphaUnpremultiply(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::AlphaUnpremultiply(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                          uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
        Base::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
}

SIMD_API void SimdBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                          uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
        Base::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
}

SIMD_API void SimdBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                           const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
                                           uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BackgroundIncrementCount(value, valueStride, width, height, loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundIncrementCount(value, valueStride, width, height, loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BackgroundIncrementCount(value, valueStride, width, height, loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundIncrementCount(value, valueStride, width, height, loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BackgroundIncrementCount(value, valueStride, width, height, loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif
        Base::BackgroundIncrementCount(value, valueStride, width, height, loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
}

SIMD_API void SimdBackgroundAdjustRange(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
                                        uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
                                        uint8_t * hiValue, size_t hiValueStride, uint8_t threshold)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
            hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
            hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    else
#endif
        Base::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold);
}

SIMD_API void SimdBackgroundAdjustRangeMasked(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
                                              uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
                                              uint8_t * hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
            hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif
        Base::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
}

SIMD_API void SimdBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                       uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
        Base::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
}

SIMD_API void SimdBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                             uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    else
#endif
        Base::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
}

SIMD_API void SimdBackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                     uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif
        Base::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
}

SIMD_API void SimdBase64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Base64Decode(src, srcSize, dst, dstSize);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable)
        Avx2::Base64Decode(src, srcSize, dst, dstSize);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::Base64Decode(src, srcSize, dst, dstSize);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        Neon::Base64Decode(src, srcSize, dst, dstSize);
    else
#endif
    Base::Base64Decode(src, srcSize, dst, dstSize);
}

SIMD_API void SimdBase64Encode(const uint8_t* src, size_t size, uint8_t* dst)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Base64Encode(src, size, dst);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable)
        Avx2::Base64Encode(src, size, dst);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::Base64Encode(src, size, dst);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        Neon::Base64Encode(src, size, dst);
    else
#endif
        Base::Base64Encode(src, size, dst);
}

SIMD_API void SimdBayerToBgr(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgr, size_t bgrStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::A + 2)
        Avx512bw::BayerToBgr(bayer, width, height, bayerStride, bayerFormat, bgr, bgrStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A + 2)
        Avx2::BayerToBgr(bayer, width, height, bayerStride, bayerFormat, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A + 2)
        Sse41::BayerToBgr(bayer, width, height, bayerStride, bayerFormat, bgr, bgrStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 2)
        Neon::BayerToBgr(bayer, width, height, bayerStride, bayerFormat, bgr, bgrStride);
    else
#endif
        Base::BayerToBgr(bayer, width, height, bayerStride, bayerFormat, bgr, bgrStride);
}

SIMD_API void SimdBayerToBgra(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::A + 2)
        Avx512bw::BayerToBgra(bayer, width, height, bayerStride, bayerFormat, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A + 2)
        Avx2::BayerToBgra(bayer, width, height, bayerStride, bayerFormat, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A + 2)
        Sse41::BayerToBgra(bayer, width, height, bayerStride, bayerFormat, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 2)
        Neon::BayerToBgra(bayer, width, height, bayerStride, bayerFormat, bgra, bgraStride, alpha);
    else
#endif
        Base::BayerToBgra(bayer, width, height, bayerStride, bayerFormat, bgra, bgraStride, alpha);
}

SIMD_API void SimdBgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgraToBayer(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BgraToBayer(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgraToBayer(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgraToBayer(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
    else
#endif
        Base::BgraToBayer(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
}

SIMD_API void SimdBgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
    else
#endif
        Base::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
}

SIMD_API void SimdBgraToGray(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *gray, size_t grayStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::HA)
        Neon::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif
        Base::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
}

SIMD_API void SimdBgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgraToRgb(bgra, width, height, bgraStride, rgb, rgbStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::BgraToRgb(bgra, width, height, bgraStride, rgb, rgbStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::BgraToRgb(bgra, width, height, bgraStride, rgb, rgbStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgraToRgb(bgra, width, height, bgraStride, rgb, rgbStride);
    else
#endif
        Base::BgraToRgb(bgra, width, height, bgraStride, rgb, rgbStride);
}

SIMD_API void SimdBgraToRgba(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgba, size_t rgbaStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgraToRgba(bgra, width, height, bgraStride, rgba, rgbaStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::BgraToRgba(bgra, width, height, bgraStride, rgba, rgbaStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::BgraToRgba(bgra, width, height, bgraStride, rgba, rgbaStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgraToRgba(bgra, width, height, bgraStride, rgba, rgbaStride);
    else
#endif
        Base::BgraToRgba(bgra, width, height, bgraStride, rgba, rgbaStride);
}

SIMD_API void SimdBgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::DA)
        Sse41::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
    uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
{
    SIMD_EMPTY();
    Base::BgraToYuv420pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
}

SIMD_API void SimdBgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::DA)
        Sse41::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, 
    uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
{
    SIMD_EMPTY();
    Base::BgraToYuv444pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
}

SIMD_API void SimdBgraToYuva420p(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, 
    uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride, uint8_t * a, size_t aStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgraToYuva420p(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::DA)
        Avx2::BgraToYuva420p(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::DA)
        Sse41::BgraToYuva420p(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::BgraToYuva420p(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride);
    else
#endif
        Base::BgraToYuva420p(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride);
}

SIMD_API void SimdBgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgrToBayer(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BgrToBayer(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgrToBayer(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgrToBayer(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
    else
#endif
        Base::BgrToBayer(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
}

SIMD_API void SimdBgrToBgra(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
    else
#endif
#if defined(SIMD_AVX2_ENABLE) && !defined(SIMD_CLANG_AVX2_BGR_TO_BGRA_ERROR)
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
    else
#endif
        Base::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
}

SIMD_API void SimdBgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
    const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::HA)
        Sse41::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
    else
#endif
        Base::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
}

SIMD_API void SimdBgrToGray(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t *gray, size_t grayStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif
#if defined(SIMD_AVX2_ENABLE) && !defined(SIMD_CLANG_AVX2_BGR_TO_BGRA_ERROR)
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif
        Base::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
}

SIMD_API void SimdBgrToHsl(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsl, size_t hslStride)
{
    SIMD_EMPTY();
    Base::BgrToHsl(bgr, width, height, bgrStride, hsl, hslStride);
}

SIMD_API void SimdBgrToHsv(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsv, size_t hsvStride)
{
    SIMD_EMPTY();
    Base::BgrToHsv(bgr, width, height, bgrStride, hsv, hsvStride);
}

SIMD_API void SimdBgrToRgb(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t * rgb, size_t rgbStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgrToRgb(bgr, width, height, bgrStride, rgb, rgbStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::BgrToRgb(bgr, width, height, bgrStride, rgb, rgbStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::BgrToRgb(bgr, width, height, bgrStride, rgb, rgbStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgrToRgb(bgr, width, height, bgrStride, rgb, rgbStride);
    else
#endif
        Base::BgrToRgb(bgr, width, height, bgrStride, rgb, rgbStride);
}

SIMD_API void SimdBgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::DA)
        Sse41::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgrToYuv422p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::DA)
        Sse41::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                  uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif
        Base::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
}

SIMD_API void SimdAveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                           uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative,
                           uint8_t * dst, size_t dstStride, SimdCompareType compareType)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif
        Base::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
}

SIMD_API void SimdAveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height,
    size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride)
{
    SIMD_EMPTY();
    Base::AveragingBinarizationV2(src, srcStride, width, height, neighborhood, shift, positive, negative, dst, dstStride);
}

SIMD_API void SimdConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height,
                                   uint8_t value, SimdCompareType compareType, uint32_t * count)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ConditionalCount8u(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ConditionalCount8u(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::ConditionalCount8u(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::ConditionalCount8u(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::ConditionalCount8u(src, stride, width, height, value, compareType, count);
    else
#endif
        Base::ConditionalCount8u(src, stride, width, height, value, compareType, count);
}

SIMD_API void SimdConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height,
                                     int16_t value, SimdCompareType compareType, uint32_t * count)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ConditionalCount16i(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::ConditionalCount16i(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::HA)
        Sse41::ConditionalCount16i(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::ConditionalCount16i(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::HA)
        Neon::ConditionalCount16i(src, stride, width, height, value, compareType, count);
    else
#endif
        Base::ConditionalCount16i(src, stride, width, height, value, compareType, count);
}

SIMD_API void SimdConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                 const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
        Base::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
}

SIMD_API void SimdConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                       const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
        Base::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
}

SIMD_API void SimdConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                       const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2)
        Avx2::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A + 2)
        Sse41::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 2)
        Vmx::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 2)
        Neon::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
        Base::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
}

SIMD_API void SimdConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if (Vmx::Enable && width >= Vmx::A)
        Vmx::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
    else
#endif
        Base::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
}

SIMD_API void SimdCopy(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    Base::Copy(src, srcStride, width, height, pixelSize, dst, dstStride);
}

SIMD_API void SimdCopyFrame(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize,
                           size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    Base::CopyFrame(src, srcStride, width, height, pixelSize, frameLeft, frameTop, frameRight, frameBottom, dst, dstStride);
}

SIMD_API void SimdDeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
                    uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif
        Base::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
}

SIMD_API void SimdDeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
    uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::DeinterleaveBgr(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::DeinterleaveBgr(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::DeinterleaveBgr(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::DeinterleaveBgr(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
    else
#endif
        Base::DeinterleaveBgr(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
}

SIMD_API void SimdDeinterleaveBgra(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height,
    uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::DeinterleaveBgra(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::DeinterleaveBgra(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::DeinterleaveBgra(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::DeinterleaveBgra(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
    else
#endif
        Base::DeinterleaveBgra(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
}

SIMD_API void * SimdDetectionLoadStringXml(char * xml)
{
    SIMD_EMPTY();
    return Base::DetectionLoadStringXml(xml);
}

SIMD_API void * SimdDetectionLoadA(const char * path)
{
    SIMD_EMPTY();
    return Base::DetectionLoadA(path);
}

SIMD_API void SimdDetectionInfo(const void * data, size_t * width, size_t * height, SimdDetectionInfoFlags * flags)
{
    SIMD_EMPTY();
    Base::DetectionInfo(data, width, height, flags);
}

SIMD_API void * SimdDetectionInit(const void * data, uint8_t * sum, size_t sumStride, size_t width, size_t height,
    uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride, int throughColumn, int int16)
{
    SIMD_EMPTY();
    return Base::DetectionInit(data, sum, sumStride, width, height, sqsum, sqsumStride, tilted, tiltedStride, throughColumn, int16);
}

SIMD_API void SimdDetectionPrepare(void * hid)
{
    SIMD_EMPTY();
    Base::DetectionPrepare(hid);
}

SIMD_API void SimdDetectionHaarDetect32fp(const void * hid, const uint8_t * mask, size_t maskStride, 
    ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    size_t width = right - left;
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::DetectionHaarDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::DetectionHaarDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::DetectionHaarDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::DetectionHaarDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
        Base::DetectionHaarDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
}

SIMD_API void SimdDetectionHaarDetect32fi(const void * hid, const uint8_t * mask, size_t maskStride,
    ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    size_t width = right - left;
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::DetectionHaarDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::DetectionHaarDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::DetectionHaarDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::DetectionHaarDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
        Base::DetectionHaarDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
}

SIMD_API void SimdDetectionLbpDetect32fp(const void * hid, const uint8_t * mask, size_t maskStride,
    ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    size_t width = right - left;
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::DetectionLbpDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::DetectionLbpDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::DetectionLbpDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::DetectionLbpDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
        Base::DetectionLbpDetect32fp(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
}

SIMD_API void SimdDetectionLbpDetect32fi(const void * hid, const uint8_t * mask, size_t maskStride,
    ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    size_t width = right - left;
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::DetectionLbpDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::DetectionLbpDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::DetectionLbpDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::DetectionLbpDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
        Base::DetectionLbpDetect32fi(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
}

SIMD_API void SimdDetectionLbpDetect16ip(const void * hid, const uint8_t * mask, size_t maskStride,
    ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    size_t width = right - left;
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::DetectionLbpDetect16ip(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::DetectionLbpDetect16ip(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::DetectionLbpDetect16ip(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::DetectionLbpDetect16ip(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
        Base::DetectionLbpDetect16ip(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
}

SIMD_API void SimdDetectionLbpDetect16ii(const void * hid, const uint8_t * mask, size_t maskStride,
    ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    size_t width = right - left;
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::DetectionLbpDetect16ii(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::DetectionLbpDetect16ii(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::DetectionLbpDetect16ii(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::DetectionLbpDetect16ii(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
    else
#endif
        Base::DetectionLbpDetect16ii(hid, mask, maskStride, left, top, right, bottom, dst, dstStride);
}

SIMD_API void SimdEdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                 uint8_t * background, size_t backgroundStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif
        Base::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
}

SIMD_API void SimdEdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                 uint8_t * background, size_t backgroundStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif
        Base::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
}

SIMD_API void SimdEdgeBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                  const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::EdgeBackgroundIncrementCount(value, valueStride, width, height, backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundIncrementCount(value, valueStride, width, height, backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::EdgeBackgroundIncrementCount(value, valueStride, width, height, backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundIncrementCount(value, valueStride, width, height, backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::EdgeBackgroundIncrementCount(value, valueStride, width, height, backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif
        Base::EdgeBackgroundIncrementCount(value, valueStride, width, height, backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
}

SIMD_API void SimdEdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
                               uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold);
    else
#endif
        Base::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold);
}

SIMD_API void SimdEdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
                               uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif
        Base::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold, mask, maskStride);
}

SIMD_API void SimdEdgeBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                              uint8_t * background, size_t backgroundStride)
{
    SIMD_EMPTY();
    Base::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
}

SIMD_API void SimdEdgeBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                              uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif
        Base::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
}

SIMD_API void SimdFill(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uint8_t value)
{
    SIMD_EMPTY();
    Base::Fill(dst, stride, width, height, pixelSize, value);
}

SIMD_API void SimdFillFrame(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize,
                           size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t value)
{
    SIMD_EMPTY();
    Base::FillFrame(dst, stride, width, height, pixelSize, frameLeft, frameTop, frameRight, frameBottom, value);
}

SIMD_API void SimdFillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::FillBgr(dst, stride, width, height, blue, green, red);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::FillBgr(dst, stride, width, height, blue, green, red);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::FillBgr(dst, stride, width, height, blue, green, red);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::FillBgr(dst, stride, width, height, blue, green, red);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::FillBgr(dst, stride, width, height, blue, green, red);
    else
#endif
        Base::FillBgr(dst, stride, width, height, blue, green, red);
}

SIMD_API void SimdFillBgra(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::F)
        Avx2::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::F)
        Sse41::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::F)
        Vmx::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F)
        Neon::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif
        Base::FillBgra(dst, stride, width, height, blue, green, red, alpha);
}

SIMD_API void SimdFillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8_t * pixel, size_t pixelSize)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::FillPixel(dst, stride, width, height, pixel, pixelSize);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::FillPixel(dst, stride, width, height, pixel, pixelSize);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::FillPixel(dst, stride, width, height, pixel, pixelSize);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::FillPixel(dst, stride, width, height, pixel, pixelSize);
    else
#endif
        Base::FillPixel(dst, stride, width, height, pixel, pixelSize);
}

SIMD_API void SimdFill32f(float * dst, size_t size, const float * value)
{
    SIMD_EMPTY();
    typedef void(*SimdFill32fPtr) (float * dst, size_t size, const float * value);
    const static SimdFill32fPtr simdFill32f = SIMD_FUNC4(Fill32f, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdFill32f(dst, size, value);
}

SIMD_API void SimdFloat32ToBFloat16(const float* src, size_t size, uint16_t* dst)
{
    SIMD_EMPTY();
    typedef void(*SimdFloat32ToBFloat16Ptr) (const float* src, size_t size, uint16_t* dst);
    const static SimdFloat32ToBFloat16Ptr simdFloat32ToBFloat16 = SIMD_FUNC4(Float32ToBFloat16, SIMD_AVX512BF16_FUNC, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC);

    simdFloat32ToBFloat16(src, size, dst);
}

SIMD_API void SimdBFloat16ToFloat32(const uint16_t* src, size_t size, float* dst)
{
    SIMD_EMPTY();
    typedef void(*SimdBFloat16ToFloat32Ptr) (const uint16_t* src, size_t size, float* dst);
    const static SimdBFloat16ToFloat32Ptr simdBFloat16ToFloat32 = SIMD_FUNC3(BFloat16ToFloat32, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC);

    simdBFloat16ToFloat32(src, size, dst);
}

SIMD_API void SimdFloat32ToFloat16(const float * src, size_t size, uint16_t * dst)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Float32ToFloat16(src, size, dst);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && size >= Avx2::F)
        Avx2::Float32ToFloat16(src, size, dst);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::Float32ToFloat16(src, size, dst);
    else
#endif
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    if (Neon::Enable && size >= Neon::F)
        Neon::Float32ToFloat16(src, size, dst);
    else
#endif
        Base::Float32ToFloat16(src, size, dst);
}

SIMD_API void SimdFloat16ToFloat32(const uint16_t * src, size_t size, float * dst)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Float16ToFloat32(src, size, dst);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && size >= Avx2::F)
        Avx2::Float16ToFloat32(src, size, dst);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::Float16ToFloat32(src, size, dst);
    else
#endif
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    if (Neon::Enable && size >= Neon::F)
        Neon::Float16ToFloat32(src, size, dst);
    else
#endif
        Base::Float16ToFloat32(src, size, dst);
}

SIMD_API void SimdSquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::SquaredDifferenceSum16f(a, b, size, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && size >= Avx2::F)
        Avx2::SquaredDifferenceSum16f(a, b, size, sum);
    else
#endif
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    if (Neon::Enable && size >= Neon::F)
        Neon::SquaredDifferenceSum16f(a, b, size, sum);
    else
#endif
        Base::SquaredDifferenceSum16f(a, b, size, sum);
}

SIMD_API void SimdCosineDistance16f(const uint16_t * a, const uint16_t * b, size_t size, float * distance)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::CosineDistance16f(a, b, size, distance);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && size >= Avx2::F)
        Avx2::CosineDistance16f(a, b, size, distance);
    else
#endif
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    if (Neon::Enable && size >= Neon::F)
        Neon::CosineDistance16f(a, b, size, distance);
    else
#endif
        Base::CosineDistance16f(a, b, size, distance);
}

SIMD_API void SimdCosineDistancesMxNa16f(size_t M, size_t N, size_t K, const uint16_t * const * A, const uint16_t * const * B, float * distances)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && K >= Avx512bw::F)
        Avx512bw::CosineDistancesMxNa16f(M, N, K, A, B, distances);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && K >= Avx2::F)
        Avx2::CosineDistancesMxNa16f(M, N, K, A, B, distances);
    else
#endif
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    if (Neon::Enable && K >= Neon::F)
        Neon::CosineDistancesMxNa16f(M, N, K, A, B, distances);
    else
#endif
        Base::CosineDistancesMxNa16f(M, N, K, A, B, distances);
}

SIMD_API void SimdCosineDistancesMxNp16f(size_t M, size_t N, size_t K, const uint16_t* A, const uint16_t* B, float* distances)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && K >= Avx512bw::F)
        Avx512bw::CosineDistancesMxNp16f(M, N, K, A, B, distances);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && K >= Avx2::F)
        Avx2::CosineDistancesMxNp16f(M, N, K, A, B, distances);
    else
#endif
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    if (Neon::Enable && K >= Neon::F)
        Neon::CosineDistancesMxNp16f(M, N, K, A, B, distances);
    else
#endif
        Base::CosineDistancesMxNp16f(M, N, K, A, B, distances);
}

SIMD_API void SimdVectorNormNa16f(size_t N, size_t K, const uint16_t* const* A, float* norms)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && K >= Avx512bw::F)
        Avx512bw::VectorNormNa16f(N, K, A, norms);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && K >= Avx2::F)
        Avx2::VectorNormNa16f(N, K, A, norms);
    else
#endif
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    if (Neon::Enable && K >= Neon::F)
        Neon::VectorNormNa16f(N, K, A, norms);
    else
#endif
        return Base::VectorNormNa16f(N, K, A, norms);
}

SIMD_API void SimdVectorNormNp16f(size_t N, size_t K, const uint16_t* A, float* norms)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && K >= Avx512bw::F)
        Avx512bw::VectorNormNp16f(N, K, A, norms);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && K >= Avx2::F)
        Avx2::VectorNormNp16f(N, K, A, norms);
    else
#endif
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    if (Neon::Enable && K >= Neon::F)
        Neon::VectorNormNp16f(N, K, A, norms);
    else
#endif
        return Base::VectorNormNp16f(N, K, A, norms);
}

SIMD_API void SimdFloat32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Float32ToUint8(src, size, lower, upper, dst);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && size >= Avx2::A)
        Avx2::Float32ToUint8(src, size, lower, upper, dst);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && size >= Sse41::A)
        Sse41::Float32ToUint8(src, size, lower, upper, dst);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && size >= Neon::A)
        Neon::Float32ToUint8(src, size, lower, upper, dst);
    else
#endif
        Base::Float32ToUint8(src, size, lower, upper, dst);
}

SIMD_API void SimdUint8ToFloat32(const uint8_t * src, size_t size, const float * lower, const float * upper, float * dst)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Uint8ToFloat32(src, size, lower, upper, dst);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && size >= Avx2::HA)
        Avx2::Uint8ToFloat32(src, size, lower, upper, dst);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && size >= Sse41::A)
        Sse41::Uint8ToFloat32(src, size, lower, upper, dst);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && size >= Neon::A)
        Neon::Uint8ToFloat32(src, size, lower, upper, dst);
    else
#endif
        Base::Uint8ToFloat32(src, size, lower, upper, dst);
}

SIMD_API void SimdCosineDistance32f(const float * a, const float * b, size_t size, float * distance)
{
    SIMD_EMPTY();
    typedef void(*SimdCosineDistance32fPtr) (const float * a, const float * b, size_t size, float * distance);
    const static SimdCosineDistance32fPtr simdCosineDistance32f = SIMD_FUNC5(CosineDistance32f, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdCosineDistance32f(a, b, size, distance);
}

SIMD_API void SimdGaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                     size_t channelCount, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && (width - 1)*channelCount >= Avx512bw::A)
        Avx512bw::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && (width - 1)*channelCount >= Avx2::A)
        Avx2::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && (width - 1)*channelCount >= Sse41::A)
        Sse41::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && (width - 1)*channelCount >= Vmx::A)
        Vmx::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && (width - 1)*channelCount >= Neon::A)
        Neon::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void* SimdGaussianBlurInit(size_t width, size_t height, size_t channels, const float* sigma, const float* epsilon)
{
    SIMD_EMPTY();
    typedef void* (*SimdGaussianBlurInitPtr) (size_t width, size_t height, size_t channels, const float* sigma, const float* epsilon);
    const static SimdGaussianBlurInitPtr simdGaussianBlurInit = SIMD_FUNC4(GaussianBlurInit, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return simdGaussianBlurInit(width, height, channels, sigma, epsilon);
}

SIMD_API void SimdGaussianBlurRun(const void* filter, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
{
    SIMD_EMPTY();
    ((GaussianBlur*)filter)->Run(src, srcStride, dst, dstStride);
}

typedef void(*SimdGemm32fPtr) (size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc);

SIMD_API void SimdGemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
{
    SIMD_EMPTY();
    const static SimdGemm32fPtr simdGemm32fNN = SIMD_FUNC5(Gemm32fNN, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdGemm32fNN(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

SIMD_API void SimdGemm32fNT(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc)
{
    SIMD_EMPTY();
    const static SimdGemm32fPtr simdGemm32fNT = SIMD_FUNC5(Gemm32fNT, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdGemm32fNT(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

SIMD_API void SimdGrayToBgr(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t * bgr, size_t bgrStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
    else
#endif
        Base::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
}

SIMD_API void SimdGrayToBgra(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif
        Base::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
}

SIMD_API void SimdAbsSecondDerivativeHistogram(const uint8_t *src, size_t width, size_t height, size_t stride, size_t step, size_t indent, uint32_t * histogram)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::A + 2 * indent)
        Avx512bw::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2*indent)
        Avx2::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A + 2*indent)
        Sse41::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 2*indent)
        Vmx::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 2 * indent)
        Neon::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif
        Base::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
}

SIMD_API void SimdHistogram(const uint8_t *src, size_t width, size_t height, size_t stride, uint32_t * histogram)
{
    SIMD_EMPTY();
    Base::Histogram(src, width, height, stride, histogram);
}

SIMD_API void SimdHistogramMasked(const uint8_t *src, size_t srcStride, size_t width, size_t height, 
                                  const uint8_t * mask, size_t maskStride, uint8_t index, uint32_t * histogram)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
    else
#endif
        Base::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
}

SIMD_API void SimdHistogramConditional(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t * histogram)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::HistogramConditional(src, srcStride, width, height, mask, maskStride, value, compareType, histogram);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::HistogramConditional(src, srcStride, width, height, mask, maskStride, value, compareType, histogram);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::HistogramConditional(src, srcStride, width, height, mask, maskStride, value, compareType, histogram);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::HistogramConditional(src, srcStride, width, height, mask, maskStride, value, compareType, histogram);
    else
#endif
        Base::HistogramConditional(src, srcStride, width, height, mask, maskStride, value, compareType, histogram);
}

SIMD_API void SimdNormalizedColors(const uint32_t * histogram, uint8_t * colors)
{
    SIMD_EMPTY();
    Base::NormalizedColors(histogram, colors);
}

SIMD_API void SimdChangeColors(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * colors, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::HA)
        Avx512bw::ChangeColors(src, srcStride, width, height, colors, dst, dstStride);
    else
#endif
        Base::ChangeColors(src, srcStride, width, height, colors, dst, dstStride);
}

SIMD_API void SimdNormalizeHistogram(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::HA)
        Avx512bw::NormalizeHistogram(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::NormalizeHistogram(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdHogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height, 
                                         size_t cellX, size_t cellY, size_t quantization, float * histograms)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::HA + 2)
        Avx512bw::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2)
        Avx2::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A + 2)
        Sse41::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
    else
#endif
#ifdef SIMD_VSX_ENABLE
    if(Vsx::Enable && width >= Vsx::A + 2)
        Vsx::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 2)
        Neon::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
    else
#endif
        Base::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
}

SIMD_API void SimdHogExtractFeatures(const uint8_t * src, size_t stride, size_t width, size_t height, float * features)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::HA + 2)
        Avx512bw::HogExtractFeatures(src, stride, width, height, features);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::HA + 2)
        Avx2::HogExtractFeatures(src, stride, width, height, features);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A + 2)
        Sse41::HogExtractFeatures(src, stride, width, height, features);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 2)
        Neon::HogExtractFeatures(src, stride, width, height, features);
    else
#endif
        Base::HogExtractFeatures(src, stride, width, height, features);
}

SIMD_API void SimdHogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F && count >= Sse41::F)
        Avx512bw::HogDeinterleave(src, srcStride, width, height, count, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F && count >= Sse41::F)
        Avx2::HogDeinterleave(src, srcStride, width, height, count, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F && count >= Sse41::F)
        Sse41::HogDeinterleave(src, srcStride, width, height, count, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F && count >= Neon::F)
        Neon::HogDeinterleave(src, srcStride, width, height, count, dst, dstStride);
    else
#endif
        Base::HogDeinterleave(src, srcStride, width, height, count, dst, dstStride);
}

SIMD_API void SimdHogFilterSeparable(const float * src, size_t srcStride, size_t width, size_t height,
    const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F + colSize - 1)
        Avx512bw::HogFilterSeparable(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F + colSize - 1)
        Avx2::HogFilterSeparable(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F + colSize - 1)
        Sse41::HogFilterSeparable(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F + colSize - 1)
        Neon::HogFilterSeparable(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
    else
#endif
        Base::HogFilterSeparable(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
}

SIMD_API void SimdHogLiteExtractFeatures(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t cell, float * features, size_t featuresStride)
{
    SIMD_EMPTY();
#if defined(SIMD_AVX2_ENABLE) || defined(SIMD_SSE41_ENABLE) || defined(SIMD_NEON_ENABLE)
    size_t size = (width / cell - 1)*cell;
#endif
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::HogLiteExtractFeatures(src, srcStride, width, height, cell, features, featuresStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && size >= Avx2::A)
        Avx2::HogLiteExtractFeatures(src, srcStride, width, height, cell, features, featuresStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && size >= Sse41::A)
        Sse41::HogLiteExtractFeatures(src, srcStride, width, height, cell, features, featuresStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && size >= Neon::A)
        Neon::HogLiteExtractFeatures(src, srcStride, width, height, cell, features, featuresStride);
    else
#endif
        Base::HogLiteExtractFeatures(src, srcStride, width, height, cell, features, featuresStride);
}

SIMD_API void SimdHogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::HogLiteFilterFeatures(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable)
        Avx2::HogLiteFilterFeatures(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable)
        Avx::HogLiteFilterFeatures(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::HogLiteFilterFeatures(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        Neon::HogLiteFilterFeatures(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
    else
#endif
        Base::HogLiteFilterFeatures(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
}

SIMD_API void SimdHogLiteResizeFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::HogLiteResizeFeatures(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable)
        Avx2::HogLiteResizeFeatures(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable)
        Avx::HogLiteResizeFeatures(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::HogLiteResizeFeatures(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        Neon::HogLiteResizeFeatures(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
    else
#endif
        Base::HogLiteResizeFeatures(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
}

SIMD_API void SimdHogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::HogLiteCompressFeatures(src, srcStride, width, height, pca, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable)
        Avx2::HogLiteCompressFeatures(src, srcStride, width, height, pca, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable)
        Avx::HogLiteCompressFeatures(src, srcStride, width, height, pca, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable)
        Sse41::HogLiteCompressFeatures(src, srcStride, width, height, pca, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        Neon::HogLiteCompressFeatures(src, srcStride, width, height, pca, dst, dstStride);
    else
#endif
        Base::HogLiteCompressFeatures(src, srcStride, width, height, pca, dst, dstStride);
}

SIMD_API void SimdHogLiteFilterSeparable(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::HogLiteFilterSeparable(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && srcWidth >= hSize - 1 + Avx2::F)
        Avx2::HogLiteFilterSeparable(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && srcWidth >= hSize - 1 + Avx::F)
        Avx::HogLiteFilterSeparable(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && srcWidth >= hSize - 1 + Sse41::F)
        Sse41::HogLiteFilterSeparable(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && srcWidth >= hSize - 1 + Neon::F)
        Neon::HogLiteFilterSeparable(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
    else
#endif
        Base::HogLiteFilterSeparable(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
}

SIMD_API void SimdHogLiteFindMax7x7(const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * value, size_t * col, size_t * row)
{
    SIMD_EMPTY();
    typedef void(*SimdHogLiteFindMax7x7Ptr) (const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * value, size_t * col, size_t * row);
    const static SimdHogLiteFindMax7x7Ptr simdHogLiteFindMax7x7 = SIMD_FUNC3(HogLiteFindMax7x7, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdHogLiteFindMax7x7(a, aStride, b, bStride, height, value, col, row);
}

SIMD_API void SimdHogLiteCreateMask(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    typedef void(*SimdHogLiteCreateMaskPtr) (const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride);
    const static SimdHogLiteCreateMaskPtr simdHogLiteCreateMask = SIMD_FUNC4(HogLiteCreateMask, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdHogLiteCreateMask(src, srcStride, srcWidth, srcHeight, threshold, scale, size, dst, dstStride);
}

SIMD_API uint8_t* SimdImageSaveToMemory(const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, size_t* size)
{
    SIMD_EMPTY();
    const static Simd::ImageSaveToMemoryPtr imageSaveToMemory = SIMD_FUNC4(ImageSaveToMemory, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return imageSaveToMemory(src, stride, width, height, format, file, quality, size);
}

SIMD_API SimdBool SimdImageSaveToFile(const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, const char* path)
{
    SIMD_EMPTY();
    const static Simd::ImageSaveToMemoryPtr imageSaveToMemory = SIMD_FUNC4(ImageSaveToMemory, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return ImageSaveToFile(imageSaveToMemory, src, stride, width, height, format, file, quality, path);
}

SIMD_API uint8_t* SimdNv12SaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* uv, size_t uvStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size)
{
    SIMD_EMPTY();
    typedef uint8_t*(*SimdNv12SaveAsJpegToMemoryPtr) (const uint8_t* y, size_t yStride, const uint8_t* uv, size_t uvStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size);
    const static SimdNv12SaveAsJpegToMemoryPtr simdNv12SaveAsJpegToMemory = SIMD_FUNC4(Nv12SaveAsJpegToMemory, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return simdNv12SaveAsJpegToMemory(y, yStride, uv, uvStride, width, height, yuvType, quality, size);
}

SIMD_API uint8_t* SimdYuv420pSaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, 
    size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size)
{
    SIMD_EMPTY();
    typedef uint8_t* (*SimdYuv420pSaveAsJpegToMemoryPtr) (const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, 
        const uint8_t* v, size_t vStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size);
    const static SimdYuv420pSaveAsJpegToMemoryPtr simdYuv420pSaveAsJpegToMemory = SIMD_FUNC4(Yuv420pSaveAsJpegToMemory, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return simdYuv420pSaveAsJpegToMemory(y, yStride, u, uStride, v, vStride, width, height, yuvType, quality, size);
}

SIMD_API uint8_t* SimdImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format)
{
    SIMD_EMPTY();
    const static Simd::ImageLoadFromMemoryPtr imageLoadFromMemory = SIMD_FUNC4(ImageLoadFromMemory, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return imageLoadFromMemory(data, size, stride, width, height, format);
}

SIMD_API uint8_t* SimdImageLoadFromFile(const char* path, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format)
{
    SIMD_EMPTY();
    const static Simd::ImageLoadFromMemoryPtr imageLoadFromMemory = SIMD_FUNC4(ImageLoadFromMemory, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return ImageLoadFromFile(imageLoadFromMemory, path, stride, width, height, format);
}

SIMD_API void SimdInt16ToGray(const uint8_t * src, size_t width, size_t height, size_t srcStride, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Int16ToGray(src, width, height, srcStride, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::Int16ToGray(src, width, height, srcStride, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::Int16ToGray(src, width, height, srcStride, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::HA)
        Neon::Int16ToGray(src, width, height, srcStride, dst, dstStride);
    else
#endif
        Base::Int16ToGray(src, width, height, srcStride, dst, dstStride);
}

SIMD_API void SimdIntegral(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                      uint8_t * sum, size_t sumStride, uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride,
                      SimdPixelFormatType sumFormat, SimdPixelFormatType sqsumFormat)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Integral(src, srcStride, width, height, sum, sumStride, sqsum, sqsumStride, tilted, tiltedStride, sumFormat, sqsumFormat);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable)
        Avx2::Integral(src, srcStride, width, height, sum, sumStride, sqsum, sqsumStride, tilted, tiltedStride, sumFormat, sqsumFormat);
    else
#endif
        Base::Integral(src, srcStride, width, height, sum, sumStride, sqsum, sqsumStride, tilted, tiltedStride, sumFormat, sqsumFormat);
}

SIMD_API void SimdInterferenceIncrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t increment, int16_t saturation)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::HA)
        Sse41::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::HA)
        Neon::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
    else
#endif
        Base::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
}

SIMD_API void SimdInterferenceIncrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height, 
                                              uint8_t increment, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
    else
#endif
        Base::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
}

SIMD_API void SimdInterferenceDecrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t decrement, int16_t saturation)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::HA)
        Sse41::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::HA)
        Neon::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
    else
#endif
        Base::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
}

SIMD_API void SimdInterferenceDecrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height, 
                                              uint8_t decrement, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
    else
#endif
        Base::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
}

SIMD_API void SimdInterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if (Vmx::Enable && width >= Vmx::A)
        Vmx::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
    else
#endif
        Base::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
}

SIMD_API void SimdInterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride,
    size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::InterleaveBgr(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::InterleaveBgr(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::InterleaveBgr(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::InterleaveBgr(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
    else
#endif
        Base::InterleaveBgr(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
}

SIMD_API void SimdInterleaveBgra(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride,
    size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::InterleaveBgra(b, bStride, g, gStride, r, rStride, a, aStride, width, height, bgra, bgraStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::InterleaveBgra(b, bStride, g, gStride, r, rStride, a, aStride, width, height, bgra, bgraStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::InterleaveBgra(b, bStride, g, gStride, r, rStride, a, aStride, width, height, bgra, bgraStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::InterleaveBgra(b, bStride, g, gStride, r, rStride, a, aStride, width, height, bgra, bgraStride);
    else
#endif
        Base::InterleaveBgra(b, bStride, g, gStride, r, rStride, a, aStride, width, height, bgra, bgraStride);
}

SIMD_API void SimdLaplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::Laplace(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::Laplace(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::Laplace(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::Laplace(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::Laplace(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::Laplace(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdLaplaceAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdLaplaceAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::LaplaceAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::LaplaceAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::LaplaceAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::LaplaceAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::LaplaceAbsSum(src, stride, width, height, sum);
    else
#endif
        Base::LaplaceAbsSum(src, stride, width, height, sum);
}

SIMD_API void SimdLbpEstimate(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::A + 2)
        Avx512bw::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2)
        Avx2::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A + 2)
        Sse41::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 2)
        Vmx::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 2)
        Neon::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::LbpEstimate(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdMeanFilter3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && (width - 1)*channelCount >= Avx512bw::A)
        Avx512bw::MeanFilter3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && (width - 1)*channelCount >= Avx2::A)
        Avx2::MeanFilter3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && (width - 1)*channelCount >= Sse41::A)
        Sse41::MeanFilter3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if (Vmx::Enable && (width - 1)*channelCount >= Vmx::A)
        Vmx::MeanFilter3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && (width - 1)*channelCount >= Neon::A)
        Neon::MeanFilter3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::MeanFilter3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && (width - 1)*channelCount >= Avx512bw::A)
        Avx512bw::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && (width - 1)*channelCount >= Avx2::A)
        Avx2::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && (width - 1)*channelCount >= Sse41::A)
        Sse41::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && (width - 1)*channelCount >= Vmx::A)
        Vmx::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && (width - 1)*channelCount >= Neon::A)
        Neon::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterRhomb5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && (width - 2)*channelCount >= Avx512bw::A)
        Avx512bw::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && (width - 2)*channelCount >= Avx2::A)
        Avx2::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && (width - 2)*channelCount >= Sse41::A)
        Sse41::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && (width - 2)*channelCount >= Vmx::A)
        Vmx::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && (width - 2)*channelCount >= Neon::A)
        Neon::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && (width - 1)*channelCount >= Avx512bw::A)
        Avx512bw::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && (width - 1)*channelCount >= Avx2::A)
        Avx2::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && (width - 1)*channelCount >= Sse41::A)
        Sse41::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && (width - 1)*channelCount >= Vmx::A)
        Vmx::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && (width - 1)*channelCount >= Neon::A)
        Neon::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && (width - 2)*channelCount >= Avx512bw::A)
        Avx512bw::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && (width - 2)*channelCount >= Avx2::A)
        Avx2::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && (width - 2)*channelCount >= Sse41::A)
        Sse41::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && (width - 2)*channelCount >= Vmx::A)
        Vmx::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && (width - 2)*channelCount >= Neon::A)
        Neon::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdNeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride, int inversion)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F)
        Avx512bw::NeuralConvert(src, srcStride, width, height, dst, dstStride, inversion);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx::F)
        Avx2::NeuralConvert(src, srcStride, width, height, dst, dstStride, inversion);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::NeuralConvert(src, srcStride, width, height, dst, dstStride, inversion);
    else
#endif
#ifdef SIMD_VSX_ENABLE
    if (Vsx::Enable && width >= Vsx::A)
        Vsx::NeuralConvert(src, srcStride, width, height, dst, dstStride, inversion);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::NeuralConvert(src, srcStride, width, height, dst, dstStride, inversion);
    else
#endif
        Base::NeuralConvert(src, srcStride, width, height, dst, dstStride, inversion);
}

SIMD_API void SimdNeuralProductSum(const float * a, const float * b, size_t size, float * sum)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralProductSumPtr) (const float * a, const float * b, size_t size, float * sum);
    const static SimdNeuralProductSumPtr simdNeuralProductSum = SIMD_FUNC6(NeuralProductSum, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_VSX_FUNC, SIMD_NEON_FUNC);

    simdNeuralProductSum(a, b, size, sum);
}

SIMD_API void SimdNeuralAddValue(const float * value, float * dst, size_t size)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralAddValuePtr) (const float * value, float * dst, size_t size);
    const static SimdNeuralAddValuePtr simdNeuralAddValue = SIMD_FUNC4(NeuralAddValue, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralAddValue(value, dst, size);
}

SIMD_API void SimdNeuralAddVector(const float * src, size_t size, float * dst)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralAddVectorPtr) (const float * src, size_t size, float * dst);
    const static SimdNeuralAddVectorPtr simdNeuralAddVector = SIMD_FUNC4(NeuralAddVector, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralAddVector(src, size, dst);
}

SIMD_API void SimdNeuralAddVectorMultipliedByValue(const float * src, size_t size, const float * value, float * dst)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralAddVectorMultipliedByValuePtr) (const float * src, size_t size, const float * value, float * dst);
    const static SimdNeuralAddVectorMultipliedByValuePtr simdNeuralAddVectorMultipliedByValue = SIMD_FUNC5(NeuralAddVectorMultipliedByValue, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralAddVectorMultipliedByValue(src, size, value, dst);
}

SIMD_API void SimdNeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralRoughSigmoidPtr) (const float * src, size_t size, const float * slope, float * dst);
    const static SimdNeuralRoughSigmoidPtr simdNeuralRoughSigmoid = SIMD_FUNC5(NeuralRoughSigmoid, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_VSX_FUNC, SIMD_NEON_FUNC);

    simdNeuralRoughSigmoid(src, size, slope, dst);
}

SIMD_API void SimdNeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralRoughSigmoid2Ptr) (const float * src, size_t size, const float * slope, float * dst);
    const static SimdNeuralRoughSigmoid2Ptr simdNeuralRoughSigmoid2 = SIMD_FUNC5(NeuralRoughSigmoid2, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralRoughSigmoid2(src, size, slope, dst);
}

SIMD_API void SimdNeuralDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralDerivativeSigmoidPtr) (const float * src, size_t size, const float * slope, float * dst);
    const static SimdNeuralDerivativeSigmoidPtr simdNeuralDerivativeSigmoid = SIMD_FUNC4(NeuralDerivativeSigmoid, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralDerivativeSigmoid(src, size, slope, dst);
}

SIMD_API void SimdNeuralRoughTanh(const float * src, size_t size, const float * slope, float * dst)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralRoughTanhPtr) (const float * src, size_t size, const float * slope, float * dst);
    const static SimdNeuralRoughTanhPtr simdNeuralRoughTanh = SIMD_FUNC4(NeuralRoughTanh, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralRoughTanh(src, size, slope, dst);
}

SIMD_API void SimdNeuralDerivativeTanh(const float * src, size_t size, const float * slope, float * dst)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralDerivativeTanhPtr) (const float * src, size_t size, const float * slope, float * dst);
    const static SimdNeuralDerivativeTanhPtr simdNeuralDerivativeTanh = SIMD_FUNC4(NeuralDerivativeTanh, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralDerivativeTanh(src, size, slope, dst);
}

SIMD_API void SimdNeuralDerivativeRelu(const float * src, size_t size, const float * slope, float * dst)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralDerivativeReluPtr) (const float * src, size_t size, const float * slope, float * dst);
    const static SimdNeuralDerivativeReluPtr simdNeuralDerivativeRelu = SIMD_FUNC4(NeuralDerivativeRelu, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralDerivativeRelu(src, size, slope, dst);
}

SIMD_API void SimdNeuralPow(const float * src, size_t size, const float * exponent, float * dst)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralPowPtr) (const float * src, size_t size, const float * exponent, float * dst);
    const static SimdNeuralPowPtr simdNeuralPow = SIMD_FUNC4(NeuralPow, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralPow(src, size, exponent, dst);
}

SIMD_API void SimdNeuralUpdateWeights(const float * x, size_t size, const float * a, const float * b, float * d, float * w)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralUpdateWeightsPtr) (const float * x, size_t size, const float * a, const float * b, float * d, float * w);
    const static SimdNeuralUpdateWeightsPtr simdNeuralUpdateWeights = SIMD_FUNC4(NeuralUpdateWeights, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralUpdateWeights(x, size, a, b, d, w);
}

SIMD_API void SimdNeuralAdaptiveGradientUpdate(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralAdaptiveGradientUpdatePtr) (const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight);
    const static SimdNeuralAdaptiveGradientUpdatePtr simdNeuralAdaptiveGradientUpdate = SIMD_FUNC4(NeuralAdaptiveGradientUpdate, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralAdaptiveGradientUpdate(delta, size, batch, alpha, epsilon, gradient, weight);
}

SIMD_API void SimdNeuralAddConvolution2x2Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F)
        Avx512bw::NeuralAddConvolution2x2Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::NeuralAddConvolution2x2Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && width >= Avx::F)
        Avx::NeuralAddConvolution2x2Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F)
        Sse41::NeuralAddConvolution2x2Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F)
        Neon::NeuralAddConvolution2x2Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
        Base::NeuralAddConvolution2x2Forward(src, srcStride, width, height, weights, dst, dstStride);
}

SIMD_API void SimdNeuralAddConvolution3x3Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F)
        Avx512bw::NeuralAddConvolution3x3Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::NeuralAddConvolution3x3Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && width >= Avx::F)
        Avx::NeuralAddConvolution3x3Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F)
        Sse41::NeuralAddConvolution3x3Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F)
        Neon::NeuralAddConvolution3x3Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
        Base::NeuralAddConvolution3x3Forward(src, srcStride, width, height, weights, dst, dstStride);
}

SIMD_API void SimdNeuralAddConvolution4x4Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F)
        Avx512bw::NeuralAddConvolution4x4Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::NeuralAddConvolution4x4Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && width >= Avx::F)
        Avx::NeuralAddConvolution4x4Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F)
        Sse41::NeuralAddConvolution4x4Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F)
        Neon::NeuralAddConvolution4x4Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
        Base::NeuralAddConvolution4x4Forward(src, srcStride, width, height, weights, dst, dstStride);
}

SIMD_API void SimdNeuralAddConvolution5x5Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F)
        Avx512bw::NeuralAddConvolution5x5Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::NeuralAddConvolution5x5Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && width >= Avx::F)
        Avx::NeuralAddConvolution5x5Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F)
        Sse41::NeuralAddConvolution5x5Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F)
        Neon::NeuralAddConvolution5x5Forward(src, srcStride, width, height, weights, dst, dstStride);
    else
#endif
        Base::NeuralAddConvolution5x5Forward(src, srcStride, width, height, weights, dst, dstStride);
}

SIMD_API void SimdNeuralAddConvolution2x2Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralAddConvolution2x2BackwardPtr) (const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);
    const static SimdNeuralAddConvolution2x2BackwardPtr simdNeuralAddConvolution2x2Backward = SIMD_FUNC5(NeuralAddConvolution2x2Backward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralAddConvolution2x2Backward(src, srcStride, width, height, weights, dst, dstStride);
}

SIMD_API void SimdNeuralAddConvolution3x3Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralAddConvolution3x3BackwardPtr) (const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);
    const static SimdNeuralAddConvolution3x3BackwardPtr simdNeuralAddConvolution3x3Backward = SIMD_FUNC5(NeuralAddConvolution3x3Backward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralAddConvolution3x3Backward(src, srcStride, width, height, weights, dst, dstStride);
}

SIMD_API void SimdNeuralAddConvolution4x4Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralAddConvolution4x4BackwardPtr) (const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);
    const static SimdNeuralAddConvolution4x4BackwardPtr simdNeuralAddConvolution4x4Backward = SIMD_FUNC5(NeuralAddConvolution4x4Backward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralAddConvolution4x4Backward(src, srcStride, width, height, weights, dst, dstStride);
}

SIMD_API void SimdNeuralAddConvolution5x5Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralAddConvolution5x5BackwardPtr) (const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);
    const static SimdNeuralAddConvolution5x5BackwardPtr simdNeuralAddConvolution5x5Backward = SIMD_FUNC5(NeuralAddConvolution5x5Backward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralAddConvolution5x5Backward(src, srcStride, width, height, weights, dst, dstStride);
}

SIMD_API void SimdNeuralAddConvolution2x2Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F)
        Avx512bw::NeuralAddConvolution2x2Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::NeuralAddConvolution2x2Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && width >= Avx::F)
        Avx::NeuralAddConvolution2x2Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F)
        Sse41::NeuralAddConvolution2x2Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F)
        Neon::NeuralAddConvolution2x2Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
        Base::NeuralAddConvolution2x2Sum(src, srcStride, dst, dstStride, width, height, sums);
}

SIMD_API void SimdNeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F)
        Avx512bw::NeuralAddConvolution3x3Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::NeuralAddConvolution3x3Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && width >= Avx::F)
        Avx::NeuralAddConvolution3x3Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F)
        Sse41::NeuralAddConvolution3x3Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F)
        Neon::NeuralAddConvolution3x3Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
        Base::NeuralAddConvolution3x3Sum(src, srcStride, dst, dstStride, width, height, sums);
}

SIMD_API void SimdNeuralAddConvolution4x4Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F)
        Avx512bw::NeuralAddConvolution4x4Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::NeuralAddConvolution4x4Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && width >= Avx::F)
        Avx::NeuralAddConvolution4x4Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F)
        Sse41::NeuralAddConvolution4x4Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F)
        Neon::NeuralAddConvolution4x4Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
        Base::NeuralAddConvolution4x4Sum(src, srcStride, dst, dstStride, width, height, sums);
}

SIMD_API void SimdNeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::F)
        Avx512bw::NeuralAddConvolution5x5Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::F)
        Avx2::NeuralAddConvolution5x5Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && width >= Avx::F)
        Avx::NeuralAddConvolution5x5Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::F)
        Sse41::NeuralAddConvolution5x5Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::F)
        Neon::NeuralAddConvolution5x5Sum(src, srcStride, dst, dstStride, width, height, sums);
    else
#endif
        Base::NeuralAddConvolution5x5Sum(src, srcStride, dst, dstStride, width, height, sums);
}

SIMD_API void SimdNeuralPooling1x1Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::F)
        Avx512bw::NeuralPooling1x1Max3x3(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width > Avx2::F)
        Avx2::NeuralPooling1x1Max3x3(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width > Sse41::F)
        Sse41::NeuralPooling1x1Max3x3(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::F)
        Neon::NeuralPooling1x1Max3x3(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::NeuralPooling1x1Max3x3(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdNeuralPooling2x2Max2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::DF)
        Avx512bw::NeuralPooling2x2Max2x2(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if (Avx::Enable && width >= Avx::DF)
        Avx::NeuralPooling2x2Max2x2(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::DF)
        Sse41::NeuralPooling2x2Max2x2(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DF)
        Neon::NeuralPooling2x2Max2x2(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::NeuralPooling2x2Max2x2(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdNeuralPooling2x2Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::DF)
        Avx512bw::NeuralPooling2x2Max3x3(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width > Avx2::DF)
        Avx2::NeuralPooling2x2Max3x3(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width > Sse41::DF)
        Sse41::NeuralPooling2x2Max3x3(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::DF)
        Neon::NeuralPooling2x2Max3x3(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::NeuralPooling2x2Max3x3(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdNeuralConvolutionForward(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, 
    const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, 
    void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add)
{
    SIMD_EMPTY();
    typedef void(*SimdNeuralConvolutionForwardPtr) (const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, 
        const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, 
        void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add);
    const static SimdNeuralConvolutionForwardPtr simdNeuralConvolutionForward = SIMD_FUNC5(NeuralConvolutionForward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdNeuralConvolutionForward(src, srcWidth, srcHeight, srcDepth, weight, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, buffer, size, dst, dstWidth, dstHeight, dstDepth, add);
}

SIMD_API void SimdOperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
               size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width*channelCount >= Sse41::A)
        Sse41::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width*channelCount >= Vmx::A)
        Vmx::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width*channelCount >= Neon::A)
        Neon::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif
        Base::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
}

SIMD_API void SimdOperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
                                    size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::HA)
        Sse41::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::HA)
        Neon::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
    else
#endif
        Base::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
}

SIMD_API void SimdVectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif
        Base::VectorProduct(vertical, horizontal, dst, stride, width, height);
}

SIMD_API void SimdReduceColor2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
    uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ReduceColor2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceColor2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && srcWidth >= Sse41::DA)
        Sse41::ReduceColor2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && srcWidth >= Neon::DA)
        Neon::ReduceColor2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
        Base::ReduceColor2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
}

SIMD_API void SimdReduceGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                   uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && srcWidth >= Sse41::DA)
        Sse41::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth >= Vmx::DA)
        Vmx::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && srcWidth >= Neon::DA)
        Neon::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
        Base::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
}

SIMD_API void SimdReduceGray3x3(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                   uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && srcWidth >= Avx512bw::DA)
        Avx512bw::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && srcWidth >= Sse41::A)
        Sse41::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth >= Vmx::DA)
        Vmx::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && srcWidth >= Neon::DA)
        Neon::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
        Base::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
}

SIMD_API void SimdReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                   uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && srcWidth > Avx512bw::DA)
        Avx512bw::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth > Avx2::DA)
        Avx2::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && srcWidth > Sse41::A)
        Sse41::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth > Vmx::DA)
        Vmx::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && srcWidth > Neon::DA)
        Neon::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
        Base::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
}

SIMD_API void SimdReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                   uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && srcWidth >= Avx512bw::DA)
        Avx512bw::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && srcWidth >= Sse41::A)
        Sse41::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth >= Vmx::DA)
        Vmx::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && srcWidth >= Neon::DA)
        Neon::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
        Base::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
}

SIMD_API void SimdReorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Reorder16bit(src, size, dst);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && size >= Avx2::A)
        Avx2::Reorder16bit(src, size, dst);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && size >= Sse41::A)
        Sse41::Reorder16bit(src, size, dst);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && size >= Vmx::A)
        Vmx::Reorder16bit(src, size, dst);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && size >= Neon::A)
        Neon::Reorder16bit(src, size, dst);
    else
#endif
        Base::Reorder16bit(src, size, dst);
}

SIMD_API void SimdReorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Reorder32bit(src, size, dst);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && size >= Avx2::A)
        Avx2::Reorder32bit(src, size, dst);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && size >= Sse41::A)
        Sse41::Reorder32bit(src, size, dst);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && size >= Vmx::A)
        Vmx::Reorder32bit(src, size, dst);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && size >= Neon::A)
        Neon::Reorder32bit(src, size, dst);
    else
#endif
        Base::Reorder32bit(src, size, dst);
}

SIMD_API void SimdReorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Reorder64bit(src, size, dst);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && size >= Avx2::A)
        Avx2::Reorder64bit(src, size, dst);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && size >= Sse41::A)
        Sse41::Reorder64bit(src, size, dst);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && size >= Vmx::A)
        Vmx::Reorder64bit(src, size, dst);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && size >= Neon::A)
        Neon::Reorder64bit(src, size, dst);
    else
#endif
        Base::Reorder64bit(src, size, dst);
}

SIMD_API void SimdResizeBilinear(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
    uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && dstWidth >= Avx512bw::A)
        Avx512bw::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && dstWidth >= Avx2::A)
        Avx2::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && dstWidth >= Sse41::A)
        Sse41::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && dstWidth >= Vmx::A)
        Vmx::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && dstWidth >= Neon::A)
        Neon::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
        Base::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
}

SIMD_API void * SimdResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
{
    SIMD_EMPTY();
    typedef void*(*SimdResizerInitPtr) (size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    const static SimdResizerInitPtr simdResizerInit = SIMD_FUNC5(ResizerInit, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return simdResizerInit(srcX, srcY, dstX, dstY, channels, type, method);
}

SIMD_API void SimdResizerRun(const void * resizer, const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
    ((Resizer*)resizer)->Run(src, srcStride, dst, dstStride);
}

SIMD_API void SimdRgbToBgra(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::RgbToBgra(rgb, width, height, rgbStride, bgra, bgraStride, alpha);
    else
#endif
#if defined(SIMD_AVX2_ENABLE) && !defined(SIMD_CLANG_AVX2_BGR_TO_BGRA_ERROR)
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::RgbToBgra(rgb, width, height, rgbStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::RgbToBgra(rgb, width, height, rgbStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::RgbToBgra(rgb, width, height, rgbStride, bgra, bgraStride, alpha);
    else
#endif
        Base::RgbToBgra(rgb, width, height, rgbStride, bgra, bgraStride, alpha);
}

SIMD_API void SimdRgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::RgbToGray(rgb, width, height, rgbStride, gray, grayStride);
    else
#endif
#if defined(SIMD_AVX2_ENABLE) && !defined(SIMD_CLANG_AVX2_BGR_TO_BGRA_ERROR)
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::RgbToGray(rgb, width, height, rgbStride, gray, grayStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::RgbToGray(rgb, width, height, rgbStride, gray, grayStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::RgbToGray(rgb, width, height, rgbStride, gray, grayStride);
    else
#endif
        Base::RgbToGray(rgb, width, height, rgbStride, gray, grayStride);
}

SIMD_API void SimdRgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::RgbaToGray(rgba, width, height, rgbaStride, gray, grayStride);
    else
#endif
#if defined(SIMD_AVX2_ENABLE)
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::RgbaToGray(rgba, width, height, rgbaStride, gray, grayStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::RgbaToGray(rgba, width, height, rgbaStride, gray, grayStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::RgbaToGray(rgba, width, height, rgbaStride, gray, grayStride);
    else
#endif
        Base::RgbaToGray(rgba, width, height, rgbaStride, gray, grayStride);
}

SIMD_API void SimdSegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
    else
#endif
        Base::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
}

SIMD_API void SimdSegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::SegmentationFillSingleHoles(mask, stride, width, height, index);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A + 2)
        Avx2::SegmentationFillSingleHoles(mask, stride, width, height, index);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A + 2)
        Sse41::SegmentationFillSingleHoles(mask, stride, width, height, index);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A + 2)
        Vmx::SegmentationFillSingleHoles(mask, stride, width, height, index);
    else
#endif
#ifdef SIMD_NEON_ENABLE
        if (Neon::Enable && width > Neon::A + 2)
            Neon::SegmentationFillSingleHoles(mask, stride, width, height, index);
        else
#endif
        Base::SegmentationFillSingleHoles(mask, stride, width, height, index);
}

SIMD_API void SimdSegmentationPropagate2x2(const uint8_t * parent, size_t parentStride, size_t width, size_t height, 
                                           uint8_t * child, size_t childStride, const uint8_t * difference, size_t differenceStride, 
                                           uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
        difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 1)
        Avx2::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
        difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A + 1)
        Sse41::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
        difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 1)
        Vmx::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
        difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A + 1)
        Neon::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
            difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
    else
#endif
        Base::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
        difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
}

SIMD_API void SimdSegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
                                           ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::SegmentationShrinkRegion(mask, stride, width, height, index, left, top, right, bottom);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A && *right - *left >= (ptrdiff_t)Avx2::A)
        Avx2::SegmentationShrinkRegion(mask, stride, width, height, index, left, top, right, bottom);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A && *right - *left >= (ptrdiff_t)Sse41::A)
        Sse41::SegmentationShrinkRegion(mask, stride, width, height, index, left, top, right, bottom);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A && *right - *left >= (ptrdiff_t)Vmx::A)
        Vmx::SegmentationShrinkRegion(mask, stride, width, height, index, left, top, right, bottom);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A && *right - *left >= (ptrdiff_t)Neon::A)
        Neon::SegmentationShrinkRegion(mask, stride, width, height, index, left, top, right, bottom);
    else
#endif
        Base::SegmentationShrinkRegion(mask, stride, width, height, index, left, top, right, bottom);
}

SIMD_API void SimdShiftBilinear(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
    const uint8_t * bkg, size_t bkgStride, const double * shiftX, const double * shiftY,
    size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable)
        Avx2::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable)
        Sse41::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable)
        Vmx::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable)
        Neon::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif
        Base::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
}

SIMD_API void SimdSobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::SobelDx(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDx(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::SobelDx(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDx(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::SobelDx(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::SobelDx(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdSobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdSobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::SobelDxAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDxAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::SobelDxAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDxAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::SobelDxAbsSum(src, stride, width, height, sum);
    else
#endif
        Base::SobelDxAbsSum(src, stride, width, height, sum);
}

SIMD_API void SimdSobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::SobelDy(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDy(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::SobelDy(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDy(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::SobelDy(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::SobelDy(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdSobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdSobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::SobelDyAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDyAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::SobelDyAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDyAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::SobelDyAbsSum(src, stride, width, height, sum);
    else
#endif
        Base::SobelDyAbsSum(src, stride, width, height, sum);
}

SIMD_API void SimdContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::ContourMetrics(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::ContourMetrics(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::ContourMetrics(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::ContourMetrics(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::ContourMetrics(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::ContourMetrics(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                       const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
    else
#endif
        Base::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
}

SIMD_API void SimdContourAnchors(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width > Avx512bw::A)
        Avx512bw::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width > Sse41::A)
        Sse41::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width > Neon::A)
        Neon::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
    else
#endif
        Base::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
}

SIMD_API void SimdSquaredDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
                          size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
        Base::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
}

SIMD_API void SimdSquaredDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
                          const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
        Base::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
}

SIMD_API void SimdSquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
{
    SIMD_EMPTY();
    typedef void (* SimdSquaredDifferenceSum32fPtr) (const float * a, const float * b, size_t size, float * sum);
    const static SimdSquaredDifferenceSum32fPtr simdSquaredDifferenceSum32f = SIMD_FUNC5(SquaredDifferenceSum32f, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_VSX_FUNC, SIMD_NEON_FUNC);

    simdSquaredDifferenceSum32f(a, b, size, sum);
}

SIMD_API void SimdSquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum)
{
    SIMD_EMPTY();
    typedef void (* SimdSquaredDifferenceKahanSum32fPtr) (const float * a, const float * b, size_t size, float * sum);
    const static SimdSquaredDifferenceKahanSum32fPtr simdSquaredDifferenceKahanSum32f = SIMD_FUNC5(SquaredDifferenceKahanSum32f, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_VSX_FUNC, SIMD_NEON_FUNC);

    simdSquaredDifferenceKahanSum32f(a, b, size, sum);
}

SIMD_API void SimdGetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height,
                  uint8_t * min, uint8_t * max, uint8_t * average)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif
        Base::GetStatistic(src, stride, width, height, min, max, average);
}

SIMD_API void SimdGetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
                uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
{
    SIMD_EMPTY();
    const bool simd = width < SHRT_MAX && height < SHRT_MAX;
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::A && simd)
        Avx512bw::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A && simd)
        Vmx::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A && simd)
        Neon::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif
        Base::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
}

SIMD_API void SimdGetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
    uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::GetObjectMoments(src, srcStride, width, height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::GetObjectMoments(src, srcStride, width, height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::GetObjectMoments(src, srcStride, width, height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::GetObjectMoments(src, srcStride, width, height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
    else
#endif
        Base::GetObjectMoments(src, srcStride, width, height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
}

SIMD_API void SimdGetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::GetRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::GetRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::GetRowSums(src, stride, width, height, sums);
    else
#endif
        Base::GetRowSums(src, stride, width, height, sums);
}

SIMD_API void SimdGetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::GetColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::GetColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::GetColSums(src, stride, width, height, sums);
    else
#endif
        Base::GetColSums(src, stride, width, height, sums);
}

SIMD_API void SimdGetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif
        Base::GetAbsDyRowSums(src, stride, width, height, sums);
}

SIMD_API void SimdGetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif
        Base::GetAbsDxColSums(src, stride, width, height, sums);
}

SIMD_API void SimdValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ValueSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ValueSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::ValueSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::ValueSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::ValueSum(src, stride, width, height, sum);
    else
#endif
        Base::ValueSum(src, stride, width, height, sum);
}

SIMD_API void SimdSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::SquareSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SquareSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::SquareSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::SquareSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::SquareSum(src, stride, width, height, sum);
    else
#endif
        Base::SquareSum(src, stride, width, height, sum);
}

SIMD_API void SimdValueSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * valueSum, uint64_t * squareSum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ValueSquareSum(src, stride, width, height, valueSum, squareSum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ValueSquareSum(src, stride, width, height, valueSum, squareSum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::ValueSquareSum(src, stride, width, height, valueSum, squareSum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::ValueSquareSum(src, stride, width, height, valueSum, squareSum);
    else
#endif
        Base::ValueSquareSum(src, stride, width, height, valueSum, squareSum);
}

SIMD_API void SimdValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::ValueSquareSums(src, stride, width, height, channels, valueSums, squareSums);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::ValueSquareSums(src, stride, width, height, channels, valueSums, squareSums);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::ValueSquareSums(src, stride, width, height, channels, valueSums, squareSums);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::ValueSquareSums(src, stride, width, height, channels, valueSums, squareSums);
    else
#endif
        Base::ValueSquareSums(src, stride, width, height, channels, valueSums, squareSums);
}

SIMD_API void SimdCorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::CorrelationSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::CorrelationSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::CorrelationSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::CorrelationSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::CorrelationSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
        Base::CorrelationSum(a, aStride, b, bStride, width, height, sum);
}

SIMD_API void SimdStretchGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                    uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::A)
        Avx2::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && srcWidth >= Sse41::A)
        Sse41::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth >= Vmx::A)
        Vmx::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && srcWidth >= Neon::A)
        Neon::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
        Base::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
}

SIMD_API void SimdSvmSumLinear(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum)
{
    SIMD_EMPTY();
    typedef void(*SimdSvmSumLinearPtr) (const float* x, const float* svs, const float* weights, size_t length, size_t count, float* sum);
    const static SimdSvmSumLinearPtr simdSvmSumLinear = SIMD_FUNC5(SvmSumLinear, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_VSX_FUNC, SIMD_NEON_FUNC);

    simdSvmSumLinear(x, svs, weights, length, count, sum);
}

SIMD_API void SimdSynetAddBias(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetAddBiasPtr) (const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);
    const static SimdSynetAddBiasPtr simdSynetAddBias = SIMD_FUNC4(SynetAddBias, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetAddBias(bias, channels, spatial, dst, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetAdd8i(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
    uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetAdd8iPtr) (const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
        uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);
    const static SimdSynetAdd8iPtr simdSynetAdd8i = SIMD_FUNC3(SynetAdd8i, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC);

    simdSynetAdd8i(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, format, compatibility);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetConvert32fTo8u(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t* dst, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetConvert32fTo8uPtr) (const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t* dst, SimdSynetCompatibilityType compatibility);
    const static SimdSynetConvert32fTo8uPtr simdSynetConvert32fTo8u = SIMD_FUNC4(SynetConvert32fTo8u, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetConvert32fTo8u(src, batch, channels, height, width, format, scale, shift, dst, compatibility);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetConvert8uTo32f(const uint8_t* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, float* dst, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetConvert8uTo32fPtr) (const uint8_t* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, float* dst, SimdSynetCompatibilityType compatibility);
    const static SimdSynetConvert8uTo32fPtr simdSynetConvert8uTo32f = SIMD_FUNC3(SynetConvert8uTo32f, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC);//, SIMD_NEON_FUNC);

    simdSynetConvert8uTo32f(src, batch, channels, height, width, format, scale, shift, dst, compatibility);
#else
    assert(0);
#endif
}

SIMD_API void * SimdSynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * params, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void* (*SimdSynetConvolution32fInitPtr) (size_t batch, const SimdConvolutionParameters * params, SimdSynetCompatibilityType compatibility);
    const static SimdSynetConvolution32fInitPtr simdSynetConvolution32fInit = SIMD_FUNC7(SynetConvolution32fInit, SIMD_AMX_FUNC, SIMD_AVX512BF16_FUNC, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return simdSynetConvolution32fInit(batch, params, compatibility);
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetConvolution32fExternalBufferSize(const void * context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetConvolution32f*)context)->ExternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetConvolution32fInternalBufferSize(const void * context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetConvolution32f*)context)->InternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API const char* SimdSynetConvolution32fInfo(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetConvolution32f*)context)->Info();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API void SimdSynetConvolution32fSetParams(void * context, const float * weight, SimdBool * internal, const float * bias, const float * params)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    ((SynetConvolution32f*)context)->SetParams(weight, internal, bias, params);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetConvolution32fForward(void * context, const float * src, float * buf, float * dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    SynetConvolution32f * c = (SynetConvolution32f*)context;
    SIMD_PERF_EXT(c);
    c->Forward(src, buf, dst);
#else
    assert(0);
#endif
}

SIMD_API void* SimdSynetConvolution8iInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void* (*SimdSynetConvolution8iInitPtr) (size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    const static SimdSynetConvolution8iInitPtr simdSynetConvolution8iInit = SIMD_FUNC5(SynetConvolution8iInit, SIMD_AVX512VNNI_FUNC, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);
    
    return simdSynetConvolution8iInit(batch, conv, compatibility);
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetConvolution8iExternalBufferSize(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetConvolution8i*)context)->ExternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetConvolution8iInternalBufferSize(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetConvolution8i*)context)->InternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API const char* SimdSynetConvolution8iInfo(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetConvolution8i*)context)->Info();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API void SimdSynetConvolution8iSetParams(void* context, const float* weight, const float* bias, const float* params, const float* const* stats)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    ((SynetConvolution8i*)context)->SetParams(weight, bias, params, stats);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetConvolution8iForward(void* context, const uint8_t* src, uint8_t* buf, uint8_t* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    SynetConvolution8i* c = (SynetConvolution8i*)context;
    SIMD_PERF_EXT(c);
    c->Forward(src, buf, dst);
#else
    assert(0);
#endif
}

SIMD_API void * SimdSynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * params, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void* (*SimdSynetDeconvolution32fInitPtr) (size_t batch, const SimdConvolutionParameters * params, SimdSynetCompatibilityType compatibility);
    const static SimdSynetDeconvolution32fInitPtr simdSynetDeconvolution32fInit = SIMD_FUNC5(SynetDeconvolution32fInit, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);
    
    return simdSynetDeconvolution32fInit(batch, params, compatibility);
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetDeconvolution32fExternalBufferSize(const void * context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetDeconvolution32f*)context)->ExternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetDeconvolution32fInternalBufferSize(const void * context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetDeconvolution32f*)context)->InternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API const char* SimdSynetDeconvolution32fInfo(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetDeconvolution32f*)context)->Info();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API void SimdSynetDeconvolution32fSetParams(void * context, const float * weight, SimdBool * internal, const float * bias, const float * params)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    ((SynetDeconvolution32f*)context)->SetParams(weight, internal, bias, params);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetDeconvolution32fForward(void * context, const float * src, float * buf, float * dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    SynetDeconvolution32f * d = (SynetDeconvolution32f*)context;
    SIMD_PERF_EXT(d);
    d->Forward(src, buf, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetEltwiseLayerForwardPtr) (float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst);
    const static SimdSynetEltwiseLayerForwardPtr simdSynetEltwiseLayerForward = SIMD_FUNC5(SynetEltwiseLayerForward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetEltwiseLayerForward(src, weight, count, size, type, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetElu32fPtr) (const float * src, size_t size, const float * alpha, float * dst);
    const static SimdSynetElu32fPtr simdSynetElu32f = SIMD_FUNC4(SynetElu32f, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetElu32f(src, size, alpha, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetFusedLayerForward0Ptr) (const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);
    const static SimdSynetFusedLayerForward0Ptr simdSynetFusedLayerForward0 = SIMD_FUNC4(SynetFusedLayerForward0, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetFusedLayerForward0(src, bias, scale, channels, spatial, dst, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetFusedLayerForward1Ptr) (const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);
    const static SimdSynetFusedLayerForward1Ptr simdSynetFusedLayerForward1 = SIMD_FUNC4(SynetFusedLayerForward1, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetFusedLayerForward1(src, bias0, scale1, bias1, channels, spatial, dst, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetFusedLayerForward2Ptr) (const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst, SimdTensorFormatType format);
    const static SimdSynetFusedLayerForward2Ptr simdSynetFusedLayerForward2 = SIMD_FUNC4(SynetFusedLayerForward2, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetFusedLayerForward2(src, scale, bias, channels, spatial, slope, dst, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetFusedLayerForward3(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetFusedLayerForward3Ptr) (const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);
    const static SimdSynetFusedLayerForward3Ptr simdSynetFusedLayerForward3 = SIMD_FUNC4(SynetFusedLayerForward3, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetFusedLayerForward3(src, bias, scale, channels, spatial, dst, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetFusedLayerForward4Ptr) (const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);
    const static SimdSynetFusedLayerForward4Ptr simdSynetFusedLayerForward4 = SIMD_FUNC4(SynetFusedLayerForward4, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetFusedLayerForward4(src, bias0, scale1, bias1, channels, spatial, dst, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetFusedLayerForward8Ptr) (const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);
    const static SimdSynetFusedLayerForward8Ptr simdSynetFusedLayerForward8 = SIMD_FUNC4(SynetFusedLayerForward8, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetFusedLayerForward8(src0, src1, src2, channels, spatial, dst, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetFusedLayerForward9(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetFusedLayerForward9Ptr) (const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format);
    const static SimdSynetFusedLayerForward9Ptr simdSynetFusedLayerForward9 = SIMD_FUNC4(SynetFusedLayerForward9, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetFusedLayerForward9(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetHardSigmoid32fPtr) (const float* src, size_t size, const float* scale, const float* shift, float* dst);
    const static SimdSynetHardSigmoid32fPtr simdSynetHardSigmoid32f = SIMD_FUNC4(SynetHardSigmoid32f, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetHardSigmoid32f(src, size, scale, shift, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetHswish32fPtr) (const float * src, size_t size, const float * shift, const float * scale, float * dst);
    const static SimdSynetHswish32fPtr simdSynetHswish32f = SIMD_FUNC4(SynetHswish32f, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetHswish32f(src, size, shift, scale, dst);
#else
assert(0);
#endif
}

SIMD_API void* SimdSynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void* (*SimdSynetInnerProduct32fInitPtr) (size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);
    const static SimdSynetInnerProduct32fInitPtr simdSynetInnerProduct32fInit = SIMD_FUNC5(SynetInnerProduct32fInit, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return simdSynetInnerProduct32fInit(batch, input, output, transpose, activation);
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetInnerProduct32fInternalBufferSize(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetInnerProduct32f*)context)->InternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API void SimdSynetInnerProduct32fSetParams(void* context, const float* weight, SimdBool* internal, const float* bias, const float* params)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    ((SynetInnerProduct32f*)context)->SetParams(weight, internal, bias, params);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetInnerProduct32fForward(void* context, const float* src, float* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    SynetInnerProduct32f* c = (SynetInnerProduct32f*)context;
    SIMD_PERF_EXT(c);
    c->Forward(src, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetInnerProductLayerForwardPtr) (const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst);
    const static SimdSynetInnerProductLayerForwardPtr simdSynetInnerProductLayerForward = SIMD_FUNC5(SynetInnerProductLayerForward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetInnerProductLayerForward(src, weight, bias, count, size, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetInnerProduct8iPtr) (size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst, SimdSynetCompatibilityType compatibility);
    const static SimdSynetInnerProduct8iPtr simdSynetInnerProduct8i = SIMD_FUNC3(SynetInnerProduct8i, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC);

    simdSynetInnerProduct8i(M, N, K, src, weight, dst, compatibility);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetLrnLayerCrossChannels(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetLrnLayerCrossChannelsPtr) (const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format);
    const static SimdSynetLrnLayerCrossChannelsPtr simdSynetLrnLayerCrossChannels = SIMD_FUNC4(SynetLrnLayerCrossChannels, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetLrnLayerCrossChannels(src, half, channels, spatial, k, dst, format);
#else
    assert(0);
#endif
}

SIMD_API void * SimdSynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void* (*SimdSynetMergedConvolution32fInitPtr) (size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add, SimdSynetCompatibilityType compatibility);
    const static SimdSynetMergedConvolution32fInitPtr simdSynetMergedConvolution32fInit = SIMD_FUNC7(SynetMergedConvolution32fInit, SIMD_AMX_FUNC, SIMD_AVX512BF16_FUNC, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    return simdSynetMergedConvolution32fInit(batch, convs, count, add, compatibility);
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetMergedConvolution32fExternalBufferSize(const void * context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetMergedConvolution32f*)context)->ExternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetMergedConvolution32fInternalBufferSize(const void * context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetMergedConvolution32f*)context)->InternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API const char* SimdSynetMergedConvolution32fInfo(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetMergedConvolution32f*)context)->Info();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API void SimdSynetMergedConvolution32fSetParams(void * context, const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    ((SynetMergedConvolution32f*)context)->SetParams(weight, internal, bias, params);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetMergedConvolution32fForward(void * context, const float * src, float * buf, float * dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    SynetMergedConvolution32f * c = (SynetMergedConvolution32f*)context;
    SIMD_PERF_EXT(c);
    c->Forward(src, buf, dst);
#else
    assert(0);
#endif
}

SIMD_API void* SimdSynetMergedConvolution8iInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void* (*SimdSynetMergedConvolution8iInitPtr) (size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdSynetCompatibilityType compatibility);
    const static SimdSynetMergedConvolution8iInitPtr simdSynetMergedConvolution8iInit = SIMD_FUNC4(SynetMergedConvolution8iInit, SIMD_AVX512VNNI_FUNC, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC);

    return simdSynetMergedConvolution8iInit(batch, convs, count, compatibility);
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetMergedConvolution8iExternalBufferSize(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetMergedConvolution8i*)context)->ExternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetMergedConvolution8iInternalBufferSize(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetMergedConvolution8i*)context)->InternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API const char* SimdSynetMergedConvolution8iInfo(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((SynetMergedConvolution8i*)context)->Info();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API void SimdSynetMergedConvolution8iSetParams(void* context, const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params, const float* const* stats)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    ((SynetMergedConvolution8i*)context)->SetParams(weight, internal, bias, params, stats);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetMergedConvolution8iForward(void* context, const uint8_t* src, uint8_t* buf, uint8_t* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    SynetMergedConvolution8i* c = (SynetMergedConvolution8i*)context;
    SIMD_PERF_EXT(c);
    c->Forward(src, buf, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetMish32fPtr) (const float* src, size_t size, const float* threshold, float* dst);
    const static SimdSynetMish32fPtr simdSynetMish32f = SIMD_FUNC4(SynetMish32f, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetMish32f(src, size, threshold, dst);
#else
    assert(0);
#endif
}

void SimdSynetPoolingAverage(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
    size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdBool excludePad, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetPoolingAveragePtr) (const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdBool exludePad, SimdTensorFormatType format);
    const static SimdSynetPoolingAveragePtr simdSynetPoolingAverage = SIMD_FUNC4(SynetPoolingAverage, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetPoolingAverage(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, excludePad, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetPoolingMax32f(const float* src, size_t srcC, size_t srcH, size_t srcW,
    size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX,
    size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetPoolingMax32fPtr) (const float* src, size_t srcC, size_t srcH, size_t srcW,
        size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX,
        size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format);
    const static SimdSynetPoolingMax32fPtr simdSynetPoolingMax32f = SIMD_FUNC4(SynetPoolingMax32f, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetPoolingMax32f(src, srcC, srcH, srcW, kernelC, kernelY, kernelX, strideC, strideY, strideX, padC, padY, padX, dst, dstC, dstH, dstW, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetPoolingMax8u(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
    size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetPoolingMax8uPtr) (const uint8_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format);
    const static SimdSynetPoolingMax8uPtr simdSynetPoolingMax8u = SIMD_FUNC4(SynetPoolingMax8u, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetPoolingMax8u(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetPreluLayerForward(const float * src, const float * slope, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetPreluLayerForwardPtr) (const float * src, const float * slope, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);
    const static SimdSynetPreluLayerForwardPtr simdSynetPreluLayerForward = SIMD_FUNC4(SynetPreluLayerForward, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetPreluLayerForward(src, slope, channels, spatial, dst, format);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetRelu32fPtr) (const float* src, size_t size, const float* slope, float* dst);
    const static SimdSynetRelu32fPtr simdSynetRelu32f = SIMD_FUNC4(SynetRelu32f, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetRelu32f(src, size, slope, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetReorderImage(size_t batch, size_t channels, size_t spatial, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetReorderImagePtr) (size_t batch, size_t channels, size_t spatial, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat);
    const static SimdSynetReorderImagePtr simdSynetReorderImage = SIMD_FUNC4(SynetReorderImage, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetReorderImage(batch, channels, spatial, src, srcFormat, dst, dstFormat);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetReorderFilter(size_t output, size_t input, size_t kernel, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetReorderFilterPtr) (size_t output, size_t input, size_t kernel, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat);
    const static SimdSynetReorderFilterPtr simdSynetReorderFilter = SIMD_FUNC4(SynetReorderFilter, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetReorderFilter(output, input, kernel, src, srcFormat, dst, dstFormat);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetRestrictRange32fPtr) (const float * src, size_t size, const float * lower, const float * upper, float * dst);
    const static SimdSynetRestrictRange32fPtr simdSynetRestrictRange32f = SIMD_FUNC4(SynetRestrictRange32f, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetRestrictRange32f(src, size, lower, upper, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetScaleLayerForwardPtr) (const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);
    const static SimdSynetScaleLayerForwardPtr simdSynetScaleLayerForward = SIMD_FUNC5(SynetScaleLayerForward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
#else
    assert(0);
#endif
}

SIMD_API void* SimdSynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void* (*SimdSynetScale8iInitPtr) (size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);
    const static SimdSynetScale8iInitPtr simdSynetScale8iInit = SIMD_FUNC3(SynetScale8iInit, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC);

    return simdSynetScale8iInit(batch, channels, spatial, srcType, dstType, format, compatibility);
#else
    assert(0);
    return 0;
#endif
}

SIMD_API size_t SimdSynetScale8iInternalBufferSize(const void* context)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return ((Base::SynetScale8i*)context)->InternalBufferSize();
#else
    assert(0);
    return 0;
#endif
}

SIMD_API void SimdSynetScale8iSetParams(void* context, const float* weight, const float* bias, const float* const* stats)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    ((Base::SynetScale8i*)context)->SetParams(weight, bias, stats);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetScale8iForward(void* context, const uint8_t* src, uint8_t* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    ((Base::SynetScale8i*)context)->Forward(src, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetSetInput(const uint8_t * src, size_t width, size_t height, size_t stride, SimdPixelFormatType srcFormat,
    const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType dstFormat)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::A)
        Avx512bw::SynetSetInput(src, width, height, stride, srcFormat, lower, upper, dst, channels, dstFormat);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::SynetSetInput(src, width, height, stride, srcFormat, lower, upper, dst, channels, dstFormat);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::SynetSetInput(src, width, height, stride, srcFormat, lower, upper, dst, channels, dstFormat);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::SynetSetInput(src, width, height, stride, srcFormat, lower, upper, dst, channels, dstFormat);
    else
#endif
        Base::SynetSetInput(src, width, height, stride, srcFormat, lower, upper, dst, channels, dstFormat);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetShuffleLayerForward(const float* src0, const float* src1, size_t channels0, size_t channels1, size_t spatial, float* dst0, float* dst1, SimdTensorFormatType format, int type)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetShuffleLayerForwardPtr) (const float* src0, const float* src1, size_t channels0, size_t channels1, size_t spatial, float* dst0, float* dst1, SimdTensorFormatType format, int type);
    const static SimdSynetShuffleLayerForwardPtr simdSynetShuffleLayerForward = SIMD_FUNC4(SynetShuffleLayerForward, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetShuffleLayerForward(src0, src1, channels0, channels1, spatial, dst0, dst1, format, type);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetSigmoid32fPtr) (const float* src, size_t size, const float* slope, float* dst);
    const static SimdSynetSigmoid32fPtr simdSynetSigmoid32f = SIMD_FUNC4(SynetSigmoid32f, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetSigmoid32f(src, size, slope, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetSoftmaxLayerForwardPtr) (const float * src, size_t outer, size_t count, size_t inner, float * dst);
    const static SimdSynetSoftmaxLayerForwardPtr simdSynetSoftmaxLayerForward = SIMD_FUNC4(SynetSoftmaxLayerForward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetSoftmaxLayerForward(src, outer, count, inner, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetSoftplus32fPtr) (const float* src, size_t size, const float* beta, const float* threshold, float* dst);
    const static SimdSynetSoftplus32fPtr simdSynetSoftplus32f = SIMD_FUNC4(SynetSoftplus32f, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetSoftplus32f(src, size, beta, threshold, dst);
#else
    assert(0);
#endif
}

SIMD_API SimdTensorFormatType SimdSynetSpecifyTensorFormat(SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return Base::SynetSpecifyTensorFormat(format);
#else
    assert(0);
    return SimdTensorFormatUnknown;
#endif
}

SIMD_API void SimdSynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetSwish32fPtr) (const float* src, size_t size, const float* slope, float* dst);
    const static SimdSynetSwish32fPtr simdSynetSwish32f = SIMD_FUNC4(SynetSwish32f, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetSwish32f(src, size, slope, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdSynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetTanh32fPtr) (const float* src, size_t size, const float* slope, float* dst);
    const static SimdSynetTanh32fPtr simdSynetTanh32f = SIMD_FUNC4(SynetTanh32f, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetTanh32f(src, size, slope, dst);
#else
    assert(0);
#endif
}


SIMD_API size_t SimdSynetTensorAlignment(SimdTensorFormatType format)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    return Base::SynetTensorAlignment(format);
#else
    assert(0);
    return 0;
#endif
}

SIMD_API void SimdSynetUnaryOperation32fLayerForward(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    typedef void(*SimdSynetUnaryOperation32fLayerForwardPtr) (const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst);
    const static SimdSynetUnaryOperation32fLayerForwardPtr simdSynetUnaryOperation32fLayerForward = SIMD_FUNC4(SynetUnaryOperation32fLayerForward, SIMD_AVX512BW_FUNC, SIMD_AVX2_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdSynetUnaryOperation32fLayerForward(src, size, type, dst);
#else
    assert(0);
#endif
}

SIMD_API void SimdTextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                     uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif
        Base::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
}

SIMD_API void SimdTextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                      uint8_t boost, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif
        Base::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
}

SIMD_API void SimdTextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                             const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif
        Base::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
}

SIMD_API void SimdTexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height,
    int32_t shift, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if (Vmx::Enable && width >= Vmx::A)
        Vmx::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    else
#endif
        Base::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
}

SIMD_API void SimdTransformImage(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, SimdTransformType transform, uint8_t * dst, size_t dstStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx2::A)
        Avx512bw::TransformImage(src, srcStride, width, height, pixelSize, transform, dst, dstStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::TransformImage(src, srcStride, width, height, pixelSize, transform, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::TransformImage(src, srcStride, width, height, pixelSize, transform, dst, dstStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::HA)
        Neon::TransformImage(src, srcStride, width, height, pixelSize, transform, dst, dstStride);
    else
#endif
        Base::TransformImage(src, srcStride, width, height, pixelSize, transform, dst, dstStride);
}

SIMD_API void SimdUyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Uyvy422ToBgr(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::DA)
        Avx2::Uyvy422ToBgr(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::DA)
        Sse41::Uyvy422ToBgr(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Uyvy422ToBgr(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
    else
#endif
        Base::Uyvy422ToBgr(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
}

SIMD_API void SimdUyvy422ToYuv420p(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Uyvy422ToYuv420p(uyvy, uyvyStride, width, height, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::DA)
        Avx2::Uyvy422ToYuv420p(uyvy, uyvyStride, width, height, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::DA)
        Sse41::Uyvy422ToYuv420p(uyvy, uyvyStride, width, height, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Uyvy422ToYuv420p(uyvy, uyvyStride, width, height, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::Uyvy422ToYuv420p(uyvy, uyvyStride, width, height, y, yStride, u, uStride, v, vStride);
}

typedef void(*SimdWinogradSetFilterPtr) (const float * src, size_t size, float * dst, SimdBool trans);
typedef void(*SimdWinogradSetInputPtr) (const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);
typedef void(*SimdWinogradSetOutputPtr) (const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

SIMD_API void SimdWinogradKernel1x3Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetFilterPtr simdWinogradKernel1x3Block1x4SetFilter = SIMD_FUNC4(WinogradKernel1x3Block1x4SetFilter, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel1x3Block1x4SetFilter(src, size, dst, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel1x3Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
    size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetInputPtr simdWinogradKernel1x3Block1x4SetInput = SIMD_FUNC4(WinogradKernel1x3Block1x4SetInput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel1x3Block1x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetOutputPtr simdWinogradKernel1x3Block1x4SetOutput = SIMD_FUNC4(WinogradKernel1x3Block1x4SetOutput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel1x3Block1x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel1x5Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetFilterPtr simdWinogradKernel1x5Block1x4SetFilter = SIMD_FUNC4(WinogradKernel1x5Block1x4SetFilter, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel1x5Block1x4SetFilter(src, size, dst, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel1x5Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
    size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetInputPtr simdWinogradKernel1x5Block1x4SetInput = SIMD_FUNC4(WinogradKernel1x5Block1x4SetInput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel1x5Block1x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel1x5Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetOutputPtr simdWinogradKernel1x5Block1x4SetOutput = SIMD_FUNC4(WinogradKernel1x5Block1x4SetOutput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel1x5Block1x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel2x2Block2x2SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetFilterPtr simdWinogradKernel2x2Block2x2SetFilter = SIMD_FUNC4(WinogradKernel2x2Block2x2SetFilter, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel2x2Block2x2SetFilter(src, size, dst, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel2x2Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
    size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetInputPtr simdWinogradKernel2x2Block2x2SetInput = SIMD_FUNC4(WinogradKernel2x2Block2x2SetInput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel2x2Block2x2SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel2x2Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetOutputPtr simdWinogradKernel2x2Block2x2SetOutput = SIMD_FUNC4(WinogradKernel2x2Block2x2SetOutput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel2x2Block2x2SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel2x2Block4x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetFilterPtr simdWinogradKernel2x2Block4x4SetFilter = SIMD_FUNC4(WinogradKernel2x2Block4x4SetFilter, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel2x2Block4x4SetFilter(src, size, dst, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel2x2Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
    size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetInputPtr simdWinogradKernel2x2Block4x4SetInput = SIMD_FUNC4(WinogradKernel2x2Block4x4SetInput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel2x2Block4x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel2x2Block4x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetOutputPtr simdWinogradKernel2x2Block4x4SetOutput = SIMD_FUNC4(WinogradKernel2x2Block4x4SetOutput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel2x2Block4x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel3x3Block2x2SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetFilterPtr simdWinogradKernel3x3Block2x2SetFilter = SIMD_FUNC4(WinogradKernel3x3Block2x2SetFilter, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel3x3Block2x2SetFilter(src, size, dst, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel3x3Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
    size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetInputPtr simdWinogradKernel3x3Block2x2SetInput = SIMD_FUNC4(WinogradKernel3x3Block2x2SetInput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel3x3Block2x2SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel3x3Block2x2SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetOutputPtr simdWinogradKernel3x3Block2x2SetOutput = SIMD_FUNC4(WinogradKernel3x3Block2x2SetOutput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel3x3Block2x2SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel3x3Block3x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetFilterPtr simdWinogradKernel3x3Block3x3SetFilter = SIMD_FUNC4(WinogradKernel3x3Block3x3SetFilter, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel3x3Block3x3SetFilter(src, size, dst, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel3x3Block3x3SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
    size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetInputPtr simdWinogradKernel3x3Block3x3SetInput = SIMD_FUNC4(WinogradKernel3x3Block3x3SetInput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC); 

    simdWinogradKernel3x3Block3x3SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel3x3Block3x3SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetOutputPtr simdWinogradKernel3x3Block3x3SetOutput = SIMD_FUNC4(WinogradKernel3x3Block3x3SetOutput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel3x3Block3x3SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel3x3Block4x4SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetFilterPtr simdWinogradKernel3x3Block4x4SetFilter = SIMD_FUNC4(WinogradKernel3x3Block4x4SetFilter, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel3x3Block4x4SetFilter(src, size, dst, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel3x3Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
    size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetInputPtr simdWinogradKernel3x3Block4x4SetInput = SIMD_FUNC4(WinogradKernel3x3Block4x4SetInput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel3x3Block4x4SetInput(src, srcChannels, srcHeight, srcWidth, padY, padX, padH, padW, dst, dstStride, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdWinogradKernel3x3Block4x4SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
{
    SIMD_EMPTY();
#if defined(SIMD_SYNET_ENABLE)
    const static SimdWinogradSetOutputPtr simdWinogradKernel3x3Block4x4SetOutput = SIMD_FUNC4(WinogradKernel3x3Block4x4SetOutput, SIMD_AVX512BW_FUNC, SIMD_AVX_FUNC, SIMD_SSE41_FUNC, SIMD_NEON_FUNC);

    simdWinogradKernel3x3Block4x4SetOutput(src, srcStride, dst, dstChannels, dstHeight, dstWidth, trans);
#else
    assert(0);
#endif
}

SIMD_API void SimdYuva420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
    const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuva420pToBgra(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuva420pToBgra(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuva420pToBgra(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuva420pToBgra(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride);
    else
#endif
        Base::Yuva420pToBgra(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride);
}

SIMD_API void SimdYuva444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
    const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, SimdYuvType yuvType)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuva444pToBgraV2(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride, yuvType);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::Yuva444pToBgraV2(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride, yuvType);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::Yuva444pToBgraV2(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride, yuvType);
    else
#endif
        Base::Yuva444pToBgraV2(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride, yuvType);
}

SIMD_API void SimdYuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                 size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
        Base::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
}

SIMD_API void SimdYuv422pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                 size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
        Base::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
}

SIMD_API void SimdYuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                               size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
        Base::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
}

SIMD_API void SimdYuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                  size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
        Base::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
}

SIMD_API void SimdYuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
    size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv420pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuv420pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuv420pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
    else
#endif
        Base::Yuv420pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
}

SIMD_API void SimdYuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                                size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
        Base::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
}

SIMD_API void SimdYuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                  size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
        Base::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
}

SIMD_API void SimdYuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
    size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv444pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::Yuv444pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::Yuv444pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
    else
#endif
        Base::Yuv444pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
}

SIMD_API void SimdYuv444pToHsl(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                               size_t width, size_t height, uint8_t * hsl, size_t hslStride)
{
    SIMD_EMPTY();
    Base::Yuv444pToHsl(y, yStride, u, uStride, v, vStride, width, height, hsl, hslStride);
}

SIMD_API void SimdYuv444pToHsv(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                               size_t width, size_t height, uint8_t * hsv, size_t hsvStride)
{
    SIMD_EMPTY();
    Base::Yuv444pToHsv(y, yStride, u, uStride, v, vStride, width, height, hsv, hsvStride);
}

SIMD_API void SimdYuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                 size_t width, size_t height, uint8_t * hue, size_t hueStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_VSX_ENABLE
    if(Vsx::Enable && width >= Vsx::DA)
        Vsx::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
        Base::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
}

SIMD_API void SimdYuv444pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                 size_t width, size_t height, uint8_t * hue, size_t hueStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable && width >= Sse41::A)
        Sse41::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_VSX_ENABLE
    if(Vsx::Enable && width >= Vsx::A)
        Vsx::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
        Base::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
}

SIMD_API void SimdYuv420pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
    size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv420pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuv420pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuv420pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
        Base::Yuv420pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
}

SIMD_API void SimdYuv422pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
    size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv422pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv422pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuv422pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuv422pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
        Base::Yuv422pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
}

SIMD_API void SimdYuv444pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
    size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable)
        Avx512bw::Yuv444pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::A)
        Sse41::Yuv444pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::A)
        Neon::Yuv444pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
    else
#endif
        Base::Yuv444pToRgb(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
}

SIMD_API void SimdYuv420pToUyvy422(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
    size_t width, size_t height, uint8_t* uyvy, size_t uyvyStride)
{
    SIMD_EMPTY();
#ifdef SIMD_AVX512BW_ENABLE
    if (Avx512bw::Enable && width >= Avx512bw::DA)
        Avx512bw::Yuv420pToUyvy422(y, yStride, u, uStride, v, vStride, width, height, uyvy, uyvyStride);
    else
#endif
#ifdef SIMD_AVX2_ENABLE
    if (Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420pToUyvy422(y, yStride, u, uStride, v, vStride, width, height, uyvy, uyvyStride);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if (Sse41::Enable && width >= Sse41::DA)
        Sse41::Yuv420pToUyvy422(y, yStride, u, uStride, v, vStride, width, height, uyvy, uyvyStride);
    else
#endif
#ifdef SIMD_NEON_ENABLE
    if (Neon::Enable && width >= Neon::DA)
        Neon::Yuv420pToUyvy422(y, yStride, u, uStride, v, vStride, width, height, uyvy, uyvyStride);
    else
#endif
        Base::Yuv420pToUyvy422(y, yStride, u, uStride, v, vStride, width, height, uyvy, uyvyStride);
}




