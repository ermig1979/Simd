/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar,
*               2014-2015 Antonenka Mikhail.
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

#if defined(WIN32) && !defined(SIMD_STATIC)

#define SIMD_EXPORTS

#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD dwReasonForCall, LPVOID lpReserved)
{
    switch(dwReasonForCall)
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
#include "Simd/SimdEnable.h"
#include "Simd/SimdVersion.h"
#include "Simd/SimdConst.h"

#include "Simd/SimdBase.h"
#include "Simd/SimdSse1.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdSsse3.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdSse42.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdVmx.h"
#include "Simd/SimdVsx.h"

using namespace Simd;

SIMD_API const char * SimdVersion()
{
    return SIMD_VERSION;
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
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable)
        return Avx2::A;
    else
#endif
#ifdef SIMD_AVX_ENABLE
    if(Avx::Enable)
        return sizeof(__m256);
    else
#endif
#ifdef SIMD_SSE41_ENABLE
    if(Sse41::Enable)
        return Sse41::A;
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable)
        return Ssse3::A;
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable)
        return Sse2::A;
    else
#endif
#ifdef SIMD_SSE_ENABLE
    if(Sse::Enable)
        return sizeof(__m128);
    else
#endif
#ifdef SIMD_VSX_ENABLE
    if(Vsx::Enable)
        return Vsx::A;
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable)
        return Vmx::A;
    else
#endif
        return sizeof(void *);
}

SIMD_API uint32_t SimdCrc32c(const void * src, size_t size)
{
#ifdef SIMD_SSE42_ENABLE
    if(Sse42::Enable)
        return Sse42::Crc32c(src, size);
    else
#endif
        return Base::Crc32c(src, size);
}

SIMD_API void SimdAbsDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t * b, size_t bStride,
                                   size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
        Base::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
}

SIMD_API void SimdAbsDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
                                         const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
        Base::AbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
}

SIMD_API void SimdAbsDifferenceSums3x3(const uint8_t *current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
                                       size_t width, size_t height, uint64_t * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A + 2)
        Avx2::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width > Sse2::A + 2)
        Sse2::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A + 2)
        Vmx::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
    else
#endif
        Base::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
}

SIMD_API void SimdAbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
                                             const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A + 2)
        Avx2::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width > Sse2::A + 2)
        Sse2::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A + 2)
        Vmx::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
        Base::AbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
}

SIMD_API void SimdAbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                          uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Simd::Avx2::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdAddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                       const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
                                       uint16_t weight, uint8_t * difference, size_t differenceStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif
        Base::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
}

SIMD_API void SimdAlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
                   const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif
        Base::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
}

SIMD_API void SimdBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                          uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
        Base::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
}

SIMD_API void SimdBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                          uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
        Base::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
}

SIMD_API void SimdBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                           const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
                                           uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundIncrementCount(value, valueStride, width, height,
        loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundIncrementCount(value, valueStride, width, height,
        loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundIncrementCount(value, valueStride, width, height,
        loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif
        Base::BackgroundIncrementCount(value, valueStride, width, height,
        loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
}

SIMD_API void SimdBackgroundAdjustRange(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
                                        uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
                                        uint8_t * hiValue, size_t hiValueStride, uint8_t threshold)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
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
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif
        Base::BackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
}

SIMD_API void SimdBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                       uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif
        Base::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
}

SIMD_API void SimdBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                             uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    else
#endif
        Base::BackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
}

SIMD_API void SimdBackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                     uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif
        Base::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
}

SIMD_API void SimdBayerToBgr(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgr, size_t bgrStride)
{
    Base::BayerToBgr(bayer, width, height, bayerStride, bayerFormat, bgr, bgrStride);
}

SIMD_API void SimdBayerToBgra(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
    Base::BayerToBgra(bayer, width, height, bayerStride, bayerFormat, bgra, bgraStride, alpha);
}

SIMD_API void SimdBgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
{
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::BgraToBayer(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgraToBayer(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
    else
#endif
        Base::BgraToBayer(bgra, width, height, bgraStride, bayer, bayerStride, bayerFormat);
}

SIMD_API void SimdBgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride)
{
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
    else
#endif
        Base::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
}

SIMD_API void SimdBgraToGray(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *gray, size_t grayStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif
        Base::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
}

SIMD_API void SimdBgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::DA)
        Ssse3::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::DA)
        Sse2::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgraToYuv420p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::DA)
        Ssse3::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::DA)
        Sse2::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgraToYuv422p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgraToYuv444p(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
{
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::BgrToBayer(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgrToBayer(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
    else
#endif
        Base::BgrToBayer(bgr, width, height, bgrStride, bayer, bayerStride, bayerFormat);
}

SIMD_API void SimdBgrToBgra(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
    else
#endif
        Base::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
}

SIMD_API void SimdBgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
    const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::HA)
        Sse2::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
    else
#endif
        Base::Bgr48pToBgra32(blue, blueStride, width, height, green, greenStride, red, redStride, bgra, bgraStride, alpha);
}

SIMD_API void SimdBgrToGray(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t *gray, size_t grayStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif
        Base::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
}

SIMD_API void SimdBgrToHsl(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsl, size_t hslStride)
{
    Base::BgrToHsl(bgr, width, height, bgrStride, hsl, hslStride);
}

SIMD_API void SimdBgrToHsv(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsv, size_t hsvStride)
{
    Base::BgrToHsv(bgr, width, height, bgrStride, hsv, hsvStride);
}

SIMD_API void SimdBgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::DA)
        Ssse3::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgrToYuv420p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgrToYuv422p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::DA)
        Ssse3::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgrToYuv422p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
    else
#endif
        Base::BgrToYuv444p(bgr, width, height, bgrStride, y, yStride, u, uStride, v, vStride);
}

SIMD_API void SimdBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                  uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif
        Base::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
}

SIMD_API void SimdAveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                           uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative,
                           uint8_t * dst, size_t dstStride, SimdCompareType compareType)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif
        Base::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
}

SIMD_API void SimdConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height,
                                   uint8_t value, SimdCompareType compareType, uint32_t * count)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ConditionalCount8u(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::ConditionalCount8u(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::ConditionalCount8u(src, stride, width, height, value, compareType, count);
    else
#endif
        Base::ConditionalCount8u(src, stride, width, height, value, compareType, count);
}

SIMD_API void SimdConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height,
                                     int16_t value, SimdCompareType compareType, uint32_t * count)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::ConditionalCount16i(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::HA)
        Sse2::ConditionalCount16i(src, stride, width, height, value, compareType, count);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::ConditionalCount16i(src, stride, width, height, value, compareType, count);
    else
#endif
        Base::ConditionalCount16i(src, stride, width, height, value, compareType, count);
}

SIMD_API void SimdConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                 const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
        Base::ConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
}

SIMD_API void SimdConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                       const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
        Base::ConditionalSquareSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
}

SIMD_API void SimdConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                       const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 3)
        Avx2::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A + 3)
        Sse2::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 3)
        Vmx::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    else
#endif
        Base::ConditionalSquareGradientSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
}

SIMD_API void SimdConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
	if (Avx2::Enable && width >= Avx2::A)
		Avx2::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
	else
#endif
#ifdef SIMD_SSE2_ENABLE
	if (Sse2::Enable && width >= Sse2::A)
		Sse2::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
	else
#endif
#ifdef SIMD_VMX_ENABLE
	if (Vmx::Enable && width >= Vmx::A)
		Vmx::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
	else
#endif
		Base::ConditionalFill(src, srcStride, width, height, threshold, compareType, value, dst, dstStride);
}

SIMD_API void SimdCopy(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, uint8_t * dst, size_t dstStride)
{
    Base::Copy(src, srcStride, width, height, pixelSize, dst, dstStride);
}

SIMD_API void SimdCopyFrame(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize,
                           size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t * dst, size_t dstStride)
{
    Base::CopyFrame(src, srcStride, width, height, pixelSize, frameLeft, frameTop, frameRight, frameBottom, dst, dstStride);
}

SIMD_API void SimdDeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
                    uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif
        Base::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
}

SIMD_API void SimdEdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                 uint8_t * background, size_t backgroundStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif
        Base::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
}

SIMD_API void SimdEdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                 uint8_t * background, size_t backgroundStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif
        Base::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
}

SIMD_API void SimdEdgeBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                                  const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundIncrementCount(value, valueStride, width, height,
        backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundIncrementCount(value, valueStride, width, height,
        backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundIncrementCount(value, valueStride, width, height,
        backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif
        Base::EdgeBackgroundIncrementCount(value, valueStride, width, height,
        backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
}

SIMD_API void SimdEdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
                               uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height,
        backgroundValue, backgroundValueStride, threshold);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height,
        backgroundValue, backgroundValueStride, threshold);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height,
        backgroundValue, backgroundValueStride, threshold);
    else
#endif
        Base::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height,
        backgroundValue, backgroundValueStride, threshold);
}

SIMD_API void SimdEdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
                               uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height,
        backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height,
        backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height,
        backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif
        Base::EdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height,
        backgroundValue, backgroundValueStride, threshold, mask, maskStride);
}

SIMD_API void SimdEdgeBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                              uint8_t * background, size_t backgroundStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
    else
#endif
        Base::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
}

SIMD_API void SimdEdgeBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
                              uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif
        Base::EdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
}

SIMD_API void SimdFill(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uint8_t value)
{
    Base::Fill(dst, stride, width, height, pixelSize, value);
}

SIMD_API void SimdFillFrame(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize,
                           size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t value)
{
    Base::FillFrame(dst, stride, width, height, pixelSize, frameLeft, frameTop, frameRight, frameBottom, value);
}

SIMD_API void SimdFillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::FillBgr(dst, stride, width, height, blue, green, red);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::FillBgr(dst, stride, width, height, blue, green, red);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::FillBgr(dst, stride, width, height, blue, green, red);
    else
#endif
        Base::FillBgr(dst, stride, width, height, blue, green, red);
}

SIMD_API void SimdFillBgra(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif
        Base::FillBgra(dst, stride, width, height, blue, green, red, alpha);
}

SIMD_API void SimdGaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                     size_t channelCount, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width*channelCount >= Ssse3::A)
        Ssse3::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width*channelCount >= Vmx::A)
        Vmx::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdGrayToBgr(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgr, size_t bgrStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
    else
#endif
        Base::GrayToBgr(gray, width, height, grayStride, bgr, bgrStride);
}

SIMD_API void SimdGrayToBgra(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif
        Base::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
}

SIMD_API void SimdAbsSecondDerivativeHistogram(const uint8_t *src, size_t width, size_t height, size_t stride, size_t step, size_t indent, uint32_t * histogram)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2*indent)
        Avx2::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A + 2*indent)
        Sse2::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 2*indent)
        Vmx::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif
        Base::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
}

SIMD_API void SimdHistogram(const uint8_t *src, size_t width, size_t height, size_t stride, uint32_t * histogram)
{
    Base::Histogram(src, width, height, stride, histogram);
}

SIMD_API void SimdHistogramMasked(const uint8_t *src, size_t srcStride, size_t width, size_t height, 
                                  const uint8_t * mask, size_t maskStride, uint8_t index, uint32_t * histogram)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
    else
#endif
        Base::HistogramMasked(src, srcStride, width, height, mask, maskStride, index, histogram);
}

SIMD_API void SimdNormalizeHistogram(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
    Base::NormalizeHistogram(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdHogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height, 
                                         size_t cellX, size_t cellY, size_t quantization, float * histograms)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2)
        Avx2::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A + 2)
        Sse2::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
    else
#endif
#ifdef SIMD_VSX_ENABLE
    if(Vsx::Enable && width >= Vsx::A + 2)
        Vsx::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
    else
#endif
        Base::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
}

SIMD_API void SimdIntegral(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                      uint8_t * sum, size_t sumStride, uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride,
                      SimdPixelFormatType sumFormat, SimdPixelFormatType sqsumFormat)
{
    Base::Integral(src, srcStride, width, height, sum, sumStride, sqsum, sqsumStride, tilted, tiltedStride, sumFormat, sqsumFormat);
}

SIMD_API void SimdInterferenceIncrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t increment, int16_t saturation)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::HA)
        Sse2::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
    else
#endif
        Base::InterferenceIncrement(statistic, stride, width, height, increment, saturation);
}

SIMD_API void SimdInterferenceIncrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height, 
                                              uint8_t increment, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
    else
#endif
        Base::InterferenceIncrementMasked(statistic, statisticStride, width, height, increment, saturation, mask, maskStride, index);
}

SIMD_API void SimdInterferenceDecrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t decrement, int16_t saturation)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::HA)
        Sse2::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
    else
#endif
        Base::InterferenceDecrement(statistic, stride, width, height, decrement, saturation);
}

SIMD_API void SimdInterferenceDecrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height, 
                                              uint8_t decrement, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
    else
#endif
        Base::InterferenceDecrementMasked(statistic, statisticStride, width, height, decrement, saturation, mask, maskStride, index);
}

SIMD_API void SimdInterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride)
{
#ifdef SIMD_AVX2_ENABLE
	if (Avx2::Enable && width >= Avx2::A)
		Avx2::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
	else
#endif
#ifdef SIMD_SSE2_ENABLE
	if (Sse2::Enable && width >= Sse2::A)
		Sse2::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
	else
#endif
#ifdef SIMD_VMX_ENABLE
	if (Vmx::Enable && width >= Vmx::A)
		Vmx::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
	else
#endif
		Base::InterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride);
}

SIMD_API void SimdLaplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::Laplace(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::Laplace(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width > Sse2::A)
        Sse2::Laplace(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::Laplace(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::Laplace(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdLaplaceAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::LaplaceAbs(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdLaplaceAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::LaplaceAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::LaplaceAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::LaplaceAbsSum(src, stride, width, height, sum);
    else
#endif
        Base::LaplaceAbsSum(src, stride, width, height, sum);
}

SIMD_API void SimdLbpEstimate(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2)
        Avx2::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A + 2)
        Sse2::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 2)
        Vmx::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::LbpEstimate(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdMedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width*channelCount >= Vmx::A)
        Vmx::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterRhomb5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width*channelCount >= Vmx::A)
        Vmx::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width*channelCount >= Vmx::A)
        Vmx::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width*channelCount >= Vmx::A)
        Vmx::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif
        Base::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdOperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
               size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width*channelCount >= Vmx::A)
        Vmx::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif
        Base::OperationBinary8u(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
}

SIMD_API void SimdOperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
                                    size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::HA)
        Avx2::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::HA)
        Sse2::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::HA)
        Vmx::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
    else
#endif
        Base::OperationBinary16i(a, aStride, b, bStride, width, height, dst, dstStride, type);
}

SIMD_API void SimdVectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif
        Base::VectorProduct(vertical, horizontal, dst, stride, width, height);
}

SIMD_API void SimdReduceGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                   uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && srcWidth >= Ssse3::DA)
        Ssse3::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth >= Sse2::DA)
        Sse2::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth >= Vmx::DA)
        Vmx::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
        Base::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
}

SIMD_API void SimdReduceGray3x3(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                   uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth >= Sse2::A)
        Sse2::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth >= Vmx::DA)
        Vmx::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
        Base::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
}

SIMD_API void SimdReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                   uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth > Avx2::DA)
        Avx2::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && srcWidth > Ssse3::A)
        Ssse3::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth > Sse2::A)
        Sse2::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth >= Vmx::DA)
        Vmx::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
        Base::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
}

SIMD_API void SimdReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                   uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth >= Sse2::A)
        Sse2::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth >= Vmx::DA)
        Vmx::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif
        Base::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
}

SIMD_API void SimdReorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && size >= Avx2::A)
        Avx2::Reorder16bit(src, size, dst);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && size >= Ssse3::A)
        Ssse3::Reorder16bit(src, size, dst);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && size >= Sse2::A)
        Sse2::Reorder16bit(src, size, dst);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && size >= Vmx::A)
        Vmx::Reorder16bit(src, size, dst);
    else
#endif
        Base::Reorder16bit(src, size, dst);
}

SIMD_API void SimdReorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && size >= Avx2::A)
        Avx2::Reorder32bit(src, size, dst);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && size >= Ssse3::A)
        Ssse3::Reorder32bit(src, size, dst);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && size >= Sse2::A)
        Sse2::Reorder32bit(src, size, dst);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && size >= Vmx::A)
        Vmx::Reorder32bit(src, size, dst);
    else
#endif
        Base::Reorder32bit(src, size, dst);
}

SIMD_API void SimdReorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && size >= Avx2::A)
        Avx2::Reorder64bit(src, size, dst);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && size >= Ssse3::A)
        Ssse3::Reorder64bit(src, size, dst);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && size >= Sse2::A)
        Sse2::Reorder64bit(src, size, dst);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && size >= Vmx::A)
        Vmx::Reorder64bit(src, size, dst);
    else
#endif
        Base::Reorder64bit(src, size, dst);
}

SIMD_API void SimdResizeBilinear(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
    uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && dstWidth >= Avx2::A)
        Avx2::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && dstWidth >= Ssse3::A)
        Ssse3::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && dstWidth >= Sse2::A)
        Sse2::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && dstWidth >= Vmx::A)
        Vmx::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif
        Base::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
}

SIMD_API void SimdSegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
    else
#endif
        Base::SegmentationChangeIndex(mask, stride, width, height, oldIndex, newIndex);
}

SIMD_API void SimdSegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A + 2)
        Avx2::SegmentationFillSingleHoles(mask, stride, width, height, index);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width > Sse2::A + 2)
        Sse2::SegmentationFillSingleHoles(mask, stride, width, height, index);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A + 2)
        Vmx::SegmentationFillSingleHoles(mask, stride, width, height, index);
    else
#endif
        Base::SegmentationFillSingleHoles(mask, stride, width, height, index);
}

SIMD_API void SimdSegmentationPropagate2x2(const uint8_t * parent, size_t parentStride, size_t width, size_t height, 
                                           uint8_t * child, size_t childStride, const uint8_t * difference, size_t differenceStride, 
                                           uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 1)
        Avx2::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
        difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A + 1)
        Sse2::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
        difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A + 1)
        Vmx::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
        difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
    else
#endif
        Base::SegmentationPropagate2x2(parent, parentStride, width, height, child, childStride,
        difference, differenceStride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
}

SIMD_API void SimdSegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
                                           ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom)
{
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
        Base::SegmentationShrinkRegion(mask, stride, width, height, index, left, top, right, bottom);
}

SIMD_API void SimdShiftBilinear(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
    const uint8_t * bkg, size_t bkgStride, const double * shiftX, const double * shiftY,
    size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable)
        Avx2::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable)
        Sse2::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable)
        Vmx::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif
        Base::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
}

SIMD_API void SimdSobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDx(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::SobelDx(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width > Sse2::A)
        Sse2::SobelDx(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDx(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::SobelDx(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdSobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::SobelDxAbs(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdSobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDxAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::SobelDxAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDxAbsSum(src, stride, width, height, sum);
    else
#endif
        Base::SobelDxAbsSum(src, stride, width, height, sum);
}

SIMD_API void SimdSobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDy(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::SobelDy(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width > Sse2::A)
        Sse2::SobelDy(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDy(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::SobelDy(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdSobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::SobelDyAbs(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdSobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::SobelDyAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::SobelDyAbsSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::SobelDyAbsSum(src, stride, width, height, sum);
    else
#endif
        Base::SobelDyAbsSum(src, stride, width, height, sum);
}

SIMD_API void SimdContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::ContourMetrics(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::ContourMetrics(src, srcStride, width, height, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::ContourMetrics(src, srcStride, width, height, dst, dstStride);
    else
#endif
        Base::ContourMetrics(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                       const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width > Ssse3::A)
        Ssse3::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
    else
#endif
        Base::ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, dst, dstStride);
}

SIMD_API void SimdContourAnchors(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A)
        Avx2::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width > Sse2::A)
        Sse2::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width > Vmx::A)
        Vmx::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
    else
#endif
        Base::ContourAnchors(src, srcStride, width, height, step, threshold, dst, dstStride);
}

SIMD_API void SimdSquaredDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
                          size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
        Base::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
}

SIMD_API void SimdSquaredDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
                          const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
        Base::SquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
}

typedef void (* SimdSquaredDifferenceSum32fPtr) (const float * a, const float * b, size_t size, float * sum);
SimdSquaredDifferenceSum32fPtr simdSquaredDifferenceSum32f = SIMD_FUNC3(SquaredDifferenceSum32f, SIMD_AVX_FUNC, SIMD_SSE_FUNC, SIMD_VSX_FUNC);

SIMD_API void SimdSquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
{
    simdSquaredDifferenceSum32f(a, b, size, sum);
}

typedef void (* SimdSquaredDifferenceKahanSum32fPtr) (const float * a, const float * b, size_t size, float * sum);
SimdSquaredDifferenceKahanSum32fPtr simdSquaredDifferenceKahanSum32f = SIMD_FUNC3(SquaredDifferenceKahanSum32f, SIMD_AVX_FUNC, SIMD_SSE_FUNC, SIMD_VSX_FUNC);

SIMD_API void SimdSquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum)
{
    simdSquaredDifferenceKahanSum32f(a, b, size, sum);
}

SIMD_API void SimdGetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height,
                  uint8_t * min, uint8_t * max, uint8_t * average)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif
        Base::GetStatistic(src, stride, width, height, min, max, average);
}

SIMD_API void SimdGetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
                uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A && width < SHRT_MAX && height < SHRT_MAX)
        Avx2::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A && width < SHRT_MAX && height < SHRT_MAX)
        Sse2::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A && width < SHRT_MAX && height < SHRT_MAX)
        Vmx::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif
        Base::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
}

SIMD_API void SimdGetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetRowSums(src, stride, width, height, sums);
    else
#endif
        Base::GetRowSums(src, stride, width, height, sums);
}

SIMD_API void SimdGetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetColSums(src, stride, width, height, sums);
    else
#endif
        Base::GetColSums(src, stride, width, height, sums);
}

SIMD_API void SimdGetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif
        Base::GetAbsDyRowSums(src, stride, width, height, sums);
}

SIMD_API void SimdGetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif
        Base::GetAbsDxColSums(src, stride, width, height, sums);
}

SIMD_API void SimdValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ValueSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::ValueSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::ValueSum(src, stride, width, height, sum);
    else
#endif
        Base::ValueSum(src, stride, width, height, sum);
}

SIMD_API void SimdSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SquareSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::SquareSum(src, stride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::SquareSum(src, stride, width, height, sum);
    else
#endif
        Base::SquareSum(src, stride, width, height, sum);
}

SIMD_API void SimdCorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::CorrelationSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::CorrelationSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::CorrelationSum(a, aStride, b, bStride, width, height, sum);
    else
#endif
        Base::CorrelationSum(a, aStride, b, bStride, width, height, sum);
}

SIMD_API void SimdStretchGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                    uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::A)
        Avx2::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth >= Sse2::A)
        Sse2::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && srcWidth >= Vmx::A)
        Vmx::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif
        Base::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
}

SIMD_API void SimdSvmSumLinear(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum)
{
#ifdef SIMD_AVX_ENABLE
    if(Avx::Enable)
        Avx::SvmSumLinear(x, svs, weights, length, count, sum);
    else
#endif
#ifdef SIMD_SSE_ENABLE
    if(Sse::Enable)
        Sse::SvmSumLinear(x, svs, weights, length, count, sum);
    else
#endif
#ifdef SIMD_VSX_ENABLE
    if(Vsx::Enable)
        Vsx::SvmSumLinear(x, svs, weights, length, count, sum);
    else
#endif
        Base::SvmSumLinear(x, svs, weights, length, count, sum);
}

SIMD_API void SimdTextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                     uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif
        Base::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
}

SIMD_API void SimdTextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                      uint8_t boost, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif
        Base::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
}

SIMD_API void SimdTextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                             const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif
        Base::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
}

SIMD_API void SimdTexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height,
                                int shift, uint8_t * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    else
#endif
        Base::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
}

SIMD_API void SimdYuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                 size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::DA)
        Ssse3::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
        Base::Yuv420pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
}

SIMD_API void SimdYuv422pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                 size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::DA)
        Ssse3::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
        Base::Yuv422pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
}

SIMD_API void SimdYuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                               size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_SSSE3_ENABLE
    if(Ssse3::Enable && width >= Ssse3::A)
        Ssse3::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    else
#endif
        Base::Yuv444pToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
}

SIMD_API void SimdYuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                  size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::DA)
        Sse2::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
        Base::Yuv420pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
}

SIMD_API void SimdYuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                                size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::DA)
        Sse2::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::DA)
        Vmx::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
        Base::Yuv422pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
}

SIMD_API void SimdYuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                  size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
#ifdef SIMD_VMX_ENABLE
    if(Vmx::Enable && width >= Vmx::A)
        Vmx::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif
        Base::Yuv444pToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
}

SIMD_API void SimdYuv444pToHsl(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                               size_t width, size_t height, uint8_t * hsl, size_t hslStride)
{
    Base::Yuv444pToHsl(y, yStride, u, uStride, v, vStride, width, height, hsl, hslStride);
}

SIMD_API void SimdYuv444pToHsv(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                               size_t width, size_t height, uint8_t * hsv, size_t hsvStride)
{
    Base::Yuv444pToHsv(y, yStride, u, uStride, v, vStride, width, height, hsv, hsvStride);
}

SIMD_API void SimdYuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                 size_t width, size_t height, uint8_t * hue, size_t hueStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::DA)
        Sse2::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_VSX_ENABLE
    if(Vsx::Enable && width >= Vsx::DA)
        Vsx::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
        Base::Yuv420pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
}

SIMD_API void SimdYuv444pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                 size_t width, size_t height, uint8_t * hue, size_t hueStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
#ifdef SIMD_VSX_ENABLE
    if(Vsx::Enable && width >= Vsx::A)
        Vsx::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif
        Base::Yuv444pToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
}



