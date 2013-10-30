/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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

#include "Simd/SimdEnable.h"
#include "Simd/SimdVersion.h"
#include "Simd/SimdConst.h"

#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdSse42.h"
#include "Simd/SimdAvx2.h"

using namespace Simd;

SIMD_API const char * SimdVersion()
{
    return SIMD_VERSION;
}

SIMD_API void SimdAbsDifferenceSum(const uchar *a, size_t aStride, const uchar * b, size_t bStride, 
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
        Base::AbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
}

SIMD_API void SimdAbsDifferenceSumMasked(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
                                         const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AbsDifferenceSum(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AbsDifferenceSum(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif
        Base::AbsDifferenceSum(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
}

SIMD_API void SimdAbsDifferenceSums3x3(const uchar *current, size_t currentStride, const uchar * background, size_t backgroundStride,
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
        Base::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
}

SIMD_API void SimdAbsDifferenceSums3x3Masked(const uchar *current, size_t currentStride, const uchar *background, size_t backgroundStride,
                                             const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width > Avx2::A + 2)
        Avx2::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width > Sse2::A + 2)
        Sse2::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
    else
#endif
        Base::AbsDifferenceSums3x3(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
}

SIMD_API void SimdAbsGradientSaturatedSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
                                          uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Simd::Avx2::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::AbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdAddFeatureDifference(const uchar * value, size_t valueStride, size_t width, size_t height, 
                                       const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride,
                                       ushort weight, uchar * difference, size_t differenceStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::AddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
}

SIMD_API void SimdAlphaBlending(const uchar *src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
                   const uchar *alpha, size_t alphaStride, uchar *dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif//SIMD_AVX2_ENABLE 
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE       
        Base::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
}

SIMD_API void SimdBackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
                                          uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::BackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
}

SIMD_API void SimdBackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
                                          uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::BackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
}

SIMD_API void SimdBackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
                                           const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
                                           uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundIncrementCount(value, valueStride, width, height,
        loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundIncrementCount(value, valueStride, width, height,
        loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::BackgroundIncrementCount(value, valueStride, width, height,
        loValue, loValueStride, hiValue, hiValueStride, loCount, loCountStride, hiCount, hiCountStride);
}

SIMD_API void SimdBackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
                                        uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
                                        uchar * hiValue, size_t hiValueStride, uchar threshold)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
        hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
        hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    else
#endif// SIMD_SSE2_ENABLE
        Base::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
        hiCount, hiCountStride, hiValue, hiValueStride, threshold);
}

SIMD_API void SimdBackgroundAdjustRangeMasked(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
                                              uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
                                              uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
        hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
        hiCount, hiCountStride,hiValue, hiValueStride, threshold, mask, maskStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::BackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride, 
        hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
}

SIMD_API void SimdBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
                                       uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
}

SIMD_API void SimdBackgroundShiftRangeMasked(const uchar * value, size_t valueStride, size_t width, size_t height,
                                             uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride, 
        mask, maskStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride, 
        mask, maskStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::BackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, 
        maskStride);
}

SIMD_API void SimdBackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
                                     uchar index, uchar value, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::BackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
}

SIMD_API void SimdBgraToBgr(const uchar *bgra, size_t width, size_t height, size_t bgraStride, uchar *bgr, size_t bgrStride)
{
    Base::BgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
}

SIMD_API void SimdBgraToGray(const uchar *bgra, size_t width, size_t height, size_t bgraStride, uchar *gray, size_t grayStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif//SIMD_AVX2_ENABLE 
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    else
#endif//SIMD_SSE2_ENABLE       
        Base::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
}

SIMD_API void SimdBgrToBgra(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *bgra, size_t bgraStride, uchar alpha)
{
    Base::BgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
}

SIMD_API void SimdBgrToGray(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *gray, size_t grayStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif//SIMD_AVX2_ENABLE 
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    else
#endif//SIMD_SSE2_ENABLE       
        Base::BgrToGray(bgr, width, height, bgrStride, gray, grayStride);
}

SIMD_API void SimdBinarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
                  uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride, SimdCompareType compareType)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    else
#endif// SIMD_SSE2_ENABLE
        Base::Binarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
}

SIMD_API void SimdAveragingBinarization(const uchar * src, size_t srcStride, size_t width, size_t height,
                           uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, 
                           uchar * dst, size_t dstStride, SimdCompareType compareType)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    else
#endif// SIMD_SSE2_ENABLE
        Base::AveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
}

SIMD_API void SimdCopy(const uchar * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, uchar * dst, size_t dstStride)
{
    Base::Copy(src, srcStride, width, height, pixelSize, dst, dstStride);
}

SIMD_API void SimdCopyFrame(const uchar * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, 
                           size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uchar * dst, size_t dstStride)
{
    Base::CopyFrame(src, srcStride, width, height, pixelSize, frameLeft, frameTop, frameRight, frameBottom, dst, dstStride);
}

SIMD_API uint32_t SimdCrc32(const void * src, size_t size)
{
#ifdef SIMD_SSE42_ENABLE
    if(Sse42::Enable)
        return Sse42::Crc32(src, size);
    else
#endif//SIMD_SSE42_ENABLE
        return Base::Crc32(src, size);
}

SIMD_API void SimdDeinterleaveUv(const uchar * uv, size_t uvStride, size_t width, size_t height, 
                    uchar * u, size_t uStride, uchar * v, size_t vStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::DeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
}

SIMD_API void SimdEdgeBackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
                                 uchar * background, size_t backgroundStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::EdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
}

SIMD_API void SimdEdgeBackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
                                 uchar * background, size_t backgroundStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::EdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
}

SIMD_API void SimdEdgeBackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
                                  const uchar * backgroundValue, size_t backgroundValueStride, uchar * backgroundCount, size_t backgroundCountStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundIncrementCount(value, valueStride, width, height,
        backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundIncrementCount(value, valueStride, width, height,
        backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::EdgeBackgroundIncrementCount(value, valueStride, width, height,
        backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
}

SIMD_API void SimdEdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
                               uchar * backgroundValue, size_t backgroundValueStride, uchar threshold)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
        backgroundValue, backgroundValueStride, threshold);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
        backgroundValue, backgroundValueStride, threshold);
    else
#endif// SIMD_SSE2_ENABLE
        Base::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
        backgroundValue, backgroundValueStride, threshold);
}

SIMD_API void SimdEdgeBackgroundAdjustRangeMasked(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
                               uchar * backgroundValue, size_t backgroundValueStride, uchar threshold, const uchar * mask, size_t maskStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
        backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
        backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::EdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, 
        backgroundValue, backgroundValueStride, threshold, mask, maskStride);
}

SIMD_API void SimdEdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
                              uchar * background, size_t backgroundStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
}

SIMD_API void SimdEdgeBackgroundShiftRangeMasked(const uchar * value, size_t valueStride, size_t width, size_t height,
                              uchar * background, size_t backgroundStride, const uchar * mask, size_t maskStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    else
#endif// SIMD_SSE2_ENABLE
        Base::EdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
}

SIMD_API void SimdFill(uchar * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uchar value)
{
    Base::Fill(dst, stride, width, height, pixelSize, value);
}

SIMD_API void SimdFillFrame(uchar * dst, size_t stride, size_t width, size_t height, size_t pixelSize, 
                           size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uchar value)
{
    Base::FillFrame(dst, stride, width, height, pixelSize, frameLeft, frameTop, frameRight, frameBottom, value);
}

SIMD_API void SimdFillBgra(uchar * dst, size_t stride, size_t width, size_t height, uchar blue, uchar green, uchar red, uchar alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    else
#endif// SIMD_SSE2_ENABLE
        Base::FillBgra(dst, stride, width, height, blue, green, red, alpha);
}

SIMD_API void SimdGaussianBlur3x3(const uchar * src, size_t srcStride, size_t width, size_t height, 
                     size_t channelCount, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::GaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdGrayToBgra(const uchar *gray, size_t width, size_t height, size_t grayStride, uchar *bgra, size_t bgraStride, uchar alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif//SIMD_AVX2_ENABLE 
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    else
#endif//SIMD_SSE2_ENABLE       
        Base::GrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
}

SIMD_API void SimdAbsSecondDerivativeHistogram(const uchar *src, size_t width, size_t height, size_t stride, size_t step, size_t indent, uint * histogram)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2*indent)
        Avx2::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A + 2*indent)
        Sse2::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    else
#endif//SIMD_SSE2_ENABLE
        Base::AbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
}

SIMD_API void SimdHistogram(const uchar *src, size_t width, size_t height, size_t stride, uint * histogram)
{
    Base::Histogram(src, width, height, stride, histogram);
}

SIMD_API void SimdInterleaveBgra(uchar *bgra, size_t size, const int *blue, int bluePrecision, bool blueSigned, 
    const int *green, int greenPrecision, bool greenSigned, const int *red, int redPrecision, bool redSigned, uchar alpha)
{
    Base::InterleaveBgra(bgra, size, blue, bluePrecision, blueSigned, green, greenPrecision, greenSigned, red, redPrecision, redSigned, alpha);
}

SIMD_API void SimdInterleaveBgra(uchar *bgra, size_t size, const int *gray, int grayPrecision, bool graySigned, uchar alpha)
{
    Base::InterleaveBgra(bgra, size, gray, grayPrecision, graySigned, alpha);
}

SIMD_API void SimdLbpEstimate(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A + 2)
        Avx2::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A + 2)
        Sse2::LbpEstimate(src, srcStride, width, height, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::LbpEstimate(src, srcStride, width, height, dst, dstStride);
}

SIMD_API void SimdMedianFilterRhomb3x3(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::MedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterRhomb5x5(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::MedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterSquare3x3(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::MedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdMedianFilterSquare5x5(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::MedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
}

SIMD_API void SimdOperation(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
               size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride, SimdOperationType type)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width*channelCount >= Avx2::A)
        Avx2::Operation(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width*channelCount >= Sse2::A)
        Sse2::Operation(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    else
#endif// SIMD_SSE2_ENABLE
        Base::Operation(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
}

SIMD_API void SimdVectorProduct(const uchar * vertical, const uchar * horizontal, uchar * dst, size_t stride, size_t width, size_t height)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::VectorProduct(vertical, horizontal, dst, stride, width, height);
    else
#endif// SIMD_SSE2_ENABLE
        Base::VectorProduct(vertical, horizontal, dst, stride, width, height);
}

SIMD_API void SimdReduceGray2x2(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
                   uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth >= Sse2::DA)
        Sse2::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
}

SIMD_API void SimdReduceGray3x3(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
                   uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth >= Sse2::A)
        Sse2::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif//SIMD_SSE2_ENABLE
        Base::ReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
}

SIMD_API void SimdReduceGray4x4(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
                   uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth > Avx2::DA)
        Avx2::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth > Sse2::A)
        Sse2::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::ReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
}

SIMD_API void SimdReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
                   uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::DA)
        Avx2::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth >= Sse2::A)
        Sse2::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    else
#endif//SIMD_SSE2_ENABLE
        Base::ReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
}

SIMD_API void SimdResizeBilinear(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
    uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::A)
        Avx2::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth >= Sse2::A)
        Sse2::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    else
#endif//SIMD_SSE2_ENABLE
        Base::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
}

SIMD_API void SimdShiftBilinear(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
    const uchar * bkg, size_t bkgStride, double shiftX, double shiftY, 
    size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && shiftX + Avx2::A < cropRight - cropLeft)
        Avx2::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && shiftX + Sse2::A < cropRight - cropLeft)
        Sse2::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::ShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
        shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
}

SIMD_API void SimdSquaredDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
                          size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    else
#endif//SIMD_SSE2_ENABLE
        Base::SquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
}

SIMD_API void SimdSquaredDifferenceSumMasked(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
                          const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::SquaredDifferenceSum(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::SquaredDifferenceSum(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    else
#endif//SIMD_SSE2_ENABLE
        Base::SquaredDifferenceSum(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
}

SIMD_API void SimdGetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
                  uchar * min, uchar * max, uchar * average)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetStatistic(src, stride, width, height, min, max, average);
    else
#endif// SIMD_SSE2_ENABLE
        Base::GetStatistic(src, stride, width, height, min, max, average);
}

SIMD_API void SimdGetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
                uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A && width < SHRT_MAX && height < SHRT_MAX)
        Avx2::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A && width < SHRT_MAX && height < SHRT_MAX)
        Sse2::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);
    else
#endif// SIMD_SSE2_ENABLE
        Base::GetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);    
}

SIMD_API void SimdGetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetRowSums(src, stride, width, height, sums);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetRowSums(src, stride, width, height, sums);
    else
#endif// SIMD_SSE2_ENABLE
        Base::GetRowSums(src, stride, width, height, sums);
}

SIMD_API void SimdGetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetColSums(src, stride, width, height, sums);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetColSums(src, stride, width, height, sums);
    else
#endif// SIMD_SSE2_ENABLE
        Base::GetColSums(src, stride, width, height, sums);
}

SIMD_API void SimdGetAbsDyRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetAbsDyRowSums(src, stride, width, height, sums);
    else
#endif// SIMD_SSE2_ENABLE
        Base::GetAbsDyRowSums(src, stride, width, height, sums);
}

SIMD_API void SimdGetAbsDxColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::GetAbsDxColSums(src, stride, width, height, sums);
    else
#endif// SIMD_SSE2_ENABLE
        Base::GetAbsDxColSums(src, stride, width, height, sums);
}

SIMD_API void SimdConditionalCount(const uchar * src, size_t stride, size_t width, size_t height, 
                                   uchar value, SimdCompareType compareType, uint * count)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::ConditionalCount(src, stride, width, height, value, compareType, count);
    else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::ConditionalCount(src, stride, width, height, value, compareType, count);
    else
#endif// SIMD_SSE2_ENABLE
        Base::ConditionalCount(src, stride, width, height, value, compareType, count);
}

SIMD_API void SimdStretchGray2x2(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
                    uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && srcWidth >= Avx2::A)
        Avx2::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && srcWidth >= Sse2::A)
        Sse2::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::StretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
}

SIMD_API void SimdTextureBoostedSaturatedGradient(const uchar * src, size_t srcStride, size_t width, size_t height, 
                                     uchar saturation, uchar boost, uchar * dx, size_t dxStride, uchar * dy, size_t dyStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::TextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
}

SIMD_API void SimdTextureBoostedUv(const uchar * src, size_t srcStride, size_t width, size_t height, 
                      uchar boost, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::TextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
}

SIMD_API void SimdTextureGetDifferenceSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
                             const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride, int64_t * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    else
#endif//SIMD_SSE2_ENABLE
        Base::TextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
}

SIMD_API void SimdTexturePerformCompensation(const uchar * src, size_t srcStride, size_t width, size_t height, 
                                int shift, uchar * dst, size_t dstStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::TexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
}

SIMD_API void SimdYuv420ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
                 size_t width, size_t height, uchar * bgr, size_t bgrStride)
{
    Base::Yuv420ToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
}

SIMD_API void SimdYuv444ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
                 size_t width, size_t height, uchar * bgr, size_t bgrStride)
{
    Base::Yuv444ToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
}

SIMD_API void SimdYuvToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
               const int *y, const int *u, const int *v, int dx, int dy, int precision, uchar alpha)
{
    Base::YuvToBgra(bgra, width, height, stride, y, u, v, dx, dy, precision, alpha);
}

SIMD_API void SimdYuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
                  size_t width, size_t height, uchar * bgra, size_t bgraStride, uchar alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::DA)
        Sse2::Yuv420ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif//SIMD_SSE2_ENABLE
        Base::Yuv420ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
}

SIMD_API void SimdYuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
                  size_t width, size_t height, uchar * bgra, size_t bgraStride, uchar alpha)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::Yuv444ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    else
#endif//SIMD_SSE2_ENABLE
        Base::Yuv444ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
}

SIMD_API void SimdYuv420ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
                 size_t width, size_t height, uchar * hue, size_t hueStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::DA)
        Avx2::Yuv420ToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::DA)
        Sse2::Yuv420ToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::Yuv420ToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
}

SIMD_API void SimdYuv444ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
                 size_t width, size_t height, uchar * hue, size_t hueStride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Avx2::Enable && width >= Avx2::A)
        Avx2::Yuv444ToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Sse2::Enable && width >= Sse2::A)
        Sse2::Yuv444ToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    else
#endif//SIMD_SSE2_ENABLE
        Base::Yuv444ToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
}



