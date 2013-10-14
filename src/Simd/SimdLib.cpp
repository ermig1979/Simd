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

#include "Simd/SimdLib.h"

#include "Simd/SimdEnable.h"
#include "Simd/SimdVersion.h"
#include "Simd/SimdConst.h"

#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdAvx2.h"

#ifdef WIN32
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

SIMD_API const char * SimdVersion()
{
    return SIMD_VERSION;
}

using namespace Simd;

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








