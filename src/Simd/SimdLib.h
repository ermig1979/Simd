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
#ifndef __SimdLib_h__
#define __SimdLib_h__

#include <stdlib.h>

#if defined(WIN32) && !defined(SIMD_STATIC)
#  ifdef SIMD_EXPORTS
#    define SIMD_API __declspec(dllexport)
#  else//SIMD_EXPORTS
#    define SIMD_API __declspec(dllimport)
#  endif//SIMD_EXPORTS
#else //WIN32
#    define SIMD_API
#endif//WIN32

#ifdef __cplusplus
extern "C" 
{
#endif//__cplusplus

    SIMD_API const char * SimdVersion();

    SIMD_API void SimdAbsDifferenceSum(const unsigned char *a, size_t aStride, const unsigned char * b, size_t bStride, 
        size_t width, size_t height, unsigned long long * sum);

    SIMD_API void SimdAbsDifferenceSumMasked(const unsigned char *a, size_t aStride, const unsigned char *b, size_t bStride, 
        const unsigned char *mask, size_t maskStride, unsigned char index, size_t width, size_t height, unsigned long long * sum);

    SIMD_API void SimdAbsGradientSaturatedSum(const unsigned char * src, size_t srcStride, size_t width, size_t height, 
        unsigned char * dst, size_t dstStride);

    SIMD_API void SimdAddFeatureDifference(const unsigned char * value, size_t valueStride, size_t width, size_t height, 
        const unsigned char * lo, size_t loStride, const unsigned char * hi, size_t hiStride,
        unsigned short weight, unsigned char * difference, size_t differenceStride);

    SIMD_API void SimdBackgroundGrowRangeSlow(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        unsigned char * lo, size_t loStride, unsigned char * hi, size_t hiStride);

    SIMD_API void SimdBackgroundGrowRangeFast(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        unsigned char * lo, size_t loStride, unsigned char * hi, size_t hiStride);

    SIMD_API void SimdBackgroundIncrementCount(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        const unsigned char * loValue, size_t loValueStride, const unsigned char * hiValue, size_t hiValueStride,
        unsigned char * loCount, size_t loCountStride, unsigned char * hiCount, size_t hiCountStride);

    SIMD_API void SimdBackgroundAdjustRange(unsigned char * loCount, size_t loCountStride, size_t width, size_t height, 
        unsigned char * loValue, size_t loValueStride, unsigned char * hiCount, size_t hiCountStride, 
        unsigned char * hiValue, size_t hiValueStride, unsigned char threshold);

    SIMD_API void SimdBackgroundAdjustRangeMasked(unsigned char * loCount, size_t loCountStride, size_t width, size_t height, 
        unsigned char * loValue, size_t loValueStride, unsigned char * hiCount, size_t hiCountStride, 
        unsigned char * hiValue, size_t hiValueStride, unsigned char threshold, const unsigned char * mask, size_t maskStride);

    SIMD_API void SimdBackgroundShiftRange(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        unsigned char * lo, size_t loStride, unsigned char * hi, size_t hiStride);

    SIMD_API void SimdBackgroundShiftRangeMasked(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        unsigned char * lo, size_t loStride, unsigned char * hi, size_t hiStride, const unsigned char * mask, size_t maskStride);

    SIMD_API void SimdBackgroundInitMask(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        unsigned char index, unsigned char value, unsigned char * dst, size_t dstStride);

    SIMD_API void SimdBgraToBgr(const unsigned char *bgra, size_t width, size_t height, size_t bgraStride, unsigned char *bgr, size_t bgrStride);

    SIMD_API void SimdBgraToGray(const unsigned char *bgra, size_t width, size_t height, size_t bgraStride, unsigned char *gray, size_t grayStride);

    SIMD_API void SimdBgrToBgra(const unsigned char *bgr, size_t width, size_t height, size_t bgrStride, unsigned char *bgra, size_t bgraStride, unsigned char alpha);

    SIMD_API void SimdBgrToGray(const unsigned char *bgr, size_t width, size_t height, size_t bgrStride, unsigned char *gray, size_t grayStride);

#ifdef __cplusplus 
}
#endif // __cplusplus

#endif//__SimdLib_h__
