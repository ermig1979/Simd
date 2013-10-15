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

    typedef enum SimdCompareType
    {
        SimdCompareGreaterThen,
        SimdCompareLesserThen,
        SimdCompareEqualTo,
    } SimdCompareType;

    typedef enum SimdOperationType
    {
        SimdOperationAverage,
        SimdOperationAnd,
        SimdOperationMaximum,
        SimdOperationSaturatedSubtraction,
    } SimdOperationType;

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

    SIMD_API void SimdAlphaBlending(const unsigned char *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
        const unsigned char *alpha, size_t alphaStride, unsigned char *dst, size_t dstStride);

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

    SIMD_API void SimdBinarization(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        unsigned char value, unsigned char positive, unsigned char negative, unsigned char * dst, size_t dstStride, SimdCompareType compareType);

    SIMD_API void SimdAveragingBinarization(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        unsigned char value, size_t neighborhood, unsigned char threshold, unsigned char positive, unsigned char negative,
        unsigned char * dst, size_t dstStride, SimdCompareType compareType);

    SIMD_API void SimdCopy(const unsigned char * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, unsigned char * dst, size_t dstStride);

    SIMD_API void SimdCopyFrame(const unsigned char * src, size_t srcStride, size_t width, size_t height, size_t pixelSize,
        size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, unsigned char * dst, size_t dstStride);

    SIMD_API unsigned int SimdCrc32(const void * src, size_t size);

    SIMD_API void SimdDeinterleaveUv(const unsigned char * uv, size_t uvStride, size_t width, size_t height,
        unsigned char * u, size_t uStride, unsigned char * v, size_t vStride);

    SIMD_API void SimdEdgeBackgroundGrowRangeSlow(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        unsigned char * background, size_t backgroundStride);

    SIMD_API void SimdEdgeBackgroundGrowRangeFast(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        unsigned char * background, size_t backgroundStride);

    SIMD_API void SimdEdgeBackgroundIncrementCount(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        const unsigned char * backgroundValue, size_t backgroundValueStride, unsigned char * backgroundCount, size_t backgroundCountStride);

    SIMD_API void SimdEdgeBackgroundAdjustRange(unsigned char * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
        unsigned char * backgroundValue, size_t backgroundValueStride, unsigned char threshold);

    SIMD_API void SimdEdgeBackgroundAdjustRangeMasked(unsigned char * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
        unsigned char * backgroundValue, size_t backgroundValueStride, unsigned char threshold, const unsigned char * mask, size_t maskStride);

    SIMD_API void SimdEdgeBackgroundShiftRange(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        unsigned char * background, size_t backgroundStride);

    SIMD_API void SimdEdgeBackgroundShiftRangeMasked(const unsigned char * value, size_t valueStride, size_t width, size_t height,
        unsigned char * background, size_t backgroundStride, const unsigned char * mask, size_t maskStride);

    SIMD_API void SimdFill(unsigned char * dst, size_t stride, size_t width, size_t height, size_t pixelSize, unsigned char value);

    SIMD_API void SimdFillFrame(unsigned char * dst, size_t stride, size_t width, size_t height, size_t pixelSize,
        size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, unsigned char value);

    SIMD_API void SimdFillBgra(unsigned char * dst, size_t stride, size_t width, size_t height,
        unsigned char blue, unsigned char green, unsigned char red, unsigned char alpha);

    SIMD_API void SimdGaussianBlur3x3(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, unsigned char * dst, size_t dstStride);

    SIMD_API void SimdGrayToBgra(const unsigned char *gray, size_t width, size_t height, size_t grayStride,
        unsigned char *bgra, size_t bgraStride, unsigned char alpha);

    SIMD_API void SimdAbsSecondDerivativeHistogram(const unsigned char *src, size_t width, size_t height, size_t stride,
        size_t step, size_t indent, unsigned int * histogram);

    SIMD_API void SimdHistogram(const unsigned char *src, size_t width, size_t height, size_t stride, unsigned int * histogram);

    SIMD_API void SimdInterleaveBgrToBgra(unsigned char *bgra, size_t size, const int *blue, int bluePrecision, bool blueSigned,
        const int *green, int greenPrecision, bool greenSigned, const int *red, int redPrecision, bool redSigned, unsigned char alpha);

    SIMD_API void SimdInterleaveGrayToBgra(unsigned char *bgra, size_t size, const int *gray, int grayPrecision, bool graySigned, unsigned char alpha);

    SIMD_API void SimdLbpEstimate(const unsigned char * src, size_t srcStride, size_t width, size_t height, unsigned char * dst, size_t dstStride);

    SIMD_API void SimdMedianFilterSquare3x3(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, unsigned char * dst, size_t dstStride);

    SIMD_API void SimdMedianFilterSquare5x5(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, unsigned char * dst, size_t dstStride);

    SIMD_API void SimdOperation(const unsigned char * a, size_t aStride, const unsigned char * b, size_t bStride,
        size_t width, size_t height, size_t channelCount, unsigned char * dst, size_t dstStride, SimdOperationType type);

    SIMD_API void SimdVectorProduct(const unsigned char * vertical, const unsigned char * horizontal,
        unsigned char * dst, size_t stride, size_t width, size_t height);

    SIMD_API void SimdReduceGray2x2(const unsigned char *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        unsigned char *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

    SIMD_API void SimdReduceGray3x3(const unsigned char *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        unsigned char *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation);

    SIMD_API void SimdReduceGray4x4(const unsigned char *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        unsigned char *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

    SIMD_API void SimdReduceGray5x5(const unsigned char *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        unsigned char *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation);

    SIMD_API void SimdResizeBilinear(const unsigned char *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        unsigned char *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

    SIMD_API void SimdShiftBilinear(const unsigned char * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
        const unsigned char * bkg, size_t bkgStride, double shiftX, double shiftY,
        size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, unsigned char * dst, size_t dstStride);

    SIMD_API void SimdSquaredDifferenceSum(const unsigned char *a, size_t aStride, const unsigned char *b, size_t bStride,
        size_t width, size_t height, unsigned long long * sum);

    SIMD_API void SimdSquaredDifferenceSumMasked(const unsigned char *a, size_t aStride, const unsigned char *b, size_t bStride,
        const unsigned char *mask, size_t maskStride, unsigned char index, size_t width, size_t height, unsigned long long * sum);

    SIMD_API void SimdGetStatistic(const unsigned char * src, size_t stride, size_t width, size_t height,
        unsigned char * min, unsigned char * max, unsigned char * average);

    SIMD_API void SimdGetMoments(const unsigned char * mask, size_t stride, size_t width, size_t height, unsigned char index,
        unsigned long long * area, unsigned long long * x, unsigned long long * y, unsigned long long * xx, unsigned long long * xy, unsigned long long * yy);

    SIMD_API void SimdGetRowSums(const unsigned char * src, size_t stride, size_t width, size_t height, unsigned int * sums);

    SIMD_API void SimdGetColSums(const unsigned char * src, size_t stride, size_t width, size_t height, unsigned int * sums);

    SIMD_API void SimdGetAbsDyRowSums(const unsigned char * src, size_t stride, size_t width, size_t height, unsigned int * sums);

    SIMD_API void SimdGetAbsDxColSums(const unsigned char * src, size_t stride, size_t width, size_t height, unsigned int * sums);

    SIMD_API void SimdStretchGray2x2(const unsigned char *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        unsigned char *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

    SIMD_API void SimdTextureBoostedSaturatedGradient(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        unsigned char saturation, unsigned char boost, unsigned char * dx, size_t dxStride, unsigned char * dy, size_t dyStride);

    SIMD_API void SimdTextureBoostedUv(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        unsigned char boost, unsigned char * dst, size_t dstStride);

    SIMD_API void SimdTextureGetDifferenceSum(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        const unsigned char * lo, size_t loStride, const unsigned char * hi, size_t hiStride, long long * sum);

    SIMD_API void SimdTexturePerformCompensation(const unsigned char * src, size_t srcStride, size_t width, size_t height,
        int shift, unsigned char * dst, size_t dstStride);

    SIMD_API void SimdYuv420ToBgr(const unsigned char * y, size_t yStride, const unsigned char * u, size_t uStride, const unsigned char * v, size_t vStride,
        size_t width, size_t height, unsigned char * bgr, size_t bgrStride);

    SIMD_API void SimdYuv444ToBgr(const unsigned char * y, size_t yStride, const unsigned char * u, size_t uStride, const unsigned char * v, size_t vStride,
        size_t width, size_t height, unsigned char * bgr, size_t bgrStride);

    SIMD_API void SimdYuvToBgra(unsigned char *bgra, size_t width, size_t height, size_t stride,
        const int *y, const int *u, const int *v, int dx, int dy, int precision, unsigned char alpha);

    SIMD_API void SimdYuv420ToBgra(const unsigned char * y, size_t yStride, const unsigned char * u, size_t uStride, const unsigned char * v, size_t vStride,
        size_t width, size_t height, unsigned char * bgra, size_t bgraStride, unsigned char alpha = 0xFF);

    SIMD_API void SimdYuv444ToBgra(const unsigned char * y, size_t yStride, const unsigned char * u, size_t uStride, const unsigned char * v, size_t vStride,
        size_t width, size_t height, unsigned char * bgra, size_t bgraStride, unsigned char alpha = 0xFF);

    SIMD_API void SimdYuv420ToHue(const unsigned char * y, size_t yStride, const unsigned char * u, size_t uStride, const unsigned char * v, size_t vStride,
        size_t width, size_t height, unsigned char * hue, size_t hueStride);

    SIMD_API void SimdYuv444ToHue(const unsigned char * y, size_t yStride, const unsigned char * u, size_t uStride, const unsigned char * v, size_t vStride,
        size_t width, size_t height, unsigned char * hue, size_t hueStride);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif//__SimdLib_h__
