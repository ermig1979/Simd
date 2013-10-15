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
#ifndef __SimdAlg_h__
#define __SimdAlg_h__

#include "Simd/SimdLib.h"
#include "Simd/SimdTypes.h"

namespace Simd
{
    SIMD_INLINE void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
        size_t width, size_t height, uint64_t * sum)
    {
        SimdAbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    }

    SIMD_INLINE void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
        const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sum)
    {
        SimdAbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    }

    SIMD_INLINE void AbsGradientSaturatedSum(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
    {
        SimdAbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    }

    SIMD_INLINE void AddFeatureDifference(const uchar * value, size_t valueStride, size_t width, size_t height, 
        const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride,
        ushort weight, uchar * difference, size_t differenceStride)
    {
        SimdAddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    }

    SIMD_INLINE void AlphaBlending(const uchar *src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
        const uchar *alpha, size_t alphaStride, uchar *dst, size_t dstStride)
    {
        SimdAlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
    }

    SIMD_INLINE void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
    {
        SimdBackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    }

    SIMD_INLINE void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
    {
        SimdBackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    }

    SIMD_INLINE void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
        const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
        uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride)
    {
        SimdBackgroundIncrementCount(value, valueStride, width, height, loValue, loValueStride, hiValue, hiValueStride, 
            loCount, loCountStride, hiCount, hiCountStride);
    }

    SIMD_INLINE void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
        uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
        uchar * hiValue, size_t hiValueStride, uchar threshold)
    {
        SimdBackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
                hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    }

    SIMD_INLINE void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
        uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
        uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride)
    {
        SimdBackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride, 
                hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
    }

    SIMD_INLINE void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
    {
        SimdBackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    }

    SIMD_INLINE void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride)
    {
        SimdBackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    }

    SIMD_INLINE void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
        uchar index, uchar value, uchar * dst, size_t dstStride)
    {
        SimdBackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    }

    SIMD_INLINE void BgraToBgr(const uchar *bgra, size_t width, size_t height, size_t bgraStride, uchar *bgr, size_t bgrStride)
    {
        SimdBgraToBgr(bgra, width, height, bgraStride, bgr, bgrStride);
    }

    SIMD_INLINE void BgraToGray(const uchar *bgra, size_t width, size_t height, size_t bgraStride, uchar *gray, size_t grayStride)
    {
        SimdBgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    }

    SIMD_INLINE void BgrToBgra(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *bgra, size_t bgraStride, uchar alpha = 0xFF)
    {
        SimdBgrToBgra(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
    }

    SIMD_INLINE void BgrToGray(const uchar *bgr, size_t width, size_t height, size_t bgrStride, uchar *gray, size_t grayStride)
    {
        SimdBgrToGray(bgr, width, height, bgrStride, gray, grayStride);
    }

    SIMD_INLINE void Binarization(const uchar * src, size_t srcStride, size_t width, size_t height, 
        uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride, SimdCompareType compareType)
    {
        SimdBinarization(src, srcStride, width, height, value, positive, negative, dst, dstStride, compareType);
    }

    SIMD_INLINE void AveragingBinarization(const uchar * src, size_t srcStride, size_t width, size_t height,
        uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, 
        uchar * dst, size_t dstStride, SimdCompareType compareType)
    {
        SimdAveragingBinarization(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride, compareType);
    }

    SIMD_INLINE void Copy(const uchar * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, uchar * dst, size_t dstStride)
    {
        SimdCopy(src, srcStride, width, height, pixelSize, dst, dstStride);
    }

    SIMD_INLINE void CopyFrame(const uchar * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, 
        size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uchar * dst, size_t dstStride)
    {
        SimdCopyFrame(src, srcStride, width, height, pixelSize, frameLeft, frameTop, frameRight, frameBottom, dst, dstStride);
    }

    SIMD_INLINE uint32_t Crc32(const void * src, size_t size)
    {
        return SimdCrc32(src, size);
    }

    SIMD_INLINE void DeinterleaveUv(const uchar * uv, size_t uvStride, size_t width, size_t height, 
        uchar * u, size_t uStride, uchar * v, size_t vStride)
    {
        SimdDeinterleaveUv(uv, uvStride, width, height, u, uStride, v, vStride);
    }

    SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * background, size_t backgroundStride)
    {
        SimdEdgeBackgroundGrowRangeSlow(value, valueStride, width, height, background, backgroundStride);
    }

    SIMD_INLINE void EdgeBackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * background, size_t backgroundStride)
    {
        SimdEdgeBackgroundGrowRangeFast(value, valueStride, width, height, background, backgroundStride);
    }

    SIMD_INLINE void EdgeBackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
        const uchar * backgroundValue, size_t backgroundValueStride, uchar * backgroundCount, size_t backgroundCountStride)
    {
        SimdEdgeBackgroundIncrementCount(value, valueStride, width, height, backgroundValue, backgroundValueStride, backgroundCount, backgroundCountStride);
    }

    SIMD_INLINE void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
        uchar * backgroundValue, size_t backgroundValueStride, uchar threshold)
    {
        SimdEdgeBackgroundAdjustRange(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold);
    }

    SIMD_INLINE void EdgeBackgroundAdjustRange(uchar * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, 
        uchar * backgroundValue, size_t backgroundValueStride, uchar threshold, const uchar * mask, size_t maskStride)
    {
        SimdEdgeBackgroundAdjustRangeMasked(backgroundCount, backgroundCountStride, width, height, backgroundValue, backgroundValueStride, threshold, mask, maskStride);
    }

    SIMD_INLINE void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * background, size_t backgroundStride)
    {
        SimdEdgeBackgroundShiftRange(value, valueStride, width, height, background, backgroundStride);
    }

    SIMD_INLINE void EdgeBackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * background, size_t backgroundStride, const uchar * mask, size_t maskStride)
    {
        SimdEdgeBackgroundShiftRangeMasked(value, valueStride, width, height, background, backgroundStride, mask, maskStride);
    }

    SIMD_INLINE void Fill(uchar * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uchar value)
    {
        SimdFill(dst, stride, width, height, pixelSize, value);
    }

    SIMD_INLINE void FillFrame(uchar * dst, size_t stride, size_t width, size_t height, size_t pixelSize, 
        size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uchar value)
    {
        SimdFillFrame(dst, stride, width, height, pixelSize, frameLeft, frameTop, frameRight, frameBottom, value);
    }

    SIMD_INLINE void FillBgra(uchar * dst, size_t stride, size_t width, size_t height, uchar blue, uchar green, uchar red, uchar alpha)
    {
        SimdFillBgra(dst, stride, width, height, blue, green, red, alpha);
    }

    SIMD_INLINE void GaussianBlur3x3(const uchar * src, size_t srcStride, size_t width, size_t height, 
        size_t channelCount, uchar * dst, size_t dstStride)
    {
        SimdGaussianBlur3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    }

    SIMD_INLINE void GrayToBgra(const uchar *gray, size_t width, size_t height, size_t grayStride, uchar *bgra, size_t bgraStride, uchar alpha = 0xFF)
    {
        SimdGrayToBgra(gray, width, height, grayStride, bgra, bgraStride, alpha);
    }

    SIMD_INLINE void AbsSecondDerivativeHistogram(const uchar *src, size_t width, size_t height, size_t stride, size_t step, size_t indent, uint * histogram)
    {
        SimdAbsSecondDerivativeHistogram(src, width, height, stride, step, indent, histogram);
    }

    SIMD_INLINE void Histogram(const uchar *src, size_t width, size_t height, size_t stride, uint * histogram)
    {
        SimdHistogram(src, width, height, stride, histogram);
    }

    SIMD_INLINE void InterleaveBgra(uchar *bgra, size_t size, const int *blue, int bluePrecision, bool blueSigned, 
        const int *green, int greenPrecision, bool greenSigned, const int *red, int redPrecision, bool redSigned, uchar alpha = 0xFF)
    {
        SimdInterleaveBgrToBgra(bgra, size, blue, bluePrecision, blueSigned, green, greenPrecision, greenSigned, red, redPrecision, redSigned, alpha);
    }

    SIMD_INLINE void InterleaveBgra(uchar *bgra, size_t size, const int *gray, int grayPrecision, bool graySigned, uchar alpha = 0xFF)
    {
        SimdInterleaveGrayToBgra(bgra, size, gray, grayPrecision, graySigned, alpha);
    }

    SIMD_INLINE void LbpEstimate(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
    {
        SimdLbpEstimate(src, srcStride, width, height, dst, dstStride);
    }

    SIMD_INLINE void MedianFilterSquare3x3(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
    {
        SimdMedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    }

    SIMD_INLINE void MedianFilterSquare5x5(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
    {
        SimdMedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    }
}

#include "Simd/SimdOperation.h"
#include "Simd/SimdReduceGray2x2.h"
#include "Simd/SimdReduceGray3x3.h"
#include "Simd/SimdReduceGray4x4.h"
#include "Simd/SimdReduceGray5x5.h"
#include "Simd/SimdResizeBilinear.h"
#include "Simd/SimdShiftBilinear.h"
#include "Simd/SimdSquaredDifferenceSum.h"
#include "Simd/SimdStatistic.h"
#include "Simd/SimdStretchGray2x2.h"
#include "Simd/SimdTexture.h"
#include "Simd/SimdYuvToBgra.h"
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdYuvToHue.h"

#endif//__SimdAlg_h__
