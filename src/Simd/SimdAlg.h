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

    SIMD_INLINE void AbsDifferenceSums3x3(const uchar *current, size_t currentStride, const uchar * background, size_t backgroundStride,
        size_t width, size_t height, uint64_t * sums)
    {
        SimdAbsDifferenceSums3x3(current, currentStride, background, backgroundStride, width, height, sums);
    }

    SIMD_INLINE void AbsDifferenceSums3x3(const uchar *current, size_t currentStride, const uchar *background, size_t backgroundStride,
        const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sums)
    {
        SimdAbsDifferenceSums3x3Masked(current, currentStride, background, backgroundStride, mask, maskStride, index, width, height, sums);
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

    SIMD_INLINE void MedianFilterRhomb3x3(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
    {
        SimdMedianFilterRhomb3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    }

    SIMD_INLINE void MedianFilterRhomb5x5(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
    {
        SimdMedianFilterRhomb5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    }

    SIMD_INLINE void MedianFilterSquare3x3(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
    {
        SimdMedianFilterSquare3x3(src, srcStride, width, height, channelCount, dst, dstStride);
    }

    SIMD_INLINE void MedianFilterSquare5x5(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride)
    {
        SimdMedianFilterSquare5x5(src, srcStride, width, height, channelCount, dst, dstStride);
    }

    SIMD_INLINE void Operation(const uchar * a, size_t aStride, const uchar * b, size_t bStride, 
        size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride, SimdOperationType type)
    {
        SimdOperation(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
    }

    SIMD_INLINE void VectorProduct(const uchar * vertical, const uchar * horizontal, uchar * dst, size_t stride, size_t width, size_t height)
    {
        SimdVectorProduct(vertical, horizontal, dst, stride, width, height);
    }

    SIMD_INLINE void ReduceGray2x2(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
        uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
    {
        SimdReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    }

    SIMD_INLINE void ReduceGray3x3(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
        uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation = true)
    {
        SimdReduceGray3x3(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    }

    SIMD_INLINE void ReduceGray4x4(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
        uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
    {
        SimdReduceGray4x4(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    }

    SIMD_INLINE void ReduceGray5x5(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
        uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation = true)
    {
        SimdReduceGray5x5(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
    }

    SIMD_INLINE void ResizeBilinear(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
    {
        SimdResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
    }

    SIMD_INLINE void ShiftBilinear(const uchar * src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
        const uchar * bkg, size_t bkgStride, double shiftX, double shiftY, 
        size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uchar * dst, size_t dstStride)
    {
        SimdShiftBilinear(src, srcStride, width, height, channelCount, bkg, bkgStride,
                shiftX, shiftY, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);
    }

    SIMD_INLINE void SquaredDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
        size_t width, size_t height, uint64_t * sum)
    {
        SimdSquaredDifferenceSum(a, aStride, b, bStride, width, height, sum);
    }

    SIMD_INLINE void SquaredDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
        const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sum)
    {
        SimdSquaredDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    }

    SIMD_INLINE void GetStatistic(const uchar * src, size_t stride, size_t width, size_t height, 
        uchar * min, uchar * max, uchar * average)
    {
        SimdGetStatistic(src, stride, width, height, min, max, average);
    }

    SIMD_INLINE void GetMoments(const uchar * mask, size_t stride, size_t width, size_t height, uchar index, 
        uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
    {
        SimdGetMoments(mask, stride, width, height, index, area, x, y, xx, xy, yy);    
    }

    SIMD_INLINE void GetRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
    {
        SimdGetRowSums(src, stride, width, height, sums);
    }

    SIMD_INLINE void GetColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
    {
        SimdGetColSums(src, stride, width, height, sums);
    }

    SIMD_INLINE void GetAbsDyRowSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
    {
        SimdGetAbsDyRowSums(src, stride, width, height, sums);
    }

    SIMD_INLINE void GetAbsDxColSums(const uchar * src, size_t stride, size_t width, size_t height, uint * sums)
    {
        SimdGetAbsDxColSums(src, stride, width, height, sums);
    }

    SIMD_INLINE void ConditionalCount(const uchar * src, size_t stride, size_t width, size_t height, 
        uchar value, SimdCompareType compareType, uint * count)
    {
        SimdConditionalCount(src, stride, width, height, value, compareType, count);
    }

    SIMD_INLINE void ConditionalSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
        const uchar * mask, size_t maskStride, uchar value, SimdCompareType compareType, uint64_t * sum)
    {
        SimdConditionalSum(src, srcStride, width, height, mask, maskStride, value, compareType, sum);
    }

    SIMD_INLINE void StretchGray2x2(const uchar *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
        uchar *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
    {
        SimdStretchGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
    }

    SIMD_INLINE void TextureBoostedSaturatedGradient(const uchar * src, size_t srcStride, size_t width, size_t height, 
        uchar saturation, uchar boost, uchar * dx, size_t dxStride, uchar * dy, size_t dyStride)
    {
        SimdTextureBoostedSaturatedGradient(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
    }

    SIMD_INLINE void TextureBoostedUv(const uchar * src, size_t srcStride, size_t width, size_t height, 
        uchar boost, uchar * dst, size_t dstStride)
    {
        SimdTextureBoostedUv(src, srcStride, width, height, boost, dst, dstStride);
    }

    SIMD_INLINE void TextureGetDifferenceSum(const uchar * src, size_t srcStride, size_t width, size_t height, 
        const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride, int64_t * sum)
    {
        SimdTextureGetDifferenceSum(src, srcStride, width, height, lo, loStride, hi, hiStride, sum);
    }

    SIMD_INLINE void TexturePerformCompensation(const uchar * src, size_t srcStride, size_t width, size_t height, 
        int shift, uchar * dst, size_t dstStride)
    {
        SimdTexturePerformCompensation(src, srcStride, width, height, shift, dst, dstStride);
    }

    SIMD_INLINE void Yuv420ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
        size_t width, size_t height, uchar * bgr, size_t bgrStride)
    {
        SimdYuv420ToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    }

    SIMD_INLINE void Yuv444ToBgr(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
        size_t width, size_t height, uchar * bgr, size_t bgrStride)
    {
        SimdYuv444ToBgr(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
    }

    SIMD_INLINE void YuvToBgra(uchar *bgra, size_t width, size_t height, size_t stride,
        const int *y, const int *u, const int *v, int dx, int dy, int precision, uchar alpha = 0xFF)
    {
        SimdYuvToBgra(bgra, width, height, stride, y, u, v, dx, dy, precision, alpha);
    }

    SIMD_INLINE void Yuv420ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
        size_t width, size_t height, uchar * bgra, size_t bgraStride, uchar alpha = 0xFF)
    {
        SimdYuv420ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    }

    SIMD_INLINE void Yuv444ToBgra(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
        size_t width, size_t height, uchar * bgra, size_t bgraStride, uchar alpha = 0xFF)
    {
        SimdYuv444ToBgra(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
    }

    SIMD_INLINE void Yuv420ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
        size_t width, size_t height, uchar * hue, size_t hueStride)
    {
        SimdYuv420ToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    }

    SIMD_INLINE void Yuv444ToHue(const uchar * y, size_t yStride, const uchar * u, size_t uStride, const uchar * v, size_t vStride, 
        size_t width, size_t height, uchar * hue, size_t hueStride)
    {
        SimdYuv444ToHue(y, yStride, u, uStride, v, vStride, width, height, hue, hueStride);
    }
}

#endif//__SimdAlg_h__
