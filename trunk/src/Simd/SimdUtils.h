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
#include "Simd/SimdView.h"

#ifndef __SimdUtils_h__
#define __SimdUtils_h__

namespace Simd
{
    SIMD_INLINE void AbsDifferenceSum(const View & a, const View & b, uint64_t & sum)
    {
        assert(Compatible(a, b) && a.format == View::Gray8);

        SimdAbsDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
    }

    SIMD_INLINE void AbsDifferenceSum(const View & a, const View & b, const View & mask, uint8_t index, uint64_t & sum)
    {
        assert(Compatible(a, b, mask) && a.format == View::Gray8);

        SimdAbsDifferenceSumMasked(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, &sum);
    }

    SIMD_INLINE void AbsDifferenceSums3x3(const View & current, const View & background, uint64_t * sums)
    {
        assert(Compatible(current, background) && current.format == View::Gray8 && current.width > 2 && current.height > 2);

        SimdAbsDifferenceSums3x3(current.data, current.stride, background.data, background.stride, current.width, current.height, sums);
    }

    SIMD_INLINE void AbsDifferenceSums3x3(const View & current, const View & background, const View & mask, uint8_t index, uint64_t * sums)
    {
        assert(Compatible(current, background, mask) && current.format == View::Gray8 && current.width > 2 && current.height > 2);

        SimdAbsDifferenceSums3x3Masked(current.data, current.stride, background.data, background.stride, 
            mask.data, mask.stride, index, current.width, current.height, sums);
    }

    SIMD_INLINE void AbsGradientSaturatedSum(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8 && src.height >= 3 && src.width >= 3);

        SimdAbsGradientSaturatedSum(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    SIMD_INLINE void AddFeatureDifference(const View & value, const View & lo, const View & hi, uint16_t weight, View & difference)
    {
        assert(Compatible(value, lo, hi, difference) && value.format == View::Gray8);

        SimdAddFeatureDifference(value.data, value.stride, value.width, value.height,
            lo.data, lo.stride, hi.data, hi.stride, weight, difference.data, difference.stride);
    }

    SIMD_INLINE void BackgroundGrowRangeSlow(const View & value, View & lo, View & hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View::Gray8);

        SimdBackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    SIMD_INLINE void BackgroundGrowRangeFast(const View & value, View & lo, View & hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View::Gray8);

        SimdBackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    SIMD_INLINE void BackgroundIncrementCount(const View & value, const View & loValue, const View & hiValue, View & loCount, View & hiCount)
    {
        assert(Compatible(value, loValue, hiValue, loCount, hiCount) && value.format == View::Gray8);

        SimdBackgroundIncrementCount(value.data, value.stride, value.width, value.height,
            loValue.data, loValue.stride, hiValue.data, hiValue.stride,
            loCount.data, loCount.stride, hiCount.data, hiCount.stride);
    }

    SIMD_INLINE void BackgroundAdjustRange(View & loCount, View & loValue, View & hiCount, View & hiValue, uint8_t threshold)
    {
        assert(Compatible(loValue, hiValue, loCount, hiCount) && loValue.format == View::Gray8);

        SimdBackgroundAdjustRange(loCount.data, loCount.stride, loCount.width, loCount.height,
            loValue.data, loValue.stride, hiCount.data, hiCount.stride, hiValue.data, hiValue.stride, threshold);
    }

    SIMD_INLINE void BackgroundAdjustRange(View & loCount, View & loValue, View & hiCount, View & hiValue, uint8_t threshold, const View & mask)
    {
        assert(Compatible(loValue, hiValue, loCount, hiCount, mask) && loValue.format == View::Gray8);

        SimdBackgroundAdjustRangeMasked(loCount.data, loCount.stride, loCount.width, loCount.height,
            loValue.data, loValue.stride, hiCount.data, hiCount.stride, hiValue.data, hiValue.stride,
            threshold, mask.data, mask.stride);
    }

    SIMD_INLINE void BackgroundShiftRange(const View & value, View & lo, View & hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View::Gray8);

        SimdBackgroundShiftRange(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    SIMD_INLINE void BackgroundShiftRange(const View & value, View & lo, View & hi, const View & mask)
    {
        assert(Compatible(value, lo, hi, mask) && value.format == View::Gray8);

        SimdBackgroundShiftRangeMasked(value.data, value.stride, value.width, value.height,
            lo.data, lo.stride, hi.data, hi.stride, mask.data, mask.stride);
    }

    SIMD_INLINE void BackgroundInitMask(const View & src, uint8_t index, uint8_t value, View & dst)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8);

        SimdBackgroundInitMask(src.data, src.stride, src.width, src.height, index, value, dst.data, dst.stride);
    }

    SIMD_INLINE void BgraToBgr(const View & bgra, View & bgr)
    {
        assert(EqualSize(bgra, bgr) && bgra.format == View::Bgra32 && bgr.format == View::Bgr24);

        SimdBgraToBgr(bgra.data, bgra.width, bgra.height, bgra.stride, bgr.data, bgr.stride);
    }

    SIMD_INLINE void BgraToGray(const View & bgra, View & gray)
    {
        assert(EqualSize(bgra, gray) && bgra.format == View::Bgra32 && gray.format == View::Gray8);

        SimdBgraToGray(bgra.data, bgra.width, bgra.height, bgra.stride, gray.data, gray.stride);
    }

    SIMD_INLINE void BgrToBgra(const View & bgr, View & bgra, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(bgr, bgra) && bgra.format == View::Bgra32 && bgr.format == View::Bgr24);

        SimdBgrToBgra(bgr.data, bgr.width, bgr.height, bgr.stride, bgra.data, bgra.stride, alpha);
    }

    SIMD_INLINE void Bgr48pToBgra32(const View & blue, const View & green, const View & red, View & bgra, uint8_t alpha = 0xFF)
    {
        assert(Compatible(blue, green, red) && EqualSize(blue, bgra) && blue.format == View::Int16 && bgra.format == View::Bgra32);

        SimdBgr48pToBgra32(blue.data, blue.stride, blue.width, blue.height, green.data, green.stride, red.data, red.stride, bgra.data, bgra.stride, alpha);
    }

    SIMD_INLINE void BgrToGray(const View & bgr, View & gray)
    {
        assert(EqualSize(bgr, gray) && bgr.format == View::Bgr24 && gray.format == View::Gray8);

        SimdBgrToGray(bgr.data, bgr.width, bgr.height, bgr.stride, gray.data, gray.stride);
    }

    SIMD_INLINE void Binarization(const View & src, uint8_t value, uint8_t positive, uint8_t negative, View & dst, SimdCompareType compareType)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8);

        SimdBinarization(src.data, src.stride, src.width, src.height, value, positive, negative, dst.data, dst.stride, compareType);
    }

    SIMD_INLINE void AveragingBinarization(const View & src, uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, View & dst, SimdCompareType compareType)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8);

        SimdAveragingBinarization(src.data, src.stride, src.width, src.height, value,
            neighborhood, threshold, positive, negative, dst.data, dst.stride, compareType);
    }

    SIMD_INLINE void ConditionalCount(const View & src, uint8_t value, SimdCompareType compareType, uint32_t & count)
    {
        assert(src.format == View::Gray8);

        SimdConditionalCount(src.data, src.stride, src.width, src.height, value, compareType, &count);
    }

    SIMD_INLINE void ConditionalSum(const View & src, const View & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)
    {
        assert(Compatible(src, mask) && src.format == View::Gray8);

        SimdConditionalSum(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
    }

    SIMD_INLINE void ConditionalSquareSum(const View & src, const View & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)
    {
        assert(Compatible(src, mask) && src.format == View::Gray8);

        SimdConditionalSquareSum(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
    }

    SIMD_INLINE void ConditionalSquareGradientSum(const View & src, const View & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)
    {
        assert(Compatible(src, mask) && src.format == View::Gray8 && src.width >= 3 && src.height >= 3);

        SimdConditionalSquareGradientSum(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
    }

    SIMD_INLINE void Copy(const View & src, View & dst)
    {
        assert(Compatible(src, dst));

        SimdCopy(src.data, src.stride, src.width, src.height, src.PixelSize(), dst.data, dst.stride);
    }

    SIMD_INLINE void CopyFrame(const View & src, const Rectangle<ptrdiff_t> & frame, View & dst)
    {
        assert(Compatible(src, dst));

        SimdCopyFrame(src.data, src.stride, src.width, src.height, src.PixelSize(),
            frame.left, frame.top, frame.right, frame.bottom, dst.data, dst.stride);
    }

    SIMD_INLINE void DeinterleaveUv(const View & uv, View & u, View & v)
    {
        assert(EqualSize(uv, u, v) && uv.format == View::Uv16 && u.format == View::Gray8 && v.format == View::Gray8);

        SimdDeinterleaveUv(uv.data, uv.stride, uv.width, uv.height, u.data, u.stride, v.data, v.stride);
    }

    SIMD_INLINE void AlphaBlending(const View & src, const View & alpha, View & dst)
    {
        assert(Compatible(src, dst) && EqualSize(src, alpha) && alpha.format == View::Gray8 && src.ChannelSize() == 1);

        SimdAlphaBlending(src.data, src.stride, src.width, src.height, src.ChannelCount(), alpha.data, alpha.stride, dst.data, dst.stride);
    }

    SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const View & value, View & background)
    {
        assert(Compatible(value, background) && value.format == View::Gray8);

        SimdEdgeBackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    SIMD_INLINE void EdgeBackgroundGrowRangeFast(const View & value, View & background)
    {
        assert(Compatible(value, background) && value.format == View::Gray8);

        SimdEdgeBackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    SIMD_INLINE void EdgeBackgroundIncrementCount(const View & value, const View & backgroundValue, View & backgroundCount)
    {
        assert(Compatible(value, backgroundValue, backgroundCount) && value.format == View::Gray8);

        SimdEdgeBackgroundIncrementCount(value.data, value.stride, value.width, value.height,
            backgroundValue.data, backgroundValue.stride, backgroundCount.data, backgroundCount.stride);
    }

    SIMD_INLINE void EdgeBackgroundAdjustRange(View & backgroundCount, View & backgroundValue, uint8_t threshold)
    {
        assert(Compatible(backgroundCount, backgroundValue) && backgroundCount.format == View::Gray8);

        SimdEdgeBackgroundAdjustRange(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height,
            backgroundValue.data, backgroundValue.stride, threshold);
    }

    SIMD_INLINE void EdgeBackgroundAdjustRange(View & backgroundCount, View & backgroundValue, uint8_t threshold, const View & mask)
    {
        assert(Compatible(backgroundCount, backgroundValue, mask) && backgroundCount.format == View::Gray8);

        SimdEdgeBackgroundAdjustRangeMasked(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height,
            backgroundValue.data, backgroundValue.stride, threshold, mask.data, mask.stride);
    }

    SIMD_INLINE void EdgeBackgroundShiftRange(const View & value, View & background)
    {
        assert(Compatible(value, background) && value.format == View::Gray8);

        SimdEdgeBackgroundShiftRange(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    SIMD_INLINE void EdgeBackgroundShiftRange(const View & value, View & background, const View & mask)
    {
        assert(Compatible(value, background, mask) && value.format == View::Gray8);

        SimdEdgeBackgroundShiftRangeMasked(value.data, value.stride, value.width, value.height,
            background.data, background.stride, mask.data, mask.stride);
    }

    SIMD_INLINE void Fill(View & dst, uint8_t value)
    {
        SimdFill(dst.data, dst.stride, dst.width, dst.height, dst.PixelSize(), value);
    }

    SIMD_INLINE void FillFrame(View & dst, const Rectangle<ptrdiff_t> & frame, uint8_t value)
    {
        SimdFillFrame(dst.data, dst.stride, dst.width, dst.height, dst.PixelSize(),
            frame.left, frame.top, frame.right, frame.bottom, value);
    }

    SIMD_INLINE void FillBgra(View & dst, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha = 0xFF)
    {
        assert(dst.format == View::Bgra32);

        SimdFillBgra(dst.data, dst.stride, dst.width, dst.height, blue, green, red, alpha);
    }

    SIMD_INLINE void GaussianBlur3x3(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdGaussianBlur3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    SIMD_INLINE void GrayToBgra(const View & gray, View & bgra, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(gray, bgra) && bgra.format == View::Bgra32 && gray.format == View::Gray8);

        SimdGrayToBgra(gray.data, gray.width, gray.height, gray.stride, bgra.data, bgra.stride, alpha);
    }

    SIMD_INLINE void AbsSecondDerivativeHistogram(const View & src, size_t step, size_t indent, uint32_t * histogram)
    {
        assert(src.format == View::Gray8);

        SimdAbsSecondDerivativeHistogram(src.data, src.width, src.height, src.stride, step, indent, histogram);
    }

    SIMD_INLINE void Histogram(const View & src, uint32_t * histogram)
    {
        assert(src.format == View::Gray8);

        SimdHistogram(src.data, src.width, src.height, src.stride, histogram);
    }

    SIMD_INLINE void MedianFilterRhomb3x3(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterRhomb3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    SIMD_INLINE void MedianFilterRhomb5x5(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterRhomb5x5(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    SIMD_INLINE void MedianFilterSquare3x3(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterSquare3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    SIMD_INLINE void MedianFilterSquare5x5(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterSquare5x5(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    SIMD_INLINE void Operation(const View & a, const View & b, View & dst, SimdOperationType type)
    {
        assert(Compatible(a, b, dst) && a.ChannelSize() == 1);

        SimdOperation(a.data, a.stride, b.data, b.stride, a.width, a.height, a.ChannelCount(), dst.data, dst.stride, type);
    }

    SIMD_INLINE void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, View & dst)
    {
        assert(dst.format == View::Gray8);

        SimdVectorProduct(vertical, horizontal, dst.data, dst.stride, dst.width, dst.height);
    }

    SIMD_INLINE void ReduceGray2x2(const View & src, View & dst)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        SimdReduceGray2x2(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    SIMD_INLINE void ReduceGray3x3(const View & src, View & dst, bool compensation = true)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        SimdReduceGray3x3(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, compensation);
    }

    SIMD_INLINE void ReduceGray4x4(const View & src, View & dst)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        SimdReduceGray4x4(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    SIMD_INLINE void ReduceGray5x5(const View & src, View & dst, bool compensation = true)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        SimdReduceGray5x5(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, compensation);
    }

    SIMD_INLINE void ResizeBilinear(const View & src, View & dst)
    {
        assert(src.format == dst.format && src.ChannelSize() == 1);

        if(EqualSize(src, dst))
        {
            Copy(src, dst);
        }
        else
        {
            SimdResizeBilinear(src.data, src.width, src.height, src.stride,
                dst.data, dst.width, dst.height, dst.stride, src.ChannelCount());
        }
    }

    SIMD_INLINE void ShiftBilinear(const View & src, const View & bkg, const Point<double> & shift, const Rectangle<ptrdiff_t> & crop, View & dst)
    {
        assert(Compatible(src, bkg, dst) && src.ChannelSize() == 1);

        SimdShiftBilinear(src.data, src.stride, src.width, src.height, src.ChannelCount(), bkg.data, bkg.stride,
            shift.x, shift.y, crop.left, crop.top, crop.right, crop.bottom, dst.data, dst.stride);
    }

    SIMD_INLINE void SquaredDifferenceSum(const View & a, const View & b, uint64_t & sum)
    {
        assert(Compatible(a, b) && a.format == View::Gray8);

        SimdSquaredDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
    }

    SIMD_INLINE void SquaredDifferenceSum(const View & a, const View & b, const View & mask, uint8_t index, uint64_t & sum)
    {
        assert(Compatible(a, b, mask) && a.format == View::Gray8);

        SimdSquaredDifferenceSumMasked(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, &sum);
    }

    SIMD_INLINE void GetStatistic(const View & src, uint8_t & min, uint8_t & max, uint8_t & average)
    {
        assert(src.format == View::Gray8);

        SimdGetStatistic(src.data, src.stride, src.width, src.height, &min, &max, &average);
    }

    SIMD_INLINE void GetMoments(const View & mask, uint8_t index, uint64_t & area, uint64_t & x, uint64_t & y, uint64_t & xx, uint64_t & xy, uint64_t & yy)
    {
        assert(mask.format == View::Gray8);

        SimdGetMoments(mask.data, mask.stride, mask.width, mask.height, index, &area, &x, &y, &xx, &xy, &yy);
    }

    SIMD_INLINE void GetRowSums(const View & src, uint32_t * sums)
    {
        assert(src.format == View::Gray8);

        SimdGetRowSums(src.data, src.stride, src.width, src.height, sums);
    }

    SIMD_INLINE void GetColSums(const View & src, uint32_t * sums)
    {
        assert(src.format == View::Gray8);

        SimdGetColSums(src.data, src.stride, src.width, src.height, sums);
    }

    SIMD_INLINE void GetAbsDyRowSums(const View & src, uint32_t * sums)
    {
        assert(src.format == View::Gray8);

        SimdGetAbsDyRowSums(src.data, src.stride, src.width, src.height, sums);
    }

    SIMD_INLINE void GetAbsDxColSums(const View & src, uint32_t * sums)
    {
        assert(src.format == View::Gray8);

        SimdGetAbsDxColSums(src.data, src.stride, src.width, src.height, sums);
    }

    SIMD_INLINE void StretchGray2x2(const View & src, View & dst)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        SimdStretchGray2x2(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    SIMD_INLINE void TextureBoostedSaturatedGradient(const View & src, uint8_t saturation, uint8_t boost, View & dx, View & dy)
    {
        assert(Compatible(src, dx, dy) && src.format == View::Gray8 && src.height >= 3 && src.width >= 3);

        SimdTextureBoostedSaturatedGradient(src.data, src.stride, src.width, src.height, saturation, boost, dx.data, dx.stride, dy.data, dy.stride);
    }

    SIMD_INLINE void TextureBoostedUv(const View & src, uint8_t boost, View & dst)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8);

        SimdTextureBoostedUv(src.data, src.stride, src.width, src.height, boost, dst.data, dst.stride);
    }

    SIMD_INLINE void TextureGetDifferenceSum(const View & src, const View & lo, const View & hi, int64_t & sum)
    {
        assert(Compatible(src, lo, hi) && src.format == View::Gray8 && sum != NULL);

        SimdTextureGetDifferenceSum(src.data, src.stride, src.width, src.height, lo.data, lo.stride, hi.data, hi.stride, &sum);
    }

    SIMD_INLINE void TexturePerformCompensation(const View & src, int shift, View & dst)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8 && shift > -0xFF && shift < 0xFF);

        SimdTexturePerformCompensation(src.data, src.stride, src.width, src.height, shift, dst.data, dst.stride);
    }

    SIMD_INLINE void Yuv444ToBgr(const View & y, const View & u, const View & v, View & bgr)
    {
        assert(Compatible(y, u, v) && EqualSize(y, bgr) && y.format == View::Gray8 && bgr.format == View::Bgr24);

        SimdYuv444ToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
    }

    SIMD_INLINE void Yuv420ToBgr(const View & y, const View & u, const View & v, View & bgr)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(y.width == bgr.width && y.height == bgr.height);
        assert(y.format == View::Gray8 && bgr.format == View::Bgr24);

        SimdYuv420ToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
    }

    SIMD_INLINE void Yuv444ToBgra(const View & y, const View & u, const View & v, View & bgra, uint8_t alpha = 0xFF)
    {
        assert(Compatible(y, u, v) && EqualSize(y, bgra) && y.format == View::Gray8 && bgra.format == View::Bgra32);

        SimdYuv444ToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha);
    }

    SIMD_INLINE void Yuv420ToBgra(const View & y, const View & u, const View & v, View & bgra, uint8_t alpha = 0xFF)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(y.width == bgra.width && y.height == bgra.height);
        assert(y.format == View::Gray8 && bgra.format == View::Bgra32);

        SimdYuv420ToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha);
    }

    SIMD_INLINE void Yuv444ToHue(const View & y, const View & u, const View & v, View & hue)
    {
        assert(Compatible(y, u, v, hue) && y.format == View::Gray8);

        SimdYuv444ToHue(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hue.data, hue.stride);
    }

    SIMD_INLINE void Yuv420ToHue(const View & y, const View & u, const View & v, View & hue)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(Compatible(y, hue) && y.format == View::Gray8);

        SimdYuv420ToHue(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hue.data, hue.stride);
    }
}

#endif//__SimdUtils_h__

