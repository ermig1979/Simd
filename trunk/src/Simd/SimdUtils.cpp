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

#include "Simd/Simd.h"

namespace Simd
{
    void AbsDifferenceSum(const View & a, const View & b, uint64_t & sum)
    {
        assert(Compatible(a, b) && a.format == View::Gray8);

        AbsDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
    }

    void AbsDifferenceSum(const View & a, const View & b, const View & mask, uchar index, uint64_t & sum)
    {
        assert(Compatible(a, b, mask) && a.format == View::Gray8);

        AbsDifferenceSum(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, &sum);
    }

    void AbsGradientSaturatedSum(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8 && src.height >= 3 && src.width >= 3);

        AbsGradientSaturatedSum(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    void AbsSecondDerivativeHistogram(const View & src, size_t step, size_t indent, uint * histogram)
    {
        assert(src.format == View::Gray8);

        AbsSecondDerivativeHistogram(src.data, src.width, src.height, src.stride, step, indent, histogram);
    }

    void AddFeatureDifference(const View & value, const View & lo, const View & hi, ushort weight, View & difference)
    {
        assert(Compatible(value, lo, hi, difference) && value.format == View::Gray8);

        AddFeatureDifference(value.data, value.stride, value.width, value.height, 
            lo.data, lo.stride, hi.data, hi.stride, weight, difference.data, difference.stride);
    }

    void BackgroundGrowRangeSlow(const View & value, View & lo, View & hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View::Gray8);

        BackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    void BackgroundGrowRangeFast(const View & value, View & lo, View & hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View::Gray8);

        BackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    void BackgroundIncrementCount(const View & value, const View & loValue, const View & hiValue, View & loCount, View & hiCount)
    {
        assert(Compatible(value, loValue, hiValue, loCount, hiCount) && value.format == View::Gray8);

        BackgroundIncrementCount(value.data, value.stride, value.width, value.height,
            loValue.data, loValue.stride, hiValue.data, hiValue.stride,
            loCount.data, loCount.stride, hiCount.data, hiCount.stride);
    }

    void BackgroundAdjustRange(View & loCount, View & loValue, View & hiCount, View & hiValue, uchar threshold)
    {
        assert(Compatible(loValue, hiValue, loCount, hiCount) && loValue.format == View::Gray8);

        BackgroundAdjustRange(loCount.data, loCount.stride, loCount.width, loCount.height, 
            loValue.data, loValue.stride, hiCount.data, hiCount.stride, hiValue.data, hiValue.stride, threshold);
    }

    void BackgroundAdjustRange(View & loCount, View & loValue, View & hiCount, View & hiValue, uchar threshold, const View & mask)
    {
        assert(Compatible(loValue, hiValue, loCount, hiCount, mask) && loValue.format == View::Gray8);

        BackgroundAdjustRange(loCount.data, loCount.stride, loCount.width, loCount.height, 
            loValue.data, loValue.stride, hiCount.data, hiCount.stride, hiValue.data, hiValue.stride, 
            threshold, mask.data, mask.stride);
    }

    void BackgroundShiftRange(const View & value, View & lo, View & hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View::Gray8);

        BackgroundShiftRange(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    void BackgroundShiftRange(const View & value, View & lo, View & hi, const View & mask)
    {
        assert(Compatible(value, lo, hi, mask) && value.format == View::Gray8);

        BackgroundShiftRange(value.data, value.stride, value.width, value.height, 
            lo.data, lo.stride, hi.data, hi.stride, mask.data, mask.stride);
    }

    void BackgroundInitMask(const View & src, uchar index, uchar value, View & dst)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8);

        BackgroundInitMask(src.data, src.stride, src.width, src.height, index, value, dst.data, dst.stride);
    }

    void BgraToBgr(const View & bgra, View & bgr)
    {
        assert(EqualSize(bgra, bgr) && bgra.format == View::Bgra32 && bgr.format == View::Bgr24);

        BgraToBgr(bgra.data, bgra.width, bgra.height, bgra.stride, bgr.data, bgr.stride);
    }

    void BgraToGray(const View & bgra, View & gray)
    {
        assert(EqualSize(bgra, gray) && bgra.format == View::Bgra32 && gray.format == View::Gray8);

        BgraToGray(bgra.data, bgra.width, bgra.height, bgra.stride, gray.data, gray.stride);
    }

    void BgrToBgra(const View & bgr, View & bgra, uchar alpha)
    {
        assert(EqualSize(bgr, bgra) && bgra.format == View::Bgra32 && bgr.format == View::Bgr24);

        BgrToBgra(bgr.data, bgr.width, bgr.height, bgr.stride, bgra.data, bgra.stride, alpha);
    }

    void BgrToGray(const View & bgr, View & gray)
    {
        assert(EqualSize(bgr, gray) && bgr.format == View::Bgr24 && gray.format == View::Gray8);

        BgrToGray(bgr.data, bgr.width, bgr.height, bgr.stride, gray.data, gray.stride);
    }

    void Binarization(const View & src, uchar value, uchar positive, uchar negative, View & dst, CompareType compareType)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8);

        Binarization(src.data, src.stride, src.width, src.height, value, positive, negative, dst.data, dst.stride, compareType);
    }

    void AveragingBinarization(const View & src, uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, View & dst, CompareType compareType)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8);

        AveragingBinarization(src.data, src.stride, src.width, src.height, value, 
            neighborhood, threshold, positive, negative, dst.data, dst.stride, compareType);
    }

    void Copy(const View & src, View & dst)
    {
        assert(Compatible(src, dst));

        Copy(src.data, src.stride, src.width, src.height, src.PixelSize(), dst.data, dst.stride);
    }

    void CopyFrame(const View & src, const Rectangle<ptrdiff_t> & frame, View & dst)
    {
        assert(Compatible(src, dst));

        CopyFrame(src.data, src.stride, src.width, src.height, src.PixelSize(), 
            frame.left, frame.top, frame.right, frame.bottom, dst.data, dst.stride);
    }

    void DeinterleaveUv(const View & uv, View & u, View & v)
    {
        assert(EqualSize(uv, u, v) && uv.format == View::Uv16 && u.format == View::Gray8 && v.format == View::Gray8);

        DeinterleaveUv(uv.data, uv.stride, uv.width, uv.height, u.data, u.stride, v.data, v.stride);
    }

    void EdgeBackgroundGrowRangeSlow(const View & value, View & background)
    {
        assert(Compatible(value, background) && value.format == View::Gray8);

        EdgeBackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    void EdgeBackgroundGrowRangeFast(const View & value, View & background)
    {
        assert(Compatible(value, background) && value.format == View::Gray8);

        EdgeBackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    void EdgeBackgroundIncrementCount(const View & value, const View & backgroundValue, View & backgroundCount)
    {
        assert(Compatible(value, backgroundValue, backgroundCount) && value.format == View::Gray8);

        EdgeBackgroundIncrementCount(value.data, value.stride, value.width, value.height,
            backgroundValue.data, backgroundValue.stride, backgroundCount.data, backgroundCount.stride);
    }

    void EdgeBackgroundAdjustRange(View & backgroundCount, View & backgroundValue, uchar threshold)
    {
        assert(Compatible(backgroundCount, backgroundValue) && backgroundCount.format == View::Gray8);

        EdgeBackgroundAdjustRange(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height, 
            backgroundValue.data, backgroundValue.stride, threshold);
    }

    void EdgeBackgroundAdjustRange(View & backgroundCount, View & backgroundValue, uchar threshold, const View & mask)
    {
        assert(Compatible(backgroundCount, backgroundValue, mask) && backgroundCount.format == View::Gray8);

        EdgeBackgroundAdjustRange(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height, 
            backgroundValue.data, backgroundValue.stride, threshold, mask.data, mask.stride);
    }

    void EdgeBackgroundShiftRange(const View & value, View & background)
    {
        assert(Compatible(value, background) && value.format == View::Gray8);

        EdgeBackgroundShiftRange(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    void EdgeBackgroundShiftRange(const View & value, View & background, const View & mask)
    {
        assert(Compatible(value, background, mask) && value.format == View::Gray8);

        EdgeBackgroundShiftRange(value.data, value.stride, value.width, value.height, 
            background.data, background.stride, mask.data, mask.stride);
    }

    void Fill(View & dst, uchar value)
    {
        Fill(dst.data, dst.stride, dst.width, dst.height, dst.PixelSize(), value);
    }

    void FillBgra(View & dst, uchar blue, uchar green, uchar red, uchar alpha)
    {
        assert(dst.format == View::Bgra32);

        FillBgra(dst.data, dst.stride, dst.width, dst.height, blue, green, red, alpha);
    }

    void GaussianBlur3x3(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        GaussianBlur3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    void MedianFilterSquare3x3(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        MedianFilterSquare3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    void MedianFilterSquare5x5(const View & src, View & dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        MedianFilterSquare5x5(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    void Operation(const View & a, const View & b, View & dst, OperationType type)
    {
        assert(Compatible(a, b, dst) && a.ChannelSize() == 1);

        Operation(a.data, a.stride, b.data, b.stride, a.width, a.height, a.ChannelCount(), dst.data, dst.stride, type);
    }

    void ReduceGray2x2(const View & src, View & dst)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        ReduceGray2x2(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    void ReduceGray3x3(const View & src, View & dst, bool compensation)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        ReduceGray3x3(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, compensation);
    }

    void ReduceGray4x4(const View & src, View & dst)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        ReduceGray4x4(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    void ReduceGray5x5(const View & src, View & dst, bool compensation)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        ReduceGray5x5(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, compensation);
    }

    void ResizeBilinear(const View & src, View & dst)
    {
        assert(src.format == dst.format && src.ChannelSize() == 1);

        if(EqualSize(src, dst))
        {
            Copy(src, dst);
        }
        else
        {
            ResizeBilinear(src.data, src.width, src.height, src.stride,
                dst.data, dst.width, dst.height, dst.stride, src.ChannelCount());
        }
    }

    void ShiftBilinear(const View & src, const View & bkg, const Point<double> & shift, const Rectangle<ptrdiff_t> & crop, View & dst)
    {
        assert(Compatible(src, bkg, dst) && src.ChannelSize() == 1);

        ShiftBilinear(src.data, src.stride, src.width, src.height, src.ChannelCount(), bkg.data, bkg.stride,
            shift.x, shift.y, crop.left, crop.top, crop.right, crop.bottom, dst.data, dst.stride);
    }

    void SquaredDifferenceSum(const View & a, const View & b, uint64_t & sum)
    {
        assert(Compatible(a, b) && a.format == View::Gray8);

        SquaredDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
    }

    void SquaredDifferenceSum(const View & a, const View & b, const View & mask, uchar index, uint64_t & sum)
    {
        assert(Compatible(a, b, mask) && a.format == View::Gray8);

        SquaredDifferenceSum(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, &sum);
    }

    void GetStatistic(const View & src, uchar * min, uchar * max, uchar * average)
    {
        assert(src.format == View::Gray8);

        GetStatistic(src.data, src.stride, src.width, src.height, min, max, average);
    }

    void GetMoments(const View & mask, uchar index, uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
    {
        assert(mask.format == View::Gray8);

        GetMoments(mask.data, mask.stride, mask.width, mask.height, index, area, x, y, xx, xy, yy);
    }

    void GetRowSums(const View & src, uint * sums)
    {
        assert(src.format == View::Gray8);

        GetRowSums(src.data, src.stride, src.width, src.height, sums);
    }

    void GetColSums(const View & src, uint * sums)
    {
        assert(src.format == View::Gray8);

        GetColSums(src.data, src.stride, src.width, src.height, sums);
    }

    void GetAbsDyRowSums(const View & src, uint * sums)
    {
        assert(src.format == View::Gray8);

        GetAbsDyRowSums(src.data, src.stride, src.width, src.height, sums);
    }

    void GetAbsDxColSums(const View & src, uint * sums)
    {
        assert(src.format == View::Gray8);

        GetAbsDxColSums(src.data, src.stride, src.width, src.height, sums);
    }

    void StretchGray2x2(const View & src, View & dst)
    {
        assert(src.format == View::Gray8 && dst.format == View::Gray8);

        StretchGray2x2(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    void TextureBoostedSaturatedGradient(const View & src, uchar saturation, uchar boost, View & dx, View & dy)
    {
        assert(Compatible(src, dx, dy) && src.format == View::Gray8 && src.height >= 3 && src.width >= 3);

        TextureBoostedSaturatedGradient(src.data, src.stride, src.width, src.height, saturation, boost, dx.data, dx.stride, dy.data, dy.stride);
    }

    void TextureBoostedUv(const View & src, uchar boost, View & dst)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8);

        TextureBoostedUv(src.data, src.stride, src.width, src.height, boost, dst.data, dst.stride);
    }

    void TextureGetDifferenceSum(const View & src, const View & lo, const View & hi, int64_t * sum)
    {
        assert(Compatible(src, lo, hi) && src.format == View::Gray8 && sum != NULL);

        TextureGetDifferenceSum(src.data, src.stride, src.width, src.height, lo.data, lo.stride, hi.data, hi.stride, sum);
    }

    void TexturePerformCompensation(const View & src, int shift, View & dst)
    {
        assert(Compatible(src, dst) && src.format == View::Gray8 && shift > -0xFF && shift < 0xFF);

        TexturePerformCompensation(src.data, src.stride, src.width, src.height, shift, dst.data, dst.stride);
    }

    void Yuv444ToBgr(const View & y, const View & u, const View & v, View & bgr)
    {
        assert(Compatible(y, u, v) && EqualSize(y, bgr) && y.format == View::Gray8 && bgr.format == View::Bgr24);

        Yuv444ToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
    }

    void Yuv420ToBgr(const View & y, const View & u, const View & v, View & bgr)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(y.width == bgr.width && y.height == bgr.height);
        assert(y.format == View::Gray8 && bgr.format == View::Bgr24);

        Yuv420ToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
    }
    
    void Yuv444ToBgra(const View & y, const View & u, const View & v, View & bgra, uchar alpha)
    {
        assert(Compatible(y, u, v) && EqualSize(y, bgra) && y.format == View::Gray8 && bgra.format == View::Bgra32);

        Yuv444ToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha);
    }

    void Yuv420ToBgra(const View & y, const View & u, const View & v, View & bgra, uchar alpha)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(y.width == bgra.width && y.height == bgra.height);
        assert(y.format == View::Gray8 && bgra.format == View::Bgra32);

        Yuv420ToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha);
    }

    void Yuv444ToHue(const View & y, const View & u, const View & v, View & hue)
    {
        assert(Compatible(y, u, v, hue) && y.format == View::Gray8);

        Yuv444ToHue(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hue.data, hue.stride);
    }

    void Yuv420ToHue(const View & y, const View & u, const View & v, View & hue)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(Compatible(y, hue) && y.format == View::Gray8);

        Yuv420ToHue(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hue.data, hue.stride);
    }
}
