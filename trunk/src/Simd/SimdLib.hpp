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

#include "Simd/SimdView.hpp"
#include "Simd/SimdPixel.hpp"

#ifndef __SimdLib_hpp__
#define __SimdLib_hpp__

namespace Simd
{
    template<class A> SIMD_INLINE void AbsDifferenceSum(const View<A>& a, const View<A>& b, uint64_t & sum)
    {
        assert(Compatible(a, b) && a.format == View<A>::Gray8);

        SimdAbsDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
    }

    template<class A> SIMD_INLINE void AbsDifferenceSum(const View<A>& a, const View<A>& b, const View<A>& mask, uint8_t index, uint64_t & sum)
    {
        assert(Compatible(a, b, mask) && a.format == View<A>::Gray8);

        SimdAbsDifferenceSumMasked(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, &sum);
    }

    template<class A> SIMD_INLINE void AbsDifferenceSums3x3(const View<A>& current, const View<A>& background, uint64_t * sums)
    {
        assert(Compatible(current, background) && current.format == View<A>::Gray8 && current.width > 2 && current.height > 2);

        SimdAbsDifferenceSums3x3(current.data, current.stride, background.data, background.stride, current.width, current.height, sums);
    }

    template<class A> SIMD_INLINE void AbsDifferenceSums3x3(const View<A>& current, const View<A>& background, const View<A>& mask, uint8_t index, uint64_t * sums)
    {
        assert(Compatible(current, background, mask) && current.format == View<A>::Gray8 && current.width > 2 && current.height > 2);

        SimdAbsDifferenceSums3x3Masked(current.data, current.stride, background.data, background.stride,
            mask.data, mask.stride, index, current.width, current.height, sums);
    }

    template<class A> SIMD_INLINE void AbsGradientSaturatedSum(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8 && src.height >= 3 && src.width >= 3);

        SimdAbsGradientSaturatedSum(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void AddFeatureDifference(const View<A>& value, const View<A>& lo, const View<A>& hi, uint16_t weight, View<A>& difference)
    {
        assert(Compatible(value, lo, hi, difference) && value.format == View<A>::Gray8);

        SimdAddFeatureDifference(value.data, value.stride, value.width, value.height,
            lo.data, lo.stride, hi.data, hi.stride, weight, difference.data, difference.stride);
    }

    template<class A> SIMD_INLINE void BackgroundGrowRangeSlow(const View<A>& value, View<A>& lo, View<A>& hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View<A>::Gray8);

        SimdBackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    template<class A> SIMD_INLINE void BackgroundGrowRangeFast(const View<A>& value, View<A>& lo, View<A>& hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View<A>::Gray8);

        SimdBackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    template<class A> SIMD_INLINE void BackgroundIncrementCount(const View<A>& value, const View<A>& loValue, const View<A>& hiValue, View<A>& loCount, View<A>& hiCount)
    {
        assert(Compatible(value, loValue, hiValue, loCount, hiCount) && value.format == View<A>::Gray8);

        SimdBackgroundIncrementCount(value.data, value.stride, value.width, value.height,
            loValue.data, loValue.stride, hiValue.data, hiValue.stride,
            loCount.data, loCount.stride, hiCount.data, hiCount.stride);
    }

    template<class A> SIMD_INLINE void BackgroundAdjustRange(View<A>& loCount, View<A>& loValue, View<A>& hiCount, View<A>& hiValue, uint8_t threshold)
    {
        assert(Compatible(loValue, hiValue, loCount, hiCount) && loValue.format == View<A>::Gray8);

        SimdBackgroundAdjustRange(loCount.data, loCount.stride, loCount.width, loCount.height,
            loValue.data, loValue.stride, hiCount.data, hiCount.stride, hiValue.data, hiValue.stride, threshold);
    }

    template<class A> SIMD_INLINE void BackgroundAdjustRange(View<A>& loCount, View<A>& loValue, View<A>& hiCount, View<A>& hiValue, uint8_t threshold, const View<A>& mask)
    {
        assert(Compatible(loValue, hiValue, loCount, hiCount, mask) && loValue.format == View<A>::Gray8);

        SimdBackgroundAdjustRangeMasked(loCount.data, loCount.stride, loCount.width, loCount.height,
            loValue.data, loValue.stride, hiCount.data, hiCount.stride, hiValue.data, hiValue.stride,
            threshold, mask.data, mask.stride);
    }

    template<class A> SIMD_INLINE void BackgroundShiftRange(const View<A>& value, View<A>& lo, View<A>& hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View<A>::Gray8);

        SimdBackgroundShiftRange(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    template<class A> SIMD_INLINE void BackgroundShiftRange(const View<A>& value, View<A>& lo, View<A>& hi, const View<A>& mask)
    {
        assert(Compatible(value, lo, hi, mask) && value.format == View<A>::Gray8);

        SimdBackgroundShiftRangeMasked(value.data, value.stride, value.width, value.height,
            lo.data, lo.stride, hi.data, hi.stride, mask.data, mask.stride);
    }

    template<class A> SIMD_INLINE void BackgroundInitMask(const View<A>& src, uint8_t index, uint8_t value, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdBackgroundInitMask(src.data, src.stride, src.width, src.height, index, value, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void BayerToBgr(const View<A>& bayer, View<A>& bgr)
    {
        assert(EqualSize(bgr, bayer) && bgr.format == View<A>::Bgr24);
        assert(bayer.format >= View<A>::BayerGrbg && bayer.format <= View<A>::BayerBggr);
        assert((bayer.width%2 == 0) && (bayer.height%2 == 0));

        SimdBayerToBgr(bayer.data, bayer.width, bayer.height, bayer.stride, (SimdPixelFormatType)bayer.format, bgr.data, bgr.stride);
    }

    template<class A> SIMD_INLINE void BayerToBgra(const View<A>& bayer, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(bgra, bayer) && bgra.format == View<A>::Bgra32);
        assert(bayer.format >= View<A>::BayerGrbg && bayer.format <= View<A>::BayerBggr);
        assert((bayer.width%2 == 0) && (bayer.height%2 == 0));

        SimdBayerToBgra(bayer.data, bayer.width, bayer.height, bayer.stride, (SimdPixelFormatType)bayer.format, bgra.data, bgra.stride, alpha);
    }

    template<class A> SIMD_INLINE void BgraToBayer(const View<A>& bgra, View<A>& bayer)
    {
        assert(EqualSize(bgra, bayer) && bgra.format == View<A>::Bgra32);
        assert(bayer.format >= View<A>::BayerGrbg && bayer.format <= View<A>::BayerBggr);
        assert((bayer.width%2 == 0) && (bayer.height%2 == 0));

        SimdBgraToBayer(bgra.data, bgra.width, bgra.height, bgra.stride, bayer.data, bayer.stride, (SimdPixelFormatType)bayer.format);
    }

    template<class A> SIMD_INLINE void BgraToBgr(const View<A>& bgra, View<A>& bgr)
    {
        assert(EqualSize(bgra, bgr) && bgra.format == View<A>::Bgra32 && bgr.format == View<A>::Bgr24);

        SimdBgraToBgr(bgra.data, bgra.width, bgra.height, bgra.stride, bgr.data, bgr.stride);
    }

    template<class A> SIMD_INLINE void BgraToGray(const View<A>& bgra, View<A>& gray)
    {
        assert(EqualSize(bgra, gray) && bgra.format == View<A>::Bgra32 && gray.format == View<A>::Gray8);

        SimdBgraToGray(bgra.data, bgra.width, bgra.height, bgra.stride, gray.data, gray.stride);
    }

    template<class A> SIMD_INLINE void BgraToYuv420p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(y.width == bgra.width && y.height == bgra.height);
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdBgraToYuv420p(bgra.data, bgra.width, bgra.height, bgra.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    template<class A> SIMD_INLINE void BgraToYuv444p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(EqualSize(bgra, y) && Compatible(y, u, v));
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdBgraToYuv444p(bgra.data, bgra.width, bgra.height, bgra.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    template<class A> SIMD_INLINE void BgrToBayer(const View<A>& bgr, View<A>& bayer)
    {
        assert(EqualSize(bgr, bayer) && bgr.format == View<A>::Bgr24);
        assert(bayer.format >= View<A>::BayerGrbg && bayer.format <= View<A>::BayerBggr);
        assert((bayer.width%2 == 0) && (bayer.height%2 == 0));

        SimdBgrToBayer(bgr.data, bgr.width, bgr.height, bgr.stride, bayer.data, bayer.stride, (SimdPixelFormatType)bayer.format);
    }

    template<class A> SIMD_INLINE void BgrToBgra(const View<A>& bgr, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(bgr, bgra) && bgra.format == View<A>::Bgra32 && bgr.format == View<A>::Bgr24);

        SimdBgrToBgra(bgr.data, bgr.width, bgr.height, bgr.stride, bgra.data, bgra.stride, alpha);
    }

    template<class A> SIMD_INLINE void Bgr48pToBgra32(const View<A>& blue, const View<A>& green, const View<A>& red, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(Compatible(blue, green, red) && EqualSize(blue, bgra) && blue.format == View<A>::Int16 && bgra.format == View<A>::Bgra32);

        SimdBgr48pToBgra32(blue.data, blue.stride, blue.width, blue.height, green.data, green.stride, red.data, red.stride, bgra.data, bgra.stride, alpha);
    }

    template<class A> SIMD_INLINE void BgrToGray(const View<A>& bgr, View<A>& gray)
    {
        assert(EqualSize(bgr, gray) && bgr.format == View<A>::Bgr24 && gray.format == View<A>::Gray8);

        SimdBgrToGray(bgr.data, bgr.width, bgr.height, bgr.stride, gray.data, gray.stride);
    }

    template<class A> SIMD_INLINE void BgrToHsl(const View<A> & bgr, View<A> & hsl)
    {
        assert(EqualSize(bgr, hsl) && bgr.format == View<A>::Bgr24 && hsl.format == View<A>::Hsl24);

        SimdBgrToHsl(bgr.data, bgr.width, bgr.height, bgr.stride, hsl.data, hsl.stride);
    }

    template<class A> SIMD_INLINE void BgrToHsv(const View<A> & bgr, View<A> & hsv)
    {
        assert(EqualSize(bgr, hsv) && bgr.format == View<A>::Bgr24 && hsv.format == View<A>::Hsv24);

        SimdBgrToHsv(bgr.data, bgr.width, bgr.height, bgr.stride, hsv.data, hsv.stride);
    }

    template<class A> SIMD_INLINE void BgrToYuv420p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(y.width == bgr.width && y.height == bgr.height);
        assert(y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdBgrToYuv420p(bgr.data, bgr.width, bgr.height, bgr.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    template<class A> SIMD_INLINE void BgrToYuv444p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(EqualSize(bgr, y) && Compatible(y, u, v));
        assert(y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdBgrToYuv444p(bgr.data, bgr.width, bgr.height, bgr.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    template<class A> SIMD_INLINE void Binarization(const View<A>& src, uint8_t value, uint8_t positive, uint8_t negative, View<A>& dst, SimdCompareType compareType)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdBinarization(src.data, src.stride, src.width, src.height, value, positive, negative, dst.data, dst.stride, compareType);
    }

    template<class A> SIMD_INLINE void AveragingBinarization(const View<A>& src, uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, View<A>& dst, SimdCompareType compareType)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdAveragingBinarization(src.data, src.stride, src.width, src.height, value,
            neighborhood, threshold, positive, negative, dst.data, dst.stride, compareType);
    }

    template<class A> SIMD_INLINE void ConditionalCount8u(const View<A> & src, uint8_t value, SimdCompareType compareType, uint32_t & count)
    {
        assert(src.format == View<A>::Gray8);

        SimdConditionalCount8u(src.data, src.stride, src.width, src.height, value, compareType, &count);
    }

    template<class A> SIMD_INLINE void ConditionalCount16i(const View<A> & src, int16_t value, SimdCompareType compareType, uint32_t & count)
    {
        assert(src.format == View<A>::Int16);

        SimdConditionalCount16i(src.data, src.stride, src.width, src.height, value, compareType, &count);
    }

    template<class A> SIMD_INLINE void ConditionalSum(const View<A> & src, const View<A> & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)
    {
        assert(Compatible(src, mask) && src.format == View<A>::Gray8);

        SimdConditionalSum(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
    }

    template<class A> SIMD_INLINE void ConditionalSquareSum(const View<A>& src, const View<A>& mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)
    {
        assert(Compatible(src, mask) && src.format == View<A>::Gray8);

        SimdConditionalSquareSum(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
    }

    template<class A> SIMD_INLINE void ConditionalSquareGradientSum(const View<A>& src, const View<A>& mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)
    {
        assert(Compatible(src, mask) && src.format == View<A>::Gray8 && src.width >= 3 && src.height >= 3);

        SimdConditionalSquareGradientSum(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
    }

    template<class A, class B> SIMD_INLINE void Copy(const View<A> & src, View<B> & dst)
    {
        assert(Compatible(src, dst));

        SimdCopy(src.data, src.stride, src.width, src.height, src.PixelSize(), dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void CopyFrame(const View<A>& src, const Rectangle<ptrdiff_t> & frame, View<A>& dst)
    {
        assert(Compatible(src, dst) && frame.Width() >= 0 && frame.Height() >= 0);
        assert(frame.left >= 0 && frame.top >= 0 && frame.right <= ptrdiff_t(src.width) && frame.bottom <= ptrdiff_t(src.height));

        SimdCopyFrame(src.data, src.stride, src.width, src.height, src.PixelSize(),
            frame.left, frame.top, frame.right, frame.bottom, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void DeinterleaveUv(const View<A>& uv, View<A>& u, View<A>& v)
    {
        assert(EqualSize(uv, u, v) && uv.format == View<A>::Uv16 && u.format == View<A>::Gray8 && v.format == View<A>::Gray8);

        SimdDeinterleaveUv(uv.data, uv.stride, uv.width, uv.height, u.data, u.stride, v.data, v.stride);
    }

    template<class A> SIMD_INLINE void AlphaBlending(const View<A>& src, const View<A>& alpha, View<A>& dst)
    {
        assert(Compatible(src, dst) && EqualSize(src, alpha) && alpha.format == View<A>::Gray8 && src.ChannelSize() == 1);

        SimdAlphaBlending(src.data, src.stride, src.width, src.height, src.ChannelCount(), alpha.data, alpha.stride, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const View<A>& value, View<A>& background)
    {
        assert(Compatible(value, background) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    template<class A> SIMD_INLINE void EdgeBackgroundGrowRangeFast(const View<A>& value, View<A>& background)
    {
        assert(Compatible(value, background) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    template<class A> SIMD_INLINE void EdgeBackgroundIncrementCount(const View<A>& value, const View<A>& backgroundValue, View<A>& backgroundCount)
    {
        assert(Compatible(value, backgroundValue, backgroundCount) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundIncrementCount(value.data, value.stride, value.width, value.height,
            backgroundValue.data, backgroundValue.stride, backgroundCount.data, backgroundCount.stride);
    }

    template<class A> SIMD_INLINE void EdgeBackgroundAdjustRange(View<A>& backgroundCount, View<A>& backgroundValue, uint8_t threshold)
    {
        assert(Compatible(backgroundCount, backgroundValue) && backgroundCount.format == View<A>::Gray8);

        SimdEdgeBackgroundAdjustRange(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height,
            backgroundValue.data, backgroundValue.stride, threshold);
    }

    template<class A> SIMD_INLINE void EdgeBackgroundAdjustRange(View<A>& backgroundCount, View<A>& backgroundValue, uint8_t threshold, const View<A>& mask)
    {
        assert(Compatible(backgroundCount, backgroundValue, mask) && backgroundCount.format == View<A>::Gray8);

        SimdEdgeBackgroundAdjustRangeMasked(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height,
            backgroundValue.data, backgroundValue.stride, threshold, mask.data, mask.stride);
    }

    template<class A> SIMD_INLINE void EdgeBackgroundShiftRange(const View<A>& value, View<A>& background)
    {
        assert(Compatible(value, background) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundShiftRange(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    template<class A> SIMD_INLINE void EdgeBackgroundShiftRange(const View<A>& value, View<A>& background, const View<A>& mask)
    {
        assert(Compatible(value, background, mask) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundShiftRangeMasked(value.data, value.stride, value.width, value.height,
            background.data, background.stride, mask.data, mask.stride);
    }

    template<class A> SIMD_INLINE void Fill(View<A>& dst, uint8_t value)
    {
        SimdFill(dst.data, dst.stride, dst.width, dst.height, dst.PixelSize(), value);
    }

    template<class A> SIMD_INLINE void FillFrame(View<A>& dst, const Rectangle<ptrdiff_t> & frame, uint8_t value)
    {
        SimdFillFrame(dst.data, dst.stride, dst.width, dst.height, dst.PixelSize(),
            frame.left, frame.top, frame.right, frame.bottom, value);
    }

    template<class A> SIMD_INLINE void FillBgr(View<A>& dst, uint8_t blue, uint8_t green, uint8_t red)
    {
        assert(dst.format == View<A>::Bgr24);

        SimdFillBgr(dst.data, dst.stride, dst.width, dst.height, blue, green, red);
    }

    template<class A> SIMD_INLINE void FillBgra(View<A>& dst, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha = 0xFF)
    {
        assert(dst.format == View<A>::Bgra32);

        SimdFillBgra(dst.data, dst.stride, dst.width, dst.height, blue, green, red, alpha);
    }

    template<class A> SIMD_INLINE void GaussianBlur3x3(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdGaussianBlur3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void GrayToBgr(const View<A>& gray, View<A>& bgr)
    {
        assert(EqualSize(gray, bgr) && bgr.format == View<A>::Bgr24 && gray.format == View<A>::Gray8);

        SimdGrayToBgr(gray.data, gray.width, gray.height, gray.stride, bgr.data, bgr.stride);
    }

    template<class A> SIMD_INLINE void GrayToBgra(const View<A>& gray, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(gray, bgra) && bgra.format == View<A>::Bgra32 && gray.format == View<A>::Gray8);

        SimdGrayToBgra(gray.data, gray.width, gray.height, gray.stride, bgra.data, bgra.stride, alpha);
    }

    template<class A> SIMD_INLINE void AbsSecondDerivativeHistogram(const View<A>& src, size_t step, size_t indent, uint32_t * histogram)
    {
        assert(src.format == View<A>::Gray8 && indent >= step && src.width > 2*indent && src.height > 2*indent);

        SimdAbsSecondDerivativeHistogram(src.data, src.width, src.height, src.stride, step, indent, histogram);
    }

    template<class A> SIMD_INLINE void Histogram(const View<A>& src, uint32_t * histogram)
    {
        assert(src.format == View<A>::Gray8);

        SimdHistogram(src.data, src.width, src.height, src.stride, histogram);
    }

    template<class A> SIMD_INLINE void HistogramMasked(const View<A> & src, const View<A> & mask, uint8_t index, uint32_t * histogram)
    {
        assert(Compatible(src, mask) && src.format == View<A>::Gray8);

        SimdHistogramMasked(src.data, src.stride, src.width, src.height, mask.data, mask.stride, index, histogram);
    }

    template<class A> SIMD_INLINE void Integral(const View<A>& src, View<A>& sum)
    {
        assert(src.width + 1 == sum.width && src.height + 1 == sum.height);
        assert(src.format == View<A>::Gray8 && sum.format == View<A>::Int32);

        SimdIntegral(src.data, src.stride, src.width, src.height, sum.data, sum.stride, NULL, 0, NULL, 0,
            (SimdPixelFormatType)sum.format, SimdPixelFormatNone);
    }

    template<class A> SIMD_INLINE void Integral(const View<A>& src, View<A>& sum, View<A>& sqsum)
    {
        assert(src.width + 1 == sum.width && src.height + 1 == sum.height && EqualSize(sum, sqsum));
        assert(src.format == View<A>::Gray8 && sum.format == View<A>::Int32 && (sqsum.format == View<A>::Int32 || sqsum.format == View<A>::Double));

        SimdIntegral(src.data, src.stride, src.width, src.height, sum.data, sum.stride, sqsum.data, sqsum.stride, NULL, 0,
            (SimdPixelFormatType)sum.format, (SimdPixelFormatType)sqsum.format);
    }

    template<class A> SIMD_INLINE void Integral(const View<A>& src, View<A>& sum, View<A>& sqsum, View<A>& tilted)
    {
        assert(src.width + 1 == sum.width && src.height + 1 == sum.height && EqualSize(sum, sqsum) && Compatible(sum, tilted));
        assert(src.format == View<A>::Gray8 && sum.format == View<A>::Int32 && (sqsum.format == View<A>::Int32 || sqsum.format == View<A>::Double));

        SimdIntegral(src.data, src.stride, src.width, src.height, sum.data, sum.stride, sqsum.data, sqsum.stride, tilted.data, tilted.stride,
            (SimdPixelFormatType)sum.format, (SimdPixelFormatType)sqsum.format);
    }

    template<class A> SIMD_INLINE void InterferenceIncrement(View<A> & dst, uint8_t increment, int16_t saturation)
    {
        assert(dst.format == View<A>::Int16);

        SimdInterferenceIncrement(dst.data, dst.stride, dst.width, dst.height, increment, saturation);
    }

    template<class A> SIMD_INLINE void InterferenceIncrementMasked(View<A> & dst, uint8_t increment, int16_t saturation, const View<A>& mask, uint8_t index)
    {
        assert(dst.format == View<A>::Int16 && mask.format == View<A>::Gray8 && EqualSize(dst, mask));

        SimdInterferenceIncrementMasked(dst.data, dst.stride, dst.width, dst.height, increment, saturation, mask.data, mask.stride, index);
    }

    template<class A> SIMD_INLINE void InterferenceDecrement(View<A> & dst, uint8_t decrement, int16_t saturation)
    {
        assert(dst.format == View<A>::Int16);

        SimdInterferenceDecrement(dst.data, dst.stride, dst.width, dst.height, decrement, saturation);
    }

    template<class A> SIMD_INLINE void InterferenceDecrementMasked(View<A> & dst, uint8_t decrement, int16_t saturation, const View<A>& mask, uint8_t index)
    {
        assert(dst.format == View<A>::Int16 && mask.format == View<A>::Gray8 && EqualSize(dst, mask));

        SimdInterferenceDecrementMasked(dst.data, dst.stride, dst.width, dst.height, decrement, saturation, mask.data, mask.stride, index);
    }

    template<class A> SIMD_INLINE void LbpEstimate(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdLbpEstimate(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void MedianFilterRhomb3x3(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterRhomb3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void MedianFilterRhomb5x5(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterRhomb5x5(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void MedianFilterSquare3x3(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterSquare3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void MedianFilterSquare5x5(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterSquare5x5(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void OperationBinary8u(const View<A>& a, const View<A>& b, View<A>& dst, SimdOperationBinary8uType type)
    {
        assert(Compatible(a, b, dst) && a.ChannelSize() == 1);

        SimdOperationBinary8u(a.data, a.stride, b.data, b.stride, a.width, a.height, a.ChannelCount(), dst.data, dst.stride, type);
    }

    template<class A> SIMD_INLINE void OperationBinary16i(const View<A>& a, const View<A>& b, View<A>& dst, SimdOperationBinary16iType type)
    {
        assert(Compatible(a, b, dst) && a.format == View<A>::Int16);

        SimdOperationBinary16i(a.data, a.stride, b.data, b.stride, a.width, a.height, dst.data, dst.stride, type);
    }

    template<class A> SIMD_INLINE void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, View<A>& dst)
    {
        assert(dst.format == View<A>::Gray8);

        SimdVectorProduct(vertical, horizontal, dst.data, dst.stride, dst.width, dst.height);
    }

    template<class A> SIMD_INLINE void ReduceGray2x2(const View<A>& src, View<A>& dst)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8);

        SimdReduceGray2x2(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    template<class A> SIMD_INLINE void ReduceGray3x3(const View<A>& src, View<A>& dst, bool compensation = true)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8);

        SimdReduceGray3x3(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, compensation ? 1 : 0);
    }

    template<class A> SIMD_INLINE void ReduceGray4x4(const View<A>& src, View<A>& dst)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8);

        SimdReduceGray4x4(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    template<class A> SIMD_INLINE void ReduceGray5x5(const View<A>& src, View<A>& dst, bool compensation = true)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8);

        SimdReduceGray5x5(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, compensation ? 1 : 0);
    }

    template<class A> SIMD_INLINE void ResizeBilinear(const View<A>& src, View<A>& dst)
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

    template<class A> SIMD_INLINE void SegmentationChangeIndex(View<A> & mask, uint8_t oldIndex, uint8_t newIndex)
    {
        assert(mask.format == View<A>::Gray8);

        SimdSegmentationChangeIndex(mask.data, mask.stride, mask.width, mask.height, oldIndex, newIndex);
    }

    template<class A> SIMD_INLINE void SegmentationFillSingleHoles(View<A> & mask, uint8_t index)
    {
        assert(mask.format == View<A>::Gray8 && mask.width > 2 && mask.height > 2);

        SimdSegmentationFillSingleHoles(mask.data, mask.stride, mask.width, mask.height, index);
    }

    template<class A> SIMD_INLINE void SegmentationPropagate2x2(const View<A> & parent, View<A> & child, const View<A> & difference, uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t thresholdDifference)
    {
        assert(parent.format == View<A>::Gray8 && parent.width >= 2 && parent.height >= 2);
        assert((child.width + 1)/2 == parent.width && (child.height + 1)/2 == parent.height);
        assert(Compatible(child, difference) && child.format == View<A>::Gray8);

        SimdSegmentationPropagate2x2(parent.data, parent.stride, parent.width, parent.height, child.data, child.stride, 
            difference.data, difference.stride, currentIndex, invalidIndex, emptyIndex, thresholdDifference);
    }

    template<class A> SIMD_INLINE void SegmentationShrinkRegion(const View<A> & mask, uint8_t index, Rectangle<ptrdiff_t> & rect)
    {
        assert(mask.format == View<A>::Gray8);
        assert(rect.Width() > 0 && rect.Height() > 0 && Rectangle<ptrdiff_t>(mask.Size()).Contains(rect));

        SimdSegmentationShrinkRegion(mask.data, mask.stride, mask.width, mask.height, index, &rect.left, &rect.top, &rect.right, &rect.bottom);
    }

    template<class A> SIMD_INLINE void ShiftBilinear(const View<A> & src, const View<A> & bkg, const Point<double> & shift, const Rectangle<ptrdiff_t> & crop, View<A> & dst)
    {
        assert(Compatible(src, bkg, dst) && src.ChannelSize() == 1);

        SimdShiftBilinear(src.data, src.stride, src.width, src.height, src.ChannelCount(), bkg.data, bkg.stride,
            shift.x, shift.y, crop.left, crop.top, crop.right, crop.bottom, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void SobelDx(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdSobelDx(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void SobelDxAbs(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdSobelDxAbs(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void SobelDy(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdSobelDy(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void SobelDyAbs(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdSobelDyAbs(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void ContourMetrics(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdContourMetrics(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void ContourMetrics(const View<A>& src, const View<A>& mask, uint8_t indexMin, View<A>& dst)
    {
        assert(Compatible(src, mask) && EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdContourMetricsMasked(src.data, src.stride, src.width, src.height, mask.data, mask.stride, indexMin, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void ContourAnchors(const View<A>& src, size_t step, int16_t threshold, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Int16 && dst.format == View<A>::Gray8);

        SimdContourAnchors(src.data, src.stride, src.width, src.height, step, threshold, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void SquaredDifferenceSum(const View<A>& a, const View<A>& b, uint64_t & sum)
    {
        assert(Compatible(a, b) && a.format == View<A>::Gray8);

        SimdSquaredDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
    }

    template<class A> SIMD_INLINE void SquaredDifferenceSum(const View<A>& a, const View<A>& b, const View<A>& mask, uint8_t index, uint64_t & sum)
    {
        assert(Compatible(a, b, mask) && a.format == View<A>::Gray8);

        SimdSquaredDifferenceSumMasked(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, &sum);
    }

    template<class A> SIMD_INLINE void GetStatistic(const View<A>& src, uint8_t & min, uint8_t & max, uint8_t & average)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetStatistic(src.data, src.stride, src.width, src.height, &min, &max, &average);
    }

    template<class A> SIMD_INLINE void GetMoments(const View<A>& mask, uint8_t index, uint64_t & area, uint64_t & x, uint64_t & y, uint64_t & xx, uint64_t & xy, uint64_t & yy)
    {
        assert(mask.format == View<A>::Gray8);

        SimdGetMoments(mask.data, mask.stride, mask.width, mask.height, index, &area, &x, &y, &xx, &xy, &yy);
    }

    template<class A> SIMD_INLINE void GetRowSums(const View<A>& src, uint32_t * sums)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetRowSums(src.data, src.stride, src.width, src.height, sums);
    }

    template<class A> SIMD_INLINE void GetColSums(const View<A>& src, uint32_t * sums)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetColSums(src.data, src.stride, src.width, src.height, sums);
    }

    template<class A> SIMD_INLINE void GetAbsDyRowSums(const View<A>& src, uint32_t * sums)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetAbsDyRowSums(src.data, src.stride, src.width, src.height, sums);
    }

    template<class A> SIMD_INLINE void GetAbsDxColSums(const View<A>& src, uint32_t * sums)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetAbsDxColSums(src.data, src.stride, src.width, src.height, sums);
    }

    template<class A> SIMD_INLINE void ValueSum(const View<A>& src, uint64_t & sum)
    {
        assert(src.format == View<A>::Gray8);

        SimdValueSum(src.data, src.stride, src.width, src.height, &sum);
    }

    template<class A> SIMD_INLINE void SquareSum(const View<A>& src, uint64_t & sum)
    {
        assert(src.format == View<A>::Gray8);

        SimdSquareSum(src.data, src.stride, src.width, src.height, &sum);
    }

    template<class A> SIMD_INLINE void StretchGray2x2(const View<A>& src, View<A>& dst)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8);

        SimdStretchGray2x2(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    template<class A> SIMD_INLINE void TextureBoostedSaturatedGradient(const View<A>& src, uint8_t saturation, uint8_t boost, View<A>& dx, View<A>& dy)
    {
        assert(Compatible(src, dx, dy) && src.format == View<A>::Gray8 && src.height >= 3 && src.width >= 3);

        SimdTextureBoostedSaturatedGradient(src.data, src.stride, src.width, src.height, saturation, boost, dx.data, dx.stride, dy.data, dy.stride);
    }

    template<class A> SIMD_INLINE void TextureBoostedUv(const View<A>& src, uint8_t boost, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdTextureBoostedUv(src.data, src.stride, src.width, src.height, boost, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void TextureGetDifferenceSum(const View<A>& src, const View<A>& lo, const View<A>& hi, int64_t & sum)
    {
        assert(Compatible(src, lo, hi) && src.format == View<A>::Gray8);

        SimdTextureGetDifferenceSum(src.data, src.stride, src.width, src.height, lo.data, lo.stride, hi.data, hi.stride, &sum);
    }

    template<class A> SIMD_INLINE void TexturePerformCompensation(const View<A>& src, int shift, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8 && shift > -0xFF && shift < 0xFF);

        SimdTexturePerformCompensation(src.data, src.stride, src.width, src.height, shift, dst.data, dst.stride);
    }

    template<class A> SIMD_INLINE void Yuv444pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr)
    {
        assert(Compatible(y, u, v) && EqualSize(y, bgr) && y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdYuv444pToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
    }

    template<class A> SIMD_INLINE void Yuv420pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(y.width == bgr.width && y.height == bgr.height);
        assert(y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdYuv420pToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
    }

    template<class A> SIMD_INLINE void Yuv444pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(Compatible(y, u, v) && EqualSize(y, bgra) && y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdYuv444pToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha);
    }

    template<class A> SIMD_INLINE void Yuv420pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(y.width == bgra.width && y.height == bgra.height);
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdYuv420pToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha);
    }

    template<class A> SIMD_INLINE void Yuv444pToHsl(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hsl)
    {
        assert(Compatible(y, u, v) && EqualSize(y, hsl) && y.format == View<A>::Gray8 && hsl.format == View<A>::Hsl24);

        SimdYuv444pToHsl(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hsl.data, hsl.stride);
    }

    template<class A> SIMD_INLINE void Yuv444pToHsv(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hsv)
    {
        assert(Compatible(y, u, v) && EqualSize(y, hsv) && y.format == View<A>::Gray8 && hsv.format == View<A>::Hsv24);

        SimdYuv444pToHsv(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hsv.data, hsv.stride);
    }

    template<class A> SIMD_INLINE void Yuv444pToHue(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hue)
    {
        assert(Compatible(y, u, v, hue) && y.format == View<A>::Gray8);

        SimdYuv444pToHue(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hue.data, hue.stride);
    }

    template<class A> SIMD_INLINE void Yuv420pToHue(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hue)
    {
        assert(y.width == 2*u.width && y.height == 2*u.height && y.format == u.format);
        assert(y.width == 2*v.width && y.height == 2*v.height && y.format == v.format);
        assert(Compatible(y, hue) && y.format == View<A>::Gray8);

        SimdYuv420pToHue(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hue.data, hue.stride);
    }
}

#endif//__SimdLib_hpp__

