/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar,
*               2014-2019 Antonenka Mikhail,
*               2019-2019 Facundo Galan.
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
#include "Simd/SimdPyramid.hpp"

#ifndef __SimdLib_hpp__
#define __SimdLib_hpp__

/*! @ingroup functions
    Simd API C++ wrappers.
*/
namespace Simd
{
    /*! @ingroup info

    \fn void PrintInfo(std::ostream & os)

    \short Prints information about %Simd Library and CPU properties.

    \param [in, out] os - output stream.
    */
    SIMD_INLINE void PrintInfo(std::ostream & os)
    {
        os << "Simd Library: " << SimdVersion();
        os << "; System Sockets: " << SimdCpuInfo(SimdCpuInfoSockets);
        os << ", Cores: " << SimdCpuInfo(SimdCpuInfoCores);
        os << ", Threads: " << SimdCpuInfo(SimdCpuInfoThreads);
        os << "; Cache L1D: " << SimdCpuInfo(SimdCpuInfoCacheL1) / 1024 << " KB";
        os << ", L2: " << SimdCpuInfo(SimdCpuInfoCacheL2) / 1024 << " KB";
        os << ", L3: " << SimdCpuInfo(SimdCpuInfoCacheL3) / 1024 << " KB";
        os << "; Available SIMD:";
        os << (SimdCpuInfo(SimdCpuInfoAvx512vnni) ? " AVX-512VNNI" : "");
        os << (SimdCpuInfo(SimdCpuInfoAvx512bw) ? " AVX-512BW" : "");
        os << (SimdCpuInfo(SimdCpuInfoAvx512f) ? " AVX-512F" : "");
        os << (SimdCpuInfo(SimdCpuInfoAvx2) ? " AVX2 FMA" : "");
        os << (SimdCpuInfo(SimdCpuInfoAvx) ? " AVX" : "");
        os << (SimdCpuInfo(SimdCpuInfoSse41) ? " SSE4.1 SSSE3 SSE3" : "");
        os << (SimdCpuInfo(SimdCpuInfoSse2) ? " SSE2 SSE" : "");
        os << (SimdCpuInfo(SimdCpuInfoVmx) ? " Altivec" : "");
        os << (SimdCpuInfo(SimdCpuInfoVsx) ? " VSX" : "");
        os << (SimdCpuInfo(SimdCpuInfoNeon) ? " NEON" : "");
        os << std::endl;
    }

    /*! @ingroup correlation

        \fn void AbsDifference(const View<A> & a, const View<A> & b, View<A> & c)

        \short Gets absolute difference of two gray 8-bit images, pyxel by pixel.

        Both images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdAbsDifference.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [out] c - a destination image.
    */
    template<template<class> class A> SIMD_INLINE void AbsDifference(const View<A> & a, const View<A> & b, View<A> & c)
    {
        assert(Compatible(a, b) && Compatible(b, c) && a.format == View<A>::Gray8);

        SimdAbsDifference(a.data, a.stride, b.data, b.stride, c.data, c.stride, a.width, a.height);
    }

    /*! @ingroup correlation

        \fn void AbsDifferenceSum(const View<A>& a, const View<A>& b, uint64_t & sum)

        \short Gets sum of absolute difference of two gray 8-bit images.

        Both images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdAbsDifferenceSum.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [out] sum - the result sum of absolute difference of two images.
    */
    template<template<class> class A> SIMD_INLINE void AbsDifferenceSum(const View<A>& a, const View<A>& b, uint64_t & sum)
    {
        assert(Compatible(a, b) && a.format == View<A>::Gray8);

        SimdAbsDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
    }

    /*! @ingroup correlation

        \fn void AbsDifferenceSum(const View<A>& a, const View<A>& b, const View<A>& mask, uint8_t index, uint64_t & sum)

        \short Gets sum of absolute difference of two gray 8-bit images based on gray 8-bit mask.

        Gets the absolute difference sum for all points when mask[i] == index.
        Both images and mask must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdAbsDifferenceSumMasked.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] mask - a mask image.
        \param [in] index - a mask index.
        \param [out] sum - the result sum of absolute difference of two images.
    */
    template<template<class> class A> SIMD_INLINE void AbsDifferenceSum(const View<A>& a, const View<A>& b, const View<A>& mask, uint8_t index, uint64_t & sum)
    {
        assert(Compatible(a, b, mask) && a.format == View<A>::Gray8);

        SimdAbsDifferenceSumMasked(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, &sum);
    }

    /*! @ingroup correlation

        \fn void AbsDifferenceSums3x3(const View<A>& current, const View<A>& background, uint64_t * sums)

        \short Gets 9 sums of absolute difference of two gray 8-bit images with various relative shifts in neighborhood 3x3.

        Both images must have the same width and height. The image height and width must be equal or greater 3.
        The sums are calculated with central part (indent width = 1) of current image and with part of background image with corresponding shift.
        The shifts are lain in the range [-1, 1] for axis x and y.

        \note This function is a C++ wrapper for function ::SimdAbsDifferenceSums3x3.

        \param [in] current - a current image.
        \param [in] background - a background image.
        \param [out] sums - the pointer to buffer with result sums. Buffer size must be equal or greater 9.
    */
    template<template<class> class A> SIMD_INLINE void AbsDifferenceSums3x3(const View<A>& current, const View<A>& background, uint64_t * sums)
    {
        assert(Compatible(current, background) && current.format == View<A>::Gray8 && current.width > 2 && current.height > 2);

        SimdAbsDifferenceSums3x3(current.data, current.stride, background.data, background.stride, current.width, current.height, sums);
    }

    /*! @ingroup correlation

        \fn void AbsDifferenceSums3x3(const View<A>& current, const View<A>& background, const View<A>& mask, uint8_t index, uint64_t * sums)

        \short Gets 9 sums of absolute difference of two gray 8-bit images with various relative shifts in neighborhood 3x3 based on gray 8-bit mask.

        Gets the absolute difference sums for all points when mask[i] == index.
        Both images and mask must have the same width and height. The image height and width must be equal or greater 3.
        The sums are calculated with central part (indent width = 1) of current image and with part of background image with corresponding shift.
        The shifts are lain in the range [-1, 1] for axis x and y.

        \note This function is a C++ wrapper for function ::SimdAbsDifferenceSums3x3Masked.

        \param [in] current - a current image.
        \param [in] background - a background image.
        \param [in] mask - a mask image.
        \param [in] index - a mask index.
        \param [out] sums - the pointer to buffer with result sums. Buffer size must be equal or greater 9.
    */
    template<template<class> class A> SIMD_INLINE void AbsDifferenceSums3x3(const View<A>& current, const View<A>& background, const View<A>& mask, uint8_t index, uint64_t * sums)
    {
        assert(Compatible(current, background, mask) && current.format == View<A>::Gray8 && current.width > 2 && current.height > 2);

        SimdAbsDifferenceSums3x3Masked(current.data, current.stride, background.data, background.stride,
            mask.data, mask.stride, index, current.width, current.height, sums);
    }

    /*! @ingroup other_filter

        \fn void AbsGradientSaturatedSum(const View<A>& src, View<A>& dst)

        \short Puts to destination 8-bit gray image saturated sum of absolute gradient for every point of source 8-bit gray image.

        Both images must have the same width and height.

        For border pixels:
        \verbatim
        dst[x, y] = 0;
        \endverbatim

        For other pixels:
        \verbatim
        dx = abs(src[x + 1, y] - src[x - 1, y]);
        dy = abs(src[x, y + 1] - src[x, y - 1]);
        dst[x, y] = min(dx + dy, 255);
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdAbsGradientSaturatedSum.

        \param [in] src - a source 8-bit gray image.
        \param [out] dst - a destination 8-bit gray image.
    */
    template<template<class> class A> SIMD_INLINE void AbsGradientSaturatedSum(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8 && src.height >= 3 && src.width >= 3);

        SimdAbsGradientSaturatedSum(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup difference_estimation

        \fn void AddFeatureDifference(const View<A>& value, const View<A>& lo, const View<A>& hi, uint16_t weight, View<A>& difference)

        \short Adds feature difference to common difference sum.

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        excess = max(lo[i] - value[i], 0) + max(value[i] - hi[i], 0);
        difference[i] += (weight * excess*excess) >> 16;
        \endverbatim

        This function is used for difference estimation in algorithm of motion detection.

        \note This function is a C++ wrapper for function ::SimdAddFeatureDifference.

        \param [in] value - a current feature value.
        \param [in] lo - a feature lower bound of dynamic background.
        \param [in] hi - a feature upper bound of dynamic background.
        \param [in] weight - a current feature weight (unsigned 16-bit value).
        \param [in, out] difference - an image with total difference.
    */
    template<template<class> class A> SIMD_INLINE void AddFeatureDifference(const View<A>& value, const View<A>& lo, const View<A>& hi, uint16_t weight, View<A>& difference)
    {
        assert(Compatible(value, lo, hi, difference) && value.format == View<A>::Gray8);

        SimdAddFeatureDifference(value.data, value.stride, value.width, value.height,
            lo.data, lo.stride, hi.data, hi.stride, weight, difference.data, difference.stride);
    }

    /*! @ingroup drawing

        \fn void AlphaBlending(const View<A>& src, const View<A>& alpha, View<A>& dst)

        \short Performs alpha blending operation.

        All images must have the same width and height. Source and destination images must have the same format (8 bit per channel, for example GRAY8, UV16, BGR24 or BGRA32). Alpha must be 8-bit gray image.

        For every point:
        \verbatim
        dst[x, y, c] = (src[x, y, c]*alpha[x, y] + dst[x, y, c]*(255 - alpha[x, y]))/255;
        \endverbatim

        This function is used for image drawing.

        \note This function is a C++ wrapper for function ::SimdAlphaBlending.

        \param [in] src - a foreground image.
        \param [in] alpha - an image with alpha channel.
        \param [in, out] dst - a background image.
    */
    template<template<class> class A> SIMD_INLINE void AlphaBlending(const View<A>& src, const View<A>& alpha, View<A>& dst)
    {
        assert(Compatible(src, dst) && EqualSize(src, alpha) && alpha.format == View<A>::Gray8 && src.ChannelSize() == 1);

        SimdAlphaBlending(src.data, src.stride, src.width, src.height, src.ChannelCount(), alpha.data, alpha.stride, dst.data, dst.stride);
    }

    /*! @ingroup drawing

        \fn void AlphaBlending(const View<A>& src, uint8_t alpha, View<A>& dst)

        \short Performs alpha blending operation.

        All images must have the same width and height. Source and destination images must have the same format (8 bit per channel, for example GRAY8, UV16, BGR24 or BGRA32).

        For every point:
        \verbatim
        dst[x, y, c] = (src[x, y, c]*alpha[x, y] + dst[x, y, c]*(255 - alpha))/255;
        \endverbatim

        This function is used for image drawing.

        \note This function is a C++ wrapper for function ::SimdAlphaBlendingUniform.

        \param [in] src - a foreground image.
        \param [in] alpha - a value of alpha.
        \param [in, out] dst - a background image.
    */
    template<template<class> class A> SIMD_INLINE void AlphaBlending(const View<A>& src, uint8_t alpha, View<A>& dst)
    {
        assert(Compatible(src, dst));

        SimdAlphaBlendingUniform(src.data, src.stride, src.width, src.height, src.ChannelCount(), alpha, dst.data, dst.stride);
    }

    /*! @ingroup drawing

        \fn void AlphaFilling(View<A> & dst, const Pixel & pixel, const View<A> & alpha)

        \short Performs alpha filling operation.

        All images must have the same width and height. Destination images must have 8 bit per channel (for example GRAY8, BGR24 or BGRA32). Alpha must be 8-bit gray image.

        For every point:
        \verbatim
        dst[x, y, c] = (pixel[c]*alpha[x, y] + dst[x, y, c]*(255 - alpha[x, y]))/255;
        \endverbatim

        This function is used for image drawing.

        \note This function is a C++ wrapper for function ::SimdAlphaFilling.

        \param [in, out] dst - a background image.
        \param [in] pixel - a foreground color.
        \param [in] alpha - an image with alpha channel.
    */
    template<template<class> class A, class Pixel> SIMD_INLINE void AlphaFilling(View<A> & dst, const Pixel & pixel, const View<A> & alpha)
    {
        assert(EqualSize(dst, alpha) && alpha.format == View<A>::Gray8 && dst.ChannelSize() == 1 && dst.ChannelCount() == sizeof(Pixel));

        SimdAlphaFilling(dst.data, dst.stride, dst.width, dst.height, (uint8_t*)&pixel, sizeof(Pixel), alpha.data, alpha.stride);
    }

    /*! @ingroup drawing

        \fn void AlphaPremultiply(const View<A>& src, View<A>& dst)

        \short Performs premultiply operation.

        All images must have the same width, height and format (BGRA32).

        For every point:
        \verbatim
         dst[x, y, 0] = src[x, y, 0] * src[x, y, 3] / 255;
         dst[x, y, 1] = src[x, y, 1] * src[x, y, 3] / 255;
         dst[x, y, 2] = src[x, y, 2] * src[x, y, 3] / 255;
         dst[x, y, 3] = src[x, y, 3];
        \endverbatim

        This function is used for image drawing as a part of alpha blending operation.

        \note This function is a C++ wrapper for function ::SimdAlphaPremultiply.

        \param [in] src - an input image.
        \param [out] dst - an output premultiplyed image.
    */
    template<template<class> class A> SIMD_INLINE void AlphaPremultiply(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Bgra32);

        SimdAlphaPremultiply(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup drawing

        \fn void AlphaUnpremultiply(const View<A>& src, View<A>& dst)

        \short Performs unpremultiply operation.

        All images must have the same width, height and format (BGRA32).

        For every point:
        \verbatim
         dst[x, y, 0] = src[x, y, 0] / src[x, y, 3] * 255;
         dst[x, y, 1] = src[x, y, 1] / src[x, y, 3] * 255;
         dst[x, y, 2] = src[x, y, 2] / src[x, y, 3] * 255;
         dst[x, y, 3] = src[x, y, 3];
        \endverbatim

        This function is used for image drawing as a part of alpha blending operation.

        \note This function is a C++ wrapper for function ::SimdAlphaUnpremultiply.

        \param [in] src - an input image.
        \param [out] dst - an output unpremultiplyed image.
    */
    template<template<class> class A> SIMD_INLINE void AlphaUnpremultiply(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Bgra32);

        SimdAlphaUnpremultiply(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup background

        \fn void BackgroundGrowRangeSlow(const View<A>& value, View<A>& lo, View<A>& hi)

        \short Performs background update (initial grow, slow mode).

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        lo[i] -= value[i] < lo[i] ? 1 : 0;
        hi[i] += value[i] > hi[i] ? 1 : 0;
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdBackgroundGrowRangeSlow.

        \param [in] value - a current feature value.
        \param [in, out] lo - a feature lower bound of dynamic background.
        \param [in, out] hi - a feature upper bound of dynamic background.
    */
    template<template<class> class A> SIMD_INLINE void BackgroundGrowRangeSlow(const View<A>& value, View<A>& lo, View<A>& hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View<A>::Gray8);

        SimdBackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    /*! @ingroup background

        \fn void BackgroundGrowRangeFast(const View<A>& value, View<A>& lo, View<A>& hi)

        \short Performs background update (initial grow, fast mode).

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        lo[i] = value[i] < lo[i] ? value[i] : lo[i];
        hi[i] = value[i] > hi[i] ? value[i] : hi[i];
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdBackgroundGrowRangeFast.

        \param [in] value - a current feature value.
        \param [in, out] lo - a feature lower bound of dynamic background.
        \param [in, out] hi - a feature upper bound of dynamic background.
    */
    template<template<class> class A> SIMD_INLINE void BackgroundGrowRangeFast(const View<A>& value, View<A>& lo, View<A>& hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View<A>::Gray8);

        SimdBackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    /*! @ingroup background

        \fn void BackgroundIncrementCount(const View<A>& value, const View<A>& loValue, const View<A>& hiValue, View<A>& loCount, View<A>& hiCount)

        \short Performs collection of background statistic.

        All images must have the same width, height and format (8-bit gray).

        Updates background statistic counters for every point:
        \verbatim
        loCount[i] += (value[i] < loValue[i] && loCount[i] < 255) ? 1 : 0;
        hiCount[i] += (value[i] > hiValue[i] && hiCount[i] < 255) ? 1 : 0;
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdBackgroundIncrementCount.

        \param [in] value - a current feature value.
        \param [in] loValue - a value of feature lower bound of dynamic background.
        \param [in] hiValue - a value of feature upper bound of dynamic background.
        \param [in, out] loCount - a count of feature lower bound of dynamic background.
        \param [in, out] hiCount - a count of feature upper bound of dynamic background.
    */
    template<template<class> class A> SIMD_INLINE void BackgroundIncrementCount(const View<A>& value, const View<A>& loValue, const View<A>& hiValue, View<A>& loCount, View<A>& hiCount)
    {
        assert(Compatible(value, loValue, hiValue, loCount, hiCount) && value.format == View<A>::Gray8);

        SimdBackgroundIncrementCount(value.data, value.stride, value.width, value.height,
            loValue.data, loValue.stride, hiValue.data, hiValue.stride,
            loCount.data, loCount.stride, hiCount.data, hiCount.stride);
    }

    /*! @ingroup background

        \fn void BackgroundAdjustRange(View<A>& loCount, View<A>& loValue, View<A>& hiCount, View<A>& hiValue, uint8_t threshold)

        \short Performs adjustment of background range.

        All images must have the same width, height and format (8-bit gray).

        Adjusts background range for every point:
        \verbatim
        loValue[i] -= (loCount[i] > threshold && loValue[i] > 0) ? 1 : 0;
        loValue[i] += (loCount[i] < threshold && loValue[i] < 255) ? 1 : 0;
        loCount[i] = 0;
        hiValue[i] += (hiCount[i] > threshold && hiValue[i] < 255) ? 1 : 0;
        hiValue[i] -= (hiCount[i] < threshold && hiValue[i] > 0) ? 1 : 0;
        hiCount[i] = 0;
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdBackgroundAdjustRange.

        \param [in, out] loCount - a count of feature lower bound of dynamic background.
        \param [in, out] hiCount - a count of feature upper bound of dynamic background.
        \param [in, out] loValue - a value of feature lower bound of dynamic background.
        \param [in, out] hiValue - a value of feature upper bound of dynamic background.
        \param [in] threshold - a count threshold.
    */
    template<template<class> class A> SIMD_INLINE void BackgroundAdjustRange(View<A>& loCount, View<A>& loValue, View<A>& hiCount, View<A>& hiValue, uint8_t threshold)
    {
        assert(Compatible(loValue, hiValue, loCount, hiCount) && loValue.format == View<A>::Gray8);

        SimdBackgroundAdjustRange(loCount.data, loCount.stride, loCount.width, loCount.height,
            loValue.data, loValue.stride, hiCount.data, hiCount.stride, hiValue.data, hiValue.stride, threshold);
    }

    /*! @ingroup background

        \fn void BackgroundAdjustRange(View<A>& loCount, View<A>& loValue, View<A>& hiCount, View<A>& hiValue, uint8_t threshold, const View<A>& mask)

        \short Performs adjustment of background range with using adjust range mask.

        All images must have the same width, height and format (8-bit gray).

        Adjusts background range for every point:
        \verbatim
        if(mask[i])
        {
            loValue[i] -= (loCount[i] > threshold && loValue[i] > 0) ? 1 : 0;
            loValue[i] += (loCount[i] < threshold && loValue[i] < 255) ? 1 : 0;
            loCount[i] = 0;
            hiValue[i] += (hiCount[i] > threshold && hiValue[i] < 255) ? 1 : 0;
            hiValue[i] -= (hiCount[i] < threshold && hiValue[i] > 0) ? 1 : 0;
            hiCount[i] = 0;
        }
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdBackgroundAdjustRangeMasked.

        \param [in, out] loCount - a count of feature lower bound of dynamic background.
        \param [in, out] hiCount - a count of feature upper bound of dynamic background.
        \param [in, out] loValue - a value of feature lower bound of dynamic background.
        \param [in, out] hiValue - a value of feature upper bound of dynamic background.
        \param [in] threshold - a count threshold.
        \param [in] mask - an adjust range mask.
    */
    template<template<class> class A> SIMD_INLINE void BackgroundAdjustRange(View<A>& loCount, View<A>& loValue, View<A>& hiCount, View<A>& hiValue, uint8_t threshold, const View<A>& mask)
    {
        assert(Compatible(loValue, hiValue, loCount, hiCount, mask) && loValue.format == View<A>::Gray8);

        SimdBackgroundAdjustRangeMasked(loCount.data, loCount.stride, loCount.width, loCount.height,
            loValue.data, loValue.stride, hiCount.data, hiCount.stride, hiValue.data, hiValue.stride,
            threshold, mask.data, mask.stride);
    }

    /*! @ingroup background

        \fn void BackgroundShiftRange(const View<A>& value, View<A>& lo, View<A>& hi)

        \short Shifts background range.

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        if (value[i] > hi[i])
        {
            lo[i] = min(lo[i] + value[i] - hi[i], 255);
            hi[i] = value[i];
        }
        if (lo[i] > value[i])
        {
            lo[i] = value[i];
            hi[i] = max(hi[i] - lo[i] + value[i], 0);
        }
        \endverbatim

        This function is used for fast background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdBackgroundShiftRange.

        \param [in] value - a current feature value.
        \param [in, out] lo - a feature lower bound of dynamic background.
        \param [in, out] hi - a feature upper bound of dynamic background.
    */
    template<template<class> class A> SIMD_INLINE void BackgroundShiftRange(const View<A>& value, View<A>& lo, View<A>& hi)
    {
        assert(Compatible(value, lo, hi) && value.format == View<A>::Gray8);

        SimdBackgroundShiftRange(value.data, value.stride, value.width, value.height, lo.data, lo.stride, hi.data, hi.stride);
    }

    /*! @ingroup background

        \fn void BackgroundShiftRange(const View<A>& value, View<A>& lo, View<A>& hi, const View<A>& mask);

        \short Shifts background range with using shift range mask.

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        if(mask[i])
        {
            if (value[i] > hi[i])
            {
                lo[i] = min(lo[i] + value[i] - hi[i], 255);
                hi[i] = value[i];
            }
            if (lo[i] > value[i])
            {
                lo[i] = value[i];
                hi[i] = max(hi[i] - lo[i] + value[i], 0);
            }
        }
        \endverbatim

        This function is used for fast background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdBackgroundShiftRangeMasked.

        \param [in] value - a current feature value.
        \param [in, out] lo - a feature lower bound of dynamic background.
        \param [in, out] hi - a feature upper bound of dynamic background.
        \param [in] mask - a shift range mask.
    */
    template<template<class> class A> SIMD_INLINE void BackgroundShiftRange(const View<A>& value, View<A>& lo, View<A>& hi, const View<A>& mask)
    {
        assert(Compatible(value, lo, hi, mask) && value.format == View<A>::Gray8);

        SimdBackgroundShiftRangeMasked(value.data, value.stride, value.width, value.height,
            lo.data, lo.stride, hi.data, hi.stride, mask.data, mask.stride);
    }

    /*! @ingroup background

        \fn void BackgroundInitMask(const View<A>& src, uint8_t index, uint8_t value, View<A>& dst);

        \short Creates background update mask.

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        if(mask[i] == index)
            dst[i] = value;
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdBackgroundInitMask.

        \param [in] src - an input mask image.
        \param [in] index - a mask index into input mask.
        \param [in] value - a value to fill the output mask.
        \param [out] dst - an output mask image.
    */
    template<template<class> class A> SIMD_INLINE void BackgroundInitMask(const View<A>& src, uint8_t index, uint8_t value, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdBackgroundInitMask(src.data, src.stride, src.width, src.height, index, value, dst.data, dst.stride);
    }

    /*! @ingroup base64

        \fn std::string Base64Decode(const std::string& src)

        \short Decode string to Base64.

        \note This function is a C++ wrapper for function ::SimdBase64Decode.

        \param [in] src - an input Base64 encoded string.
        \return the output decoded string.
    */
    SIMD_INLINE std::string Base64Decode(const std::string& src)
    {
        size_t dstSize = src.length() / 4 * 3;
        std::string dst(dstSize, 0);
        SimdBase64Decode((uint8_t*)src.c_str(), src.length(), (uint8_t*)dst.c_str(), &dstSize);
        dst.resize(dstSize);
        return dst;
    }

    /*! @ingroup base64

        \fn std::string Base64Encode(const std::string& src)

        \short Encode string to Base64.

        \note This function is a C++ wrapper for function ::SimdBase64Encode.

        \param [in] src - an input original string.
        \return the output Base64 encoded string.
    */
    SIMD_INLINE std::string Base64Encode(const std::string& src)
    {
        std::string dst((src.length() + 2) / 3 * 4, 0);
        SimdBase64Encode((uint8_t*)src.c_str(), src.length(), (uint8_t*)dst.c_str());
        return dst;
    }

    /*! @ingroup bayer_conversion

        \fn void BayerToBgr(const View<A>& bayer, View<A>& bgr);

        \short Converts 8-bit Bayer image to 24-bit BGR.

        All images must have the same width and height. The width and the height must be even.

        \note This function is a C++ wrapper for function ::SimdBayerToBgr.

        \param [in] bayer - an input 8-bit Bayer image.
        \param [out] bgr - an output 24-bit BGR image.
    */
    template<template<class> class A> SIMD_INLINE void BayerToBgr(const View<A>& bayer, View<A>& bgr)
    {
        assert(EqualSize(bgr, bayer) && bgr.format == View<A>::Bgr24);
        assert(bayer.format >= View<A>::BayerGrbg && bayer.format <= View<A>::BayerBggr);
        assert((bayer.width % 2 == 0) && (bayer.height % 2 == 0));

        SimdBayerToBgr(bayer.data, bayer.width, bayer.height, bayer.stride, (SimdPixelFormatType)bayer.format, bgr.data, bgr.stride);
    }

    /*! @ingroup bayer_conversion

        \fn void BayerToBgra(const View<A>& bayer, View<A>& bgra, uint8_t alpha = 0xFF);

        \short Converts 8-bit Bayer image to 32-bit BGRA.

        All images must have the same width and height. The width and the height must be even.

        \note This function is a C++ wrapper for function ::SimdBayerToBgra.

        \param [in] bayer - an input 8-bit Bayer image.
        \param [out] bgra - an output 32-bit BGRA image.
        \param [in] alpha - a value of alpha channel. It is equal to 256 by default.
    */
    template<template<class> class A> SIMD_INLINE void BayerToBgra(const View<A>& bayer, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(bgra, bayer) && bgra.format == View<A>::Bgra32);
        assert(bayer.format >= View<A>::BayerGrbg && bayer.format <= View<A>::BayerBggr);
        assert((bayer.width % 2 == 0) && (bayer.height % 2 == 0));

        SimdBayerToBgra(bayer.data, bayer.width, bayer.height, bayer.stride, (SimdPixelFormatType)bayer.format, bgra.data, bgra.stride, alpha);
    }

    /*! @ingroup bgra_conversion

        \fn void BgraToBayer(const View<A>& bgra, View<A>& bayer)

        \short Converts 32-bit BGRA image to 8-bit Bayer image.

        All images must have the same width and height. The width and the height must be even.

        \note This function is a C++ wrapper for function ::SimdBgraToBayer.

        \param [in] bgra - an input 32-bit BGRA image.
        \param [out] bayer - an output 8-bit Bayer image.
    */
    template<template<class> class A> SIMD_INLINE void BgraToBayer(const View<A>& bgra, View<A>& bayer)
    {
        assert(EqualSize(bgra, bayer) && bgra.format == View<A>::Bgra32);
        assert(bayer.format >= View<A>::BayerGrbg && bayer.format <= View<A>::BayerBggr);
        assert((bayer.width % 2 == 0) && (bayer.height % 2 == 0));

        SimdBgraToBayer(bgra.data, bgra.width, bgra.height, bgra.stride, bayer.data, bayer.stride, (SimdPixelFormatType)bayer.format);
    }

    /*! @ingroup bgra_conversion

        \fn void BgraToBgr(const View<A>& bgra, View<A>& bgr)

        \short Converts 32-bit BGRA image to 24-bit BGR image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgraToBgr.

        \param [in] bgra - an input 32-bit BGRA image.
        \param [out] bgr - an output 24-bit BGR image.
    */
    template<template<class> class A> SIMD_INLINE void BgraToBgr(const View<A>& bgra, View<A>& bgr)
    {
        assert(EqualSize(bgra, bgr) && bgra.format == View<A>::Bgra32 && bgr.format == View<A>::Bgr24);

        SimdBgraToBgr(bgra.data, bgra.width, bgra.height, bgra.stride, bgr.data, bgr.stride);
    }

    /*! @ingroup bgra_conversion

        \fn void BgraToGray(const View<A>& bgra, View<A>& gray)

        \short Converts 32-bit BGRA image to 8-bit gray image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgraToGray.

        \param [in] bgra - an input 32-bit BGRA image.
        \param [out] gray - an output 8-bit gray image.
    */
    template<template<class> class A> SIMD_INLINE void BgraToGray(const View<A>& bgra, View<A>& gray)
    {
        assert(EqualSize(bgra, gray) && bgra.format == View<A>::Bgra32 && gray.format == View<A>::Gray8);

        SimdBgraToGray(bgra.data, bgra.width, bgra.height, bgra.stride, gray.data, gray.stride);
    }

    /*! @ingroup bgra_conversion

        \fn void BgraToRgb(const View<A>& bgra, View<A>& rgb)

        \short Converts 32-bit BGRA image to 24-bit RGB image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgraToRgb.

        \param [in] bgra - an input 32-bit BGRA image.
        \param [out] rgb - an output 24-bit RGB image.
    */
    template<template<class> class A> SIMD_INLINE void BgraToRgb(const View<A>& bgra, View<A>& rgb)
    {
        assert(EqualSize(bgra, rgb) && bgra.format == View<A>::Bgra32 && rgb.format == View<A>::Rgb24);

        SimdBgraToRgb(bgra.data, bgra.width, bgra.height, bgra.stride, rgb.data, rgb.stride);
    }

    /*! @ingroup bgra_conversion

        \fn void BgraToRgba(const View<A>& bgra, View<A>& rgba)

        \short Converts 32-bit BGRA image to 32-bit RGBA image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgraToRgba.

        \param [in] bgra - an input 32-bit BGRA image.
        \param [out] rgba - an output 32-bit RGBA image.
    */
    template<template<class> class A> SIMD_INLINE void BgraToRgba(const View<A>& bgra, View<A>& rgba)
    {
        assert(EqualSize(bgra, rgba) && bgra.format == View<A>::Bgra32 && rgba.format == View<A>::Rgba32);

        SimdBgraToRgba(bgra.data, bgra.width, bgra.height, bgra.stride, rgba.data, rgba.stride);
    }

    /*! @ingroup bgra_conversion

        \fn void BgraToYuv420p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v)

        \short Converts 32-bit BGRA image to YUV420P.

        The input BGRA and output Y images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function is a C++ wrapper for function ::SimdBgraToYuv420p.

        \param [in] bgra - an input 32-bit BGRA image.
        \param [out] y - an output 8-bit image with Y color plane.
        \param [out] u - an output 8-bit image with U color plane.
        \param [out] v - an output 8-bit image with V color plane.
    */
    template<template<class> class A> SIMD_INLINE void BgraToYuv420p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(y.width == 2 * u.width && y.height == 2 * u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == 2 * v.height && y.format == v.format);
        assert(y.width == bgra.width && y.height == bgra.height);
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdBgraToYuv420p(bgra.data, bgra.width, bgra.height, bgra.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    /*! @ingroup bgra_conversion

        \fn void BgraToYuv422p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v)

        \short Converts 32-bit BGRA image to YUV422P.

        The input BGRA and output Y images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function is a C++ wrapper for function ::SimdBgraToYuv422p.

        \param [in] bgra - an input 32-bit BGRA image.
        \param [out] y - an output 8-bit image with Y color plane.
        \param [out] u - an output 8-bit image with U color plane.
        \param [out] v - an output 8-bit image with V color plane.
    */
    template<template<class> class A> SIMD_INLINE void BgraToYuv422p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(y.width == 2 * u.width && y.height == u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == v.height && y.format == v.format);
        assert(y.width == bgra.width && y.height == bgra.height);
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdBgraToYuv422p(bgra.data, bgra.width, bgra.height, bgra.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    /*! @ingroup bgra_conversion

        \fn void BgraToYuv444p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v)

        \short Converts 32-bit BGRA image to YUV444P.

        The input BGRA and output Y, U and V images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgraToYuv444p.

        \param [in] bgra - an input 32-bit BGRA image.
        \param [out] y - an output 8-bit image with Y color plane.
        \param [out] u - an output 8-bit image with U color plane.
        \param [out] v - an output 8-bit image with V color plane.
    */
    template<template<class> class A> SIMD_INLINE void BgraToYuv444p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(EqualSize(bgra, y) && Compatible(y, u, v));
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdBgraToYuv444p(bgra.data, bgra.width, bgra.height, bgra.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    /*! @ingroup bgra_conversion

        \fn void BgraToYuva420p(const View<A> & bgra, View<A> & y, View<A> & u, View<A> & v, View<A> & a)

        \short Converts 32-bit BGRA image to YUVA420P.

        The input BGRA and output Y and A images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function is a C++ wrapper for function ::SimdBgraToYuva420p.

        \param [in] bgra - an input 32-bit BGRA image.
        \param [out] y - an output 8-bit image with Y color plane.
        \param [out] u - an output 8-bit image with U color plane.
        \param [out] v - an output 8-bit image with V color plane.
        \param [out] a - an output 8-bit image with alpha plane.
    */
    template<template<class> class A> SIMD_INLINE void BgraToYuva420p(const View<A> & bgra, View<A> & y, View<A> & u, View<A> & v, View<A> & a)
    {
        assert(y.width == 2 * u.width && y.height == 2 * u.height && y.format == u.format);
        assert(Compatible(y, a) && Compatible(u, v) && EqualSize(y, bgra));
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdBgraToYuva420p(bgra.data, bgra.stride, bgra.width, bgra.height, y.data, y.stride, u.data, u.stride, v.data, v.stride, a.data, a.stride);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToBayer(const View<A>& bgr, View<A>& bayer)

        \short Converts 24-bit BGR image to 8-bit Bayer image.

        All images must have the same width and height. The width and the height must be even.

        \note This function is a C++ wrapper for function ::SimdBgrToBayer.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] bayer - an output 8-bit Bayer image.
    */
    template<template<class> class A> SIMD_INLINE void BgrToBayer(const View<A>& bgr, View<A>& bayer)
    {
        assert(EqualSize(bgr, bayer) && bgr.format == View<A>::Bgr24);
        assert(bayer.format >= View<A>::BayerGrbg && bayer.format <= View<A>::BayerBggr);
        assert((bayer.width % 2 == 0) && (bayer.height % 2 == 0));

        SimdBgrToBayer(bgr.data, bgr.width, bgr.height, bgr.stride, bayer.data, bayer.stride, (SimdPixelFormatType)bayer.format);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToBgra(const View<A>& bgr, View<A>& bgra, uint8_t alpha = 0xFF)

        \short Converts 24-bit BGR image to 32-bit BGRA image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgrToBgra.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] bgra - an output 32-bit BGRA image.
        \param [in] alpha - a value of alpha channel. It is equal to 256 by default.
    */
    template<template<class> class A> SIMD_INLINE void BgrToBgra(const View<A>& bgr, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(bgr, bgra) && bgra.format == View<A>::Bgra32 && bgr.format == View<A>::Bgr24);

        SimdBgrToBgra(bgr.data, bgr.width, bgr.height, bgr.stride, bgra.data, bgra.stride, alpha);
    }

    /*! @ingroup other_conversion

        \fn void Bgr48pToBgra32(const View<A>& blue, const View<A>& green, const View<A>& red, View<A>& bgra, uint8_t alpha = 0xFF)

        \short Converts 48-bit planar BGR image to 32-bit BGRA image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgr48pToBgra32.

        \param [in] blue - an input 16-bit image with blue color plane.
        \param [in] green - an input 16-bit image with green color plane.
        \param [in] red - an input 16-bit image with red color plane.
        \param [out] bgra - an output 32-bit BGRA image.
        \param [in] alpha - a value of alpha channel. It is equal to 256 by default.
    */
    template<template<class> class A> SIMD_INLINE void Bgr48pToBgra32(const View<A>& blue, const View<A>& green, const View<A>& red, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(Compatible(blue, green, red) && EqualSize(blue, bgra) && blue.format == View<A>::Int16 && bgra.format == View<A>::Bgra32);

        SimdBgr48pToBgra32(blue.data, blue.stride, blue.width, blue.height, green.data, green.stride, red.data, red.stride, bgra.data, bgra.stride, alpha);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToGray(const View<A>& bgr, View<A>& gray)

        \short Converts 24-bit BGR image to 8-bit gray image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgrToGray.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] gray - an output 8-bit gray image.
    */
    template<template<class> class A> SIMD_INLINE void BgrToGray(const View<A>& bgr, View<A>& gray)
    {
        assert(EqualSize(bgr, gray) && bgr.format == View<A>::Bgr24 && gray.format == View<A>::Gray8);

        SimdBgrToGray(bgr.data, bgr.width, bgr.height, bgr.stride, gray.data, gray.stride);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToHsl(const View<A> & bgr, View<A> & hsl)

        \short Converts 24-bit BGR image to 24-bit HSL(Hue, Saturation, Lightness) image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgrToHsl.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] hsl - an output 24-bit HSL image.
    */
    template<template<class> class A> SIMD_INLINE void BgrToHsl(const View<A> & bgr, View<A> & hsl)
    {
        assert(EqualSize(bgr, hsl) && bgr.format == View<A>::Bgr24 && hsl.format == View<A>::Hsl24);

        SimdBgrToHsl(bgr.data, bgr.width, bgr.height, bgr.stride, hsl.data, hsl.stride);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToHsv(const View<A> & bgr, View<A> & hsv)

        \short Converts 24-bit BGR image to 24-bit HSV(Hue, Saturation, Value) image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgrToHsv.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] hsv - an output 24-bit HSV image.
    */
    template<template<class> class A> SIMD_INLINE void BgrToHsv(const View<A> & bgr, View<A> & hsv)
    {
        assert(EqualSize(bgr, hsv) && bgr.format == View<A>::Bgr24 && hsv.format == View<A>::Hsv24);

        SimdBgrToHsv(bgr.data, bgr.width, bgr.height, bgr.stride, hsv.data, hsv.stride);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToRgb(const View<A> & bgr, View<A> & rgb)

        \short Converts 24-bit BGR image to 24-bit RGB image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgrToRgb.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] rgb - an output 24-bit RGB image.
    */
    template<template<class> class A> SIMD_INLINE void BgrToRgb(const View<A> & bgr, View<A> & rgb)
    {
        assert(EqualSize(bgr, rgb) && bgr.format == View<A>::Bgr24 && rgb.format == View<A>::Rgb24);

        SimdBgrToRgb(bgr.data, bgr.width, bgr.height, bgr.stride, rgb.data, rgb.stride);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToRgba(const View<A>& bgr, View<A>& rgba, uint8_t alpha = 0xFF)

        \short Converts 24-bit BGR image to 32-bit RGBA image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdRgbToBgra.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] rgba - an output 32-bit RGBA image.
        \param [in] alpha - a value of alpha channel. It is equal to 256 by default.
    */
    template<template<class> class A> SIMD_INLINE void BgrToRgba(const View<A>& bgr, View<A>& rgba, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(bgr, rgba) && rgba.format == View<A>::Rgba32 && bgr.format == View<A>::Bgr24);

        SimdRgbToBgra(bgr.data, bgr.width, bgr.height, bgr.stride, rgba.data, rgba.stride, alpha);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToYuv420p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v)

        \short Converts 24-bit BGR image to YUV420P.

        The input BGR and output Y images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function is a C++ wrapper for function ::SimdBgrToYuv420p.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] y - an output 8-bit image with Y color plane.
        \param [out] u - an output 8-bit image with U color plane.
        \param [out] v - an output 8-bit image with V color plane.
    */
    template<template<class> class A> SIMD_INLINE void BgrToYuv420p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(y.width == 2 * u.width && y.height == 2 * u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == 2 * v.height && y.format == v.format);
        assert(y.width == bgr.width && y.height == bgr.height);
        assert(y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdBgrToYuv420p(bgr.data, bgr.width, bgr.height, bgr.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToYuv422p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v)

        \short Converts 24-bit BGR image to YUV422P.

        The input BGR and output Y images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function is a C++ wrapper for function ::SimdBgrToYuv422p.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] y - an output 8-bit image with Y color plane.
        \param [out] u - an output 8-bit image with U color plane.
        \param [out] v - an output 8-bit image with V color plane.
    */
    template<template<class> class A> SIMD_INLINE void BgrToYuv422p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(y.width == 2 * u.width && y.height == u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == v.height && y.format == v.format);
        assert(y.width == bgr.width && y.height == bgr.height);
        assert(y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdBgrToYuv422p(bgr.data, bgr.width, bgr.height, bgr.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    /*! @ingroup bgr_conversion

        \fn void BgrToYuv444p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v)

        \short Converts 24-bit BGR image to YUV444P.

        The input BGR and output Y, U and V images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgrToYuv444p.

        \param [in] bgr - an input 24-bit BGR image.
        \param [out] y - an output 8-bit image with Y color plane.
        \param [out] u - an output 8-bit image with U color plane.
        \param [out] v - an output 8-bit image with V color plane.
    */
    template<template<class> class A> SIMD_INLINE void BgrToYuv444p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(EqualSize(bgr, y) && Compatible(y, u, v));
        assert(y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdBgrToYuv444p(bgr.data, bgr.width, bgr.height, bgr.stride, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    /*! @ingroup binarization

        \fn void Binarization(const View<A>& src, uint8_t value, uint8_t positive, uint8_t negative, View<A>& dst, SimdCompareType compareType)

        \short Performs binarization of 8-bit gray image.

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \verbatim
        dst[i] = compare(src[i], value) ? positive : negative;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function is a C++ wrapper for function ::SimdBinarization.

        \param [in] src - an input 8-bit gray image (first value for compare operation).
        \param [in] value - a second value for compare operation.
        \param [in] positive - a destination value if comparison operation has a positive result.
        \param [in] negative - a destination value if comparison operation has a negative result.
        \param [out] dst - an output 8-bit gray binarized image.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
    */
    template<template<class> class A> SIMD_INLINE void Binarization(const View<A>& src, uint8_t value, uint8_t positive, uint8_t negative, View<A>& dst, SimdCompareType compareType)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdBinarization(src.data, src.stride, src.width, src.height, value, positive, negative, dst.data, dst.stride, compareType);
    }

    /*! @ingroup binarization

        \fn void AveragingBinarization(const View<A>& src, uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, View<A>& dst, SimdCompareType compareType)

        \short Performs averaging binarization of 8-bit gray image.

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \verbatim
        sum = 0; area = 0;
        for(dy = -neighborhood; dy <= neighborhood; ++dy)
        {
            for(dx = -neighborhood; dx <= neighborhood; ++dx)
            {
                if(x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height)
                {
                    area++;
                    if(compare(src[x + dx, x + dy], value))
                        sum++;
                }
            }
        }
        dst[x, y] = sum*255 > area*threshold ? positive : negative;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function is a C++ wrapper for function ::SimdAveragingBinarization.

        \param [in] src - an input 8-bit gray image (first value for compare operation).
        \param [in] value - a second value for compare operation.
        \param [in] neighborhood - an averaging neighborhood.
        \param [in] threshold - a threshold value for binarization. It can range from 0 to 255.
        \param [in] positive - a destination value if for neighborhood of this point number of positive comparison is greater then threshold.
        \param [in] negative - a destination value if for neighborhood of this point number of positive comparison is lesser or equal then threshold.
        \param [out] dst - an output 8-bit gray binarized image.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
    */
    template<template<class> class A> SIMD_INLINE void AveragingBinarization(const View<A>& src, uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, View<A>& dst, SimdCompareType compareType)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdAveragingBinarization(src.data, src.stride, src.width, src.height, value,
            neighborhood, threshold, positive, negative, dst.data, dst.stride, compareType);
    }

    /*! @ingroup binarization

        \fn void AveragingBinarizationV2(const View<A>& src, size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, View<A>& dst)

        \short Performs averaging binarization of 8-bit gray image.

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \verbatim
        sum = 0; area = 0;
        for(dy = -neighborhood; dy <= neighborhood; ++dy)
        {
            for(dx = -neighborhood; dx <= neighborhood; ++dx)
            {
                if(x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height)
                {
                    area++;
                    sum += src[x + dx, x + dy];
                }
            }
        }
        dst[x, y] = (src[x, y] + shift)*area > sum ? positive : negative;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdAveragingBinarizationV2.

        \param [in] src - an input 8-bit gray image (first value for compare operation).
        \param [in] neighborhood - an averaging neighborhood.
        \param [in] shift - a shift value for binarization. It can range from -255 to 255.
        \param [in] positive - a destination value for positive value of condition (seen before).
        \param [in] negative - a destination value for negative value of condition (seen before).
        \param [out] dst - an output 8-bit gray binarized image.
    */
    template<template<class> class A> SIMD_INLINE void AveragingBinarizationV2(const View<A>& src, size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdAveragingBinarizationV2(src.data, src.stride, src.width, src.height, neighborhood, shift, positive, negative, dst.data, dst.stride);
    }

    /*! @ingroup conditional

        \fn void ConditionalCount8u(const View<A> & src, uint8_t value, SimdCompareType compareType, uint32_t & count)

        \short Calculates number of points satisfying certain condition for 8-bit gray image.

        For every point:
        \verbatim
        if(compare(src[i], value))
            count++;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function is a C++ wrapper for function ::SimdConditionalCount8u.

        \param [in] src - an input 8-bit gray image (first value for compare operation).
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] count - a pointer to result unsigned 32-bit value.
    */
    template<template<class> class A> SIMD_INLINE void ConditionalCount8u(const View<A> & src, uint8_t value, SimdCompareType compareType, uint32_t & count)
    {
        assert(src.format == View<A>::Gray8);

        SimdConditionalCount8u(src.data, src.stride, src.width, src.height, value, compareType, &count);
    }

    /*! @ingroup conditional

        \fn void ConditionalCount16i(const View<A> & src, int16_t value, SimdCompareType compareType, uint32_t & count)

        \short Calculates number of points satisfying certain condition for 16-bit signed integer image.

        For every point:
        \verbatim
        if(compare(src[i], value))
            count++;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function is a C++ wrapper for function ::SimdConditionalCount16i.

        \param [in] src - an input 16-bit signed integer image (first value for compare operation).
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] count - a pointer to result unsigned 32-bit value.
    */
    template<template<class> class A> SIMD_INLINE void ConditionalCount16i(const View<A> & src, int16_t value, SimdCompareType compareType, uint32_t & count)
    {
        assert(src.format == View<A>::Int16);

        SimdConditionalCount16i(src.data, src.stride, src.width, src.height, value, compareType, &count);
    }

    /*! @ingroup conditional

        \fn void ConditionalSum(const View<A> & src, const View<A> & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)

        \short Calculates sum of image points when mask points satisfying certain condition.

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \verbatim
        if(compare(mask[i], value))
            sum += src[i];
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function is a C++ wrapper for function ::SimdConditionalSum.

        \param [in] src - an input 8-bit gray image.
        \param [in] mask - a 8-bit gray mask (first value for compare operation).
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] sum - a pointer to result unsigned 64-bit value.
    */
    template<template<class> class A> SIMD_INLINE void ConditionalSum(const View<A> & src, const View<A> & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)
    {
        assert(Compatible(src, mask) && src.format == View<A>::Gray8);

        SimdConditionalSum(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
    }

    /*! @ingroup conditional

        \fn void ConditionalSquareSum(const View<A>& src, const View<A>& mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)

        \short Calculates sum of squared image points when mask points satisfying certain condition.

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \verbatim
        if(compare(mask[i], value))
            sum += src[i]*src[i];
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function is a C++ wrapper for function ::SimdConditionalSquareSum.

        \param [in] src - an input 8-bit gray image.
        \param [in] mask - a 8-bit gray mask (first value for compare operation).
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] sum - a pointer to result unsigned 64-bit value.
    */
    template<template<class> class A> SIMD_INLINE void ConditionalSquareSum(const View<A>& src, const View<A>& mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)
    {
        assert(Compatible(src, mask) && src.format == View<A>::Gray8);

        SimdConditionalSquareSum(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
    }

    /*! @ingroup conditional

        \fn void ConditionalSquareGradientSum(const View<A>& src, const View<A>& mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)

        \short Calculates sum of squared gradient of image points when mask points satisfying certain condition.

        All images must have 8-bit gray format and must have the same width and height. The image height and width must be equal or greater 3.

        For every point except border:
        \verbatim
        if(compare(mask[x, y], value))
        {
            dx = src[x + 1, y] - src[x - 1, y];
            dy = src[x, y + 1] - src[x, y - 1];
            sum += dx*dx + dy*dy;
        }
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function is a C++ wrapper for function ::SimdConditionalSquareGradientSum.

        \param [in] src - an input 8-bit gray image.
        \param [in] mask - a 8-bit gray mask (first value for compare operation).
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] sum - a pointer to result unsigned 64-bit value.
    */
    template<template<class> class A> SIMD_INLINE void ConditionalSquareGradientSum(const View<A>& src, const View<A>& mask, uint8_t value, SimdCompareType compareType, uint64_t & sum)
    {
        assert(Compatible(src, mask) && src.format == View<A>::Gray8 && src.width >= 3 && src.height >= 3);

        SimdConditionalSquareGradientSum(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, &sum);
    }

    /*! @ingroup conditional

        \fn void ConditionalFill(const View<A> & src, uint8_t threshold, SimdCompareType compareType, uint8_t value, View<A> & dst);

        \short Fills pixels of 8-bit gray image by given value if corresponding pixels of input 8-bit gray image satisfy certain condition.

        All images must have the same width and height.

        For every point:
        \verbatim
        if(compare(src[i], threshold))
            dst[i] = value;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function is a C++ wrapper for function ::SimdConditionalFill

        \param [in] src - an input 8-bit gray image.
        \param [in] threshold - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [in] value - a value for fill operation.
        \param [in, out] dst - an output 8-bit gray image.
        */
    template<template<class> class A> SIMD_INLINE void ConditionalFill(const View<A> & src, uint8_t threshold, SimdCompareType compareType, uint8_t value, View<A> & dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdConditionalFill(src.data, src.stride, src.width, src.height, threshold, compareType, value, dst.data, dst.stride);
    }

    /*! @ingroup copying

        \fn void Copy(const View<A> & src, View<B> & dst)

        \short Copies pixels data of image from source to destination.

        All images must have the same width, height and format.

        \note This function is a C++ wrapper for function ::SimdCopy.

        \param [in] src - a source image.
        \param [out] dst - a destination image.
    */
    template<template<class> class A, template<class> class B> SIMD_INLINE void Copy(const View<A> & src, View<B> & dst)
    {
        assert(Compatible(src, dst));

        if (src.format)
        {
            SimdCopy(src.data, src.stride, src.width, src.height, src.PixelSize(), dst.data, dst.stride);
        }
    }

    /*! @ingroup copying

        \fn void CopyFrame(const View<A>& src, const Rectangle<ptrdiff_t> & frame, View<A>& dst)

        \short Copies pixels data of image from source to destination except for the portion bounded frame.

        All images must have the same width, height and format.

        \note This function is a C++ wrapper for function ::SimdCopyFrame.

        \param [in] src - a source image.
        \param [in] frame - a frame rectangle.
        \param [out] dst - a destination image.
    */
    template<template<class> class A> SIMD_INLINE void CopyFrame(const View<A>& src, const Rectangle<ptrdiff_t> & frame, View<A>& dst)
    {
        assert(Compatible(src, dst) && frame.Width() >= 0 && frame.Height() >= 0);
        assert(frame.left >= 0 && frame.top >= 0 && frame.right <= ptrdiff_t(src.width) && frame.bottom <= ptrdiff_t(src.height));

        SimdCopyFrame(src.data, src.stride, src.width, src.height, src.PixelSize(),
            frame.left, frame.top, frame.right, frame.bottom, dst.data, dst.stride);
    }

    /*! @ingroup deinterleave_conversion

        \fn void DeinterleaveUv(const View<A>& uv, View<A>& u, View<A>& v)

        \short Deinterleaves 16-bit UV interleaved image into separated 8-bit U and V planar images.

        All images must have the same width and height.
        This function used for NV12 to YUV420P conversion.

        \note This function is a C++ wrapper for function ::SimdDeinterleaveUv.

        \param [in] uv - an input 16-bit UV interleaved image.
        \param [out] u - an output 8-bit U planar image.
        \param [out] v - an output 8-bit V planar image.
    */
    template<template<class> class A> SIMD_INLINE void DeinterleaveUv(const View<A>& uv, View<A>& u, View<A>& v)
    {
        assert(EqualSize(uv, u, v) && uv.format == View<A>::Uv16 && u.format == View<A>::Gray8 && v.format == View<A>::Gray8);

        SimdDeinterleaveUv(uv.data, uv.stride, uv.width, uv.height, u.data, u.stride, v.data, v.stride);
    }

    /*! @ingroup deinterleave_conversion

        \fn void DeinterleaveBgr(const View<A>& bgr, View<A>& b, View<A>& g, View<A>& r)

        \short Deinterleaves 24-bit BGR interleaved image into separated 8-bit Blue, Green and Red planar images.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdDeinterleaveBgr.

        \param [in] bgr - an input 24-bit BGR interleaved image.
        \param [out] b - an output 8-bit Blue planar image.
        \param [out] g - an output 8-bit Green planar image.
        \param [out] r - an output 8-bit Red planar image.
        */
    template<template<class> class A> SIMD_INLINE void DeinterleaveBgr(const View<A>& bgr, View<A>& b, View<A>& g, View<A>& r)
    {
        assert(EqualSize(bgr, b) && Compatible(b, g, r) && bgr.format == View<A>::Bgr24 && b.format == View<A>::Gray8);

        SimdDeinterleaveBgr(bgr.data, bgr.stride, bgr.width, bgr.height, b.data, b.stride, g.data, g.stride, r.data, r.stride);
    }

    /*! @ingroup deinterleave_conversion

        \fn void DeinterleaveBgra(const View<A>& bgra, View<A>& b, View<A>& g, View<A>& r, View<A>& a)

        \short Deinterleaves 32-bit BGRA interleaved image into separated 8-bit Blue, Green, Red and Alpha planar images.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdDeinterleaveBgra.

        \param [in] bgra - an input 32-bit BGRA interleaved image.
        \param [out] b - an output 8-bit Blue planar image.
        \param [out] g - an output 8-bit Green planar image.
        \param [out] r - an output 8-bit Red planar image.
        \param [out] a - an output 8-bit Alpha planar image.
    */
    template<template<class> class A> SIMD_INLINE void DeinterleaveBgra(const View<A>& bgra, View<A>& b, View<A>& g, View<A>& r, View<A>& a)
    {
        assert(EqualSize(bgra, b) && Compatible(b, g, r, a) && bgra.format == View<A>::Bgra32 && b.format == View<A>::Gray8);

        SimdDeinterleaveBgra(bgra.data, bgra.stride, bgra.width, bgra.height, b.data, b.stride, g.data, g.stride, r.data, r.stride, a.data, a.stride);
    }

    /*! @ingroup deinterleave_conversion

        \fn void DeinterleaveBgra(const View<A>& bgra, View<A>& b, View<A>& g, View<A>& r)

        \short Deinterleaves 32-bit BGRA interleaved image into separated 8-bit Blue, Green and Red planar images (Alpha channel is ignored).

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdDeinterleaveBgra.

        \param [in] bgra - an input 32-bit BGRA interleaved image.
        \param [out] b - an output 8-bit Blue planar image.
        \param [out] g - an output 8-bit Green planar image.
        \param [out] r - an output 8-bit Red planar image.
    */
    template<template<class> class A> SIMD_INLINE void DeinterleaveBgra(const View<A>& bgra, View<A>& b, View<A>& g, View<A>& r)
    {
        assert(EqualSize(bgra, b) && Compatible(b, g, r) && bgra.format == View<A>::Bgra32 && b.format == View<A>::Gray8);

        SimdDeinterleaveBgra(bgra.data, bgra.stride, bgra.width, bgra.height, b.data, b.stride, g.data, g.stride, r.data, r.stride, NULL, 0);
    }

    /*! @ingroup deinterleave_conversion

        \fn void DeinterleaveRgb(const View<A>& rgb, View<A>& r, View<A>& g, View<A>& b)

        \short Deinterleaves 24-bit RGB interleaved image into separated 8-bit Red, Green and Blue planar images.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdDeinterleaveBgr.

        \param [in] rgb - an input 24-bit RGB interleaved image.
        \param [out] r - an output 8-bit Red planar image.
        \param [out] g - an output 8-bit Green planar image.
        \param [out] b - an output 8-bit Blue planar image.
        */
    template<template<class> class A> SIMD_INLINE void DeinterleaveRgb(const View<A>& rgb, View<A>& r, View<A>& g, View<A>& b)
    {
        assert(EqualSize(rgb, b) && Compatible(b, g, r) && rgb.format == View<A>::Rgb24 && b.format == View<A>::Gray8);

        SimdDeinterleaveBgr(rgb.data, rgb.stride, rgb.width, rgb.height, r.data, r.stride, g.data, g.stride, b.data, b.stride);
    }

    /*! @ingroup deinterleave_conversion

        \fn void DeinterleaveRgba(const View<A>& rgba, View<A>& r, View<A>& g, View<A>& b, View<A>& a)

        \short Deinterleaves 32-bit RGBA interleaved image into separated 8-bit Red, Green, Blue and Alpha planar images.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdDeinterleaveBgra.

        \param [in] rgba - an input 32-bit RGBA interleaved image.
        \param [out] r - an output 8-bit Red planar image.
        \param [out] g - an output 8-bit Green planar image.
        \param [out] b - an output 8-bit Blue planar image.
        \param [out] a - an output 8-bit Alpha planar image.
    */
    template<template<class> class A> SIMD_INLINE void DeinterleaveRgba(const View<A>& rgba, View<A>& r, View<A>& g, View<A>& b, View<A>& a)
    {
        assert(EqualSize(rgba, b) && Compatible(b, g, r, a) && rgba.format == View<A>::Rgba32 && b.format == View<A>::Gray8);

        SimdDeinterleaveBgra(rgba.data, rgba.stride, rgba.width, rgba.height, r.data, r.stride, g.data, g.stride, b.data, b.stride, a.data, a.stride);
    }

    /*! @ingroup deinterleave_conversion

        \fn void DeinterleaveRgba(const View<A>& rgba, View<A>& r, View<A>& g, View<A>& b)

        \short Deinterleaves 32-bit RGBA interleaved image into separated 8-bit Red, Green and Blue planar images (Alpha channel is ignored).

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdDeinterleaveBgra.

        \param [in] rgba - an input 32-bit RGBA interleaved image.
        \param [out] r - an output 8-bit Red planar image.
        \param [out] g - an output 8-bit Green planar image.
        \param [out] b - an output 8-bit Blue planar image.
    */
    template<template<class> class A> SIMD_INLINE void DeinterleaveRgba(const View<A>& rgba, View<A>& r, View<A>& g, View<A>& b)
    {
        assert(EqualSize(rgba, b) && Compatible(b, g, r) && rgba.format == View<A>::Rgba32 && b.format == View<A>::Gray8);

        SimdDeinterleaveBgra(rgba.data, rgba.stride, rgba.width, rgba.height, r.data, r.stride, g.data, g.stride, b.data, b.stride, NULL, 0);
    }

    /*! @ingroup edge_background

        \fn void EdgeBackgroundGrowRangeSlow(const View<A>& value, View<A>& background)

        \short Performs edge background update (initial grow, slow mode).

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        background[i] += value[i] > background[i] ? 1 : 0;
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdEdgeBackgroundGrowRangeSlow.

        \param [in] value - a current feature value.
        \param [in, out] background - a feature value of edge dynamic background.
    */
    template<template<class> class A> SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const View<A>& value, View<A>& background)
    {
        assert(Compatible(value, background) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundGrowRangeSlow(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    /*! @ingroup edge_background

        \fn void EdgeBackgroundGrowRangeFast(const View<A>& value, View<A>& background)

        \short Performs edge background update (initial grow, fast mode).

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        background[i] = value[i] > background[i] ? value[i] : background[i];
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdEdgeBackgroundGrowRangeFast.

        \param [in] value - a current feature value.
        \param [in, out] background - a feature value of edge dynamic background.
    */
    template<template<class> class A> SIMD_INLINE void EdgeBackgroundGrowRangeFast(const View<A>& value, View<A>& background)
    {
        assert(Compatible(value, background) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundGrowRangeFast(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    /*! @ingroup edge_background

        \fn void EdgeBackgroundIncrementCount(const View<A>& value, const View<A>& backgroundValue, View<A>& backgroundCount)

        \short Performs collection of edge background statistic.

        All images must have the same width, height and format (8-bit gray).

        Updates background statistic counters for every point:
        \verbatim
        backgroundCount[i] += (value[i] > backgroundValue[i] && backgroundCount[i] < 255) ? 1 : 0;
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdEdgeBackgroundIncrementCount.

        \param [in] value - a current feature value.
        \param [in] backgroundValue - a value of feature of edge dynamic background.
        \param [in, out] backgroundCount - a count of feature of edge dynamic background.
    */
    template<template<class> class A> SIMD_INLINE void EdgeBackgroundIncrementCount(const View<A>& value, const View<A>& backgroundValue, View<A>& backgroundCount)
    {
        assert(Compatible(value, backgroundValue, backgroundCount) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundIncrementCount(value.data, value.stride, value.width, value.height,
            backgroundValue.data, backgroundValue.stride, backgroundCount.data, backgroundCount.stride);
    }

    /*! @ingroup edge_background

        \fn void EdgeBackgroundAdjustRange(View<A>& backgroundCount, View<A>& backgroundValue, uint8_t threshold)

        \short Performs adjustment of edge background range.

        All images must have the same width, height and format (8-bit gray).

        Adjusts edge background range for every point:
        \verbatim
        backgroundValue[i] += (backgroundCount[i] > threshold && backgroundValue[i] < 255) ? 1 : 0;
        backgroundValue[i] -= (backgroundCount[i] < threshold && backgroundValue[i] > 0) ? 1 : 0;
        backgroundCount[i] = 0;
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdEdgeBackgroundAdjustRange.

        \param [in, out] backgroundCount - a count of feature of edge dynamic background.
        \param [in, out] backgroundValue - a value of feature of edge dynamic background.
        \param [in] threshold - a count threshold.
    */
    template<template<class> class A> SIMD_INLINE void EdgeBackgroundAdjustRange(View<A>& backgroundCount, View<A>& backgroundValue, uint8_t threshold)
    {
        assert(Compatible(backgroundCount, backgroundValue) && backgroundCount.format == View<A>::Gray8);

        SimdEdgeBackgroundAdjustRange(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height,
            backgroundValue.data, backgroundValue.stride, threshold);
    }

    /*! @ingroup edge_background

        \fn void EdgeBackgroundAdjustRange(View<A>& backgroundCount, View<A>& backgroundValue, uint8_t threshold, const View<A>& mask)

        \short Performs adjustment of edge background range with using adjust range mask.

        All images must have the same width, height and format (8-bit gray).

        Adjusts edge background range for every point:
        \verbatim
        if(mask[i])
        {
            backgroundValue[i] += (backgroundCount[i] > threshold && backgroundValue[i] < 255) ? 1 : 0;
            backgroundValue[i] -= (backgroundCount[i] < threshold && backgroundValue[i] > 0) ? 1 : 0;
            backgroundCount[i] = 0;
        }
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdEdgeBackgroundAdjustRangeMasked.

        \param [in, out] backgroundCount - a count of feature of edge dynamic background.
        \param [in, out] backgroundValue - a value of feature of edge dynamic background.
        \param [in] threshold - a count threshold.
        \param [in] mask - an adjust range mask.
    */
    template<template<class> class A> SIMD_INLINE void EdgeBackgroundAdjustRange(View<A>& backgroundCount, View<A>& backgroundValue, uint8_t threshold, const View<A>& mask)
    {
        assert(Compatible(backgroundCount, backgroundValue, mask) && backgroundCount.format == View<A>::Gray8);

        SimdEdgeBackgroundAdjustRangeMasked(backgroundCount.data, backgroundCount.stride, backgroundCount.width, backgroundCount.height,
            backgroundValue.data, backgroundValue.stride, threshold, mask.data, mask.stride);
    }

    /*! @ingroup edge_background

        \fn void EdgeBackgroundShiftRange(const View<A>& value, View<A>& background)

        \short Shifts edge background range.

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        background[i] = value[i];
        \endverbatim

        This function is used for fast edge background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdEdgeBackgroundShiftRange.

        \param [in] value - a current feature value.
        \param [in, out] background - a feature of the edge dynamic background.
    */
    template<template<class> class A> SIMD_INLINE void EdgeBackgroundShiftRange(const View<A>& value, View<A>& background)
    {
        assert(Compatible(value, background) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundShiftRange(value.data, value.stride, value.width, value.height, background.data, background.stride);
    }

    /*! @ingroup edge_background

        \fn void EdgeBackgroundShiftRange(const View<A>& value, View<A>& background, const View<A>& mask)

        \short Shifts edge background range with using shift range mask.

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        if(mask[i]])
            background[i] = value[i];
        \endverbatim

        This function is used for fast edge background updating in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdEdgeBackgroundShiftRangeMasked.

        \param [in] value - a current feature value.
        \param [in, out] background - a feature of the edge dynamic background.
        \param [in] mask - a shift range mask.
    */
    template<template<class> class A> SIMD_INLINE void EdgeBackgroundShiftRange(const View<A>& value, View<A>& background, const View<A>& mask)
    {
        assert(Compatible(value, background, mask) && value.format == View<A>::Gray8);

        SimdEdgeBackgroundShiftRangeMasked(value.data, value.stride, value.width, value.height,
            background.data, background.stride, mask.data, mask.stride);
    }

    /*! @ingroup filling

        \fn void Fill(View<A>& dst, uint8_t value)

        \short Fills pixels data of image by given value.

        \note This function is a C++ wrapper for function ::SimdFill.

        \param [out] dst - a destination image.
        \param [in] value - a value to fill image.
    */
    template<template<class> class A> SIMD_INLINE void Fill(View<A>& dst, uint8_t value)
    {
        SimdFill(dst.data, dst.stride, dst.width, dst.height, dst.PixelSize(), value);
    }

    /*! @ingroup filling

        \fn void FillFrame(View<A>& dst, const Rectangle<ptrdiff_t> & frame, uint8_t value)

        \short Fills pixels data of image except for the portion bounded frame by given value.

        \note This function is a C++ wrapper for function ::SimdFillFrame.

        \param [out] dst - a destination image.
        \param [in] frame - a frame rectangle.
        \param [in] value - a value to fill image.
    */
    template<template<class> class A> SIMD_INLINE void FillFrame(View<A>& dst, const Rectangle<ptrdiff_t> & frame, uint8_t value)
    {
        SimdFillFrame(dst.data, dst.stride, dst.width, dst.height, dst.PixelSize(),
            frame.left, frame.top, frame.right, frame.bottom, value);
    }

    /*! @ingroup filling

        \fn void FillBgr(View<A>& dst, uint8_t blue, uint8_t green, uint8_t red)

        \short Fills pixels data of 24-bit BGR image by given color(blue, green, red).

        \note This function is a C++ wrapper for function ::SimdFillBgr.

        \param [out] dst - a destination image.
        \param [in] blue - a blue channel of BGR to fill image.
        \param [in] green - a green channel of BGR to fill image.
        \param [in] red - a red channel of BGR to fill image.
    */
    template<template<class> class A> SIMD_INLINE void FillBgr(View<A>& dst, uint8_t blue, uint8_t green, uint8_t red)
    {
        assert(dst.format == View<A>::Bgr24);

        SimdFillBgr(dst.data, dst.stride, dst.width, dst.height, blue, green, red);
    }

    /*! @ingroup filling

        \fn void FillBgra(View<A>& dst, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha = 0xFF)

        \short Fills pixels data of 32-bit BGRA image by given color(blue, green, red, alpha).

        \note This function is a C++ wrapper for function ::SimdFillBgra.

        \param [out] dst - a destination image.
        \param [in] blue - a blue channel of BGRA to fill image.
        \param [in] green - a green channel of BGRA to fill image.
        \param [in] red - a red channel of BGRA to fill image.
        \param [in] alpha - a alpha channel of BGRA to fill image. It is equal to 255 by default.
    */
    template<template<class> class A> SIMD_INLINE void FillBgra(View<A>& dst, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha = 0xFF)
    {
        assert(dst.format == View<A>::Bgra32);

        SimdFillBgra(dst.data, dst.stride, dst.width, dst.height, blue, green, red, alpha);
    }

    /*! @ingroup filling

        \fn void FillPixel(View<A> & dst, const Pixel & pixel)

        \short Fills image by value of given pixel.

        \note This function is a C++ wrapper for function ::SimdFillPixel.

        \param [out] dst - a destination image.
        \param [in] pixel - a pixel of type which correspond to image format. The size of the type is restricted by range [1, 4].
    */
    template<template<class> class A, class Pixel> SIMD_INLINE void FillPixel(View<A> & dst, const Pixel & pixel)
    {
        assert(dst.PixelSize() == sizeof(Pixel));

        SimdFillPixel(dst.data, dst.stride, dst.width, dst.height, (uint8_t*)&pixel, sizeof(Pixel));
    }

    /*! @ingroup other_filter

        \fn void GaussianBlur3x3(const View<A>& src, View<A>& dst)

        \short Performs Gaussian blur filtration with window 3x3.

        For every point:
        \verbatim
        dst[x, y] = (src[x-1, y-1] + 2*src[x, y-1] + src[x+1, y-1] +
                    2*(src[x-1, y] + 2*src[x, y] + src[x+1, y]) +
                    src[x-1, y+1] + 2*src[x, y+1] + src[x+1, y+1] + 8) / 16;
        \endverbatim
        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function is a C++ wrapper for function ::SimdGaussianBlur3x3.

        \param [in] src - a source image.
        \param [out] dst - a destination image.
    */
    template<template<class> class A> SIMD_INLINE void GaussianBlur3x3(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdGaussianBlur3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    /*! @ingroup gray_conversion

        \fn void GrayToBgr(const View<A>& gray, View<A>& bgr)

        \short Converts 8-bit gray image to 24-bit BGR image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdGrayToBgr.

        \param [in] gray - an input 8-bit gray image.
        \param [out] bgr - an output 24-bit BGR image.
    */
    template<template<class> class A> SIMD_INLINE void GrayToBgr(const View<A>& gray, View<A>& bgr)
    {
        assert(EqualSize(gray, bgr) && bgr.format == View<A>::Bgr24 && gray.format == View<A>::Gray8);

        SimdGrayToBgr(gray.data, gray.width, gray.height, gray.stride, bgr.data, bgr.stride);
    }

    /*! @ingroup gray_conversion

        \fn void GrayToRgb(const View<A>& gray, View<A>& rgb)

        \short Converts 8-bit gray image to 24-bit RGB image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdGrayToBgr.

        \param [in] gray - an input 8-bit gray image.
        \param [out] rgb - an output 24-bit RGB image.
    */
    template<template<class> class A> SIMD_INLINE void GrayToRgb(const View<A>& gray, View<A>& rgb)
    {
        assert(EqualSize(gray, rgb) && rgb.format == View<A>::Rgb24 && gray.format == View<A>::Gray8);

        SimdGrayToBgr(gray.data, gray.width, gray.height, gray.stride, rgb.data, rgb.stride);
    }

    /*! @ingroup gray_conversion

        \fn void GrayToBgra(const View<A>& gray, View<A>& bgra, uint8_t alpha = 0xFF)

        \short Converts 8-bit gray image to 32-bit BGRA image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdGrayToBgra.

        \param [in] gray - an input 8-bit gray image.
        \param [out] bgra - an output 32-bit BGRA image.
        \param [in] alpha - a value of alpha channel. It is equal to 255 by default.
    */
    template<template<class> class A> SIMD_INLINE void GrayToBgra(const View<A>& gray, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(gray, bgra) && bgra.format == View<A>::Bgra32 && gray.format == View<A>::Gray8);

        SimdGrayToBgra(gray.data, gray.width, gray.height, gray.stride, bgra.data, bgra.stride, alpha);
    }

    /*! @ingroup gray_conversion

        \fn void GrayToRgba(const View<A>& gray, View<A>& rgba, uint8_t alpha = 0xFF)

        \short Converts 8-bit gray image to 32-bit RGBA image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdGrayToBgra.

        \param [in] gray - an input 8-bit gray image.
        \param [out] rgba - an output 32-bit RGBA image.
        \param [in] alpha - a value of alpha channel. It is equal to 255 by default.
    */
    template<template<class> class A> SIMD_INLINE void GrayToRgba(const View<A>& gray, View<A>& rgba, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(gray, rgba) && rgba.format == View<A>::Rgba32 && gray.format == View<A>::Gray8);

        SimdGrayToBgra(gray.data, gray.width, gray.height, gray.stride, rgba.data, rgba.stride, alpha);
    }

    /*! @ingroup histogram

        \fn void AbsSecondDerivativeHistogram(const View<A>& src, size_t step, size_t indent, uint32_t * histogram)

        \short Calculates histogram of second derivative for 8-bit gray image.

        For all points except the boundary (defined by parameter indent):
        \verbatim
        dx = abs(src[x, y] - average(src[x+step, y], src[x-step, y]));
        dy = abs(src[x, y] - average(src[x, y+step], src[x, y-step]));
        histogram[max(dx, dy)]++;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdAbsSecondDerivativeHistogram.

        \param [in] src - an input 8-bit gray image.
        \param [in] step - a step for second derivative calculation.
        \param [in] indent - a indent from image boundary.
        \param [out] histogram - a pointer to histogram (array of 256 unsigned 32-bit values).
    */
    template<template<class> class A> SIMD_INLINE void AbsSecondDerivativeHistogram(const View<A>& src, size_t step, size_t indent, uint32_t * histogram)
    {
        assert(src.format == View<A>::Gray8 && indent >= step && src.width > 2 * indent && src.height > 2 * indent);

        SimdAbsSecondDerivativeHistogram(src.data, src.width, src.height, src.stride, step, indent, histogram);
    }

    /*! @ingroup histogram

        \fn void Histogram(const View<A>& src, uint32_t * histogram)

        \short Calculates histogram for 8-bit gray image.

        For all points:
        \verbatim
        histogram[src[i]]++.
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdHistogram.

        \param [in] src - an input 8-bit gray image.
        \param [out] histogram - a pointer to histogram (array of 256 unsigned 32-bit values).
    */
    template<template<class> class A> SIMD_INLINE void Histogram(const View<A>& src, uint32_t * histogram)
    {
        assert(src.format == View<A>::Gray8);

        SimdHistogram(src.data, src.width, src.height, src.stride, histogram);
    }

    /*! @ingroup histogram

        \fn void HistogramMasked(const View<A> & src, const View<A> & mask, uint8_t index, uint32_t * histogram)

        \short Calculates histogram for 8-bit gray image with using mask.

        For every point:
        \verbatim
        if(mask[i] == index)
            histogram[src[i]]++.
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdHistogramMasked.

        \param [in] src - an input 8-bit gray image.
        \param [in] mask - a mask 8-bit image.
        \param [in] index - a mask index.
        \param [out] histogram - a pointer to histogram (array of 256 unsigned 32-bit values).
    */
    template<template<class> class A> SIMD_INLINE void HistogramMasked(const View<A> & src, const View<A> & mask, uint8_t index, uint32_t * histogram)
    {
        assert(Compatible(src, mask) && src.format == View<A>::Gray8);

        SimdHistogramMasked(src.data, src.stride, src.width, src.height, mask.data, mask.stride, index, histogram);
    }

    /*! @ingroup histogram

        \fn void HistogramConditional(const View<A>& src, const View<A>& mask, uint8_t value, SimdCompareType compareType, uint32_t * histogram)

        \short Calculates histogram of 8-bit gray image for those points when mask points satisfying certain condition.

        For every point:
        \verbatim
        if(compare(mask[x, y], value))
            histogram[src[x, y]]++.
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdHistogramConditional.

        \param [in] src - an input 8-bit gray image.
        \param [in] mask - a mask 8-bit image.
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] histogram - a pointer to histogram (array of 256 unsigned 32-bit values).
    */
    template<template<class> class A> SIMD_INLINE void HistogramConditional(const View<A>& src, const View<A>& mask, uint8_t value, SimdCompareType compareType, uint32_t * histogram)
    {
        assert(Compatible(src, mask) && src.format == View<A>::Gray8);

        SimdHistogramConditional(src.data, src.stride, src.width, src.height, mask.data, mask.stride, value, compareType, histogram);
    }

    /*! @ingroup histogram

        \fn void ChangeColors(const View<A> & src, const uint8_t * colors, View<A> & dst)

        \short Changes colors for 8-bit gray image with using of color map.

        The input and output 8-bit gray images must have the same size.
        Algorithm description:
        \verbatim
        for(y = 0; y < height; ++y)
            for(x = 0; x < width; ++x)
                dst[x, y] = colors[src[x, y]];
        \endverbatim
        \note This function is a C++ wrapper for function ::SimdChangeColors.

        \param [in] src - an input 8-bit gray image.
        \param [in] colors - a pointer to the color map (array of 256 unsigned 8-bit values).
        \param [out] dst - an output 8-bit gray image.
    */
    template<template<class> class A> SIMD_INLINE void ChangeColors(const View<A> & src, const uint8_t * colors, View<A> & dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdChangeColors(src.data, src.stride, src.width, src.height, colors, dst.data, dst.stride);
    }

    /*! @ingroup histogram

        \fn void NormalizeHistogram(const View<A> & src, View<A> & dst)

        \short Normalizes histogram for 8-bit gray image.

        The input and output 8-bit gray images must have the same size.

        \note This function is a C++ wrapper for function ::SimdNormalizeHistogram.

        \param [in] src - an input 8-bit gray image.
        \param [out] dst - an output 8-bit image with normalized histogram.
    */
    template<template<class> class A> SIMD_INLINE void NormalizeHistogram(const View<A> & src, View<A> & dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdNormalizeHistogram(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup hog

        \fn void SimdHogDirectionHistograms(const View<A> & src, const Point<ptrdiff_t> & cell, size_t quantization, float * histograms);

        \short Calculates HOG direction histograms for 8-bit gray image.

        Calculates HOG direction histogram for every cell of 8-bit gray image. This function is useful for face recognition.

        \note This function is a C++ wrapper for function ::SimdHogDirectionHistograms.

        \param [in] src - an input 8-bit gray image. Its size must be a multiple of cell size.
        \param [in] cell - a size of cell.
        \param [in] quantization - a direction quantization. Must be even.
        \param [out] histograms - a pointer to buffer with histograms. Array must has size grater or equal to (src.width/cell.x)*(src.height/cell.y)*quantization.
    */
    template<template<class> class A> SIMD_INLINE void HogDirectionHistograms(const View<A> & src, const Point<ptrdiff_t> & cell, size_t quantization, float * histograms)
    {
        assert(src.format == View<A>::Gray8 && src.width%cell.x == 0 && src.height%cell.y == 0 && quantization % 2 == 0);

        SimdHogDirectionHistograms(src.data, src.stride, src.width, src.height, cell.x, cell.y, quantization, histograms);
    }

    /*! @ingroup hog

        \fn void HogExtractFeatures(const View<A> & src, float * features)

        \short Extracts HOG features for 8-bit gray image.

        Extracts HOG features 8-bit gray image. 31 features are extracted for 8x8 cell size and 2x2 block size. This function is useful for face recognition.

        \note This function is a C++ wrapper for function ::SimdHogExtractFeatures.

        \param [in] src - an input 8-bit gray image. Its width and height must be a multiple of 8 and greater or equal to 16.
        \param [out] features - a pointer to buffer with features. Array must has size grater or equal to (width/8)*(height/8)*31.
    */
    template<template<class> class A> SIMD_INLINE void HogExtractFeatures(const View<A> & src, float * features)
    {
        assert(src.format == View<A>::Gray8 && src.width % 8 == 0 && src.height % 8 == 0 && src.width >= 16 && src.height >= 16);

        SimdHogExtractFeatures(src.data, src.stride, src.width, src.height, features);
    }

    /*! @ingroup hog

        \fn void HogLiteExtractFeatures(const View<A> & src, size_t cell, float * features, size_t featuresStride)

        \short Extracts lite HOG features for 8-bit gray image.

        Extracts lite (for 8 directions) HOG features 8-bit gray image. 16 features are extracted for 8x8 or 4x4 cell size and 2x2 block size.

        \note This function is a C++ wrapper for function ::SimdHogLiteExtractFeatures.

        \param [in] src - an input 8-bit gray image. Its width and height must be a multiple of cell and greater or equal to cell*3.
        \param [in] cell - a size of cell. It must be 4 or 8.
        \param [out] features - a pointer to buffer with features. Array must has size grater or equal to (height/cell - 2)*featuresStride.
        \param [in] featuresStride - a row size of the buffer with features. It must be grater or equal to (width/cell - 2)*16.
    */
    template<template<class> class A> SIMD_INLINE void HogLiteExtractFeatures(const View<A> & src, size_t cell, float * features, size_t featuresStride)
    {
        assert((cell == 4 || cell == 8) && featuresStride >= (src.width / cell - 2) * 16);
        assert(src.format == View<A>::Gray8 && src.width >= cell * 3 && src.height >= cell * 3);

        SimdHogLiteExtractFeatures(src.data, src.stride, src.width, src.height, cell, features, featuresStride);
    }

    /*! @ingroup other_conversion

        \fn void Int16ToGray(const View<A> & src, View<A> & dst)

        \short Converts 16-bit signed integer image to 8-bit gray image with saturation.

        All images must have the same width and height.

        For every point:
        \verbatim
        dst[i] = Max(0, Min(255, src[i]));
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdInt16ToGray.

        \param [in] src - an input 16-bit signed integer image
        \param [out] dst - an output 8-bit gray image.
    */
    template<template<class> class A> SIMD_INLINE void Int16ToGray(const View<A> & src, View<A> & dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Int16 && dst.format == View<A>::Gray8);

        SimdInt16ToGray(src.data, src.width, src.height, src.stride, dst.data, dst.stride);
    }

    /*! @ingroup integral

        \fn void Integral(const View<A>& src, View<A>& sum)

        \short Calculates integral images for input 8-bit gray image.

        The function can calculates sum integral image.
        A integral image must have width and height per unit greater than that of the input image.

        \note This function is a C++ wrapper for function ::SimdIntegral.

        \param [in] src - an input 8-bit gray image.
        \param [out] sum - a 32-bit integer sum image.
    */
    template<template<class> class A> SIMD_INLINE void Integral(const View<A>& src, View<A>& sum)
    {
        assert(src.width + 1 == sum.width && src.height + 1 == sum.height);
        assert(src.format == View<A>::Gray8 && sum.format == View<A>::Int32);

        SimdIntegral(src.data, src.stride, src.width, src.height, sum.data, sum.stride, NULL, 0, NULL, 0,
            (SimdPixelFormatType)sum.format, SimdPixelFormatNone);
    }

    /*! @ingroup integral

        \fn void Integral(const View<A>& src, View<A>& sum, View<A>& sqsum)

        \short Calculates integral images for input 8-bit gray image.

        The function can calculates sum integral image and square sum integral image.
        A integral images must have width and height per unit greater than that of the input image.

        \note This function is a C++ wrapper for function ::SimdIntegral.

        \param [in] src - an input 8-bit gray image.
        \param [out] sum - a 32-bit integer sum image.
        \param [out] sqsum - a 32-bit integer or 64-bit float point square sum image.
    */
    template<template<class> class A> SIMD_INLINE void Integral(const View<A>& src, View<A>& sum, View<A>& sqsum)
    {
        assert(src.width + 1 == sum.width && src.height + 1 == sum.height && EqualSize(sum, sqsum));
        assert(src.format == View<A>::Gray8 && sum.format == View<A>::Int32 && (sqsum.format == View<A>::Int32 || sqsum.format == View<A>::Double));

        SimdIntegral(src.data, src.stride, src.width, src.height, sum.data, sum.stride, sqsum.data, sqsum.stride, NULL, 0,
            (SimdPixelFormatType)sum.format, (SimdPixelFormatType)sqsum.format);
    }

    /*! @ingroup integral

        \fn void Integral(const View<A>& src, View<A>& sum, View<A>& sqsum, View<A>& tilted)

        \short Calculates integral images for input 8-bit gray image.

        The function can calculates sum integral image, square sum integral image and tilted sum integral image.
        A integral images must have width and height per unit greater than that of the input image.

        \note This function is a C++ wrapper for function ::SimdIntegral.

        \param [in] src - an input 8-bit gray image.
        \param [out] sum - a 32-bit integer sum image.
        \param [out] sqsum - a 32-bit integer or 64-bit float point square sum image.
        \param [out] tilted - a 32-bit integer tilted sum image.
    */
    template<template<class> class A> SIMD_INLINE void Integral(const View<A>& src, View<A>& sum, View<A>& sqsum, View<A>& tilted)
    {
        assert(src.width + 1 == sum.width && src.height + 1 == sum.height && EqualSize(sum, sqsum) && Compatible(sum, tilted));
        assert(src.format == View<A>::Gray8 && sum.format == View<A>::Int32 && (sqsum.format == View<A>::Int32 || sqsum.format == View<A>::Double));

        SimdIntegral(src.data, src.stride, src.width, src.height, sum.data, sum.stride, sqsum.data, sqsum.stride, tilted.data, tilted.stride,
            (SimdPixelFormatType)sum.format, (SimdPixelFormatType)sqsum.format);
    }

    /*! @ingroup interference

        \fn void InterferenceIncrement(View<A> & dst, uint8_t increment, int16_t saturation)

        \short Increments statistic of interference detector.

        For every point:
        \verbatim
        statistic[i] = min(statistic[i] + increment, saturation);
        \endverbatim

        This function is used for interference detection in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdInterferenceIncrement.

        \param [in, out] dst - a 16-bit signed integer image with statistic.
        \param [in] increment - an increment of statistic.
        \param [in] saturation - an upper saturation of statistic.
    */
    template<template<class> class A> SIMD_INLINE void InterferenceIncrement(View<A> & dst, uint8_t increment, int16_t saturation)
    {
        assert(dst.format == View<A>::Int16);

        SimdInterferenceIncrement(dst.data, dst.stride, dst.width, dst.height, increment, saturation);
    }

    /*! @ingroup interference

        \fn void InterferenceIncrementMasked(View<A> & dst, uint8_t increment, int16_t saturation, const View<A>& mask, uint8_t index)

        \short Increments statistic of interference detector with using segmentation mask.

        For every point:
        \verbatim
        if(mask[i] == index)
            statistic[i] = min(statistic[i] + increment, saturation);
        \endverbatim

        All images must have the same width, height.
        This function is used for interference detection in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdInterferenceIncrementMasked.

        \param [in, out] dst - a 16-bit signed integer image with statistic.
        \param [in] increment - an increment of statistic.
        \param [in] saturation - an upper saturation of statistic.
        \param [in] mask - a 8-bit gray image with mask.
        \param [in] index - an index of mask.
    */
    template<template<class> class A> SIMD_INLINE void InterferenceIncrementMasked(View<A> & dst, uint8_t increment, int16_t saturation, const View<A>& mask, uint8_t index)
    {
        assert(dst.format == View<A>::Int16 && mask.format == View<A>::Gray8 && EqualSize(dst, mask));

        SimdInterferenceIncrementMasked(dst.data, dst.stride, dst.width, dst.height, increment, saturation, mask.data, mask.stride, index);
    }

    /*! @ingroup interference

        \fn void InterferenceDecrement(View<A> & dst, uint8_t decrement, int16_t saturation)

        \short Decrements statistic of interference detector.

        For every point:
        \verbatim
        statistic[i] = max(statistic[i] - decrement, saturation);
        \endverbatim

        This function is used for interference detection in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdInterferenceDecrement.

        \param [in, out] dst - a 16-bit signed integer image with statistic.
        \param [in] decrement - a decrement of statistic.
        \param [in] saturation - a lower saturation of statistic.
    */
    template<template<class> class A> SIMD_INLINE void InterferenceDecrement(View<A> & dst, uint8_t decrement, int16_t saturation)
    {
        assert(dst.format == View<A>::Int16);

        SimdInterferenceDecrement(dst.data, dst.stride, dst.width, dst.height, decrement, saturation);
    }

    /*! @ingroup interference

        \fn void InterferenceDecrementMasked(View<A> & dst, uint8_t decrement, int16_t saturation, const View<A>& mask, uint8_t index)

        \short Decrements statistic of interference detector with using segmentation mask.

        For every point:
        \verbatim
        if(mask[i] == index)
            statistic[i] = max(statistic[i] - decrement, saturation);
        \endverbatim

        All images must have the same width, height.
        This function is used for interference detection in motion detection algorithm.

        \note This function is a C++ wrapper for function ::SimdInterferenceDecrementMasked.

        \param [in, out] dst - a 16-bit signed integer image with statistic.
        \param [in] decrement - a decrement of statistic.
        \param [in] saturation - a lower saturation of statistic.
        \param [in] mask - a 8-bit gray image with mask.
        \param [in] index - an index of mask.
    */
    template<template<class> class A> SIMD_INLINE void InterferenceDecrementMasked(View<A> & dst, uint8_t decrement, int16_t saturation, const View<A>& mask, uint8_t index)
    {
        assert(dst.format == View<A>::Int16 && mask.format == View<A>::Gray8 && EqualSize(dst, mask));

        SimdInterferenceDecrementMasked(dst.data, dst.stride, dst.width, dst.height, decrement, saturation, mask.data, mask.stride, index);
    }

    /*! @ingroup interleave_conversion

        \fn void InterleaveUv(const View<A>& u, const View<A>& v, View<A>& uv)

        \short Interleaves 8-bit U and V planar images into one 16-bit UV interleaved image.

        All images must have the same width and height.
        This function used for YUV420P to NV12 conversion.

        \note This function is a C++ wrapper for function ::SimdInterleaveUv.

        \param [in] u - an input 8-bit U planar image.
        \param [in] v - an input 8-bit V planar image.
        \param [out] uv - an output 16-bit UV interleaved image.
    */
    template<template<class> class A> SIMD_INLINE void InterleaveUv(const View<A>& u, const View<A>& v, View<A>& uv)
    {
        assert(EqualSize(uv, u, v) && uv.format == View<A>::Uv16 && u.format == View<A>::Gray8 && v.format == View<A>::Gray8);

        SimdInterleaveUv(u.data, u.stride, v.data, v.stride, u.width, u.height, uv.data, uv.stride);
    }

    /*! @ingroup interleave_conversion

        \fn void InterleaveBgr(const View<A> & b, const View<A> & g, const View<A> & r, View<A> & bgr)

        \short Interleaves 8-bit Blue, Green and Red planar images into one 24-bit BGR interleaved image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdInterleaveBgr.

        \param [in] b - an input 8-bit Blue planar image.
        \param [in] g - an input 8-bit Green planar image.
        \param [in] r - an input 8-bit Red planar image.
        \param [out] bgr - an output 24-bit BGR interleaved image.
    */
    template<template<class> class A> SIMD_INLINE void InterleaveBgr(const View<A> & b, const View<A> & g, const View<A> & r, View<A> & bgr)
    {
        assert(EqualSize(bgr, b, g, r) && Compatible(b, g, r) && bgr.format == View<A>::Bgr24 && b.format == View<A>::Gray8);

        SimdInterleaveBgr(b.data, b.stride, g.data, g.stride, r.data, r.stride, bgr.width, bgr.height, bgr.data, bgr.stride);
    }

    /*! @ingroup interleave_conversion

        \fn void InterleaveBgra(const View<A>& b, const View<A>& g, const View<A>& r, const View<A>& a, View<A>& bgra)

        \short Interleaves 8-bit Blue, Green, Red and Alpha planar images into one 32-bit BGRA interleaved image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdInterleaveBgra.

        \param [in] b - an input 8-bit Blue planar image.
        \param [in] g - an input 8-bit Green planar image.
        \param [in] r - an input 8-bit Red planar image.
        \param [in] a - an input 8-bit Alpha planar image.
        \param [out] bgra - an output 32-bit BGRA interleaved image.
    */
    template<template<class> class A> SIMD_INLINE void InterleaveBgra(const View<A>& b, const View<A>& g, const View<A>& r, const View<A>& a, View<A>& bgra)
    {
        assert(EqualSize(bgra, b) && Compatible(b, g, r, a) && bgra.format == View<A>::Bgra32 && b.format == View<A>::Gray8);

        SimdInterleaveBgra(b.data, b.stride, g.data, g.stride, r.data, r.stride, a.data, a.stride, bgra.width, bgra.height, bgra.data, bgra.stride);
    }

    /*! @ingroup laplace_filter

        \fn void Laplace(const View<A>& src, View<A>& dst)

        \short Calculates Laplace's filter.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \verbatim
        dst[x, y] =
            - src[x-1, y-1] -   src[x, y-1] - src[x+1, y-1]
            - src[x-1, y]   + 8*src[x, y]   - src[x+1, y]
            - src[x-1, y+1] -   src[x, y+1] - src[x+1, y+1].
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdLaplace.

        \param [in] src - an input image.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void Laplace(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdLaplace(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup laplace_filter

        \fn void LaplaceAbs(const View<A>& src, View<A>& dst)

        \short Calculates absolute value of Laplace's filter.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \verbatim
        dst[x, y] = abs(
            - src[x-1, y-1] -   src[x, y-1] - src[x+1, y-1]
            - src[x-1, y]   + 8*src[x, y]   - src[x+1, y]
            - src[x-1, y+1] -   src[x, y+1] - src[x+1, y+1]).
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdLaplaceAbs.

        \param [in] src - an input image.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void LaplaceAbs(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdLaplaceAbs(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup other_statistic

        \fn void LaplaceAbsSum(const View<A>& src, uint64_t & sum)

        \short Calculates sum of absolute value of Laplace's filter.

        Input image must has 8-bit gray format.

        For every point:
        \verbatim
        sum += abs(
            - src[x-1, y-1] -   src[x, y-1] - src[x+1, y-1]
            - src[x-1, y]   + 8*src[x, y]   - src[x+1, y]
            - src[x-1, y+1] -   src[x, y+1] - src[x+1, y+1]).
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdLaplaceAbsSum.

        \param [in] src - an input image.
        \param [out] sum - a result sum.
    */
    template<template<class> class A> SIMD_INLINE void LaplaceAbsSum(const View<A> & src, uint64_t & sum)
    {
        assert(src.format == View<A>::Gray8);

        SimdLaplaceAbsSum(src.data, src.stride, src.width, src.height, &sum);
    }

    /*! @ingroup other_filter

        \fn void LbpEstimate(const View<A>& src, View<A>& dst)

        \short Calculates LBP (Local Binary Patterns) for 8-bit gray image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdLbpEstimate.

        \param [in] src - an input 8-bit gray image.
        \param [out] dst - an output 8-bit gray image with LBP.
    */
    template<template<class> class A> SIMD_INLINE void LbpEstimate(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdLbpEstimate(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup memory

        \fn void LitterCpuCache(size_t k = 2)

        \short It creates a large buffer and fills it. 

        This function litters CPU cache. It is useful for test purposes.

        \param [in] k - a boosting coefficient of stub buffer size relative to CPU L3 cache size. Its default value is 2.
    */
    SIMD_INLINE void LitterCpuCache(size_t k = 2)
    {
        size_t size = SimdCpuInfo(SimdCpuInfoCacheL3)*k;
        uint8_t * buffer = (uint8_t*)SimdAllocate(size, SimdAlignment());
        SimdFillBgra(buffer, size, size / 4, 1, 0, 1, 2, 3);
        SimdFree(buffer);
    }

    /*! @ingroup other_filter

        \fn void MeanFilter3x3(const View<A>& src, View<A>& dst)

        \short Performs an averaging with window 3x3.

        For every point:
        \verbatim
        dst[x, y] = (src[x-1, y-1] + src[x, y-1] + src[x+1, y-1] +
                     src[x-1, y] + src[x, y] + src[x+1, y] +
                     src[x-1, y+1] + src[x, y+1] + src[x+1, y+1] + 4) / 9;
        \endverbatim

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function is a C++ wrapper for function ::SimdMeanFilter3x3.

        \param [in] src - a source image.
        \param [out] dst - a destination image.
    */
    template<template<class> class A> SIMD_INLINE void MeanFilter3x3(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMeanFilter3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    /*! @ingroup median_filter

        \fn void MedianFilterRhomb3x3(const View<A>& src, View<A>& dst)

        \short Performs median filtration of input image (filter window is a rhomb 3x3).

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function is a C++ wrapper for function ::SimdMedianFilterRhomb3x3.

        \param [in] src - an original input image.
        \param [out] dst - a filtered output image.
    */
    template<template<class> class A> SIMD_INLINE void MedianFilterRhomb3x3(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterRhomb3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    /*! @ingroup median_filter

        \fn void MedianFilterRhomb5x5(const View<A>& src, View<A>& dst)

        \short Performs median filtration of input image (filter window is a rhomb 5x5).

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function is a C++ wrapper for function ::SimdMedianFilterRhomb5x5.

        \param [in] src - an original input image.
        \param [out] dst - a filtered output image.
    */
    template<template<class> class A> SIMD_INLINE void MedianFilterRhomb5x5(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterRhomb5x5(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    /*! @ingroup median_filter

        \fn void MedianFilterSquare3x3(const View<A>& src, View<A>& dst)

        \short Performs median filtration of input image (filter window is a square 3x3).

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function is a C++ wrapper for function ::SimdMedianFilterSquare3x3.

        \param [in] src - an original input image.
        \param [out] dst - a filtered output image.
    */
    template<template<class> class A> SIMD_INLINE void MedianFilterSquare3x3(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterSquare3x3(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    /*! @ingroup median_filter

        \fn void MedianFilterSquare5x5(const View<A>& src, View<A>& dst)

        \short Performs median filtration of input image (filter window is a square 5x5).

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function is a C++ wrapper for function ::SimdMedianFilterSquare5x5.

        \param [in] src - an original input image.
        \param [out] dst - a filtered output image.
    */
    template<template<class> class A> SIMD_INLINE void MedianFilterSquare5x5(const View<A>& src, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.ChannelSize() == 1);

        SimdMedianFilterSquare5x5(src.data, src.stride, src.width, src.height, src.ChannelCount(), dst.data, dst.stride);
    }

    /*! @ingroup neural

        \fn void NeuralConvert(const View<A> & src, float * dst, size_t stride, bool inversion)

        \short Converts a 8-bit gray image to the 32-bit float array.

        The length of output array must be equal to the area of input image.

        For every point:
        \verbatim
        dst[i] = inversion ? (255 - src[col]) / 255 : src[i]/255;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdNeuralConvert.

        \param [in] src - an input image.
        \param [out] dst - a pointer to output array.
        \param [in] stride - a row size of the output array.
        \param [in] inversion - a flag of color inversion.
    */
    template<template<class> class A> SIMD_INLINE void NeuralConvert(const View<A> & src, float * dst, size_t stride, bool inversion)
    {
        assert(src.format == View<A>::Gray8);

        SimdNeuralConvert(src.data, src.stride, src.width, src.height, dst, stride, inversion ? 1 : 0);
    }

    /*! @ingroup operation

        \fn void OperationBinary8u(const View<A>& a, const View<A>& b, View<A>& dst, SimdOperationBinary8uType type)

        \short Performs given operation between two images.

        All images must have the same width, height and format (8-bit gray, 16-bit UV (UV plane of NV12 pixel format), 24-bit BGR or 32-bit BGRA).

        \note This function is a C++ wrapper for function ::SimdOperationBinary8u.

        \param [in] a - a first input image.
        \param [in] b - a second input image.
        \param [out] dst - an output image.
        \param [in] type - a type of operation (see ::SimdOperationBinary8uType).
    */
    template<template<class> class A> SIMD_INLINE void OperationBinary8u(const View<A>& a, const View<A>& b, View<A>& dst, SimdOperationBinary8uType type)
    {
        assert(Compatible(a, b, dst) && a.ChannelSize() == 1);

        SimdOperationBinary8u(a.data, a.stride, b.data, b.stride, a.width, a.height, a.ChannelCount(), dst.data, dst.stride, type);
    }

    /*! @ingroup operation

        \fn void OperationBinary16i(const View<A>& a, const View<A>& b, View<A>& dst, SimdOperationBinary16iType type)

        \short Performs given operation between two images.

        All images must have the same width, height and Simd::View::Int16 pixel format.

        \note This function is a C++ wrapper for function ::SimdOperationBinary16i.

        \param [in] a - a first input image.
        \param [in] b - a second input image.
        \param [out] dst - an output image.
        \param [in] type - a type of operation (see ::SimdOperationBinary16iType).
    */
    template<template<class> class A> SIMD_INLINE void OperationBinary16i(const View<A>& a, const View<A>& b, View<A>& dst, SimdOperationBinary16iType type)
    {
        assert(Compatible(a, b, dst) && a.format == View<A>::Int16);

        SimdOperationBinary16i(a.data, a.stride, b.data, b.stride, a.width, a.height, dst.data, dst.stride, type);
    }

    /*! @ingroup operation

        \fn void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, View<A>& dst)

        \short Calculates result 8-bit gray image as product of two vectors.

        For all points:
        \verbatim
        dst[x, y] = horizontal[x]*vertical[y]/255;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdVectorProduct.

        \param [in] vertical - a pointer to pixels data of vertical vector. It length is equal to result image height.
        \param [in] horizontal - a pointer to pixels data of horizontal vector. It length is equal to result image width.
        \param [out] dst - a result image.
    */
    template<template<class> class A> SIMD_INLINE void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, View<A>& dst)
    {
        assert(dst.format == View<A>::Gray8);

        SimdVectorProduct(vertical, horizontal, dst.data, dst.stride, dst.width, dst.height);
    }

    /*! @ingroup resizing

        \fn void ReduceGray2x2(const View<A>& src, View<A>& dst)

        \short Performs reducing (in 2 times) and Gaussian blurring a 8-bit gray image with using window 2x2.

        For input and output image must be performed: dst.width = (src.width + 1)/2,  dst.height = (src.height + 1)/2.

        For all points:
        \verbatim
        dst[x, y] = (src[2*x, 2*y] + src[2*x, 2*y + 1] + src[2*x + 1, 2*y] + src[2*x + 1, 2*y + 1] + 2)/4;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdReduceGray2x2.

        \param [in] src - an original input image.
        \param [out] dst - a reduced output image.
    */
    template<template<class> class A> SIMD_INLINE void ReduceGray2x2(const View<A>& src, View<A>& dst)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8 && Scale(src.Size()) == dst.Size());

        SimdReduceGray2x2(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    /*! @ingroup resizing

        \fn void ReduceGray3x3(const View<A>& src, View<A>& dst, bool compensation = true)

        \short Performs reducing (in 2 times) and Gaussian blurring a 8-bit gray image with using window 3x3.

        For input and output image must be performed: dst.width = (src.width + 1)/2,  dst.height = (src.height + 1)/2.

        For every point:
        \verbatim
        dst[x, y] = (src[2*x-1, 2*y-1] + 2*src[2*x, 2*y-1] + src[2*x+1, 2*y-1] +
                  2*(src[2*x-1, 2*y]   + 2*src[2*x, 2*y]   + src[2*x+1, 2*y]) +
                     src[2*x-1, 2*y+1] + 2*src[2*x, 2*y+1] + src[2*x+1, 2*y+1] + compensation ? 8 : 0) / 16;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdReduceGray3x3.

        \param [in] src - an original input image.
        \param [out] dst - a reduced output image.
        \param [in] compensation - a flag of compensation of rounding. It is equal to 'true' by default.
    */
    template<template<class> class A> SIMD_INLINE void ReduceGray3x3(const View<A>& src, View<A>& dst, bool compensation = true)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8 && Scale(src.Size()) == dst.Size());

        SimdReduceGray3x3(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, compensation ? 1 : 0);
    }

    /*! @ingroup resizing

        \fn void ReduceGray4x4(const View<A>& src, View<A>& dst)

        \short Performs reducing (in 2 times) and Gaussian blurring a 8-bit gray image with using window 4x4.

        For input and output image must be performed: dst.width = (src.width + 1)/2,  dst.height = (src.height + 1)/2.

        For every point:
        \verbatim
        dst[x, y] =   (src[2*x-1, 2*y-1] + 3*src[2*x, 2*y-1] + 3*src[2*x+1, 2*y-1] + src[2*x+2, 2*y-1]
                    3*(src[2*x-1, 2*y]   + 3*src[2*x, 2*y]   + 3*src[2*x+1, 2*y]   + src[2*x+2, 2*y]) +
                    3*(src[2*x-1, 2*y+1] + 3*src[2*x, 2*y+1] + 3*src[2*x+1, 2*y+1] + src[2*x+2, 2*y+1]) +
                       src[2*x-1, 2*y+2] + 3*src[2*x, 2*y+2] + 3*src[2*x+1, 2*y+2] + src[2*x+2, 2*y+2] + 32) / 64;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdReduceGray4x4.

        \param [in] src - an original input image.
        \param [out] dst - a reduced output image.
    */
    template<template<class> class A> SIMD_INLINE void ReduceGray4x4(const View<A>& src, View<A>& dst)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8 && Scale(src.Size()) == dst.Size());

        SimdReduceGray4x4(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    /*! @ingroup resizing

        \fn void ReduceGray5x5(const View<A>& src, View<A>& dst, bool compensation = true)

        \short Performs reducing (in 2 times) and Gaussian blurring a 8-bit gray image with using window 5x5.

        For input and output image must be performed: dst.width = (src.width + 1)/2,  dst.height = (src.height + 1)/2.

        For every point:
        \verbatim
        dst[x, y] = (
               src[2*x-2, 2*y-2] + 4*src[2*x-1, 2*y-2] + 6*src[2*x, 2*y-2] + 4*src[2*x+1, 2*y-2] + src[2*x+2, 2*y-2] +
            4*(src[2*x-2, 2*y-1] + 4*src[2*x-1, 2*y-1] + 6*src[2*x, 2*y-1] + 4*src[2*x+1, 2*y-1] + src[2*x+2, 2*y-1]) +
            6*(src[2*x-2, 2*y]   + 4*src[2*x-1, 2*y]   + 6*src[2*x, 2*y]   + 4*src[2*x+1, 2*y]   + src[2*x+2, 2*y]) +
            4*(src[2*x-2, 2*y+1] + 4*src[2*x-1, 2*y+1] + 6*src[2*x, 2*y+1] + 4*src[2*x+1, 2*y+1] + src[2*x+2, 2*y+1]) +
               src[2*x-2, 2*y+2] + 4*src[2*x-1, 2*y+2] + 6*src[2*x, 2*y+2] + 4*src[2*x+1, 2*y+2] + src[2*x+2, 2*y+2] +
            compensation ? 128 : 0) / 256;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdReduceGray5x5.

        \param [in] src - an original input image.
        \param [out] dst - a reduced output image.
        \param [in] compensation - a flag of compensation of rounding. It is equal to 'true' by default.
    */
    template<template<class> class A> SIMD_INLINE void ReduceGray5x5(const View<A>& src, View<A>& dst, bool compensation = true)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8 && Scale(src.Size()) == dst.Size());

        SimdReduceGray5x5(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, compensation ? 1 : 0);
    }

    /*! @ingroup resizing

        \fn void ReduceGray(const View<A> & src, View<A> & dst, ::SimdReduceType reduceType, bool compensation = true)

        \short Performs reducing (in 2 times) and Gaussian blurring a 8-bit gray image.

        For input and output image must be performed: dst.width = (src.width + 1)/2,  dst.height = (src.height + 1)/2.

        \param [in] src - an original input image.
        \param [out] dst - a reduced output image.
        \param [in] reduceType - a type of function used for image reducing.
        \param [in] compensation - a flag of compensation of rounding. It is relevant only for ::SimdReduce3x3 and ::SimdReduce5x5. It is equal to 'true' by default.
    */
    template<template<class> class A> SIMD_INLINE void ReduceGray(const View<A> & src, View<A> & dst, ::SimdReduceType reduceType, bool compensation = true)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8 && Scale(src.Size()) == dst.Size());

        switch (reduceType)
        {
        case SimdReduce2x2:
            Simd::ReduceGray2x2(src, dst);
            break;
        case SimdReduce3x3:
            Simd::ReduceGray3x3(src, dst, compensation);
            break;
        case SimdReduce4x4:
            Simd::ReduceGray4x4(src, dst);
            break;
        case SimdReduce5x5:
            Simd::ReduceGray5x5(src, dst, compensation);
            break;
        default:
            assert(0);
        }
    }

    /*! @ingroup resizing

        \fn void Reduce2x2(const View<A> & src, View<A> & dst)

        \short Performs reducing of image (in 2 times).

        For input and output image must be performed: dst.width = (src.width + 1)/2,  dst.height = (src.height + 1)/2.

        \param [in] src - an original input image.
        \param [out] dst - a reduced output image.
    */
    template<template<class> class A> SIMD_INLINE void Reduce2x2(const View<A> & src, View<A> & dst)
    {
        assert(src.format == dst.format && Scale(src.Size()) == dst.Size() && src.ChannelSize() == 1);

        SimdReduceColor2x2(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, src.ChannelCount());
    }

    /*! @ingroup resizing

        \fn void ResizeBilinear(const View<A>& src, View<A>& dst)

        \short Performs resizing of input image with using bilinear interpolation.

        All images must have the same format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function is a C++ wrapper for function ::SimdResizeBilinear.

        \param [in] src - an original input image.
        \param [out] dst - a resized output image.
    */
    template<template<class> class A> SIMD_INLINE void ResizeBilinear(const View<A> & src, View<A> & dst)
    {
        assert(src.format == dst.format && src.ChannelSize() == 1);

        if (EqualSize(src, dst))
        {
            Copy(src, dst);
        }
        else
        {
            SimdResizeBilinear(src.data, src.width, src.height, src.stride,
                dst.data, dst.width, dst.height, dst.stride, src.ChannelCount());
        }
    }

    /*! @ingroup resizing

        \fn void ResizeAreaGray(const View<A> & src, View<A> & dst)

        \short Performs resizing of input image with using area interpolation.

        All images must have the same format (8-bit gray).

        \param [in] src - an original input image.
        \param [out] dst - a resized output image.
    */
    template<template<class> class A> SIMD_INLINE void ResizeAreaGray(const View<A> & src, View<A> & dst)
    {
        assert(src.format == dst.format && src.format == View<A>::Gray8);

        if (EqualSize(src, dst))
        {
            Copy(src, dst);
        }
        else
        {
            size_t level = 0;
            for (; (dst.width << (level + 1)) < (size_t)src.width; level++);
            Point<ptrdiff_t> size = src.Size() << level;
            if (level)
            {
                Pyramid<A> pyramid(size, level + 1);
                Simd::ResizeBilinear(src, pyramid[0]);
                for (size_t i = 0; i < level; ++i)
                    Simd::ReduceGray(pyramid.At(i), pyramid.At(i + 1), ::SimdReduce2x2);
                Simd::Copy(pyramid[level], dst);
            }
            else
                Simd::ResizeBilinear(src, dst);
        }
    }

    /*! @ingroup resizing

        \fn void ResizeArea(const View<A> & src, View<A> & dst)

        \short Performs resizing of input image with using area interpolation.

        All images must have the same format.

        \param [in] src - an original input image.
        \param [out] dst - a resized output image.
    */
    template<template<class> class A> SIMD_INLINE void ResizeArea(const View<A> & src, View<A> & dst)
    {
        assert(src.format == dst.format);

        if (EqualSize(src, dst))
        {
            Copy(src, dst);
        }
        else
        {
            size_t level = 0;
            for (; (dst.width << (level + 1)) < (size_t)src.width; level++);
            Point<ptrdiff_t> size = src.Size() << level;
            if (level)
            {
                std::vector<View<A>> pyramid(level);
                pyramid[0].Resize(size, src.format);
                Simd::ResizeBilinear(src, pyramid[0]);
                for (size_t i = 1; i < level; ++i)
                {
                    size = Simd::Scale(size);
                    pyramid[i].Resize(size, src.format);
                    Simd::Reduce2x2(pyramid.At(i - 1), pyramid.At(i));
                }
                Simd::Reduce2x2(pyramid.At(level - 1), dst);
            }
            else
                Simd::ResizeBilinear(src, dst);
        }
    }

    /*! @ingroup resizing

        \fn void Resize(const View<A> & src, View<A> & dst, ::SimdResizeMethodType method = ::SimdResizeMethodBilinear)

        \short Performs resizing of image.

        All images must have the same format.

        \param [in] src - an original input image.
        \param [out] dst - a resized output image.
        \param [in] method - a resizing method. By default it is equal to ::SimdResizeMethodBilinear.
    */
    template<template<class> class A> SIMD_INLINE void Resize(const View<A> & src, View<A> & dst, ::SimdResizeMethodType method = ::SimdResizeMethodBilinear)
    {
        assert(src.format == dst.format && (src.format == View<A>::Float || src.ChannelSize() == 1 || src.ChannelSize() == 2));

        if (EqualSize(src, dst))
        {
            Copy(src, dst);
        }
        else
        {
            SimdResizeChannelType type = src.format == View<A>::Float ? SimdResizeChannelFloat : (src.ChannelSize() == 2 ? SimdResizeChannelShort : SimdResizeChannelByte);
            void * resizer = SimdResizerInit(src.width, src.height, dst.width, dst.height, src.ChannelCount(), type, method);
            if (resizer)
            {
                SimdResizerRun(resizer, src.data, src.stride, dst.data, dst.stride);
                SimdRelease(resizer);
            }
            else
                assert(0);
        }
    }

    /*! @ingroup resizing

        \fn void Resize(const View<A> & src, View<A> & dst, const Point<ptrdiff_t> & size, ::SimdResizeMethodType method = ::SimdResizeMethodBilinear)

        \short Performs resizing of image.

        \param [in] src - an original input image.
        \param [out] dst - a resized output image. The input image can be the output.
        \param [in] size - a size of output image.
        \param [in] method - a resizing method. By default it is equal to ::SimdResizeMethodBilinear.
    */
    template<template<class> class A> SIMD_INLINE void Resize(const View<A>& src, View<A>& dst, const Point<ptrdiff_t> & size, ::SimdResizeMethodType method = ::SimdResizeMethodBilinear)
    {
        assert(src.format == View<A>::Float || src.ChannelSize() == 1);

        if (&src == &dst)
        {
            if (src.Size() != size)
            {
                View<A> tmp(size, src.format);
                Resize(src, tmp, method);
                dst.Swap(tmp);
            }
        }
        else
        {
            if (dst.Size() != size)
                dst.Recreate(size, src.format);
            Resize(src, dst, method);
        }
    }

    /*! @ingroup rgb_conversion

        \fn void RgbToBgr(const View<A> & rgb, View<A> & bgr)

        \short Converts 24-bit RGB image to 24-bit BGR image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgrToRgb.

        \param [in] rgb - an input 24-bit RGB image.
        \param [out] bgr - an output 24-bit BGR image.
    */
    template<template<class> class A> SIMD_INLINE void RgbToBgr(const View<A>& rgb, View<A>& bgr)
    {
        assert(EqualSize(bgr, rgb) && rgb.format == View<A>::Rgb24 || bgr.format == View<A>::Bgr24);

        SimdBgrToRgb(rgb.data, rgb.width, rgb.height, rgb.stride, bgr.data, bgr.stride);
    }

    /*! @ingroup rgb_conversion

        \fn void RgbToBgra(const View<A>& rgb, View<A>& bgra, uint8_t alpha = 0xFF)

        \short Converts 24-bit RGB image to 32-bit BGRA image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdRgbToBgra.

        \param [in] rgb - an input 24-bit RGB image.
        \param [out] bgra - an output 32-bit BGRA image.
        \param [in] alpha - a value of alpha channel. It is equal to 256 by default.
    */
    template<template<class> class A> SIMD_INLINE void RgbToBgra(const View<A>& rgb, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(rgb, bgra) && bgra.format == View<A>::Bgra32 && rgb.format == View<A>::Rgb24);

        SimdRgbToBgra(rgb.data, rgb.width, rgb.height, rgb.stride, bgra.data, bgra.stride, alpha);
    }

    /*! @ingroup rgb_conversion

        \fn void RgbToGray(const View<A>& rgb, View<A>& gray)

        \short Converts 24-bit RGB image to 8-bit gray image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdRgbToGray.

        \param [in] rgb - an input 24-bit RGB image.
        \param [out] gray - an output 8-bit gray image.
    */
    template<template<class> class A> SIMD_INLINE void RgbToGray(const View<A>& rgb, View<A>& gray)
    {
        assert(EqualSize(rgb, gray) && rgb.format == View<A>::Rgb24 && gray.format == View<A>::Gray8);

        SimdRgbToGray(rgb.data, rgb.width, rgb.height, rgb.stride, gray.data, gray.stride);
    }

    /*! @ingroup rgb_conversion

        \fn void RgbToRgba(const View<A>& rgb, View<A>& rgba, uint8_t alpha = 0xFF)

        \short Converts 24-bit RGB image to 32-bit RGBA image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgrToBgra.

        \param [in] rgb - an input 24-bit RGB image.
        \param [out] rgba - an output 32-bit RGBA image.
        \param [in] alpha - a value of alpha channel. It is equal to 256 by default.
    */
    template<template<class> class A> SIMD_INLINE void RgbToRgba(const View<A>& rgb, View<A>& rgba, uint8_t alpha = 0xFF)
    {
        assert(EqualSize(rgb, rgba) && rgba.format == View<A>::Rgba32 && rgb.format == View<A>::Rgb24);

        SimdBgrToBgra(rgb.data, rgb.width, rgb.height, rgb.stride, rgba.data, rgba.stride, alpha);
    }

    /*! @ingroup rgba_conversion

        \fn void RgbaToBgr(const View<A>& rgba, View<A>& bgr)

        \short Converts 32-bit RGBA image to 24-bit BGR image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgraToRgb.

        \param [in] rgba - an input 32-bit RGBA image.
        \param [out] bgr - an output 24-bit RGB image.
    */
    template<template<class> class A> SIMD_INLINE void RgbaToBgr(const View<A>& rgba, View<A>& bgr)
    {
        assert(EqualSize(rgba, bgr) && rgba.format == View<A>::Rgba32 && bgr.format == View<A>::Bgr24);

        SimdBgraToRgb(rgba.data, rgba.width, rgba.height, rgba.stride, bgr.data, bgr.stride);
    }

    /*! @ingroup rgba_conversion

        \fn void RgbaToBgra(const View<A>& rgba, View<A>& bgra)

        \short Converts 32-bit RGBA image to 32-bit BGRA image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgraToRgba.

        \param [in] rgba - an input 32-bit RGBA image.
        \param [out] bgra - an output 32-bit BGRA image.
    */
    template<template<class> class A> SIMD_INLINE void RgbaToBgra(const View<A>& rgba, View<A>& bgra)
    {
        assert(EqualSize(bgra, rgba) && bgra.format == View<A>::Bgra32 && rgba.format == View<A>::Rgba32);

        SimdBgraToRgba(rgba.data, rgba.width, rgba.height, rgba.stride, bgra.data, bgra.stride);
    }

    /*! @ingroup rgba_conversion

        \fn void RgbaToGray(const View<A>& rgba, View<A>& gray)

        \short Converts 32-bit RGBA image to 8-bit gray image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdRgbaToGray.

        \param [in] rgba - an input 32-bit RGBA image.
        \param [out] gray - an output 8-bit gray image.
    */
    template<template<class> class A> SIMD_INLINE void RgbaToGray(const View<A>& rgba, View<A>& gray)
    {
        assert(EqualSize(rgba, gray) && rgba.format == View<A>::Rgba32 && gray.format == View<A>::Gray8);

        SimdRgbaToGray(rgba.data, rgba.width, rgba.height, rgba.stride, gray.data, gray.stride);
    }

    /*! @ingroup rgba_conversion

        \fn void RgbaToRgb(const View<A>& rgba, View<A>& rgb)

        \short Converts 32-bit RGBA image to 24-bit RGB image.

        All images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdBgraToBgr.

        \param [in] rgba - an input 32-bit RGBA image.
        \param [out] rgb - an output 24-bit RGB image.
    */
    template<template<class> class A> SIMD_INLINE void RgbaToRgb(const View<A>& rgba, View<A>& rgb)
    {
        assert(EqualSize(rgba, rgb) && rgba.format == View<A>::Rgba32 && rgb.format == View<A>::Rgb24);

        SimdBgraToBgr(rgba.data, rgba.width, rgba.height, rgba.stride, rgb.data, rgb.stride);
    }

    /*! @ingroup segmentation

        \fn void SegmentationChangeIndex(View<A> & mask, uint8_t oldIndex, uint8_t newIndex)

        \short Changes certain index in mask.

        Mask must has 8-bit gray pixel format.

        For every point:
        \verbatim
        if(mask[i] == oldIndex)
            mask[i] = newIndex;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSegmentationChangeIndex.

        \param [in, out] mask - a 8-bit gray mask image.
        \param [in] oldIndex - a mask old index.
        \param [in] newIndex - a mask new index.
    */
    template<template<class> class A> SIMD_INLINE void SegmentationChangeIndex(View<A> & mask, uint8_t oldIndex, uint8_t newIndex)
    {
        assert(mask.format == View<A>::Gray8);

        SimdSegmentationChangeIndex(mask.data, mask.stride, mask.width, mask.height, oldIndex, newIndex);
    }

    /*! @ingroup segmentation

        \fn void SegmentationFillSingleHoles(View<A> & mask, uint8_t index)

        \short Fill single holes in mask.

        Mask must has 8-bit gray pixel format.

        \note This function is a C++ wrapper for function ::SimdSegmentationFillSingleHoles.

        \param [in, out] mask - a 8-bit gray mask image.
        \param [in] index - a mask index.
    */
    template<template<class> class A> SIMD_INLINE void SegmentationFillSingleHoles(View<A> & mask, uint8_t index)
    {
        assert(mask.format == View<A>::Gray8 && mask.width > 2 && mask.height > 2);

        SimdSegmentationFillSingleHoles(mask.data, mask.stride, mask.width, mask.height, index);
    }

    /*! @ingroup segmentation

        \fn void SegmentationPropagate2x2(const View<A> & parent, View<A> & child, const View<A> & difference, uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)

        \short Propagates mask index from parent (upper) to child (lower) level of mask pyramid with using 2x2 scan window.

        For parent and child image must be performed: parent.width = (child.width + 1)/2, parent.height = (child.height + 1)/2.
        All images must have 8-bit gray pixel format. Size of different image is equal to child image.

        \note This function is a C++ wrapper for function ::SimdSegmentationPropagate2x2.

        \param [in] parent - a 8-bit gray parent mask image.
        \param [in, out] child - a 8-bit gray child mask image.
        \param [in] difference - a 8-bit gray difference image.
        \param [in] currentIndex - propagated mask index.
        \param [in] invalidIndex - invalid mask index.
        \param [in] emptyIndex - empty mask index.
        \param [in] differenceThreshold - a difference threshold for conditional index propagating.
    */
    template<template<class> class A> SIMD_INLINE void SegmentationPropagate2x2(const View<A> & parent, View<A> & child, const View<A> & difference, uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold)
    {
        assert(parent.format == View<A>::Gray8 && parent.width >= 2 && parent.height >= 2);
        assert((child.width + 1) / 2 == parent.width && (child.height + 1) / 2 == parent.height);
        assert(Compatible(child, difference) && child.format == View<A>::Gray8);

        SimdSegmentationPropagate2x2(parent.data, parent.stride, parent.width, parent.height, child.data, child.stride,
            difference.data, difference.stride, currentIndex, invalidIndex, emptyIndex, differenceThreshold);
    }

    /*! @ingroup segmentation

        \fn void SegmentationShrinkRegion(const View<A> & mask, uint8_t index, Rectangle<ptrdiff_t> & rect)

        \short Finds actual region of mask index location.

        Mask must has 8-bit gray pixel format.

        \note This function is a C++ wrapper for function ::SimdSegmentationShrinkRegion.

        \param [in] mask - a 8-bit gray mask image.
        \param [in] index - a mask index.
        \param [in, out] rect - a region bounding box rectangle.
    */
    template<template<class> class A> SIMD_INLINE void SegmentationShrinkRegion(const View<A> & mask, uint8_t index, Rectangle<ptrdiff_t> & rect)
    {
        assert(mask.format == View<A>::Gray8);
        assert(rect.Width() > 0 && rect.Height() > 0 && Rectangle<ptrdiff_t>(mask.Size()).Contains(rect));

        SimdSegmentationShrinkRegion(mask.data, mask.stride, mask.width, mask.height, index, &rect.left, &rect.top, &rect.right, &rect.bottom);
    }

    /*! @ingroup shifting

        \fn void ShiftBilinear(const View<A> & src, const View<A> & bkg, const Point<double> & shift, const Rectangle<ptrdiff_t> & crop, View<A> & dst)

        \short Performs shifting of input image with using bilinear interpolation.

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function is a C++ wrapper for function ::SimdShiftBilinear.

        \param [in] src - a foreground input image.
        \param [in] bkg - a background input image.
        \param [in] shift - an image shift.
        \param [in] crop - a crop rectangle.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void ShiftBilinear(const View<A> & src, const View<A> & bkg, const Point<double> & shift, const Rectangle<ptrdiff_t> & crop, View<A> & dst)
    {
        assert(Compatible(src, bkg, dst) && src.ChannelSize() == 1);

        SimdShiftBilinear(src.data, src.stride, src.width, src.height, src.ChannelCount(), bkg.data, bkg.stride,
            &shift.x, &shift.y, crop.left, crop.top, crop.right, crop.bottom, dst.data, dst.stride);
    }

    /*! @ingroup sobel_filter

        \fn void SobelDx(const View<A>& src, View<A>& dst)

        \short Calculates Sobel's filter along x axis.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \verbatim
        dst[x, y] = (src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1]).
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSobelDx.

        \param [in] src - an input image.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void SobelDx(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdSobelDx(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup sobel_filter

        \fn void SobelDxAbs(const View<A>& src, View<A>& dst)

        \short Calculates absolute value of Sobel's filter along x axis.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \verbatim
        dst[x, y] = abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1])).
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSobelDxAbs.

        \param [in] src - an input image.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void SobelDxAbs(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdSobelDxAbs(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup sobel_statistic

        \fn void SobelDxAbsSum(const View<A>& src, uint64_t & sum)

        \short Calculates sum of absolute value of Sobel's filter along x axis.

        Input image must has 8-bit gray format.

        For every point:
        \verbatim
        sum += abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1]));
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSobelDxAbsSum.

        \param [in] src - an input image.
        \param [out] sum - an unsigned 64-bit integer value with result sum.
    */
    template<template<class> class A> SIMD_INLINE void SobelDxAbsSum(const View<A>& src, uint64_t & sum)
    {
        assert(src.format == View<A>::Gray8);

        SimdSobelDxAbsSum(src.data, src.stride, src.width, src.height, &sum);
    }

    /*! @ingroup sobel_filter

        \fn void SobelDy(const View<A>& src, View<A>& dst)

        \short Calculates Sobel's filter along y axis.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \verbatim
        dst[x, y] = (src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]);
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSobelDy.

        \param [in] src - an input image.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void SobelDy(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdSobelDy(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup sobel_filter

        \fn void SobelDyAbs(const View<A>& src, View<A>& dst)

        \short Calculates absolute value of Sobel's filter along y axis.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \verbatim
        dst[x, y] = abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]));
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSobelDyAbs.

        \param [in] src - an input image.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void SobelDyAbs(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdSobelDyAbs(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup sobel_statistic

        \fn void SobelDyAbsSum(const View<A>& src, uint64_t & sum)

        \short Calculates sum of absolute value of Sobel's filter along y axis.

        Input image must has 8-bit gray format.

        For every point:
        \verbatim
        sum += abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]));
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSobelDyAbsSum.

        \param [in] src - an input image.
        \param [out] sum - an unsigned 64-bit integer value with result sum.
    */
    template<template<class> class A> SIMD_INLINE void SobelDyAbsSum(const View<A>& src, uint64_t & sum)
    {
        assert(src.format == View<A>::Gray8);

        SimdSobelDyAbsSum(src.data, src.stride, src.width, src.height, &sum);
    }

    /*! @ingroup contour

        \fn void ContourMetrics(const View<A>& src, View<A>& dst)

        \short Calculates contour metrics based on absolute value and direction of Sobel's filter along y and y axis.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.
        This function is used for contour extraction.

        For every point:
        \verbatim
        dy = abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]));
        dx = abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1]));
        dst[x, y] = (dx + dy)*2 + (dx >= dy ? 0 : 1);
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdContourMetrics.

        \param [in] src - a gray 8-bit input image.
        \param [out] dst - an output 16-bit image.
    */
    template<template<class> class A> SIMD_INLINE void ContourMetrics(const View<A>& src, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdContourMetrics(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
    }

    /*! @ingroup contour

        \fn void ContourMetrics(const View<A>& src, const View<A>& mask, uint8_t indexMin, View<A>& dst)

        \short Calculates contour metrics based on absolute value and direction of Sobel's filter along y and y axis with using mask.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.
        This function is used for contour extraction.

        For every point:
        \verbatim
        dy = abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]));
        dx = abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1]));
        dst[x, y] = mask[x, y] < indexMin ? 0 : (dx + dy)*2 + (dx >= dy ? 0 : 1);
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdContourMetricsMasked.

        \param [in] src - a  gray 8-bit input image.
        \param [in] mask - a mask 8-bit image.
        \param [in] indexMin - a mask minimal permissible index.
        \param [out] dst - an output 16-bit image.
    */
    template<template<class> class A> SIMD_INLINE void ContourMetrics(const View<A>& src, const View<A>& mask, uint8_t indexMin, View<A>& dst)
    {
        assert(Compatible(src, mask) && EqualSize(src, dst) && src.format == View<A>::Gray8 && dst.format == View<A>::Int16);

        SimdContourMetricsMasked(src.data, src.stride, src.width, src.height, mask.data, mask.stride, indexMin, dst.data, dst.stride);
    }

    /*! @ingroup contour

        \fn void ContourAnchors(const View<A>& src, size_t step, int16_t threshold, View<A>& dst)

        \short Extract contour anchors from contour metrics.

        All images must have the same width and height. Input image must has 16-bit integer format, output image must has 8-bit gray format.
        Input image with metrics can be estimated by using ::SimdContourMetrics or ::SimdContourMetricsMasked functions.
        This function is used for contour extraction.

        For every point (except border):
        \verbatim
        a[x, y] = src[x, y] >> 1.
        if(src[x, y] & 1)
            dst[x, y] = a[x, y] > 0 && (a[x, y] - a[x + 1, y] >= threshold) && (a[x, y] - a[x - 1, y] >= threshold) ? 255 : 0;
        else
            dst[x, y] = a[x, y] > 0 && (a[x, y] - a[x, y + 1] >= threshold) && (a[x, y] - a[x, y - 1] >= threshold) ? 255 : 0;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdContourAnchors.

        \param [in] src - a 16-bit input image.
        \param [in] step - a row step (to skip some rows).
        \param [in] threshold - a threshold of anchor creation.
        \param [out] dst - an output 8-bit gray image.
    */
    template<template<class> class A> SIMD_INLINE void ContourAnchors(const View<A>& src, size_t step, int16_t threshold, View<A>& dst)
    {
        assert(EqualSize(src, dst) && src.format == View<A>::Int16 && dst.format == View<A>::Gray8);

        SimdContourAnchors(src.data, src.stride, src.width, src.height, step, threshold, dst.data, dst.stride);
    }

    /*! @ingroup correlation

        \fn void SquaredDifferenceSum(const View<A>& a, const View<A>& b, uint64_t & sum)

        \short Calculates sum of squared differences for two 8-bit gray images.

        All images must have the same width and height.

        For every point:
        \verbatim
        sum += (a[i] - b[i])*(a[i] - b[i]);
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSquaredDifferenceSum.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [out] sum - a reference to unsigned 64-bit integer value with result sum.
    */
    template<template<class> class A> SIMD_INLINE void SquaredDifferenceSum(const View<A>& a, const View<A>& b, uint64_t & sum)
    {
        assert(Compatible(a, b) && a.format == View<A>::Gray8);

        SimdSquaredDifferenceSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
    }

    /*! @ingroup correlation

        \fn void SquaredDifferenceSum(const View<A>& a, const View<A>& b, const View<A>& mask, uint8_t index, uint64_t & sum)

        \short Calculates sum of squared differences for two images with using mask.

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        if(mask[i] == index)
            sum += (a[i] - b[i])*(a[i] - b[i]);
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSquaredDifferenceSumMasked.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] mask - a mask image.
        \param [in] index - a mask index.
        \param [out] sum - a reference to unsigned 64-bit integer value with result sum.
    */
    template<template<class> class A> SIMD_INLINE void SquaredDifferenceSum(const View<A>& a, const View<A>& b, const View<A>& mask, uint8_t index, uint64_t & sum)
    {
        assert(Compatible(a, b, mask) && a.format == View<A>::Gray8);

        SimdSquaredDifferenceSumMasked(a.data, a.stride, b.data, b.stride, mask.data, mask.stride, index, a.width, a.height, &sum);
    }

    /*! @ingroup other_statistic

        \fn void GetStatistic(const View<A>& src, uint8_t & min, uint8_t & max, uint8_t & average)

        \short Finds minimal, maximal and average pixel values for given image.

        The image must has 8-bit gray format.

        \note This function is a C++ wrapper for function ::SimdGetStatistic.

        \param [in] src - an input image.
        \param [out] min - a reference to unsigned 8-bit integer value with found minimal pixel value.
        \param [out] max - a reference to unsigned 8-bit integer value with found maximal pixel value.
        \param [out] average - a reference to unsigned 8-bit integer value with found average pixel value.
    */
    template<template<class> class A> SIMD_INLINE void GetStatistic(const View<A>& src, uint8_t & min, uint8_t & max, uint8_t & average)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetStatistic(src.data, src.stride, src.width, src.height, &min, &max, &average);
    }

    /*! @ingroup other_statistic

        \fn void GetMoments(const View<A>& mask, uint8_t index, uint64_t & area, uint64_t & x, uint64_t & y, uint64_t & xx, uint64_t & xy, uint64_t & yy)

        \short Calculate statistical characteristics (moments) of pixels with given index.

        The image must has 8-bit gray format.

        For every point:
        \verbatim
        if(mask[X, Y] == index)
        {
            area += 1.
            x += X.
            y += Y.
            xx += X*X.
            xy += X*Y.
            yy += Y*Y.
        }
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdGetMoments.

        \param [in] mask - a mask image.
        \param [in] index - a mask index.
        \param [out] area - a reference to unsigned 64-bit integer value with found area (number of pixels with given index).
        \param [out] x - a reference to unsigned 64-bit integer value with found first-order moment x.
        \param [out] y - a reference to unsigned 64-bit integer value with found first-order moment y.
        \param [out] xx - a reference to unsigned 64-bit integer value with found second-order moment xx.
        \param [out] xy - a reference to unsigned 64-bit integer value with found second-order moment xy.
        \param [out] yy - a reference to unsigned 64-bit integer value with found second-order moment yy.
    */
    template<template<class> class A> SIMD_INLINE void GetMoments(const View<A>& mask, uint8_t index, uint64_t & area, uint64_t & x, uint64_t & y, uint64_t & xx, uint64_t & xy, uint64_t & yy)
    {
        assert(mask.format == View<A>::Gray8);

        SimdGetMoments(mask.data, mask.stride, mask.width, mask.height, index, &area, &x, &y, &xx, &xy, &yy);
    }


    /*! @ingroup other_statistic

        \fn void GetObjectMoments(const View<A> & src, const View<A> & mask, uint8_t index, uint64_t & n, uint64_t & s, uint64_t & sx, uint64_t & sy, uint64_t & sxx, uint64_t & sxy, uint64_t & syy)

        \short Calculate statistical characteristics (moments) of given object.

        The images must has 8-bit gray format and equal size. One of them can be empty.

        For every point:
        \verbatim
        if(mask[X, Y] == index || mask == 0)
        {
            S = src ? src[X, Y] : 1;
            n += 1.
            s += S;
            sx += S*X.
            sy += S*Y.
            sxx += S*X*X.
            sxy += S*X*Y.
            syy += S*Y*Y.
        }
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdGetObjectMoments.

        \param [in] src - an input image.
        \param [in] mask - a mask image. Can be empty.
        \param [in] index - an object index.
        \param [out] n - a reference to unsigned 64-bit integer value with found are of given object.
        \param [out] s - a reference to unsigned 64-bit integer value with sum of image values of given object.
        \param [out] sx - a reference to unsigned 64-bit integer value with found first-order moment x of given object.
        \param [out] sy - a reference to unsigned 64-bit integer value with found first-order moment y of given object.
        \param [out] sxx - a reference to unsigned 64-bit integer value with found second-order moment xx of given object.
        \param [out] sxy - a reference to unsigned 64-bit integer value with found second-order moment xy of given object.
        \param [out] syy - a reference to unsigned 64-bit integer value with found second-order moment yy of given object.
    */
    template<template<class> class A> SIMD_INLINE void GetObjectMoments(const View<A> & src, const View<A> & mask, uint8_t index, uint64_t & n, uint64_t & s, uint64_t & sx, uint64_t & sy, uint64_t & sxx, uint64_t & sxy, uint64_t & syy)
    {
        assert(src.format == View<A>::Empty || src.format == View<A>::Gray8);
        assert(mask.format == View<A>::Empty || mask.format == View<A>::Gray8);
        assert(src.format == View<A>::Gray8 || mask.format == View<A>::Gray8);
        assert(src.format == mask.format ? EqualSize(src, mask) : true);

        if (src.format)
            SimdGetObjectMoments(src.data, src.stride, src.width, src.height, mask.data, mask.stride, index, &n, &s, &sx, &sy, &sxx, &sxy, &syy);
        else
            SimdGetObjectMoments(src.data, src.stride, mask.width, mask.height, mask.data, mask.stride, index, &n, &s, &sx, &sy, &sxx, &sxy, &syy);
    }

    /*! @ingroup row_statistic

        \fn void GetRowSums(const View<A>& src, uint32_t * sums)

        \short Calculate sums of rows for given 8-bit gray image.

        For all rows:
        \verbatim
        for(x = 0; x < width; ++x)
            sums[y] += src[x, y];
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdGetRowSums.

        \param [in] src - an input image.
        \param [out] sums - a pointer to array of unsigned 32-bit integers result sums of rows. It length must be equal to image height.
    */
    template<template<class> class A> SIMD_INLINE void GetRowSums(const View<A>& src, uint32_t * sums)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetRowSums(src.data, src.stride, src.width, src.height, sums);
    }

    /*! @ingroup col_statistic

        \fn void GetColSums(const View<A>& src, uint32_t * sums)

        \short Calculate sums of columns for given 8-bit gray image.

        For all columns:
        \verbatim
        for(y = 0; y < height; ++y)
            sums[x] += src[x, y];
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdGetColSums.

        \param [in] src - an input image.
        \param [out] sums - a pointer to array of unsigned 32-bit integers result sums of columns. It length must be equal to image width.
    */
    template<template<class> class A> SIMD_INLINE void GetColSums(const View<A>& src, uint32_t * sums)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetColSums(src.data, src.stride, src.width, src.height, sums);
    }

    /*! @ingroup row_statistic

        \fn void GetAbsDyRowSums(const View<A>& src, uint32_t * sums)

        \short Calculate sums of absolute derivate along y axis for rows for given 8-bit gray image.

        For all rows except the last:
        \verbatim
        for(x = 0; x < width; ++x)
            sums[y] += abs(src[x, y+1] - src[x, y]);
        \endverbatim
        For the last row:
        \verbatim
        sums[height-1] = 0;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdGetAbsDyRowSums.

        \param [in] src - an input image.
        \param [out] sums - a pointer to array of unsigned 32-bit integers result sums. It length must be equal to image height.
    */
    template<template<class> class A> SIMD_INLINE void GetAbsDyRowSums(const View<A>& src, uint32_t * sums)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetAbsDyRowSums(src.data, src.stride, src.width, src.height, sums);
    }

    /*! @ingroup col_statistic

        \fn void GetAbsDxColSums(const View<A>& src, uint32_t * sums)

        \short Calculate sums of absolute derivate along x axis for columns for given 8-bit gray image.

        For all columns except the last:
        \verbatim
        for(y = 0; y < height; ++y)
            sums[y] += abs(src[x+1, y] - src[x, y]);
        \endverbatim
        For the last column:
        \verbatim
        sums[width-1] = 0;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdGetAbsDxColSums.

        \param [in] src - an input image.
        \param [out] sums - a pointer to array of unsigned 32-bit integers result columns. It length must be equal to image width.
    */
    template<template<class> class A> SIMD_INLINE void GetAbsDxColSums(const View<A>& src, uint32_t * sums)
    {
        assert(src.format == View<A>::Gray8);

        SimdGetAbsDxColSums(src.data, src.stride, src.width, src.height, sums);
    }

    /*! @ingroup other_statistic

        \fn void ValueSum(const View<A>& src, uint64_t & sum)

        \short Gets sum of value of pixels for gray 8-bit image.

        \note This function is a C++ wrapper for function ::SimdValueSum.

        \param [in] src - an input image.
        \param [out] sum - a result sum.
    */
    template<template<class> class A> SIMD_INLINE void ValueSum(const View<A>& src, uint64_t & sum)
    {
        assert(src.format == View<A>::Gray8);

        SimdValueSum(src.data, src.stride, src.width, src.height, &sum);
    }

    /*! @ingroup other_statistic

        \fn void SquareSum(const View<A>& src, uint64_t & sum)

        \short Gets sum of squared value of pixels for gray 8-bit image.

        \note This function is a C++ wrapper for function ::SimdSquareSum.

        \param [in] src - an input image.
        \param [out] sum - a result sum.
    */
    template<template<class> class A> SIMD_INLINE void SquareSum(const View<A> & src, uint64_t & sum)
    {
        assert(src.format == View<A>::Gray8);

        SimdSquareSum(src.data, src.stride, src.width, src.height, &sum);
    }
    
    /*! @ingroup other_statistic

        \fn void ValueSquareSum(const View<A>& src, uint64_t & valueSum, uint64_t & squareSum)

        \short Gets sum and sum of squared value of pixels for gray 8-bit image.

        \note This function is a C++ wrapper for function ::SimdValueSquareSum.

        \param [in] src - an input image.
        \param [out] valueSum - a result value sum.
        \param [out] squareSum - a result square sum.
    */
    template<template<class> class A> SIMD_INLINE void ValueSquareSum(const View<A>& src, uint64_t & valueSum, uint64_t & squareSum)
    {
        assert(src.format == View<A>::Gray8);

        SimdValueSquareSum(src.data, src.stride, src.width, src.height, &valueSum, &squareSum);
    }

    /*! @ingroup other_statistic

        \fn void ValueSquareSums(const View<A>& src, uint64_t * valueSums, uint64_t * squareSums)

        \short Gets image channels value sums and squared value sums for image. The image must have 8-bit depth per channel.

        \note This function is a C++ wrapper for function ::SimdValueSquareSums.

        \param [in] src - an input image.
        \param [out] valueSums - the pointer to output buffer with value sums. Size of the buffer must be equal to count of image channels.
        \param [out] squareSums - the pointer to output buffer with square sums. Size of the buffer must be equal to count of image channels.
    */
    template<template<class> class A> SIMD_INLINE void ValueSquareSums(const View<A>& src, uint64_t * valueSums, uint64_t * squareSums)
    {
        assert(src.ChannelSize() == 1);

        SimdValueSquareSums(src.data, src.stride, src.width, src.height, src.ChannelCount(), valueSums, squareSums);
    }

    /*! @ingroup other_statistic

        \fn void CorrelationSum(const View<A> & a, const View<A> & b, uint64_t & sum)

        \short Gets sum of pixel correlation for two gray 8-bit images.

        For all points:
        \verbatim
        sum += a[i]*b[i];
        \endverbatim

        All images must have the same width and height and 8-bit gray pixel format.

        \note This function is a C++ wrapper for function ::SimdCorrelationSum.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [out] sum - a result sum.
    */
    template<template<class> class A> SIMD_INLINE void CorrelationSum(const View<A> & a, const View<A> & b, uint64_t & sum)
    {
        assert(Compatible(a, b) && a.format == View<A>::Gray8);

        SimdCorrelationSum(a.data, a.stride, b.data, b.stride, a.width, a.height, &sum);
    }

    /*! @ingroup resizing

        \fn void StretchGray2x2(const View<A>& src, View<A>& dst)

        \short Stretches input 8-bit gray image in two times.

        \note This function is a C++ wrapper for function ::SimdStretchGray2x2.

        \param [in] src - an original input image.
        \param [out] dst - a stretched output image.
    */
    template<template<class> class A> SIMD_INLINE void StretchGray2x2(const View<A> & src, View<A> & dst)
    {
        assert(src.format == View<A>::Gray8 && dst.format == View<A>::Gray8);
        assert(src.width * 2 == dst.width && src.height * 2 == dst.height);

        SimdStretchGray2x2(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
    }

    /*! @ingroup synet_conversion

        \fn void SynetSetInput(const View<A> & src, const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType format)

        \short Sets image to the input of neural network of <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        Algorithm's details (example for BGRA pixel format and NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(y = 0; y < src.height; ++y)
                for(x = 0; x < src.width; ++x)
                    dst[(c*height + y)*width + x] = src.data[src.stride*y + src.width*4 + c]*(upper[c] - lower[c])/255 + lower[c];
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdSynetSetInput.

        \param [in] src - an input image.There are supported following image formats: View<A>::Gray8, View<A>::Bgr24, View<A>::Bgra32, View<A>::Rgb24.
        \param [in] lower - a pointer to the array with lower bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
        \param [in] upper - a pointer to the array with upper bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
        \param [out] dst - a pointer to the output 32-bit float image tensor.
        \param [in] channels - a number of channels in the output image tensor. It can be 1 or 3.
        \param [in] format - a format of output image tensor. There are supported following tensor formats: ::SimdTensorFormatNchw, ::SimdTensorFormatNhwc.
    */
    template<template<class> class A> SIMD_INLINE void SynetSetInput(const View<A> & src, const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType format)
    {
        assert(src.format == View<A>::Gray8 || src.format == View<A>::Bgr24 || src.format == View<A>::Bgra32 || src.format == View<A>::Rgb24);
        assert(format == SimdTensorFormatNchw || format == SimdTensorFormatNhwc);

        SimdSynetSetInput(src.data, src.width, src.height, src.stride, (SimdPixelFormatType)src.format, lower, upper, dst, channels, format);
    }

    /*! @ingroup texture_estimation

        \fn void TextureBoostedSaturatedGradient(const View<A>& src, uint8_t saturation, uint8_t boost, View<A>& dx, View<A>& dy)

        \short Calculates boosted saturated gradients for given input image.

        All images must have the same width, height and format (8-bit gray).

        For border pixels:
        \verbatim
        dx[x, y] = 0;
        dy[x, y] = 0;
        \endverbatim
        For other pixels:
        \verbatim
        dx[x, y] = (saturation + max(-saturation, min(saturation, (src[x + 1, y] - src[x - 1, y]))))*boost;
        dy[x, y] = (saturation + max(-saturation, min(saturation, (src[x, y + 1] - src[x, y - 1]))))*boost;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdTextureBoostedSaturatedGradient.

        \param [in] src - a source 8-bit gray image.
        \param [in] saturation - a saturation of gradient.
        \param [in] boost - a boost coefficient.
        \param [out] dx - an image with boosted saturated gradient along x axis.
        \param [out] dy - an image with boosted saturated gradient along y axis.
    */
    template<template<class> class A> SIMD_INLINE void TextureBoostedSaturatedGradient(const View<A>& src, uint8_t saturation, uint8_t boost, View<A>& dx, View<A>& dy)
    {
        assert(Compatible(src, dx, dy) && src.format == View<A>::Gray8 && src.height >= 3 && src.width >= 3);

        SimdTextureBoostedSaturatedGradient(src.data, src.stride, src.width, src.height, saturation, boost, dx.data, dx.stride, dy.data, dy.stride);
    }

    /*! @ingroup texture_estimation

        \fn void TextureBoostedUv(const View<A>& src, uint8_t boost, View<A>& dst)

        \short Calculates boosted colorized texture feature of input image (actual for U and V components of YUV format).

        All images must have the same width, height and format (8-bit gray).

        For every pixel:
        \verbatim
        lo = 128 - (128/boost);
        hi = 255 - lo;
        dst[x, y] = max(lo, min(hi, src[i]))*boost;
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdTextureBoostedUv.

        \param [in] src - a source 8-bit gray image.
        \param [in] boost - a boost coefficient.
        \param [out] dst - a result image.
    */
    template<template<class> class A> SIMD_INLINE void TextureBoostedUv(const View<A>& src, uint8_t boost, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8);

        SimdTextureBoostedUv(src.data, src.stride, src.width, src.height, boost, dst.data, dst.stride);
    }

    /*! @ingroup texture_estimation

        \fn void TextureGetDifferenceSum(const View<A>& src, const View<A>& lo, const View<A>& hi, int64_t & sum)

        \short Calculates difference between current image and background.

        All images must have the same width, height and format (8-bit gray).

        For every pixel:
        \verbatim
        sum += current - average(lo[i], hi[i]);
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdTextureGetDifferenceSum.

        \param [in] src - a current image.
        \param [in] lo - an image with lower bound of background feature.
        \param [in] hi - an image with upper bound of background feature.
        \param [out] sum - a reference to 64-bit integer with result sum.
    */
    template<template<class> class A> SIMD_INLINE void TextureGetDifferenceSum(const View<A>& src, const View<A>& lo, const View<A>& hi, int64_t & sum)
    {
        assert(Compatible(src, lo, hi) && src.format == View<A>::Gray8);

        SimdTextureGetDifferenceSum(src.data, src.stride, src.width, src.height, lo.data, lo.stride, hi.data, hi.stride, &sum);
    }

    /*! @ingroup texture_estimation

        \fn void TexturePerformCompensation(const View<A>& src, int shift, View<A>& dst)

        \short Performs brightness compensation of input image.

        All images must have the same width, height and format (8-bit gray).

        For every pixel:
        \verbatim
        dst[i] = max(0, min(255, src[i] + shift));
        \endverbatim

        \note This function is a C++ wrapper for function ::SimdTexturePerformCompensation.

        \param [in] src - an input image.
        \param [in] shift - a compensation shift.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void TexturePerformCompensation(const View<A>& src, int shift, View<A>& dst)
    {
        assert(Compatible(src, dst) && src.format == View<A>::Gray8 && shift > -0xFF && shift < 0xFF);

        SimdTexturePerformCompensation(src.data, src.stride, src.width, src.height, shift, dst.data, dst.stride);
    }

    /*! @ingroup transform

        \fn Point<ptrdiff_t> TransformSize(const Point<ptrdiff_t> & size, ::SimdTransformType transform);

        \short Gets size of transformed image.

        \param [in] size - a size of input image.
        \param [in] transform - a type of image transformation.
        \return - the size of transformed image.
    */
    SIMD_INLINE Point<ptrdiff_t> TransformSize(const Point<ptrdiff_t> & size, ::SimdTransformType transform)
    {
        switch (transform)
        {
        case ::SimdTransformRotate0:
        case ::SimdTransformRotate180:
        case ::SimdTransformTransposeRotate90:
        case ::SimdTransformTransposeRotate270:
            return size;
        case ::SimdTransformRotate90:
        case ::SimdTransformRotate270:
        case ::SimdTransformTransposeRotate0:
        case ::SimdTransformTransposeRotate180:
            return Point<ptrdiff_t>(size.y, size.x);
        default:
            assert(0);
            return Point<ptrdiff_t>();
        }
    }

    /*! @ingroup transform

        \fn void TransformImage(const View<A> & src, ::SimdTransformType transform, View<A> & dst);

        \short Performs transformation of input image. The type of transformation is defined by ::SimdTransformType enumeration.

        \note This function is a C++ wrapper for function ::SimdTransformImage.

        \param [in] src - an input image.
        \param [in] transform - a type of image transformation.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void TransformImage(const View<A> & src, ::SimdTransformType transform, View<A> & dst)
    {
        assert(src.format == dst.format && TransformSize(src.Size(), transform) == dst.Size());

        SimdTransformImage(src.data, src.stride, src.width, src.height, src.PixelSize(), transform, dst.data, dst.stride);
    }

    /*! @ingroup uyvy_conversion

        \fn void Uyvy422ToBgr(const View<A>& uyvy, View<A>& bgr, SimdYuvType yuvType = SimdYuvBt601);

        \short Converts 16-bit UYVY422 image to 24-bit BGR image.

        The input and output images must have the same width and height. Width must be even number.

        \note This function is a C++ wrapper for function ::SimdUyvy422ToBgr.

        \param [in] uyvy - an input 16-bit UYVY422 image.
        \param [out] bgr - an output 24-bit BGR image.
        \param [in] yuvType - a type of input YUV image (see descriprion of ::SimdYuvType). By default is equal to ::SimdYuvBt601.
    */
    template<template<class> class A> SIMD_INLINE void Uyvy422ToBgr(const View<A>& uyvy, View<A>& bgr, SimdYuvType yuvType = SimdYuvBt601)
    {
        assert(EqualSize(uyvy, bgr) && uyvy.format == View<A>::Uyvy32 && bgr.format == View<A>::Bgr24);

        SimdUyvy422ToBgr(uyvy.data, uyvy.stride, uyvy.width, uyvy.height, bgr.data, bgr.stride, yuvType);
    }

    /*! @ingroup uyvy_conversion

        \fn void Uyvy422ToYuv420p(const View<A>& uyvy, View<A>& y, View<A>& u, View<A>& v);

        \short Converts 16-bit UYVY422 image to YUV420P.

        The input UYVY422 and output Y images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function is a C++ wrapper for function ::SimdUyvy422ToYuv420p.

        \param [in] uyvy - an input 16-bit UYVY422 image.
        \param [out] y - an output 8-bit image with Y color plane.
        \param [out] u - an output 8-bit image with U color plane.
        \param [out] v - an output 8-bit image with V color plane.
    */
    template<template<class> class A> SIMD_INLINE void Uyvy422ToYuv420p(const View<A>& uyvy, View<A>& y, View<A>& u, View<A>& v)
    {
        assert(y.width == uyvy.width && y.height == uyvy.height);
        assert(y.width == 2 * u.width && y.height == 2 * u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == 2 * v.height && y.format == v.format);
        assert(uyvy.format == View<A>::Uyvy32 && y.format == View<A>::Gray8);

        SimdUyvy422ToBgr(uyvy.data, uyvy.stride, uyvy.width, uyvy.height, y.data, y.stride, u.data, u.stride, v.data, v.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuva420pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, const View<A>& a, View<A>& bgra)

        \short Converts YUVA420P image to 32-bit BGRA image.

        The input Y, A and output BGRA images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function is a C++ wrapper for function ::SimdYuva420pToBgra.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [in] a - an input 8-bit image with alpha channel.
        \param [out] bgra - an output 32-bit BGRA image.
    */
    template<template<class> class A> SIMD_INLINE void Yuva420pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, const View<A>& a, View<A>& bgra)
    {
        assert(y.width == 2 * u.width && y.height == 2 * u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == 2 * v.height && y.format == v.format);
        assert(Compatible(y, a) && EqualSize(y, bgra));
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdYuva420pToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, a.data, a.stride, y.width, y.height, bgra.data, bgra.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv420pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr)

        \short Converts YUV420P image to 24-bit BGR image.

        The input Y and output BGR images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function is a C++ wrapper for function ::SimdYuv420pToBgr.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] bgr - an output 24-bit BGR image.
    */
    template<template<class> class A> SIMD_INLINE void Yuv420pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr)
    {
        assert(y.width == 2 * u.width && y.height == 2 * u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == 2 * v.height && y.format == v.format);
        assert(y.width == bgr.width && y.height == bgr.height);
        assert(y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdYuv420pToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv422pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr)

        \short Converts YUV422P image to 24-bit BGR image.

        The input Y and output BGR images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function is a C++ wrapper for function ::SimdYuv422pToBgr.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] bgr - an output 24-bit BGR image.
    */
    template<template<class> class A> SIMD_INLINE void Yuv422pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr)
    {
        assert(y.width == 2 * u.width && y.height == u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == v.height && y.format == v.format);
        assert(y.width == bgr.width && y.height == bgr.height);
        assert(y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdYuv422pToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv444pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr)

        \short Converts YUV444P image to 24-bit BGR image.

        The input Y, U, V and output BGR images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdYuv444pToBgr.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] bgr - an output 24-bit BGR image.
    */
    template<template<class> class A> SIMD_INLINE void Yuv444pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr)
    {
        assert(Compatible(y, u, v) && EqualSize(y, bgr) && y.format == View<A>::Gray8 && bgr.format == View<A>::Bgr24);

        SimdYuv444pToBgr(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv420pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha = 0xFF)

        \short Converts YUV420P image to 32-bit BGRA image.

        The input Y and output BGRA images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function is a C++ wrapper for function ::SimdYuv420pToBgra.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] bgra - an output 32-bit BGRA image.
        \param [in] alpha - a value of alpha channel. It is equal to 255 by default.
    */
    template<template<class> class A> SIMD_INLINE void Yuv420pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(y.width == 2 * u.width && y.height == 2 * u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == 2 * v.height && y.format == v.format);
        assert(y.width == bgra.width && y.height == bgra.height);
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdYuv420pToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv422pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha = 0xFF)

        \short Converts YUV422P image to 32-bit BGRA image.

        The input Y and output BGRA images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function is a C++ wrapper for function ::SimdYuv422pToBgra.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] bgra - an output 32-bit BGRA image.
        \param [in] alpha - a value of alpha channel. It is equal to 255 by default.
    */
    template<template<class> class A> SIMD_INLINE void Yuv422pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(y.width == 2 * u.width && y.height == u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == v.height && y.format == v.format);
        assert(y.width == bgra.width && y.height == bgra.height);
        assert(y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdYuv422pToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv444pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha = 0xFF)

        \short Converts YUV444P image to 32-bit BGRA image.

        The input Y, U, V and output BGRA images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdYuv444pToBgra.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] bgra - an output 32-bit BGRA image.
        \param [in] alpha - a value of alpha channel. It is equal to 255 by default.
    */
    template<template<class> class A> SIMD_INLINE void Yuv444pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha = 0xFF)
    {
        assert(Compatible(y, u, v) && EqualSize(y, bgra) && y.format == View<A>::Gray8 && bgra.format == View<A>::Bgra32);

        SimdYuv444pToBgra(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv444pToHsl(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hsl)

        \short Converts YUV444P image to 24-bit HSL(Hue, Saturation, Lightness) image.

        The input Y, U, V and output HSL images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdYuv444pToHsl.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] hsl - an output 24-bit HSL image.
    */
    template<template<class> class A> SIMD_INLINE void Yuv444pToHsl(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hsl)
    {
        assert(Compatible(y, u, v) && EqualSize(y, hsl) && y.format == View<A>::Gray8 && hsl.format == View<A>::Hsl24);

        SimdYuv444pToHsl(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hsl.data, hsl.stride);
    }

    /*! @ingroup yuv_conversion

       \fn void Yuv444pToHsv(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hsv)

       \short Converts YUV444P image to 24-bit HSV(Hue, Saturation, Value) image.

       The input Y, U, V and output HSV images must have the same width and height.

       \note This function is a C++ wrapper for function ::SimdYuv444pToHsv.

       \param [in] y - an input 8-bit image with Y color plane.
       \param [in] u - an input 8-bit image with U color plane.
       \param [in] v - an input 8-bit image with V color plane.
       \param [out] hsv - an output 24-bit HSV image.
   */
    template<template<class> class A> SIMD_INLINE void Yuv444pToHsv(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hsv)
    {
        assert(Compatible(y, u, v) && EqualSize(y, hsv) && y.format == View<A>::Gray8 && hsv.format == View<A>::Hsv24);

        SimdYuv444pToHsv(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hsv.data, hsv.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv420pToHue(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hue)

        \short Converts YUV420P image to 8-bit image with Hue component of HSV or HSL color space.

        The input Y and output Hue images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function is a C++ wrapper for function ::SimdYuv420pToHue.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] hue - an output 8-bit Hue image.
    */
    template<template<class> class A> SIMD_INLINE void Yuv420pToHue(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hue)
    {
        assert(y.width == 2 * u.width && y.height == 2 * u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == 2 * v.height && y.format == v.format);
        assert(Compatible(y, hue) && y.format == View<A>::Gray8);

        SimdYuv420pToHue(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hue.data, hue.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv444pToHue(const View<A> & y, const View<A> & u, const View<A> & v, View<A> & hue)

        \short Converts YUV444P image to 8-bit image with Hue component of HSV or HSL color space.

        The input Y, U, V and output Hue images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdYuv444pToHue.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] hue - an output 8-bit Hue image.
    */
    template<template<class> class A> SIMD_INLINE void Yuv444pToHue(const View<A> & y, const View<A> & u, const View<A> & v, View<A> & hue)
    {
        assert(Compatible(y, u, v, hue) && y.format == View<A>::Gray8);

        SimdYuv444pToHue(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, hue.data, hue.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv420pToRgb(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& rgb)

        \short Converts YUV420P image to 24-bit RGB image.

        The input Y and output RGB images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function is a C++ wrapper for function ::SimdYuv420pToRgb.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] rgb - an output 24-bit RGB image.
    */
    template<template<class> class A> SIMD_INLINE void Yuv420pToRgb(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& rgb)
    {
        assert(y.width == 2 * u.width && y.height == 2 * u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == 2 * v.height && y.format == v.format);
        assert(y.width == rgb.width && y.height == rgb.height);
        assert(y.format == View<A>::Gray8 && rgb.format == View<A>::Rgb24);

        SimdYuv420pToRgb(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, rgb.data, rgb.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv422pToRgb(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& rgb)

        \short Converts YUV422P image to 24-bit RGB image.

        The input Y and output RGB images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function is a C++ wrapper for function ::SimdYuv422pToRgb.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] rgb - an output 24-bit RGB image.
    */
    template<template<class> class A> SIMD_INLINE void Yuv422pToRgb(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& rgb)
    {
        assert(y.width == 2 * u.width && y.height == u.height && y.format == u.format);
        assert(y.width == 2 * v.width && y.height == v.height && y.format == v.format);
        assert(y.width == rgb.width && y.height == rgb.height);
        assert(y.format == View<A>::Gray8 && rgb.format == View<A>::Rgb24);

        SimdYuv422pToRgb(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, rgb.data, rgb.stride);
    }

    /*! @ingroup yuv_conversion

        \fn void Yuv444pToRgb(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& rgb)

        \short Converts YUV444P image to 24-bit RGB image.

        The input Y, U, V and output RGB images must have the same width and height.

        \note This function is a C++ wrapper for function ::SimdYuv444pToRgb.

        \param [in] y - an input 8-bit image with Y color plane.
        \param [in] u - an input 8-bit image with U color plane.
        \param [in] v - an input 8-bit image with V color plane.
        \param [out] rgb - an output 24-bit RGB image.
    */
    template<template<class> class A> SIMD_INLINE void Yuv444pToRgb(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& rgb)
    {
        assert(Compatible(y, u, v) && EqualSize(y, rgb) && y.format == View<A>::Gray8 && rgb.format == View<A>::Rgb24);

        SimdYuv444pToRgb(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, rgb.data, rgb.stride);
    }

    /*! @ingroup universal_conversion

        \fn void Convert(const View<A> & src, View<A> & dst)

        \short Converts an image of one format to an image of another format.

        The input and output images must have the same width and height.

        \note This function supports conversion between View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24 and View::Rgba32 image formats.

        \param [in] src - an input image.
        \param [out] dst - an output image.
    */
    template<template<class> class A> SIMD_INLINE void Convert(const View<A> & src, View<A> & dst)
    {
        assert(EqualSize(src, dst) && src.format && dst.format);

        if (src.format == dst.format)
        {
            Copy(src, dst);
            return;
        }

        switch (src.format)
        {
        case View<A>::Gray8:
            switch (dst.format)
            {
            case View<A>::Bgra32:
                GrayToBgra(src, dst);
                break;
            case View<A>::Rgba32:
                GrayToRgba(src, dst);
                break;
            case View<A>::Bgr24:
                GrayToBgr(src, dst);
                break;
            case View<A>::Rgb24:
                GrayToRgb(src, dst);
                break;
            default:
                assert(0);
            }
            break;

        case View<A>::Bgr24:
            switch (dst.format)
            {
            case View<A>::Bgra32:
                BgrToBgra(src, dst);
                break;
            case View<A>::Gray8:
                BgrToGray(src, dst);
                break;
            case View<A>::Rgb24:
                BgrToRgb(src, dst);
                break;
            case View<A>::Rgba32:
                BgrToRgba(src, dst);
                break;
            default:
                assert(0);
            }
            break;

        case View<A>::Rgb24:
            switch (dst.format)
            {
            case View<A>::Bgra32:
                RgbToBgra(src, dst);
                break;
            case View<A>::Bgr24:
                RgbToBgr(src, dst);
                break;
            case View<A>::Gray8:
                RgbToGray(src, dst);
                break;
            case View<A>::Rgba32:
                RgbToRgba(src, dst);
                break;
            default:
                assert(0);
            }
            break;

        case View<A>::Bgra32:
            switch (dst.format)
            {
            case View<A>::Bgr24:
                BgraToBgr(src, dst);
                break;
            case View<A>::Gray8:
                BgraToGray(src, dst);
                break;
            case View<A>::Rgb24:
                BgraToRgb(src, dst);
                break;
            case View<A>::Rgba32:
                BgraToRgba(src, dst);
                break;
            default:
                assert(0);
            }
            break;

        case View<A>::Rgba32:
            switch (dst.format)
            {
            case View<A>::Bgra32:
                RgbaToBgra(src, dst);
                break;
            case View<A>::Bgr24:
                RgbaToBgr(src, dst);
                break;
            case View<A>::Gray8:
                RgbaToGray(src, dst);
                break;
            case View<A>::Rgb24:
                RgbaToRgb(src, dst);
                break;
            default:
                assert(0);
            }
            break;

        default:
            assert(0);
        }
    }

    /*! @ingroup cpp_pyramid_functions

        \fn void Fill(Pyramid<A> & pyramid, uint8_t value)

        \short Fills pixels data of images in the pyramid by given value.

        \param [out] pyramid - a pyramid.
        \param [in] value - a value to fill the pyramid.
    */
    template<template<class> class A> SIMD_INLINE void Fill(Pyramid<A> & pyramid, uint8_t value)
    {
        for (size_t level = 0; level < pyramid.Size(); ++level)
            Simd::Fill(pyramid.At(level), value);
    }

    /*! @ingroup cpp_pyramid_functions

        \fn void Copy(const Pyramid<A> & src, Pyramid<A> & dst)

        \short Copies one pyramid to another pyramid.

        \note Input and output pyramids must have the same size.

        \param [in] src - an input pyramid.
        \param [out] dst - an output pyramid.
    */
    template<template<class> class A> SIMD_INLINE void Copy(const Pyramid<A> & src, Pyramid<A> & dst)
    {
        assert(src.Size() == dst.Size());
        for (size_t level = 0; level < src.Size(); ++level)
            Simd::Copy(src.At(level), dst.At(level));
    }

    /*! @ingroup cpp_pyramid_functions

        \fn void Build(Pyramid<A> & pyramid, ::SimdReduceType reduceType, bool compensation = true)

        \short Builds the pyramid (fills upper levels on the base of the lowest level).

        \param [out] pyramid - a built pyramid.
        \param [in] reduceType - a type of function used for image reducing.
        \param [in] compensation - a flag of compensation of rounding. It is relevant only for ::SimdReduce3x3 and ::SimdReduce5x5. It is equal to 'true' by default.
    */
    template<template<class> class A> SIMD_INLINE void Build(Pyramid<A> & pyramid, ::SimdReduceType reduceType, bool compensation = true)
    {
        for (size_t level = 1; level < pyramid.Size(); ++level)
            Simd::ReduceGray(pyramid.At(level - 1), pyramid.At(level), reduceType, compensation);
    }
}

#endif//__SimdLib_hpp__

