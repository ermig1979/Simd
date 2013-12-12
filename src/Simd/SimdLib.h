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

/** \file SimdLib.h
* This file contains a Simd Library API functions.
*/

#ifndef __SimdLib_h__
#define __SimdLib_h__

#include "Simd/SimdTypes.h"

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

    /*!
    * \fn const char * SimdVersion();
    *
    * \short Gets version of Simd Library.
    *
    * \return string with version of Simd Library (major version number, minor version number, release number, number of SVN's commits).
    */
    SIMD_API const char * SimdVersion();

    /*!
    * \fn uint32_t SimdCrc32c(const void * src, size_t size);
    *
    * \short Gets 32-bit cyclic redundancy check (CRC32c) for current data.
    *
    * Calculation is performed for for polynomial 0x1EDC6F41 (Castagnoli-crc).
    *
    * \param [in] src - a pointer to data.
    * \param [in] size - a size of the data.
    * \return 32-bit cyclic redundancy check (CRC32c).
    */
    SIMD_API uint32_t SimdCrc32c(const void * src, size_t size);

    /*!
    * \fn void SimdAbsDifferenceSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum);
    *
    * \short Gets sum of absolute difference of two gray 8-bit images. 
    *
    * Both images must have the same width and height.
    *
    * \param [in] a - a pointer to pixels data of first image.
    * \param [in] aStride - a row size of first image.
    * \param [in] b - a pointer to pixels data of second image.
    * \param [in] bStride - a row size of second image.
    * \param [in] width - an image width.
    * \param [in] height - an image height.
    * \param [out] sum - the result sum of absolute difference of two images.
    */
    SIMD_API void SimdAbsDifferenceSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        size_t width, size_t height, uint64_t * sum);

    /*!
    * \fn void SimdAbsDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum);
    *
    * \short Gets sum of absolute difference of two gray 8-bit images based on gray 8-bit mask. 
    *
    * Gets the absolute difference sum for all points where mask[i] == index.
    * Both images and mask must have the same width and height.
    *
    * \param [in] a - a pointer to pixels data of first image.
    * \param [in] aStride - a row size of first image.
    * \param [in] b - a pointer to pixels data of second image.
    * \param [in] bStride - a row size of second image.
    * \param [in] mask - a pointer to pixels data of mask image.
    * \param [in] maskStride - a row size of mask image.
    * \param [in] index - a mask index.
    * \param [in] width - an image width.
    * \param [in] height - an image height.
    * \param [out] sum - the result sum of absolute difference of two images.
    */
    SIMD_API void SimdAbsDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum);

    /*!
    * \fn void SimdAbsDifferenceSums3x3(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride, size_t width, size_t height, uint64_t * sums);
    *
    * \short Gets 9 sums of absolute difference of two gray 8-bit images with various relative shifts in neighborhood 3x3. 
    *
    * Both images must have the same width and height. The image height and width must be equal or greater 3.
    * The sums are calculated with central part (indent width = 1) of current image and with part of background image with corresponding shift.
    * The shifts are lain in the range [-1, 1] for axis x and y.
    *
    * \param [in] current - a pointer to pixels data of current image.
    * \param [in] currentStride - a row size of the current image.
    * \param [in] background - a pointer to pixels data of the background image.
    * \param [in] backgroundStride - a row size of the background image.
    * \param [in] width - an image width.
    * \param [in] height - an image height.
    * \param [out] sums - the pointer to buffer with result sums. Buffer size must be equal or greater 9.
    */
    SIMD_API void SimdAbsDifferenceSums3x3(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
        size_t width, size_t height, uint64_t * sums);

    /*!
    * \fn void SimdAbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride, const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums);
    *
    * \short Gets 9 sums of absolute difference of two gray 8-bit images with various relative shifts in neighborhood 3x3 based on gray 8-bit mask. 
    *
    * Gets the absolute difference sums for all points where mask[i] == index.
    * Both images and mask must have the same width and height. The image height and width must be equal or greater 3.
    * The sums are calculated with central part (indent width = 1) of current image and with part of background image with corresponding shift.
    * The shifts are lain in the range [-1, 1] for axis x and y.
    *
    * \param [in] current - a pointer to pixels data of current image.
    * \param [in] currentStride - a row size of the current image.
    * \param [in] background - a pointer to pixels data of the background image.
    * \param [in] backgroundStride - a row size of the background image.
    * \param [in] mask - a pointer to pixels data of mask image.
    * \param [in] maskStride - a row size of mask image.
    * \param [in] index - a mask index.
    * \param [in] width - an image width.
    * \param [in] height - an image height.
    * \param [out] sums - the pointer to buffer with result sums. Buffer size must be equal or greater 9.
    */
    SIMD_API void SimdAbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
        const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums);

    /*!
    * \fn void SimdAbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);
    *
    * \short Puts to destination 8-bit gray image saturated sum of absolute gradient for every point of source 8-bit gray image. 
    *
    * Both images must have the same width and height.
    *
    * For border pixels dst[x, y] = 0, for other pixels: 
    * dst[x, y] = min(dx[x, y] + dy[x, y], 255), where 
    * dx[x, y] = abs(src[x + 1, y] - src[x - 1, y]), 
    * dy[x, y] = abs(src[x, y + 1] - src[x, y - 1]).
    *
    * \param [in] src - a pointer to pixels data of source 8-bit gray image.
    * \param [in] srcStride - a row size of source image.
    * \param [in] width - an image width.
    * \param [in] height - an image height.
    * \param [out] dst - a pointer to pixels data of destination 8-bit gray image.
    * \param [in] dstStride - a row size of destination image.
    */
    SIMD_API void SimdAbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t * dst, size_t dstStride);

    /*!
    * \fn void SimdAddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height, const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, uint16_t weight, uint8_t * difference, size_t differenceStride);
    *
    * \short Adds feature difference to common difference sum. 
    *
    * All images must have the same width, height and format (8-bit gray).
    *
    * For every point: difference[i] += (weight * excess[i]*excess[i]) >> 16, where excess[i] = max(lo[i] - value[i], 0) + max(value[i] - hi[i], 0).
    *
    * This function is used for difference estimation in algorithm of motion detection.
    *
    * \param [in] value - a pointer to pixels data of current feature value.
    * \param [in] valueStride - a row size of the value image.
    * \param [in] width - an image width.
    * \param [in] height - an image height.
    * \param [in] lo - a pointer to pixels data of feature lower bound of dynamic background.
    * \param [in] loStride - a row size of the lo image.
    * \param [in] hi - a pointer to pixels data of feature upper bound of dynamic background.
    * \param [in] hiStride - a row size of the hi image.
    * \param [in] weight - a current feature weight (unsigned 16-bit value).
    * \param [in, out] difference- a pointer to pixels data of image with total difference.
    * \param [in] differenceStride - a row size of difference image.
    */
    SIMD_API void SimdAddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
        uint16_t weight, uint8_t * difference, size_t differenceStride);

    SIMD_API void SimdAlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
        const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride);

    SIMD_API void SimdBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

    SIMD_API void SimdBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

    SIMD_API void SimdBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
        uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride);

    SIMD_API void SimdBackgroundAdjustRange(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
        uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
        uint8_t * hiValue, size_t hiValueStride, uint8_t threshold);

    SIMD_API void SimdBackgroundAdjustRangeMasked(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
        uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
        uint8_t * hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride);

    SIMD_API void SimdBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

    SIMD_API void SimdBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride);

    SIMD_API void SimdBackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdBgraToBgr(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *bgr, size_t bgrStride);

    SIMD_API void SimdBgraToGray(const uint8_t *bgra, size_t width, size_t height, size_t bgraStride, uint8_t *gray, size_t grayStride);

    SIMD_API void SimdBgrToBgra(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t *bgra, size_t bgraStride, uint8_t alpha);

    SIMD_API void SimdBgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
        const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

    SIMD_API void SimdBgrToGray(const uint8_t *bgr, size_t width, size_t height, size_t bgrStride, uint8_t *gray, size_t grayStride);

    SIMD_API void SimdBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType);

    SIMD_API void SimdAveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative,
        uint8_t * dst, size_t dstStride, SimdCompareType compareType);

    SIMD_API void SimdConditionalCount(const uint8_t * src, size_t stride, size_t width, size_t height, 
        uint8_t value, SimdCompareType compareType, uint32_t * count);

    SIMD_API void SimdConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

    SIMD_API void SimdConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

    SIMD_API void SimdConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

    SIMD_API void SimdCopy(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdCopyFrame(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize,
        size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdDeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
        uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    SIMD_API void SimdEdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * background, size_t backgroundStride);

    SIMD_API void SimdEdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * background, size_t backgroundStride);

    SIMD_API void SimdEdgeBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride);

    SIMD_API void SimdEdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
        uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold);

    SIMD_API void SimdEdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
        uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride);

    SIMD_API void SimdEdgeBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * background, size_t backgroundStride);

    SIMD_API void SimdEdgeBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride);

    SIMD_API void SimdFill(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uint8_t value);

    SIMD_API void SimdFillFrame(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize,
        size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t value);

    SIMD_API void SimdFillBgra(uint8_t * dst, size_t stride, size_t width, size_t height,
        uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha);

    SIMD_API void SimdGaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdGrayToBgra(const uint8_t *gray, size_t width, size_t height, size_t grayStride,
        uint8_t *bgra, size_t bgraStride, uint8_t alpha);

    SIMD_API void SimdAbsSecondDerivativeHistogram(const uint8_t *src, size_t width, size_t height, size_t stride,
        size_t step, size_t indent, uint32_t * histogram);

    SIMD_API void SimdHistogram(const uint8_t *src, size_t width, size_t height, size_t stride, uint32_t * histogram);

    SIMD_API void SimdIntegralSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * sum, size_t sumStride);

    SIMD_API void SimdLbpEstimate(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdMedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdMedianFilterRhomb5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdMedianFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdMedianFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdOperation(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationType type);

    SIMD_API void SimdVectorProduct(const uint8_t * vertical, const uint8_t * horizontal,
        uint8_t * dst, size_t stride, size_t width, size_t height);

    SIMD_API void SimdReduceGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

    SIMD_API void SimdReduceGray3x3(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation);

    SIMD_API void SimdReduceGray4x4(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

    SIMD_API void SimdReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, bool compensation);

    SIMD_API void SimdResizeBilinear(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

    SIMD_API void SimdShiftBilinear(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
        const uint8_t * bkg, size_t bkgStride, double shiftX, double shiftY,
        size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdSobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdSobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdSquaredDifferenceSum(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
        size_t width, size_t height, uint64_t * sum);

    SIMD_API void SimdSquaredDifferenceSumMasked(const uint8_t *a, size_t aStride, const uint8_t *b, size_t bStride,
        const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum);

    SIMD_API void SimdGetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height,
        uint8_t * min, uint8_t * max, uint8_t * average);

    SIMD_API void SimdGetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
        uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

    SIMD_API void SimdGetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

    SIMD_API void SimdGetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

    SIMD_API void SimdGetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

    SIMD_API void SimdGetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

    SIMD_API void SimdStretchGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

    SIMD_API void SimdTextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride);

    SIMD_API void SimdTextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t boost, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdTextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum);

    SIMD_API void SimdTexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        int32_t shift, uint8_t * dst, size_t dstStride);

    SIMD_API void SimdYuv420ToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

    SIMD_API void SimdYuv444ToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

    SIMD_API void SimdYuv420ToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha = 0xFF);

    SIMD_API void SimdYuv444ToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha = 0xFF);

    SIMD_API void SimdYuv420ToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * hue, size_t hueStride);

    SIMD_API void SimdYuv444ToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * hue, size_t hueStride);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif//__SimdLib_h__
