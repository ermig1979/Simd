/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#ifndef __SimdSve2_h__
#define __SimdSve2_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        void AddFeatureDifference(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            const uint8_t* lo, size_t loStride, const uint8_t* hi, size_t hiStride,
            uint16_t weight, uint8_t* difference, size_t differenceStride);

        void AlphaBlending(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t* alpha, size_t alphaStride, uint8_t* dst, size_t dstStride);

        void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride);

        void AlphaBlendingBgraToYuv420p(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

        void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            uint8_t alpha, uint8_t* dst, size_t dstStride);

        void AlphaFilling(uint8_t* dst, size_t dstStride, size_t width, size_t height, const uint8_t* channel,
            size_t channelCount, const uint8_t* alpha, size_t alphaStride);

        void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t* dst, size_t dstStride, SimdBool argb);

        void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t* dst, size_t dstStride, SimdBool argb);

        void Binarization(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride, SimdCompareType compareType);

        void AveragingBinarization(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative,
            uint8_t* dst, size_t dstStride, SimdCompareType compareType);

        void AveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride);

        void AbsSecondDerivativeHistogram(const uint8_t* src, size_t width, size_t height, size_t stride,
            size_t step, size_t indent, uint32_t* histogram);

        void BackgroundIncrementCount(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            const uint8_t* loValue, size_t loValueStride, const uint8_t* hiValue, size_t hiValueStride,
            uint8_t* loCount, size_t loCountStride, uint8_t* hiCount, size_t hiCountStride);

        void BackgroundAdjustRangeMasked(uint8_t* loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t* loValue, size_t loValueStride, uint8_t* hiCount, size_t hiCountStride,
            uint8_t* hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t* mask, size_t maskStride);
      
        void BackgroundAdjustRange(uint8_t* loCount, size_t loCountStride, size_t width, size_t height,
            uint8_t* loValue, size_t loValueStride, uint8_t* hiCount, size_t hiCountStride,
            uint8_t* hiValue, size_t hiValueStride, uint8_t threshold);

        void BackgroundShiftRange(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            uint8_t* lo, size_t loStride, uint8_t* hi, size_t hiStride);

        void BackgroundShiftRangeMasked(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            uint8_t* lo, size_t loStride, uint8_t* hi, size_t hiStride, const uint8_t* mask, size_t maskStride);

        void BackgroundInitMask(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t index, uint8_t value, uint8_t* dst, size_t dstStride);

        void BayerToBgr(const uint8_t* bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t* bgr, size_t bgrStride);

        void BayerToBgra(const uint8_t* bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t* bgra, size_t bgraStride, uint8_t alpha);

        void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize);

        void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst);

        void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst);
      
        void BgraToBgr(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* bgr, size_t bgrStride);

        void BgraToBayer(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* bayer, size_t bayerStride, SimdPixelFormatType bayerFormat);

        void BgraToGray(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* gray, size_t grayStride);

        void BgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride);

        void BgraToRgba(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgba, size_t rgbaStride);

        void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

        void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

        void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

        void Bgr48pToBgra32(const uint8_t* blue, size_t blueStride, size_t width, size_t height,
            const uint8_t* green, size_t greenStride, const uint8_t* red, size_t redStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha);

        void BgrToBayer(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* bayer, size_t bayerStride, SimdPixelFormatType bayerFormat);

        void BgrToBgra(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha);

        void BgrToGray(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* gray, size_t grayStride);

        void BgrToHsl(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* hsl, size_t hslStride);

        void BgrToHsv(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* hsv, size_t hsvStride);

        void BgrToLab(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* lab, size_t labStride);

        void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* rgb, size_t rgbStride);

        void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

        void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

        void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

        void ConditionalCount8u(const uint8_t* src, size_t stride, size_t width, size_t height, uint8_t value, SimdCompareType compareType, uint32_t* count);

        void ConditionalCount16i(const uint8_t* src, size_t stride, size_t width, size_t height, int16_t value, SimdCompareType compareType, uint32_t* count);

        void ConditionalSum(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t* sum);

        void ConditionalSquareSum(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t* sum);

        void ConditionalSquareGradientSum(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t* sum);

        void ConditionalFill(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t* dst, size_t dstStride);

        void CorrelationSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, size_t width, size_t height, uint64_t* sum);

        void Reorder16bit(const uint8_t* src, size_t size, uint8_t* dst);

        void Reorder32bit(const uint8_t* src, size_t size, uint8_t* dst);

        void Reorder64bit(const uint8_t* src, size_t size, uint8_t* dst);

        void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride);

        void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride);

        void SquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum);

        void ValueSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum);

        void ValueSquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSum, uint64_t* squareSum);

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums);
    }
#endif
}
#endif
