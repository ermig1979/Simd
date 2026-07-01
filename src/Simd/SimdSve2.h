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

        void HistogramMasked(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t index, uint32_t* histogram);

        void HistogramConditional(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t* histogram);

        void BackgroundGrowRangeSlow(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            uint8_t* lo, size_t loStride, uint8_t* hi, size_t hiStride);

        void BackgroundGrowRangeFast(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            uint8_t* lo, size_t loStride, uint8_t* hi, size_t hiStride);

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

        void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst);

        void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst);

        void Float32ToUint8(const float* src, size_t size, const float* lower, const float* upper, uint8_t* dst);

        void Float32ToFloat16(const float* src, size_t size, uint16_t* dst);

        void Float16ToFloat32(const uint16_t* src, size_t size, float* dst);

        void SquaredDifferenceSum16f(const uint16_t* a, const uint16_t* b, size_t size, float* sum);

        void NeuralConvert(const uint8_t* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride, int inversion);

        void NeuralProductSum(const float* a, const float* b, size_t size, float* sum);

        void NeuralAddVectorMultipliedByValue(const float* src, size_t size, const float* value, float* dst);

        void NeuralAddVector(const float* src, size_t size, float* dst);

        void NeuralAddValue(const float* value, float* dst, size_t size);

        void NeuralDerivativeSigmoid(const float* src, size_t size, const float* slope, float* dst);

        void NeuralDerivativeTanh(const float* src, size_t size, const float* slope, float* dst);

        void NeuralDerivativeRelu(const float* src, size_t size, const float* slope, float* dst);

        void NeuralPow(const float* src, size_t size, const float* exponent, float* dst);

        void NeuralUpdateWeights(const float* x, size_t size, const float* a, const float* b, float* d, float* w);

        void NeuralAdaptiveGradientUpdate(const float* delta, size_t size, size_t batch, const float* alpha, const float* epsilon, float* gradient, float* weight);

        void NeuralAddConvolution2x2Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride);

        void NeuralAddConvolution3x3Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride);

        void NeuralAddConvolution4x4Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride);

        void NeuralAddConvolution5x5Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride);

        void NeuralAddConvolution2x2Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride);

        void NeuralAddConvolution3x3Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride);

        void NeuralAddConvolution4x4Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride);

        void NeuralAddConvolution5x5Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride);

        void NeuralAddConvolution2x2Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums);

        void NeuralAddConvolution3x3Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums);

        void NeuralAddConvolution4x4Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums);

        void NeuralAddConvolution5x5Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums);

        void NeuralConvolutionForward(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, const float* weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, void* buffer, size_t* size, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add);

        void NeuralPooling1x1Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride);

        void NeuralPooling2x2Max2x2(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride);

        void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride);

        void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride);

        void Gemm32fNN(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc);

        void Gemm32fNT(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc);

        void HogDirectionHistograms(const uint8_t* src, size_t stride, size_t width, size_t height,
            size_t cellX, size_t cellY, size_t quantization, float* histograms);

        void HogExtractFeatures(const uint8_t* src, size_t stride, size_t width, size_t height, float* features);

        void HogDeinterleave(const float* src, size_t srcStride, size_t width, size_t height, size_t count, float** dst, size_t dstStride);

        void HogFilterSeparable(const float* src, size_t srcStride, size_t width, size_t height,
            const float* rowFilter, size_t rowSize, const float* colFilter, size_t colSize, float* dst, size_t dstStride, int add);

        void LbpEstimate(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        void TextureBoostedSaturatedGradient(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t* dx, size_t dxStride, uint8_t* dy, size_t dyStride);

        void TextureBoostedUv(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t boost, uint8_t* dst, size_t dstStride);

        void SobelDx(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        void SobelDxAbs(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        void SobelDxAbsSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum);

        void SobelDy(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        void SobelDyAbs(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        void SobelDyAbsSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum);

        void Laplace(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        void LaplaceAbs(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        void LaplaceAbsSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum);

        void MaxFilterSquare3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold);

        void MaxFilterSquare5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold);

        void MedianFilterRhomb3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride);

        void MedianFilterRhomb5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride);

        void MedianFilterSquare3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride);

        void MedianFilterSquare5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            uint8_t* dst, size_t dstStride);

        void MinFilterSquare3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold);

        void MinFilterSquare5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride, int threshold);

        void MeanFilter3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride);

        void MidpointFilterSquare3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride);

        void MidpointFilterSquare5x5(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride);

        void CosineDistance16f(const uint16_t* a, const uint16_t* b, size_t size, float* distance);

        void CosineDistancesMxNa16f(size_t M, size_t N, size_t K, const uint16_t* const* A, const uint16_t* const* B, float* distances);

        void CosineDistancesMxNp16f(size_t M, size_t N, size_t K, const uint16_t* A, const uint16_t* B, float* distances);

        void CosineDistance32f(const float* a, const float* b, size_t size, float* distance);
      
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

        void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride,
            uint8_t* a, size_t aStride, SimdYuvType yuvType);

        void Bgr48pToBgra32(const uint8_t* blue, size_t blueStride, size_t width, size_t height,
            const uint8_t* green, size_t greenStride, const uint8_t* red, size_t redStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha);

        void GrayToBgr(const uint8_t* gray, size_t width, size_t height, size_t grayStride, uint8_t* bgr, size_t bgrStride);

        void GrayToBgra(const uint8_t* gray, size_t width, size_t height, size_t grayStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha);

        void GrayToY(const uint8_t* gray, size_t grayStride, size_t width, size_t height, uint8_t* y, size_t yStride);

        void ReduceColor2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

        void ReduceGray2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

        void ReduceGray3x3(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation);

        void ReduceGray4x4(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

        void ReduceGray5x5(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation);

        void StretchGray2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

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

        void ContourMetrics(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        void ContourMetricsMasked(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t indexMin, uint8_t* dst, size_t dstStride);

        void ContourAnchors(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t* dst, size_t dstStride);

        void CorrelationSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, size_t width, size_t height, uint64_t* sum);

        void SquaredDifferenceSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint64_t* sum);

        void SquaredDifferenceSumMasked(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            const uint8_t* mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t* sum);

        void SquaredDifferenceSum32f(const float* a, const float* b, size_t size, float* sum);

        void SquaredDifferenceKahanSum32f(const float* a, const float* b, size_t size, float* sum);

        void DetectionHaarDetect32fp(const void* hid, const uint8_t* mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t* dst, size_t dstStride);

        void DetectionHaarDetect32fi(const void* hid, const uint8_t* mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t* dst, size_t dstStride);

        void DetectionLbpDetect32fp(const void* hid, const uint8_t* mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t* dst, size_t dstStride);

        void DetectionLbpDetect32fi(const void* hid, const uint8_t* mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t* dst, size_t dstStride);

        void DetectionLbpDetect16ip(const void* hid, const uint8_t* mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t* dst, size_t dstStride);

        void DetectionLbpDetect16ii(const void* hid, const uint8_t* mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t* dst, size_t dstStride);

        void FillBgr(uint8_t* dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red);

        void FillBgra(uint8_t* dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha);

        void FillPixel(uint8_t* dst, size_t stride, size_t width, size_t height, const uint8_t* pixel, size_t pixelSize);

        void Fill32f(float* dst, size_t size, const float* value);

        void Reorder16bit(const uint8_t* src, size_t size, uint8_t* dst);

        void Reorder32bit(const uint8_t* src, size_t size, uint8_t* dst);

        void Reorder64bit(const uint8_t* src, size_t size, uint8_t* dst);

        void RgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride);

        void RgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride);

        void RgbToBgra(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha);

        void SegmentationChangeIndex(uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex);
        void SegmentationFillSingleHoles(uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t index);
        void SegmentationPropagate2x2(const uint8_t* parent, size_t parentStride, size_t width, size_t height,
            uint8_t* child, size_t childStride, const uint8_t* difference, size_t differenceStride,
            uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold);
        void SegmentationShrinkRegion(const uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t index,
            ptrdiff_t* left, ptrdiff_t* top, ptrdiff_t* right, ptrdiff_t* bottom);

        void ShiftBilinear(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t* bkg, size_t bkgStride, const double* shiftX, const double* shiftY,
            size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t* dst, size_t dstStride);

        void GetRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums);

        void GetColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums);

        void GetAbsDyRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums);

        void GetAbsDxColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums);

        void GetMoments(const uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t index,
            uint64_t* area, uint64_t* x, uint64_t* y, uint64_t* xx, uint64_t* xy, uint64_t* yy);

        void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy);

        void SquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum);

        void ValueSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum);

        void ValueSquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSum, uint64_t* squareSum);

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums);
    }
#endif
}
#endif
