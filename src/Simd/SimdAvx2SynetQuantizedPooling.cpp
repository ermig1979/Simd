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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Avx2
    {
        SIMD_INLINE __m256i QuantizeSumLinear(__m256i sum, const __m256i& bias, const __m256& norm, const __m256i& zero)
        {
            return _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum, bias)), norm)), zero);
        }

        SIMD_INLINE void QuantizeSumLinear8(__m256i sum, const __m256i& bias, const __m256& norm, const __m256i& zero, uint8_t* dst)
        {
            __m256i d0 = QuantizeSumLinear(sum, bias, norm, zero);
            _mm_storel_epi64((__m128i*)dst, _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizeSumLinear32(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3,
            const __m256i& bias, const __m256& norm, const __m256i& zero, uint8_t* dst)
        {
            __m256i d0 = QuantizeSumLinear(sum0, bias, norm, zero);
            __m256i d1 = QuantizeSumLinear(sum1, bias, norm, zero);
            __m256i d2 = QuantizeSumLinear(sum2, bias, norm, zero);
            __m256i d3 = QuantizeSumLinear(sum3, bias, norm, zero);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        SIMD_INLINE int32_t Sum8u(const uint8_t* src, size_t size)
        {
            size_t i = 0, sizeQA = AlignLo(size, QA), sizeA = AlignLo(size, A);
            __m256i sums[4] = { K_ZERO, K_ZERO, K_ZERO, K_ZERO };
            for (; i < sizeQA; i += QA)
            {
                sums[0] = _mm256_add_epi64(sums[0], _mm256_sad_epu8(_mm256_loadu_si256((__m256i*)(src + i + 0 * A)), K_ZERO));
                sums[1] = _mm256_add_epi64(sums[1], _mm256_sad_epu8(_mm256_loadu_si256((__m256i*)(src + i + 1 * A)), K_ZERO));
                sums[2] = _mm256_add_epi64(sums[2], _mm256_sad_epu8(_mm256_loadu_si256((__m256i*)(src + i + 2 * A)), K_ZERO));
                sums[3] = _mm256_add_epi64(sums[3], _mm256_sad_epu8(_mm256_loadu_si256((__m256i*)(src + i + 3 * A)), K_ZERO));
            }
            sums[0] = _mm256_add_epi64(_mm256_add_epi64(sums[0], sums[1]), _mm256_add_epi64(sums[2], sums[3]));
            for (; i < sizeA; i += A)
                sums[0] = _mm256_add_epi64(sums[0], _mm256_sad_epu8(_mm256_loadu_si256((__m256i*)(src + i)), K_ZERO));
            __m128i sum = _mm_add_epi64(_mm256_castsi256_si128(sums[0]), _mm256_extracti128_si256(sums[0], 1));
            int32_t total = (int32_t)Sse41::ExtractInt64Sum(sum);
            for (; i < size; ++i)
                total += src[i];
            return total;
        }

        SIMD_INLINE void QuantizedPoolingAverageNhwc(const uint8_t* src, size_t srcS, size_t srcC, size_t srcCF8, size_t srcCF32,
            size_t kH, size_t kW, const __m256i& bias, const __m256& norm, const __m256i& zero, uint8_t* dst)
        {
            size_t c = 0;
            for (; c < srcCF32; c += A)
            {
                __m256i sum0 = K_ZERO, sum1 = K_ZERO, sum2 = K_ZERO, sum3 = K_ZERO;
                for (size_t h = 0; h < kH; ++h)
                {
                    for (size_t w = 0; w < kW; ++w)
                    {
                        const uint8_t* ps = src + h * srcS + w * srcC + c;
                        __m128i s0 = _mm_loadu_si128((__m128i*)ps + 0);
                        __m128i s1 = _mm_loadu_si128((__m128i*)ps + 1);
                        sum0 = _mm256_add_epi32(sum0, _mm256_cvtepu8_epi32(_mm_srli_si128(s0, 0)));
                        sum1 = _mm256_add_epi32(sum1, _mm256_cvtepu8_epi32(_mm_srli_si128(s0, 8)));
                        sum2 = _mm256_add_epi32(sum2, _mm256_cvtepu8_epi32(_mm_srli_si128(s1, 0)));
                        sum3 = _mm256_add_epi32(sum3, _mm256_cvtepu8_epi32(_mm_srli_si128(s1, 8)));
                    }
                }
                QuantizeSumLinear32(sum0, sum1, sum2, sum3, bias, norm, zero, dst + c);
            }
            for (; c < srcCF8; c += F)
            {
                __m256i sum = K_ZERO;
                for (size_t h = 0; h < kH; ++h)
                    for (size_t w = 0; w < kW; ++w)
                        sum = _mm256_add_epi32(sum, _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src + h * srcS + w * srcC + c))));
                QuantizeSumLinear8(sum, bias, norm, zero, dst + c);
            }
            for (; c < srcC; ++c)
            {
                int32_t sum = 0;
                for (size_t h = 0; h < kH; ++h)
                    for (size_t w = 0; w < kW; ++w)
                        sum += src[h * srcS + w * srcC + c];
                dst[c] = (uint8_t)Base::QuantizeSumLinear(sum, _mm256_cvtsi256_si32(bias), _mm_cvtss_f32(_mm256_castps256_ps128(norm)), _mm256_cvtsi256_si32(zero), 0, 255);
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageGlobalNhwc(const uint8_t* src, int srcZero, const float* srcScale,
            size_t batch, size_t channels, size_t spatial, uint8_t* dst, const float* dstScale, int dstZero)
        {
            int32_t bias = -srcZero * int32_t(spatial);
            __m256i _bias = _mm256_set1_epi32(bias), _zero = _mm256_set1_epi32(dstZero);
            __m256 _norm = _mm256_set1_ps(srcScale[0] / (dstScale[0] * float(spatial)));
            size_t channels8 = AlignLo(channels, F), channels32 = AlignLo(channels, A);
            Array32u sum(channels);
            for (size_t b = 0; b < batch; ++b)
            {
                GetColSums(src, channels, channels, spatial, sum.data);
                size_t c = 0;
                uint32_t* sums = sum.data;
                for (; c < channels32; c += A, sums += A)
                    QuantizeSumLinear32(_mm256_loadu_si256((__m256i*)(sums + 0 * F)), _mm256_loadu_si256((__m256i*)(sums + 1 * F)),
                        _mm256_loadu_si256((__m256i*)(sums + 2 * F)), _mm256_loadu_si256((__m256i*)(sums + 3 * F)), _bias, _norm, _zero, dst + c);
                for (; c < channels8; c += F, sums += F)
                    QuantizeSumLinear8(_mm256_loadu_si256((__m256i*)sums), _bias, _norm, _zero, dst + c);
                for (; c < channels; ++c)
                {
                    int32_t sum = 0;
                    for (size_t s = 0; s < spatial; ++s)
                        sum += src[s * channels + c];
                    dst[c] = (uint8_t)Base::QuantizeSumLinear(sum, bias, _mm_cvtss_f32(_mm256_castps256_ps128(_norm)), dstZero, 0, 255);
                }
                src += spatial * channels;
                dst += channels;
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageGlobalNchw(const uint8_t* src, int srcZero, const float* srcScale,
            size_t batch, size_t channels, size_t spatial, uint8_t* dst, const float* dstScale, int dstZero)
        {
            int32_t bias = -srcZero * int32_t(spatial);
            const __m256i _bias = _mm256_set1_epi32(bias), _zero = _mm256_set1_epi32(dstZero);
            const __m256 _norm = _mm256_set1_ps(srcScale[0] / (dstScale[0] * float(spatial)));
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256i sum = _mm256_set1_epi32(Sum8u(src, spatial));
                    dst[c] = (uint8_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(QuantizeSumLinear(sum, _bias, _norm, _zero), K_ZERO), K_ZERO)));
                    src += spatial;
                }
                dst += channels;
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageNchw2x2(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW,
            uint8_t* dst, size_t dstH, size_t dstW, const __m256i& bias, const __m256& norm, const __m256i& zero)
        {
            size_t dstWF = AlignLo(dstW, F);
            const __m256i one = _mm256_set1_epi16(1);
            for (size_t b = 0; b < srcC; ++b)
            {
                for (size_t dy = 0; dy < dstH; ++dy)
                {
                    size_t dx = 0, sx = 0;
                    const uint8_t* src0 = src + dy * 2 * srcW;
                    const uint8_t* src1 = src0 + srcW;
                    for (; dx < dstWF; dx += F, sx += DF)
                    {
                        __m256i s0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src0 + sx)));
                        __m256i s1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(src1 + sx)));
                        __m256i sum = _mm256_madd_epi16(_mm256_add_epi16(s0, s1), one);
                        QuantizeSumLinear8(sum, bias, norm, zero, dst + dx);
                    }
                    for (; dx < dstW; ++dx, sx += 2)
                    {
                        int32_t sum = src0[sx] + src0[sx + 1] + src1[sx] + src1[sx + 1];
                        dst[dx] = (uint8_t)Base::QuantizeSumLinear(sum, _mm256_cvtsi256_si32(bias), _mm_cvtss_f32(_mm256_castps256_ps128(norm)), _mm256_cvtsi256_si32(zero), 0, 255);
                    }
                    dst += dstW;
                }
                src += srcH * srcW;
            }
        }

        void SynetQuantizedPoolingAverage(const uint8_t* src, const float* srcScale, int srcZero, size_t batch, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, SimdBool excludePad,
            uint8_t* dst, const float* dstScale, int dstZero, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc && srcC >= F)
            {
                if (kernelY == srcH && kernelX == srcW && strideY == 1 && strideX == 1 && padY == 0 && padX == 0)
                {
                    QuantizedPoolingAverageGlobalNhwc(src, srcZero, srcScale, batch, srcC, srcH * srcW, dst, dstScale, dstZero);
                    return;
                }
                size_t srcS = srcW * srcC, srcCF8 = AlignLo(srcC, F), srcCF32 = AlignLo(srcC, A);
                __m256i zero = _mm256_set1_epi32(dstZero);
                int32_t bias = -srcZero * int32_t(kernelY * kernelX);
                float norm = srcScale[0] / (dstScale[0] * float(kernelY * kernelX));
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t ph = 0; ph < dstH; ++ph)
                    {
                        size_t hStart = ph * strideY - padY;
                        size_t hEnd = Simd::Min(hStart + kernelY, srcH);
                        hStart = Simd::Max<ptrdiff_t>(0, hStart);
                        for (size_t pw = 0; pw < dstW; ++pw)
                        {
                            size_t wStart = pw * strideX - padX;
                            size_t wEnd = Simd::Min(wStart + kernelX, srcW);
                            wStart = Simd::Max<ptrdiff_t>(0, wStart);
                            const uint8_t* ps = src + hStart * srcS + wStart * srcC;
                            if (excludePad)
                            {
                                int area = int(hEnd - hStart) * int(wEnd - wStart);
                                QuantizedPoolingAverageNhwc(ps, srcS, srcC, srcCF8, srcCF32, hEnd - hStart, wEnd - wStart,
                                    _mm256_set1_epi32(-srcZero * area), _mm256_set1_ps(srcScale[0] / (dstScale[0] * float(area))), zero, dst);
                            }
                            else
                                QuantizedPoolingAverageNhwc(ps, srcS, srcC, srcCF8, srcCF32, hEnd - hStart, wEnd - wStart,
                                    _mm256_set1_epi32(bias), _mm256_set1_ps(norm), zero, dst);
                            dst += srcC;
                        }
                    }
                    src += srcC * srcH * srcW;
                }
                return;
            }
            else if (format == SimdTensorFormatNchw)
            {
                if (kernelY == srcH && kernelX == srcW && strideY == 1 && strideX == 1 && padY == 0 && padX == 0)
                {
                    QuantizedPoolingAverageGlobalNchw(src, srcZero, srcScale, batch, srcC, srcH * srcW, dst, dstScale, dstZero);
                    return;
                }
                if (kernelY == 2 && kernelX == 2 && strideY == 2 && strideX == 2 && padY == 0 && padX == 0 && dstH * 2 <= srcH && dstW * 2 <= srcW)
                {
                    QuantizedPoolingAverageNchw2x2(src, srcC * batch, srcH, srcW, dst, dstH, dstW, _mm256_set1_epi32(-srcZero * 4),
                        _mm256_set1_ps(srcScale[0] / (dstScale[0] * 4.0f)), _mm256_set1_epi32(dstZero));
                    return;
                }
            }
            Base::SynetQuantizedPoolingAverage(src, srcScale, srcZero, batch, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX,
                padY, padX, excludePad, dst, dstScale, dstZero, dstH, dstW, format);
        }
    }
#endif
}
