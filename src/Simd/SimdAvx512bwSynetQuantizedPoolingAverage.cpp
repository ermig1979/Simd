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
#include "Simd/SimdAvx512bw.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Avx512bw
    {
        SIMD_INLINE int32_t Sum8u(const uint8_t* src, size_t size)
        {
            size_t i = 0, sizeQA = AlignLo(size, QA), sizeA = AlignLo(size, A);
            __m512i sums[4] = { K_ZERO, K_ZERO, K_ZERO, K_ZERO };
            for (; i < sizeQA; i += QA)
            {
                sums[0] = _mm512_add_epi64(sums[0], _mm512_sad_epu8(_mm512_loadu_si512((__m512i*)(src + i + 0 * A)), K_ZERO));
                sums[1] = _mm512_add_epi64(sums[1], _mm512_sad_epu8(_mm512_loadu_si512((__m512i*)(src + i + 1 * A)), K_ZERO));
                sums[2] = _mm512_add_epi64(sums[2], _mm512_sad_epu8(_mm512_loadu_si512((__m512i*)(src + i + 2 * A)), K_ZERO));
                sums[3] = _mm512_add_epi64(sums[3], _mm512_sad_epu8(_mm512_loadu_si512((__m512i*)(src + i + 3 * A)), K_ZERO));
            }
            sums[0] = _mm512_add_epi64(_mm512_add_epi64(sums[0], sums[1]), _mm512_add_epi64(sums[2], sums[3]));
            for (; i < sizeA; i += A)
                sums[0] = _mm512_add_epi64(sums[0], _mm512_sad_epu8(_mm512_loadu_si512((__m512i*)(src + i)), K_ZERO));
            int32_t total = (int32_t)ExtractSum<uint64_t>(sums[0]);
            for (; i < size; ++i)
                total += src[i];
            return total;
        }

        SIMD_INLINE void QuantizedPoolingAverageNhwc(const uint8_t* src, size_t srcS, size_t srcC, size_t srcCF16, size_t srcCF64,
            size_t kH, size_t kW, const __m512i& bias, const __m512& norm, const __m512i& zero, uint8_t* dst)
        {
            size_t c = 0;
            for (; c < srcCF64; c += A)
            {
                __m512i sum0 = K_ZERO, sum1 = K_ZERO, sum2 = K_ZERO, sum3 = K_ZERO;
                for (size_t h = 0; h < kH; ++h)
                {
                    for (size_t w = 0; w < kW; ++w)
                    {
                        const uint8_t* ps = src + h * srcS + w * srcC + c;
                        sum0 = _mm512_add_epi32(sum0, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)ps + 0)));
                        sum1 = _mm512_add_epi32(sum1, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)ps + 1)));
                        sum2 = _mm512_add_epi32(sum2, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)ps + 2)));
                        sum3 = _mm512_add_epi32(sum3, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)ps + 3)));
                    }
                }
                QuantizeSumLinear64(sum0, sum1, sum2, sum3, bias, norm, zero, dst + c);
            }
            for (; c < srcCF16; c += F)
            {
                __m512i sum = K_ZERO;
                for (size_t h = 0; h < kH; ++h)
                    for (size_t w = 0; w < kW; ++w)
                        sum = _mm512_add_epi32(sum, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(src + h * srcS + w * srcC + c))));
                QuantizeSumLinear16(sum, bias, norm, zero, dst + c);
            }
            for (; c < srcC; ++c)
            {
                int32_t sum = 0;
                for (size_t h = 0; h < kH; ++h)
                    for (size_t w = 0; w < kW; ++w)
                        sum += src[h * srcS + w * srcC + c];
                dst[c] = (uint8_t)Base::QuantizeSumLinear(sum, _mm512_cvtsi512_si32(bias), _mm_cvtss_f32(_mm512_castps512_ps128(norm)), _mm512_cvtsi512_si32(zero), 0, 255);
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageGlobalNhwc(const uint8_t* src, int srcZero, const float* srcScale,
            size_t batch, size_t channels, size_t spatial, uint8_t* dst, const float* dstScale, int dstZero)
        {
            int32_t bias = -srcZero * int32_t(spatial);
            __m512i _bias = _mm512_set1_epi32(bias), _zero = _mm512_set1_epi32(dstZero);
            __m512 _norm = _mm512_set1_ps(srcScale[0] / (dstScale[0] * float(spatial)));
            size_t channels16 = AlignLo(channels, F), channels64 = AlignLo(channels, A);
            __mmask16 tailC = TailMask16(channels - channels16);
            Array32u sum(channels);
            for (size_t b = 0; b < batch; ++b)
            {
                GetColSums(src, channels, channels, spatial, sum.data);
                size_t c = 0;
                uint32_t* sums = sum.data;
                for (; c < channels64; c += A, sums += A)
                    QuantizeSumLinear64(_mm512_loadu_si512(sums + 0 * F), _mm512_loadu_si512(sums + 1 * F),
                        _mm512_loadu_si512(sums + 2 * F), _mm512_loadu_si512(sums + 3 * F), _bias, _norm, _zero, dst + c);
                for (; c < channels16; c += F, sums += F)
                    QuantizeSumLinear16(_mm512_loadu_si512(sums), _bias, _norm, _zero, dst + c);
                if (c < channels)
                    QuantizeSumLinear16(_mm512_loadu_si512(sums), _bias, _norm, _zero, dst + c, tailC);
                src += spatial * channels;
                dst += channels;
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageGlobalNchw(const uint8_t* src, int srcZero, const float* srcScale,
            size_t batch, size_t channels, size_t spatial, uint8_t* dst, const float* dstScale, int dstZero)
        {
            int32_t bias = -srcZero * int32_t(spatial);
            const __m512i _bias = _mm512_set1_epi32(bias), _zero = _mm512_set1_epi32(dstZero);
            const __m512 _norm = _mm512_set1_ps(srcScale[0] / (dstScale[0] * float(spatial)));
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512i sum = _mm512_set1_epi32(Sum8u(src, spatial));
                    dst[c] = (uint8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(QuantizeSumLinear(sum, _bias, _norm, _zero), K_ZERO), K_ZERO)));
                    src += spatial;
                }
                dst += channels;
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageNchw2x2(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW,
            uint8_t* dst, size_t dstH, size_t dstW, const __m512i& bias, const __m512& norm, const __m512i& zero)
        {
            size_t dstWF = AlignLo(dstW, F);
            const __m512i one = _mm512_set1_epi16(1);
            for (size_t b = 0; b < srcC; ++b)
            {
                for (size_t dy = 0; dy < dstH; ++dy)
                {
                    size_t dx = 0, sx = 0;
                    const uint8_t* src0 = src + dy * 2 * srcW;
                    const uint8_t* src1 = src0 + srcW;
                    for (; dx < dstWF; dx += F, sx += DF)
                    {
                        __m512i s0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(src0 + sx)));
                        __m512i s1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(src1 + sx)));
                        __m512i sum = _mm512_madd_epi16(_mm512_add_epi16(s0, s1), one);
                        QuantizeSumLinear16(sum, bias, norm, zero, dst + dx);
                    }
                    for (; dx < dstW; ++dx, sx += 2)
                    {
                        int32_t sum = src0[sx] + src0[sx + 1] + src1[sx] + src1[sx + 1];
                        dst[dx] = (uint8_t)Base::QuantizeSumLinear(sum, _mm512_cvtsi512_si32(bias), _mm_cvtss_f32(_mm512_castps512_ps128(norm)), _mm512_cvtsi512_si32(zero), 0, 255);
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
                size_t srcS = srcW * srcC, srcCF16 = AlignLo(srcC, F), srcCF64 = AlignLo(srcC, A);
                __m512i zero = _mm512_set1_epi32(dstZero);
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
                                QuantizedPoolingAverageNhwc(ps, srcS, srcC, srcCF16, srcCF64, hEnd - hStart, wEnd - wStart,
                                    _mm512_set1_epi32(-srcZero * area), _mm512_set1_ps(srcScale[0] / (dstScale[0] * float(area))), zero, dst);
                            }
                            else
                                QuantizedPoolingAverageNhwc(ps, srcS, srcC, srcCF16, srcCF64, hEnd - hStart, wEnd - wStart,
                                    _mm512_set1_epi32(bias), _mm512_set1_ps(norm), zero, dst);
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
                    QuantizedPoolingAverageNchw2x2(src, srcC * batch, srcH, srcW, dst, dstH, dstW, _mm512_set1_epi32(-srcZero * 4),
                        _mm512_set1_ps(srcScale[0] / (dstScale[0] * 4.0f)), _mm512_set1_epi32(dstZero));
                    return;
                }
            }
            Base::SynetQuantizedPoolingAverage(src, srcScale, srcZero, batch, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX,
                padY, padX, excludePad, dst, dstScale, dstZero, dstH, dstW, format);
        }
    }
#endif
}
