/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdSynetScale16b.h"
#include "Simd/SimdSynetAdd16bCommon.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512bw
    {
        template <typename S, typename D> void NormBias16bDF(const S* src, const float * norm,  const float * bias, D* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1);

        template <> SIMD_INLINE void NormBias16bDF(const float* src, const float* norm, const float* bias, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail0, src + 0), _mm512_maskz_loadu_ps(tail0, norm + 0), _mm512_maskz_loadu_ps(tail0, bias + 0)));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail1, src + F), _mm512_maskz_loadu_ps(tail1, norm + F), _mm512_maskz_loadu_ps(tail1, bias + F)));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, const float* norm, const float* bias, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_fmadd_ps(BFloat16ToFloat32<0>(_src), _mm512_maskz_loadu_ps(tail0, norm + 0), _mm512_maskz_loadu_ps(tail0, bias + 0)));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_fmadd_ps(BFloat16ToFloat32<1>(_src), _mm512_maskz_loadu_ps(tail1, norm + F), _mm512_maskz_loadu_ps(tail1, bias + F)));
        }

        template <> SIMD_INLINE void NormBias16bDF(const float* src, const float* norm, const float* bias, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512 dst0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail0, src + 0), _mm512_maskz_loadu_ps(tail0, norm + 0), _mm512_maskz_loadu_ps(tail0, bias + 0));
            __m512 dst1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail1, src + F), _mm512_maskz_loadu_ps(tail1, norm + F), _mm512_maskz_loadu_ps(tail1, bias + F));
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, const float* norm, const float* bias, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            __m512 dst0 = _mm512_fmadd_ps(BFloat16ToFloat32<0>(_src), _mm512_maskz_loadu_ps(tail0, norm + 0), _mm512_maskz_loadu_ps(tail0, bias + 0));
            __m512 dst1 = _mm512_fmadd_ps(BFloat16ToFloat32<1>(_src), _mm512_maskz_loadu_ps(tail1, norm + F), _mm512_maskz_loadu_ps(tail1, bias + F));
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <typename S, typename D> void NormBias16bDF(const S* src, __m512 norm, __m512 bias, D* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1);

        template <> SIMD_INLINE void NormBias16bDF(const float* src, __m512 norm, __m512 bias, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail0, src + 0), norm, bias));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail1, src + F), norm, bias));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, __m512 norm, __m512 bias, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_fmadd_ps(BFloat16ToFloat32<0>(_src), norm, bias));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_fmadd_ps(BFloat16ToFloat32<1>(_src), norm, bias));
        }

        template <> SIMD_INLINE void NormBias16bDF(const float* src, __m512 norm, __m512 bias, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512 dst0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail0, src + 0), norm, bias);
            __m512 dst1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail1, src + F), norm, bias);
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, __m512 norm, __m512 bias, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            __m512 dstE = _mm512_fmadd_ps(BFloat16ToFloat32Even(_src), norm, bias);
            __m512 dstO = _mm512_fmadd_ps(BFloat16ToFloat32Odd(_src), norm, bias);
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16Interlived(dstE, dstO));
        }

        template <typename S, typename D> void NormBias16bDF(const S* src, const __m512& norm0, const __m512& bias0, const __m512& norm1, const __m512& bias1, D* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1);

        template <> SIMD_INLINE void NormBias16bDF(const float* src, const __m512& norm0, const __m512& bias0, const __m512& norm1, const __m512& bias1, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail0, src + 0), norm0, bias0));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail1, src + F), norm1, bias1));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, const __m512& norm0, const __m512& bias0, const __m512& norm1, const __m512& bias1, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_fmadd_ps(BFloat16ToFloat32<0>(_src), norm0, bias0));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_fmadd_ps(BFloat16ToFloat32<1>(_src), norm1, bias1));
        }

        template <> SIMD_INLINE void NormBias16bDF(const float* src, const __m512& norm0, const __m512& bias0, const __m512& norm1, const __m512& bias1, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512 dst0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail0, src + 0), norm0, bias0);
            __m512 dst1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail1, src + F), norm1, bias1);
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, const __m512& norm0, const __m512& bias0, const __m512& norm1, const __m512& bias1, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            __m512 dst0 = _mm512_fmadd_ps(BFloat16ToFloat32<0>(_src), norm0, bias0);
            __m512 dst1 = _mm512_fmadd_ps(BFloat16ToFloat32<1>(_src), norm1, bias1);
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template<class S, class D> void SynetNormBias16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            if (format == SimdTensorFormatNchw)
            {
                size_t spatialDF = AlignLo(spatial, DF);
                __mmask32 tail = TailMask32(spatial - spatialDF);
                __mmask16 tail0 = TailMask16(spatial - spatialDF), tail1 = TailMask16(spatial - spatialDF - F);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _norm = _mm512_set1_ps(norm[c]);
                    __m512 _bias = _mm512_set1_ps(bias[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        NormBias16bDF<S, D>(src + s, _norm, _bias, dst + s, __mmask32(-1), __mmask16(-1), __mmask16(-1));
                    if(s < spatial)
                        NormBias16bDF<S, D>(src + s, _norm, _bias, dst + s, tail, tail0, tail1);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                if (channels == 3)
                {
                    size_t spatialDF = AlignLo(spatial, DF) * 3;
                    spatial *= 3;
                    __m512 _norm[3];
                    _norm[0] = _mm512_setr_ps(norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0]);
                    _norm[1] = _mm512_setr_ps(norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1]);
                    _norm[2] = _mm512_setr_ps(norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2], norm[0], norm[1], norm[2]);
                    __m512 _bias[3];
                    _bias[0] = _mm512_setr_ps(bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0]);
                    _bias[1] = _mm512_setr_ps(bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1]);
                    _bias[2] = _mm512_setr_ps(bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2], bias[0], bias[1], bias[2]);
                    __mmask32 tail = __mmask32(-1);
                    __mmask16 tail0 = __mmask16(-1), tail1 = __mmask16(-1);
                    size_t s = 0;
                    for (; s < spatialDF; s += 3 * DF)
                    {
                        NormBias16bDF<S, D>(src + s + 0 * DF, _norm[0], _bias[0], _norm[1], _bias[1], dst + s + 0 * DF, tail, tail0, tail1);
                        NormBias16bDF<S, D>(src + s + 1 * DF, _norm[2], _bias[2], _norm[0], _bias[0], dst + s + 1 * DF, tail, tail0, tail1);
                        NormBias16bDF<S, D>(src + s + 2 * DF, _norm[1], _bias[1], _norm[2], _bias[2], dst + s + 2 * DF, tail, tail0, tail1);
                    }
                    for (size_t t = 0; s < spatial; s += DF, t += 2)
                    {
                        tail = TailMask32(spatial - s), tail0 = TailMask16(spatial - s), tail1 = TailMask16(spatial - s - F);
                        NormBias16bDF<S, D>(src + s, _norm[(t + 0) % 3], _bias[(t + 0) % 3], _norm[(t + 1) % 3], _bias[(t + 1) % 3], dst + s, tail, tail0, tail1);
                    }
                }
                else if (channels == 8)
                {
                    spatial *= 8;
                    size_t spatialDF = AlignLo(spatial, DF);
                    __m512 _norm = Load<false>(norm, norm);
                    __m512 _bias = Load<false>(bias, bias);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        NormBias16bDF<S, D>(src + s, _norm, _bias, _norm, _bias, dst + s, __mmask32(-1), __mmask16(-1), __mmask16(-1));
                    if (s < spatial)
                    {
                        __mmask32 tail = TailMask32(spatial - spatialDF);
                        __mmask16 tail0 = TailMask16(spatial - spatialDF), tail1 = TailMask16(spatial - spatialDF - F);
                        NormBias16bDF<S, D>(src + s, _norm, _bias, _norm, _bias, dst + s, tail, tail0, tail1);
                    }
                }
                else if (channels == 16)
                {
                    spatial *= 16;
                    size_t spatialDF = AlignLo(spatial, DF);
                    __m512 _norm = _mm512_loadu_ps(norm);
                    __m512 _bias = _mm512_loadu_ps(bias);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        NormBias16bDF<S, D>(src + s, _norm, _bias, _norm, _bias, dst + s, __mmask32(-1), __mmask16(-1), __mmask16(-1));
                    if (s < spatial)
                    {
                        __mmask32 tail = TailMask32(spatial - spatialDF);
                        __mmask16 tail0 = TailMask16(spatial - spatialDF), tail1 = TailMask16(spatial - spatialDF - F);
                        NormBias16bDF<S, D>(src + s, _norm, _bias, _norm, _bias, dst + s, tail, tail0, tail1);
                    }
                }
                else
                {
                    size_t channelsDF = AlignLo(channels, DF);
                    __mmask32 tail = TailMask32(channels - channelsDF);
                    __mmask16 tail0 = TailMask16(channels - channelsDF), tail1 = TailMask16(channels - channelsDF - F);
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t c = 0;
                        for (; c < channelsDF; c += DF)
                            NormBias16bDF<S, D>(src + c, norm + c, bias + c, dst + c, __mmask32(-1), __mmask16(-1), __mmask16(-1));
                        if (c < channels)
                            NormBias16bDF<S, D>(src + c, norm + c, bias + c, dst + c, tail, tail0, tail1);
                        src += channels;
                        dst += channels;
                    }
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename S, typename D> void Norm16bDF(const S* src, const float* norm, D* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1);

        template <> SIMD_INLINE void Norm16bDF(const float* src, const float* norm, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_mul_ps(_mm512_maskz_loadu_ps(tail0, src + 0), _mm512_maskz_loadu_ps(tail0, norm + 0)));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_mul_ps(_mm512_maskz_loadu_ps(tail1, src + F), _mm512_maskz_loadu_ps(tail1, norm + F)));
        }

        template <> SIMD_INLINE void Norm16bDF(const uint16_t* src, const float* norm, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_mul_ps(BFloat16ToFloat32<0>(_src), _mm512_maskz_loadu_ps(tail0, norm + 0)));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_mul_ps(BFloat16ToFloat32<1>(_src), _mm512_maskz_loadu_ps(tail1, norm + F)));
        }

        template <> SIMD_INLINE void Norm16bDF(const float* src, const float* norm, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512 dst0 = _mm512_mul_ps(_mm512_maskz_loadu_ps(tail0, src + 0), _mm512_maskz_loadu_ps(tail0, norm + 0));
            __m512 dst1 = _mm512_mul_ps(_mm512_maskz_loadu_ps(tail1, src + F), _mm512_maskz_loadu_ps(tail1, norm + F));
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void Norm16bDF(const uint16_t* src, const float* norm, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            __m512 dst0 = _mm512_mul_ps(BFloat16ToFloat32<0>(_src), _mm512_maskz_loadu_ps(tail0, norm + 0));
            __m512 dst1 = _mm512_mul_ps(BFloat16ToFloat32<1>(_src), _mm512_maskz_loadu_ps(tail1, norm + F));
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <typename S, typename D> void Norm16bDF(const S* src, __m512 norm, D* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1);

        template <> SIMD_INLINE void Norm16bDF(const float* src, __m512 norm, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_mul_ps(_mm512_maskz_loadu_ps(tail0, src + 0), norm));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_mul_ps(_mm512_maskz_loadu_ps(tail1, src + F), norm));
        }

        template <> SIMD_INLINE void Norm16bDF(const uint16_t* src, __m512 norm, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_mul_ps(BFloat16ToFloat32<0>(_src), norm));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_mul_ps(BFloat16ToFloat32<1>(_src), norm));
        }

        template <> SIMD_INLINE void Norm16bDF(const float* src, __m512 norm, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512 dst0 = _mm512_mul_ps(_mm512_maskz_loadu_ps(tail0, src + 0), norm);
            __m512 dst1 = _mm512_mul_ps(_mm512_maskz_loadu_ps(tail1, src + F), norm);
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void Norm16bDF(const uint16_t* src, __m512 norm, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            __m512 dstE = _mm512_mul_ps(BFloat16ToFloat32Even(_src), norm);
            __m512 dstO = _mm512_mul_ps(BFloat16ToFloat32Odd(_src), norm);
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16Interlived(dstE, dstO));
        }

        template<class S, class D> void SynetNorm16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            if (format == SimdTensorFormatNchw)
            {
                size_t spatialDF = AlignLo(spatial, DF);
                __mmask32 tail = TailMask32(spatial - spatialDF);
                __mmask16 tail0 = TailMask16(spatial - spatialDF), tail1 = TailMask16(spatial - spatialDF - F);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _norm = _mm512_set1_ps(norm[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        Norm16bDF<S, D>(src + s, _norm, dst + s, __mmask32(-1), __mmask16(-1), __mmask16(-1));
                    if (s < spatial)
                        Norm16bDF<S, D>(src + s, _norm, dst + s, tail, tail0, tail1);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                size_t channelsDF = AlignLo(channels, DF);
                __mmask32 tail = TailMask32(channels - channelsDF);
                __mmask16 tail0 = TailMask16(channels - channelsDF), tail1 = TailMask16(channels - channelsDF - F);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsDF; c += DF)
                        Norm16bDF<S, D>(src + c, norm + c, dst + c, __mmask32(-1), __mmask16(-1), __mmask16(-1));
                    if (c < channels)
                        Norm16bDF<S, D>(src + c, norm + c, dst + c, tail, tail0, tail1);
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename S, typename D> void Bias16bDF(const S* src, const float* bias, D* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1);

        template <> SIMD_INLINE void Bias16bDF(const float* src, const float* bias, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_add_ps(_mm512_maskz_loadu_ps(tail0, src + 0), _mm512_maskz_loadu_ps(tail0, bias + 0)));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_add_ps(_mm512_maskz_loadu_ps(tail1, src + F), _mm512_maskz_loadu_ps(tail1, bias + F)));
        }

        template <> SIMD_INLINE void Bias16bDF(const uint16_t* src, const float* bias, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_add_ps(BFloat16ToFloat32<0>(_src), _mm512_maskz_loadu_ps(tail0, bias + 0)));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_add_ps(BFloat16ToFloat32<1>(_src), _mm512_maskz_loadu_ps(tail1, bias + F)));
        }

        template <> SIMD_INLINE void Bias16bDF(const float* src, const float* bias, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512 dst0 = _mm512_add_ps(_mm512_maskz_loadu_ps(tail0, src + 0), _mm512_maskz_loadu_ps(tail0, bias + 0));
            __m512 dst1 = _mm512_add_ps(_mm512_maskz_loadu_ps(tail1, src + F), _mm512_maskz_loadu_ps(tail1, bias + F));
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void Bias16bDF(const uint16_t* src, const float* bias, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            __m512 dst0 = _mm512_add_ps(BFloat16ToFloat32<0>(_src), _mm512_maskz_loadu_ps(tail0, bias + 0));
            __m512 dst1 = _mm512_add_ps(BFloat16ToFloat32<1>(_src), _mm512_maskz_loadu_ps(tail1, bias + F));
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <typename S, typename D> void Bias16bDF(const S* src, __m512 bias, D* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1);

        template <> SIMD_INLINE void Bias16bDF(const float* src, __m512 bias, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_add_ps(_mm512_maskz_loadu_ps(tail0, src + 0), bias));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_add_ps(_mm512_maskz_loadu_ps(tail1, src + F), bias));
        }

        template <> SIMD_INLINE void Bias16bDF(const uint16_t* src, __m512 bias, float* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            _mm512_mask_storeu_ps(dst + 0, tail0, _mm512_add_ps(BFloat16ToFloat32<0>(_src), bias));
            _mm512_mask_storeu_ps(dst + F, tail1, _mm512_add_ps(BFloat16ToFloat32<1>(_src), bias));
        }

        template <> SIMD_INLINE void Bias16bDF(const float* src, __m512 bias, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512 dst0 = _mm512_add_ps(_mm512_maskz_loadu_ps(tail0, src + 0), bias);
            __m512 dst1 = _mm512_add_ps(_mm512_maskz_loadu_ps(tail1, src + F), bias);
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void Bias16bDF(const uint16_t* src, __m512 bias, uint16_t* dst, __mmask32 tail, __mmask16 tail0, __mmask16 tail1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            __m512 dstE = _mm512_add_ps(BFloat16ToFloat32Even(_src), bias);
            __m512 dstO = _mm512_add_ps(BFloat16ToFloat32Odd(_src), bias);
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16Interlived(dstE, dstO));
        }

        template<class S, class D> void SynetBias16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            if (format == SimdTensorFormatNchw)
            {
                size_t spatialDF = AlignLo(spatial, DF);
                __mmask32 tail = TailMask32(spatial - spatialDF);
                __mmask16 tail0 = TailMask16(spatial - spatialDF), tail1 = TailMask16(spatial - spatialDF - F);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _bias = _mm512_set1_ps(bias[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        Bias16bDF<S, D>(src + s, _bias, dst + s, __mmask32(-1), __mmask16(-1), __mmask16(-1));
                    if (s < spatial)
                        Bias16bDF<S, D>(src + s, _bias, dst + s, tail, tail0, tail1);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                size_t channelsDF = AlignLo(channels, DF);
                __mmask32 tail = TailMask32(channels - channelsDF);
                __mmask16 tail0 = TailMask16(channels - channelsDF), tail1 = TailMask16(channels - channelsDF - F);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsDF; c += DF)
                        Bias16bDF<S, D>(src + c, bias + c, dst + c, __mmask32(-1), __mmask16(-1), __mmask16(-1));
                    if (c < channels)
                        Bias16bDF<S, D>(src + c, bias + c, dst + c, tail, tail0, tail1);
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<class S, class D> static SynetScale16b::WorkerPtr GetScale16bWorker(SimdBool norm, SimdBool bias)
        {
            if (norm)
                return bias ? SynetNormBias16b<S, D> : SynetNorm16b<S, D>;
            else
                return bias ? SynetBias16b<S, D> : NULL;
        }

        template<class S> static SynetScale16b::WorkerPtr GetScale16bWorker(SimdTensorDataType dType, SimdBool norm, SimdBool bias)
        {
            switch (dType)
            {
            case SimdTensorData32f: return GetScale16bWorker<S, float>(norm, bias);
            case SimdTensorData16b: return GetScale16bWorker<S, uint16_t>(norm, bias);
            default:
                return NULL;
            }
        }

        static SynetScale16b::WorkerPtr GetScale16bWorker(SimdTensorDataType sType, SimdTensorDataType dType, SimdBool norm, SimdBool bias)
        {
            switch (sType)
            {
            case SimdTensorData32f: return GetScale16bWorker<float>(dType, norm, bias);
            case SimdTensorData16b: return GetScale16bWorker<uint16_t>(dType, norm, bias);
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetScale16b::SynetScale16b(const Scale16bParam& p)
            : Avx2::SynetScale16b(p)
        {
            _worker = GetScale16bWorker(p.sType, p.dType, p.norm, p.bias);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetScale16bInit(size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdBool norm, SimdBool bias)
        {
            Scale16bParam param(channels, spatial, srcType, dstType, format, norm, bias);
            if (!param.Valid())
                return NULL;
            if (SynetScale16b::Preferable(param))
                return new SynetScale16b(param);
            return NULL;
        }
    }
#endif
}
