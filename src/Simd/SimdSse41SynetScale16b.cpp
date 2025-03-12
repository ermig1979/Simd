/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse41
    {
        template <typename S, typename D> void NormBias16bDF(const S* src, const float * norm,  const float * bias, D* dst);

        template <> SIMD_INLINE void NormBias16bDF(const float* src, const float* norm, const float* bias, float* dst)
        {
            _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 0), _mm_loadu_ps(norm + 0)), _mm_loadu_ps(bias + 0)));
            _mm_storeu_ps(dst + F, _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + F), _mm_loadu_ps(norm + F)), _mm_loadu_ps(bias + F)));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, const float* norm, const float* bias, float* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(BFloat16ToFloat32<0>(_src), _mm_loadu_ps(norm + 0)), _mm_loadu_ps(bias + 0)));
            _mm_storeu_ps(dst + F, _mm_add_ps(_mm_mul_ps(BFloat16ToFloat32<1>(_src), _mm_loadu_ps(norm + F)), _mm_loadu_ps(bias + F)));
        }

        template <> SIMD_INLINE void NormBias16bDF(const float* src, const float* norm, const float* bias, uint16_t* dst)
        {
            __m128 dst0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 0), _mm_loadu_ps(norm + 0)), _mm_loadu_ps(bias + 0));
            __m128 dst1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + F), _mm_loadu_ps(norm + F)), _mm_loadu_ps(bias + F));
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, const float* norm, const float* bias, uint16_t* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128 dst0 = _mm_add_ps(_mm_mul_ps(BFloat16ToFloat32<0>(_src), _mm_loadu_ps(norm + 0)), _mm_loadu_ps(bias + 0));
            __m128 dst1 = _mm_add_ps(_mm_mul_ps(BFloat16ToFloat32<1>(_src), _mm_loadu_ps(norm + F)), _mm_loadu_ps(bias + F));
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16(dst0, dst1));
        }

        template <typename S, typename D> void NormBias16bDF(const S* src, __m128 norm, __m128 bias, D* dst);

        template <> SIMD_INLINE void NormBias16bDF(const float* src, __m128 norm, __m128 bias, float* dst)
        {
            _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 0), norm), bias));
            _mm_storeu_ps(dst + F, _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + F), norm), bias));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, __m128 norm, __m128 bias, float* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(BFloat16ToFloat32<0>(_src), norm), bias));
            _mm_storeu_ps(dst + F, _mm_add_ps(_mm_mul_ps(BFloat16ToFloat32<1>(_src), norm), bias));
        }

        template <> SIMD_INLINE void NormBias16bDF(const float* src, __m128 norm, __m128 bias, uint16_t* dst)
        {
            __m128 dst0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 0), norm), bias);
            __m128 dst1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + F), norm), bias);
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void NormBias16bDF(const uint16_t* src, __m128 norm, __m128 bias, uint16_t* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128 dstE = _mm_add_ps(_mm_mul_ps(BFloat16ToFloat32Even(_src), norm), bias);
            __m128 dstO = _mm_add_ps(_mm_mul_ps(BFloat16ToFloat32Odd(_src), norm), bias);
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16Interlived(dstE, dstO));
        }

        template<class S, class D> void SynetNormBias16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            if (format == SimdTensorFormatNchw)
            {
                size_t spatialDF = AlignLo(spatial, DF);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 _norm = _mm_set1_ps(norm[c]);
                    __m128 _bias = _mm_set1_ps(bias[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        NormBias16bDF<S, D>(src + s, _norm, _bias, dst + s);
                    for (; s < spatial; ++s)
                        Base::NormBias16b<S, D>(src[s], norm[c], bias[c], dst[s]);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                size_t channelsDF = AlignLo(channels, DF);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsDF; c += DF)
                        NormBias16bDF<S, D>(src + c, norm + c, bias + c, dst + c);
                    for (; c < channels; ++c)
                        Base::NormBias16b<S, D>(src[c], norm[c], bias[c], dst[c]);
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename S, typename D> void Norm16bDF(const S* src, const float* norm, D* dst);

        template <> SIMD_INLINE void Norm16bDF(const float* src, const float* norm, float* dst)
        {
            _mm_storeu_ps(dst + 0, _mm_mul_ps(_mm_loadu_ps(src + 0), _mm_loadu_ps(norm + 0)));
            _mm_storeu_ps(dst + F, _mm_mul_ps(_mm_loadu_ps(src + F), _mm_loadu_ps(norm + F)));
        }

        template <> SIMD_INLINE void Norm16bDF(const uint16_t* src, const float* norm, float* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            _mm_storeu_ps(dst + 0, _mm_mul_ps(BFloat16ToFloat32<0>(_src), _mm_loadu_ps(norm + 0)));
            _mm_storeu_ps(dst + F, _mm_mul_ps(BFloat16ToFloat32<1>(_src), _mm_loadu_ps(norm + F)));
        }

        template <> SIMD_INLINE void Norm16bDF(const float* src, const float* norm, uint16_t* dst)
        {
            __m128 dst0 = _mm_mul_ps(_mm_loadu_ps(src + 0), _mm_loadu_ps(norm + 0));
            __m128 dst1 = _mm_mul_ps(_mm_loadu_ps(src + F), _mm_loadu_ps(norm + F));
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void Norm16bDF(const uint16_t* src, const float* norm, uint16_t* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128 dst0 = _mm_mul_ps(BFloat16ToFloat32<0>(_src), _mm_loadu_ps(norm + 0));
            __m128 dst1 = _mm_mul_ps(BFloat16ToFloat32<1>(_src), _mm_loadu_ps(norm + F));
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16(dst0, dst1));
        }

        template <typename S, typename D> void Norm16bDF(const S* src, __m128 norm, D* dst);

        template <> SIMD_INLINE void Norm16bDF(const float* src, __m128 norm, float* dst)
        {
            _mm_storeu_ps(dst + 0, _mm_mul_ps(_mm_loadu_ps(src + 0), norm));
            _mm_storeu_ps(dst + F, _mm_mul_ps(_mm_loadu_ps(src + F), norm));
        }

        template <> SIMD_INLINE void Norm16bDF(const uint16_t* src, __m128 norm, float* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            _mm_storeu_ps(dst + 0, _mm_mul_ps(BFloat16ToFloat32<0>(_src), norm));
            _mm_storeu_ps(dst + F, _mm_mul_ps(BFloat16ToFloat32<1>(_src), norm));
        }

        template <> SIMD_INLINE void Norm16bDF(const float* src, __m128 norm, uint16_t* dst)
        {
            __m128 dst0 = _mm_mul_ps(_mm_loadu_ps(src + 0), norm);
            __m128 dst1 = _mm_mul_ps(_mm_loadu_ps(src + F), norm);
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void Norm16bDF(const uint16_t* src, __m128 norm, uint16_t* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128 dstE = _mm_mul_ps(BFloat16ToFloat32Even(_src), norm);
            __m128 dstO = _mm_mul_ps(BFloat16ToFloat32Odd(_src), norm);
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16Interlived(dstE, dstO));
        }

        template<class S, class D> void SynetNorm16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            if (format == SimdTensorFormatNchw)
            {
                size_t spatialDF = AlignLo(spatial, DF);
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 _norm = _mm_set1_ps(norm[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        Norm16bDF<S, D>(src + s, _norm, dst + s);
                    for (; s < spatial; ++s)
                        Base::Norm16b<S, D>(src[s], norm[c], dst[s]);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                size_t channelsDF = AlignLo(channels, DF);
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsDF; c += DF)
                        Norm16bDF<S, D>(src + c, norm + c, dst + c);
                    for (; c < channels; ++c)
                        Base::Norm16b<S, D>(src[c], norm[c], dst[c]);
                    src += channels;
                    dst += channels;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<class S, class D> void SynetBias16b(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8)
        {
            const S* src = (const S*)src8;
            D* dst = (D*)dst8;
            if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _bias = bias[c];
                    for (size_t s = 0; s < spatial; ++s)
                        Base::Bias16b<S, D>(src[s], _bias, dst[s]);
                    src += spatial;
                    dst += spatial;
                }
            }
            else if (format == SimdTensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t c = 0; c < channels; ++c)
                        Base::Bias16b<S, D>(src[c], bias[c], dst[c]);
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
            : Base::SynetScale16b(p)
        {
            //_worker = GetScale16bWorker(p.sType, p.dType, p.norm, p.bias);
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
