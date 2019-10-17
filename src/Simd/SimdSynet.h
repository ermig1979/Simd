/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#ifndef __SimdSynet_h__
#define __SimdSynet_h__

#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE bool NchwCompatible(size_t channels, size_t spatial, SimdTensorFormatType format)
        {
            return (format == SimdTensorFormatNchw && spatial != 1) || (format == SimdTensorFormatNhwc && channels == 1);
        }

        SIMD_INLINE bool NhwcCompatible(size_t channels, size_t spatial, SimdTensorFormatType format)
        {
            return (format == SimdTensorFormatNhwc && channels != 1) || (format == SimdTensorFormatNchw && spatial == 1);
        }

        template <SimdSynetEltwiseOperationType type> float SynetEltwiseLayerForward(float a, float b);

        template <> SIMD_INLINE float SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(float a, float b)
        {
            return a * b;
        }

        template <> SIMD_INLINE float SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(float a, float b)
        {
            return Simd::Max(a, b);
        }

        template <> SIMD_INLINE float SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(float a, float b)
        {
            return Simd::Min(a, b);
        }

        SIMD_INLINE float SynetElu32f(float src, float alpha)
        {
            return src >= 0.0f ? src : alpha * (::exp(src) - 1.0f);
        }

        SIMD_INLINE float SynetFusedLayerForward0(float x, float s)
        {
            return (x - Simd::Abs(x))*s + Simd::Max(0.0f, x);
        }

        SIMD_INLINE float SynetFusedLayerForward1(float x, float s, float b)
        {
            return Simd::Max(0.0f, -x)*s + b + Simd::Max(0.0f, x);
        }

        SIMD_INLINE float SynetFusedLayerForward2(float src, float scale, float bias, float slope)
        {
            float x = src * scale + bias;
            return Simd::Max(0.0f, x) + Simd::Min(0.0f, x)*slope;
        }

        SIMD_INLINE float SynetFusedLayerForward3(float x, float s)
        {
            return Simd::Max(0.0f, x) + Simd::Min(x, 0.0f) * s;
        }

        SIMD_INLINE void SynetFusedLayerForward4(float src, float bias0, float scale1, float bias1, float * dst0, float * dst1)
        {
            float x = src + bias0;
            dst0[0] = Simd::Max(0.0f, x);
            dst1[0] = Simd::Max(0.0f, x*scale1 + bias1);
        }

        SIMD_INLINE float SynetFusedLayerForward8(float src0, float src1, float src2)
        {
            return src0 + src1 * src2;
        }

        SIMD_INLINE float SynetFusedLayerForward9(float src, float scale, float bias)
        {
            return Simd::Max(0.0f, src * scale + bias);
        }

        SIMD_INLINE float SynetHswish32f(float value, float shift, float scale)
        {
            return Simd::Max(Simd::Min(value, shift) + shift, 0.0f)*scale*value;
        }

        SIMD_INLINE float SynetPreluLayerForward(float value, float slope)
        {
            return Simd::Max(0.0f, value) + slope*Simd::Min(value, 0.0f);
        }
    }

#ifdef SIMD_SSE_ENABLE
    namespace Sse
    {
        SIMD_INLINE __m128 SynetHswish32f(__m128 value, __m128 shift, __m128 scale)
        {
            return _mm_mul_ps(_mm_mul_ps(_mm_max_ps(_mm_add_ps(_mm_min_ps(value, shift), shift), _mm_setzero_ps()), scale), value);
        }

        SIMD_INLINE __m128 SynetPreluLayerForward(const __m128 & value, const __m128 & slope)
        {
            __m128 positive = _mm_max_ps(_mm_setzero_ps(), value);
            __m128 negative = _mm_min_ps(_mm_setzero_ps(), value);
            return _mm_add_ps(positive, _mm_mul_ps(slope, negative));
        }
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        SIMD_INLINE __m256 SynetHswish32f(__m256 value, __m256 shift, __m256 scale)
        {
            return _mm256_mul_ps(_mm256_mul_ps(_mm256_max_ps(_mm256_add_ps(_mm256_min_ps(value, shift), shift), _mm256_setzero_ps()), scale), value);
        }

        SIMD_INLINE __m256 SynetPreluLayerForward(const __m256 & value, const __m256 & slope)
        {
            __m256 positive = _mm256_max_ps(_mm256_setzero_ps(), value);
            __m256 negative = _mm256_min_ps(_mm256_setzero_ps(), value);
            return _mm256_add_ps(positive, _mm256_mul_ps(slope, negative));
        }
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX512F_ENABLE
    namespace Avx512f
    {
        SIMD_INLINE __m512 SynetHswish32f(__m512 value, __m512 shift, __m512 scale)
        {
            return _mm512_mul_ps(_mm512_mul_ps(_mm512_max_ps(_mm512_add_ps(_mm512_min_ps(value, shift), shift), _mm512_setzero_ps()), scale), value);
        }

        SIMD_INLINE __m512 SynetPreluLayerForward(const __m512 & value, const __m512 & slope)
        {
            __m512 positive = _mm512_max_ps(_mm512_setzero_ps(), value);
            __m512 negative = _mm512_min_ps(_mm512_setzero_ps(), value);
            return _mm512_add_ps(positive, _mm512_mul_ps(slope, negative));
        }
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        SIMD_INLINE float32x4_t SynetHswish32f(float32x4_t value, float32x4_t shift, float32x4_t scale)
        {
            return vmulq_f32(vmulq_f32(vmaxq_f32(vaddq_f32(vminq_f32(value, shift), shift), vdupq_n_f32(0.0f)), scale), value);
        }

        SIMD_INLINE float32x4_t SynetPreluLayerForward(const float32x4_t & value, const float32x4_t & slope, const float32x4_t & zero)
        {
            float32x4_t positive = vmaxq_f32(zero, value);
            float32x4_t negative = vminq_f32(zero, value);
            return vmlaq_f32(positive, slope, negative);
        }
    }
#endif//SIMD_NEON_ENABLE
}

#endif//__SimdSynet_h__
