/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
        const int U8_PRECISE_MIN = 0;
        const int U8_PRECISE_MAX = 255;
        const int I8_PRECISE_MAX = 127;
        const int I8_PRECISE_MIN = -128;

        const int U8_NARROWED_MIN = 0;
        const int U8_NARROWED_MAX = 180;
        const int I8_NARROWED_MAX = 90;
        const int I8_NARROWED_MIN = -90;

        //---------------------------------------------------------------------

        SIMD_INLINE bool NchwCompatible(size_t channels, size_t spatial, SimdTensorFormatType format)
        {
            return (format == SimdTensorFormatNchw && spatial != 1) || (format == SimdTensorFormatNhwc && channels == 1);
        }

        SIMD_INLINE bool NhwcCompatible(size_t channels, size_t spatial, SimdTensorFormatType format)
        {
            return (format == SimdTensorFormatNhwc && channels != 1) || (format == SimdTensorFormatNchw && spatial == 1);
        }

#if defined(SIMD_INT8_DEBUG_ENABLE)
        SIMD_INLINE bool FmaAvoid(SimdSynetCompatibilityType compatibility)
        {
            return (compatibility & SimdSynetCompatibilityFmaMask) == SimdSynetCompatibilityFmaAvoid;
        }

        SIMD_INLINE bool FmaNoTail(SimdSynetCompatibilityType compatibility)
        {
            return (compatibility & SimdSynetCompatibilityFmaMask) == SimdSynetCompatibilityFmaNoTail;
        }

        SIMD_INLINE bool Precise(SimdSynetCompatibilityType compatibility)
        {
            return (compatibility & SimdSynetCompatibility8iMask) == SimdSynetCompatibility8iPrecise;
        }

        SIMD_INLINE bool Overflow(SimdSynetCompatibilityType compatibility)
        {
            return (compatibility & SimdSynetCompatibility8iMask) == SimdSynetCompatibility8iOverflow;
        }

        SIMD_INLINE bool Narrowed(SimdSynetCompatibilityType compatibility)
        {
            return (compatibility & SimdSynetCompatibility8iMask) == SimdSynetCompatibility8iNarrowed;
        }
#else
        SIMD_INLINE constexpr bool FmaAvoid(SimdSynetCompatibilityType compatibility)
        {
            return false;
        }

        SIMD_INLINE constexpr bool FmaNoTail(SimdSynetCompatibilityType compatibility)
        {
            return false;
        }

        SIMD_INLINE constexpr bool Precise(SimdSynetCompatibilityType compatibility)
        {
            return false;
        }

        SIMD_INLINE constexpr bool Overflow(SimdSynetCompatibilityType compatibility)
        {
            return false;
        }

        SIMD_INLINE constexpr bool Narrowed(SimdSynetCompatibilityType compatibility)
        {
            return true;
        }
#endif

        //---------------------------------------------------------------------

        SIMD_INLINE uint8_t SynetConvert32fTo8u(float value, float scale, float shift, int lower, int upper)
        {
            return (uint8_t)Simd::RestrictRange(Round(value * scale + shift), lower, upper);
        }

        template<bool narrow> SIMD_INLINE uint8_t SynetConvert32fTo8u(float value, float scale, float shift);

        template<> SIMD_INLINE uint8_t SynetConvert32fTo8u<false>(float value, float scale, float shift)
        {
            return SynetConvert32fTo8u(value, scale, shift, U8_PRECISE_MIN, U8_PRECISE_MAX);
        }

        template<> SIMD_INLINE uint8_t SynetConvert32fTo8u<true>(float value, float scale, float shift)
        {
            return SynetConvert32fTo8u(value, scale, shift, U8_NARROWED_MIN, U8_NARROWED_MAX);
        }

        SIMD_INLINE int8_t SynetConvert32fTo8i(float value, float scale, float shift, int lower, int upper)
        {
            return (int8_t)Simd::RestrictRange(Round(value * scale + shift), lower, upper);
        }

        template<bool narrow> SIMD_INLINE int8_t SynetConvert32fTo8i(float value, float scale, float shift);

        template<> SIMD_INLINE int8_t SynetConvert32fTo8i<false>(float value, float scale, float shift)
        {
            return SynetConvert32fTo8i(value, scale, shift, I8_PRECISE_MIN, I8_PRECISE_MAX);
        }

        template<> SIMD_INLINE int8_t SynetConvert32fTo8i<true>(float value, float scale, float shift)
        {
            return SynetConvert32fTo8i(value, scale, shift, I8_NARROWED_MIN, I8_NARROWED_MAX);
        }

        //---------------------------------------------------------------------

        SIMD_INLINE float SynetConvert8uTo32f(int value, float scale, float shift)
        {
            return value * scale + shift;
        }

        //---------------------------------------------------------------------

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

        SIMD_INLINE float SynetHardSigmoid32f(float value, float scale, float shift)
        {
            return Simd::Max(0.0f, Simd::Min(value * scale + shift, 1.0f));
        }

        SIMD_INLINE float SynetHswish32f(float value, float shift, float scale)
        {
            return Simd::Max(Simd::Min(value, shift) + shift, 0.0f)*scale*value;
        }

        SIMD_INLINE float SynetMish32f(float value, float threshold)
        {
            return value > threshold ? value : value * (1.0f - 2.0f / (Simd::Square(::exp(value) + 1.0f) + 1.0f));
        }

        SIMD_INLINE float SynetRelu32f(float value, float slope)
        {
            return Simd::Max(0.0f, value) + slope * Simd::Min(value, 0.0f);
        }

        SIMD_INLINE float SynetSigmoid32f(float value, float slope)
        {
            return 1.0f / (1.0f + ::exp(-value*slope));
        }

        SIMD_INLINE float SynetSoftplus32f(float value, float beta, float threshold)
        {
            return value > threshold ? value : ::log(1.0f + ::exp(value * beta)) / beta;
        }

        SIMD_INLINE float SynetSwish32f(float value, float slope)
        {
            return value / (1.0f + ::exp(-value * slope));
        }

        SIMD_INLINE float SynetTanh32f(float value, float slope)
        {
            return ::tanh(value*slope);
        }

        //---------------------------------------------------------------------

        template<SimdSynetUnaryOperation32fType type> float SynetUnaryOperation32f(float value);

        template<> SIMD_INLINE float SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(float value)
        {
            return value > 0 ? value : -value;
        }

        template<> SIMD_INLINE float SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(float value)
        {
            return ::exp(value);
        }

        template<> SIMD_INLINE float SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(float value)
        {
            return ::log(value);
        }

        template<> SIMD_INLINE float SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(float value)
        {
            return -value;
        }

        template<> SIMD_INLINE float SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(float value)
        {
            return 1.0f / ::sqrt(value);
        }

        template<> SIMD_INLINE float SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(float value)
        {
            return ::sqrt(value);
        }

        template<> SIMD_INLINE float SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(float value)
        {
            return ::tanh(value);
        }

        template<> SIMD_INLINE float SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(float value)
        {
            return 0.0f;
        }

        //---------------------------------------------------------------------

        static SIMD_INLINE int32_t Set4(uint8_t value)
        {
            return int32_t(value) | (int32_t(value) << 8) | (int32_t(value) << 16) | (int32_t(value) << 24);
        }
    }

#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        SIMD_INLINE __m128 SynetHardSigmoid32f(__m128 value, __m128 scale, __m128 shift)
        {
            return _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(_mm_add_ps(_mm_mul_ps(value, scale), shift), _mm_set1_ps(1.0f)));
        }

        SIMD_INLINE __m128 SynetHswish32f(__m128 value, __m128 shift, __m128 scale)
        {
            return _mm_mul_ps(_mm_mul_ps(_mm_max_ps(_mm_add_ps(_mm_min_ps(value, shift), shift), _mm_setzero_ps()), scale), value);
        }

        SIMD_INLINE __m128 SynetRelu32f(__m128 value, __m128 slope)
        {
            __m128 positive = _mm_max_ps(_mm_setzero_ps(), value);
            __m128 negative = _mm_min_ps(_mm_setzero_ps(), value);
            return _mm_add_ps(positive, _mm_mul_ps(slope, negative));
        }
    }
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        SIMD_INLINE __m128i Set4(const uint8_t* src)
        {
            return _mm_set1_epi32(*(int32_t*)src);
        }

        template<bool overflow> void Madd4(__m128i& i32, __m128i u8, __m128i i8);

        template<> SIMD_INLINE void Madd4<true>(__m128i& i32, __m128i u8, __m128i i8)
        {
            i32 = _mm_add_epi32(i32, _mm_madd_epi16(_mm_maddubs_epi16(u8, i8), Sse2::K16_0001));
        }

        template<> SIMD_INLINE void Madd4<false>(__m128i& i32, __m128i u8, __m128i i8)
        {
            __m128i lo = _mm_madd_epi16(UnpackU8<0>(u8), UnpackI8<0>(i8));
            __m128i hi = _mm_madd_epi16(UnpackU8<1>(u8), UnpackI8<1>(i8));
            i32 = _mm_add_epi32(i32, _mm_hadd_epi32(lo, hi));
        }
    }
#endif//SIMD_SSE41_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        SIMD_INLINE __m256 SynetHardSigmoid32f(__m256 value, __m256 scale, __m256 shift)
        {
            return _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_add_ps(_mm256_mul_ps(value, scale), shift), _mm256_set1_ps(1.0f)));
        }

        SIMD_INLINE __m256 SynetHswish32f(__m256 value, __m256 shift, __m256 scale)
        {
            return _mm256_mul_ps(_mm256_mul_ps(_mm256_max_ps(_mm256_add_ps(_mm256_min_ps(value, shift), shift), _mm256_setzero_ps()), scale), value);
        }

        SIMD_INLINE __m256 SynetRelu32f(const __m256 & value, const __m256 & slope)
        {
            __m256 positive = _mm256_max_ps(_mm256_setzero_ps(), value);
            __m256 negative = _mm256_min_ps(_mm256_setzero_ps(), value);
            return _mm256_add_ps(positive, _mm256_mul_ps(slope, negative));
        }
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        SIMD_INLINE __m256i Set4(const uint8_t* src)
        {
            return _mm256_set1_epi32(*(int32_t*)src);
        }

        template<bool overflow> void Madd4(__m256i& i32, __m256i u8, __m256i i8);

        template<> SIMD_INLINE void Madd4<true>(__m256i& i32, __m256i u8, __m256i i8)
        {
            i32 = _mm256_add_epi32(i32, _mm256_madd_epi16(_mm256_maddubs_epi16(u8, i8), Avx2::K16_0001));
        }

        template<> SIMD_INLINE void Madd4<false>(__m256i& i32, __m256i u8, __m256i i8)
        {
            __m256i lo = _mm256_madd_epi16(Cvt8uTo16i<0>(u8), Cvt8iTo16i<0>(i8));
            __m256i hi = _mm256_madd_epi16(Cvt8uTo16i<1>(u8), Cvt8iTo16i<1>(i8));
            i32 = _mm256_add_epi32(i32, PermutedHadd32i(lo, hi));
        }
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE
    namespace Avx512f
    {
        SIMD_INLINE __m512 SynetHardSigmoid32f(__m512 value, __m512 scale, __m512 shift)
        {
            return _mm512_max_ps(_mm512_setzero_ps(), _mm512_min_ps(_mm512_add_ps(_mm512_mul_ps(value, scale), shift), _mm512_set1_ps(1.0f)));
        }

        SIMD_INLINE __m512 SynetHswish32f(__m512 value, __m512 shift, __m512 scale)
        {
            return _mm512_mul_ps(_mm512_mul_ps(_mm512_max_ps(_mm512_add_ps(_mm512_min_ps(value, shift), shift), _mm512_setzero_ps()), scale), value);
        }

        SIMD_INLINE __m512 SynetRelu32f(const __m512 & value, const __m512 & slope)
        {
            __m512 positive = _mm512_max_ps(_mm512_setzero_ps(), value);
            __m512 negative = _mm512_min_ps(_mm512_setzero_ps(), value);
            return _mm512_add_ps(positive, _mm512_mul_ps(slope, negative));
        }
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        SIMD_INLINE __m512i Set4(const uint8_t* src)
        {
            return _mm512_set1_epi32(*(int32_t*)src);
        }

        template<bool overflow> void Madd4(__m512i& i32, __m512i u8, __m512i i8);

        template<> SIMD_INLINE void Madd4<true>(__m512i& i32, __m512i u8, __m512i i8)
        {
            i32 = _mm512_add_epi32(i32, _mm512_madd_epi16(_mm512_maddubs_epi16(u8, i8), Avx512bw::K16_0001));
        }

        template<> SIMD_INLINE void Madd4<false>(__m512i& i32, __m512i u8, __m512i i8)
        {
            __m512i lo = _mm512_madd_epi16(Cvt8uTo16i<0>(u8), Cvt8iTo16i<0>(i8));
            __m512i hi = _mm512_madd_epi16(Cvt8uTo16i<1>(u8), Cvt8iTo16i<1>(i8));
            i32 = _mm512_add_epi32(i32, Hadd32(lo, hi));
        }
    }
#endif//SIMD_AVX512BW_ENABLE

#ifdef SIMD_AVX512VNNI_ENABLE
    namespace Avx512vnni
    {
        template<bool overflow> void Madd4(__m512i& i32, __m512i u8, __m512i i8);

        template<> SIMD_INLINE void Madd4<true>(__m512i& i32, __m512i u8, __m512i i8)
        {
            i32 = _mm512_add_epi32(i32, _mm512_madd_epi16(_mm512_maddubs_epi16(u8, i8), Avx512bw::K16_0001));
        }

        template<> SIMD_INLINE void Madd4<false>(__m512i& i32, __m512i u8, __m512i i8)
        {
            i32 = _mm512_dpbusd_epi32(i32, u8, i8);
        }
    }
#endif//SIMD_AVX512VNNI_ENABLE

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        SIMD_INLINE float32x4_t SynetHardSigmoid32f(float32x4_t value, float32x4_t scale, float32x4_t shift)
        {
            return vmaxq_f32(vdupq_n_f32(0.0f), vminq_f32(vaddq_f32(vmulq_f32(value, scale), shift), vdupq_n_f32(1.0f)));
        }

        SIMD_INLINE float32x4_t SynetHswish32f(float32x4_t value, float32x4_t shift, float32x4_t scale)
        {
            return vmulq_f32(vmulq_f32(vmaxq_f32(vaddq_f32(vminq_f32(value, shift), shift), vdupq_n_f32(0.0f)), scale), value);
        }

        SIMD_INLINE float32x4_t SynetRelu32f(const float32x4_t & value, const float32x4_t & slope, const float32x4_t & zero)
        {
            float32x4_t positive = vmaxq_f32(zero, value);
            float32x4_t negative = vminq_f32(zero, value);
            return vmlaq_f32(positive, slope, negative);
        }
    }
#endif//SIMD_NEON_ENABLE
}

#endif//__SimdSynet_h__
