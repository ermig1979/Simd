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
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__AVX512FP16__) && _MSC_VER >= 1933
#define __AVX512FP16__
#endif

#include "Simd/SimdSynetAdd16b.h"
#include "Simd/SimdSynetAdd16bCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)
    namespace AmxBf16
    {
#if defined(__AVX512FP16__)
        SIMD_INLINE __m512 Bf16ToFloat32(__m256i value)
        {
            return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(value), 16));
        }

        template<bool mask> SIMD_INLINE __m512i Load16(const uint16_t* src, __mmask32 tail)
        {
            return mask ? _mm512_maskz_loadu_epi16(tail, src) : _mm512_loadu_si512((__m512i*)src);
        }

        template<bool mask> SIMD_INLINE void LoadAdd16b(__m512& d0, __m512& d1, const float* src, __mmask32 tail)
        {
            d0 = mask ? _mm512_maskz_loadu_ps(__mmask16(tail), src) : _mm512_loadu_ps(src);
            d1 = mask ? _mm512_maskz_loadu_ps(__mmask16(tail >> 16), src + F) : _mm512_loadu_ps(src + F);
        }

        template<bool mask> SIMD_INLINE void LoadAdd16b(__m512& d0, __m512& d1, const uint16_t* src, __mmask32 tail)
        {
            __m512i value = Load16<mask>(src, tail);
            d0 = Bf16ToFloat32(_mm512_castsi512_si256(value));
            d1 = Bf16ToFloat32(_mm512_extracti64x4_epi64(value, 1));
        }

        template<bool mask> SIMD_INLINE void LoadAdd16b(__m512& d0, __m512& d1, const Float16* src, __mmask32 tail)
        {
            __m512i value = Load16<mask>((const uint16_t*)src, tail);
            d0 = _mm512_cvtxph_ps(_mm512_castph512_ph256(_mm512_castsi512_ph(value)));
            d1 = _mm512_cvtxph_ps(_mm256_castsi256_ph(_mm512_extracti64x4_epi64(value, 1)));
        }

        SIMD_INLINE __m512h PackFloat16(__m256h lo, __m256h hi)
        {
            return _mm512_castsi512_ph(_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castph_si256(lo)), _mm256_castph_si256(hi), 1));
        }

        template<bool mask> SIMD_INLINE void StoreAdd16b(float* dst, __m512 d0, __m512 d1, __mmask32 tail)
        {
            if (mask)
            {
                _mm512_mask_storeu_ps(dst, __mmask16(tail), d0);
                _mm512_mask_storeu_ps(dst + F, __mmask16(tail >> 16), d1);
            }
            else
            {
                _mm512_storeu_ps(dst, d0);
                _mm512_storeu_ps(dst + F, d1);
            }
        }

        template<bool mask> SIMD_INLINE void StoreAdd16b(uint16_t* dst, __m512 d0, __m512 d1, __mmask32 tail)
        {
            __m512i value = (__m512i)_mm512_cvtne2ps_pbh(d1, d0);
            if (mask)
                _mm512_mask_storeu_epi16(dst, tail, value);
            else
                _mm512_storeu_si512((__m512i*)dst, value);
        }

        template<bool mask> SIMD_INLINE void StoreAdd16b(Float16* dst, __m512 d0, __m512 d1, __mmask32 tail)
        {
            __m512i value = _mm512_castph_si512(PackFloat16(_mm512_cvtxps_ph(d0), _mm512_cvtxps_ph(d1)));
            if (mask)
                _mm512_mask_storeu_epi16((uint16_t*)dst, tail, value);
            else
                _mm512_storeu_ph(dst, _mm512_castsi512_ph(value));
        }

        template<bool mask, typename A, typename B, typename D> SIMD_INLINE void Add16bDF(const A* a, const B* b, D* dst, __mmask32 tail)
        {
            __m512 a0, a1, b0, b1;
            LoadAdd16b<mask>(a0, a1, a, tail);
            LoadAdd16b<mask>(b0, b1, b, tail);
            StoreAdd16b<mask>(dst, _mm512_add_ps(a0, b0), _mm512_add_ps(a1, b1), tail);
        }

        template <typename A, typename B, typename D> static void Add16bUniform(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
        {
            const A* a = (const A*)a8;
            const B* b = (const B*)b8;
            D* dst = (D*)dst8;
            size_t sizeDF = AlignLo(size, DF), i = 0;

            for (; i < sizeDF; i += DF)
                Add16bDF<false>(a + i, b + i, dst + i, __mmask32(-1));
            if (i < size)
                Add16bDF<true>(a + i, b + i, dst + i, TailMask32(size - i));
        }

        template <> void Add16bUniform<Float16, Float16, Float16>(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
        {
            const uint16_t* a = (const uint16_t*)a8;
            const uint16_t* b = (const uint16_t*)b8;
            uint16_t* dst = (uint16_t*)dst8;
            size_t sizeQF = AlignLo(size, QF), sizeDF = AlignLo(size, DF), i = 0;

            for (; i < sizeQF; i += QF)
            {
                _mm512_storeu_ph(dst + i, _mm512_add_ph(_mm512_loadu_ph(a + i), _mm512_loadu_ph(b + i)));
                _mm512_storeu_ph(dst + i + DF, _mm512_add_ph(_mm512_loadu_ph(a + i + DF), _mm512_loadu_ph(b + i + DF)));
            }
            for (; i < sizeDF; i += DF)
                _mm512_storeu_ph(dst + i, _mm512_add_ph(_mm512_loadu_ph(a + i), _mm512_loadu_ph(b + i)));
            if (i < size)
            {
                __mmask32 tail = TailMask32(size - i);
                __m512h sum = _mm512_add_ph(_mm512_castsi512_ph(_mm512_maskz_loadu_epi16(tail, a + i)), _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(tail, b + i)));
                _mm512_mask_storeu_epi16(dst + i, tail, _mm512_castph_si512(sum));
            }
        }

        template<class A, class B> static SynetAdd16bUniform::UniformPtr GetAdd16bUniform(SimdTensorDataType dType)
        {
            switch (dType)
            {
            case SimdTensorData32f: return Add16bUniform<A, B, float>;
            case SimdTensorData16b: return Add16bUniform<A, B, uint16_t>;
            case SimdTensorData16f: return Add16bUniform<A, B, Float16>;
            default:
                return NULL;
            }
        }

        template<class A> static SynetAdd16bUniform::UniformPtr GetAdd16bUniform(SimdTensorDataType bType, SimdTensorDataType dType)
        {
            switch (bType)
            {
            case SimdTensorData32f: return GetAdd16bUniform<A, float>(dType);
            case SimdTensorData16b: return GetAdd16bUniform<A, uint16_t>(dType);
            case SimdTensorData16f: return GetAdd16bUniform<A, Float16>(dType);
            default:
                return NULL;
            }
        }

        static SynetAdd16bUniform::UniformPtr GetAdd16bUniform(SimdTensorDataType aType, SimdTensorDataType bType, SimdTensorDataType dType)
        {
            switch (aType)
            {
            case SimdTensorData32f: return GetAdd16bUniform<float>(bType, dType);
            case SimdTensorData16b: return GetAdd16bUniform<uint16_t>(bType, dType);
            case SimdTensorData16f: return GetAdd16bUniform<Float16>(bType, dType);
            default:
                return NULL;
            }
        }

        static bool SynetAdd16bUseFp16()
        {
#if defined(SIMD_AMX_EMULATE)
            return Base::CheckBit(7, 0, Cpuid::Edx, Cpuid::AVX512_FP16);
#else
            return true;
#endif
        }
#endif

        //-------------------------------------------------------------------------------------------------

        SynetAdd16bUniform::SynetAdd16bUniform(const Add16bParam& p)
            : Avx512bw::SynetAdd16bUniform(p)
        {
#if defined(__AVX512FP16__)
            if (SynetAdd16bUseFp16() && (p.aType == SimdTensorData16f || p.bType == SimdTensorData16f || p.dType == SimdTensorData16f))
                _uniform = GetAdd16bUniform(p.aType, p.bType, p.dType);
#endif
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetAdd16bInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const size_t* bShape, size_t bCount, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format)
        {
            Add16bParam param(aShape, aCount, aType, bShape, bCount, bType, dstType, format);
            if (!param.Valid())
                return NULL;
            if (Base::SynetAdd16bUniform::Preferable(param))
                return new SynetAdd16bUniform(param);
            return NULL;
        }
    }
#endif
}
