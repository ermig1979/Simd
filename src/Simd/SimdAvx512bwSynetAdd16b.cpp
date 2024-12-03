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
#include "Simd/SimdSynetAdd16b.h"
#include "Simd/SimdSynetAdd16bCommon.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512bw
    {
        template <typename A, typename B, typename D> void Add16bF(const A* a, const B* b, D* dst, __mmask16 tail);

        template <> SIMD_INLINE void Add16bF(const float* a, const float* b, float* dst, __mmask16 tail)
        {
            _mm512_mask_storeu_ps(dst, tail, _mm512_add_ps(_mm512_maskz_loadu_ps(tail, a), _mm512_maskz_loadu_ps(tail, b)));
        }

        template <> SIMD_INLINE void Add16bF(const float* a, const uint16_t* b, float* dst, __mmask16 tail)
        {
            __m512 _b = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail, b)));
            _mm512_mask_storeu_ps(dst, tail, _mm512_add_ps(_mm512_maskz_loadu_ps(tail, a), _b));
        }

        template <> SIMD_INLINE void Add16bF(const uint16_t* a, const float* b, float* dst, __mmask16 tail)
        {
            __m512 _a = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail, a)));
            _mm512_mask_storeu_ps(dst, tail, _mm512_add_ps(_a, _mm512_maskz_loadu_ps(tail, b)));
        }

        template <> SIMD_INLINE void Add16bF(const uint16_t* a, const uint16_t* b, float* dst, __mmask16 tail)
        {
            __m512 _a = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail, a)));
            __m512 _b = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail, b)));
            _mm512_mask_storeu_ps(dst, tail, _mm512_add_ps(_a, _b));
        }

        static const __m512i K16_ADD16B_PERM = SIMD_MM512_SETR_EPI16(
            0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

        template <> SIMD_INLINE void Add16bF(const float* a, const float* b, uint16_t* dst, __mmask16 tail)
        {
            __m512i _dst = Float32ToBFloat16(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, a), _mm512_maskz_loadu_ps(tail, b)));
            _mm256_mask_storeu_epi16(dst, tail, _mm512_castsi512_si256(_mm512_permutexvar_epi16(K16_ADD16B_PERM, _dst)));
        }

        template <> SIMD_INLINE void Add16bF(const float* a, const uint16_t* b, uint16_t* dst, __mmask16 tail)
        {
            __m512 _b = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail, b)));
            __m512i _dst = Float32ToBFloat16(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, a), _b));
            _mm256_mask_storeu_epi16(dst, tail, _mm512_castsi512_si256(_mm512_permutexvar_epi16(K16_ADD16B_PERM, _dst)));
        }

        template <> SIMD_INLINE void Add16bF(const uint16_t* a, const float* b, uint16_t* dst, __mmask16 tail)
        {
            __m512 _a = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail, a)));
            __m512i _dst = Float32ToBFloat16(_mm512_add_ps(_a, _mm512_maskz_loadu_ps(tail, b)));
            _mm256_mask_storeu_epi16(dst, tail, _mm512_castsi512_si256(_mm512_permutexvar_epi16(K16_ADD16B_PERM, _dst)));
        }

        template <typename A, typename B, typename D> static void Add16bUniform(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
        {
            const A* a = (const A*)a8;
            const B* b = (const B*)b8;
            D* dst = (D*)dst8;
            size_t sizeF = AlignLo(size, F), i = 0;
            __mmask16 tail = TailMask16(size - sizeF);

            for (; i < sizeF; i += F)
                Add16bF(a + i, b + i, dst + i, __mmask16(-1));
            if (i < size)
                Add16bF(a + i, b + i, dst + i, tail);
        }

        SIMD_INLINE void Add16bDF(const uint16_t* a, const uint16_t* b, uint16_t* dst, __mmask32 tail)
        {
            __m512i _a = _mm512_maskz_loadu_epi16(tail, (__m512i*)a);
            __m512i _b = _mm512_maskz_loadu_epi16(tail, (__m512i*)b);
            __m512 even = _mm512_add_ps(BFloat16ToFloat32Even(_a), BFloat16ToFloat32Even(_b));
            __m512 odd = _mm512_add_ps(BFloat16ToFloat32Odd(_a), BFloat16ToFloat32Odd(_b));
            _mm512_mask_storeu_epi16((__m512i*)dst, tail, Float32ToBFloat16Interlived(even, odd));
        }

        template <> void Add16bUniform<uint16_t, uint16_t, uint16_t>(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
        {
            const uint16_t* a = (const uint16_t*)a8;
            const uint16_t* b = (const uint16_t*)b8;
            uint16_t* dst = (uint16_t*)dst8;
            size_t sizeDF = AlignLo(size, DF), i = 0;
            __mmask32 tail = TailMask32(size - sizeDF);

            for (; i < sizeDF; i += DF)
                Add16bDF(a + i, b + i, dst + i, __mmask32(-1));
            if(i < size)
                Add16bDF(a + i, b + i, dst + i, tail);
        }

        template<class A, class B> static SynetAdd16bUniform::UniformPtr GetAdd16bUniform(SimdTensorDataType dType)
        {
            switch (dType)
            {
            case SimdTensorData32f: return Add16bUniform<A, B, float>;
            case SimdTensorData16b: return Add16bUniform<A, B, uint16_t>;
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
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetAdd16bUniform::SynetAdd16bUniform(const Add16bParam& p)
            : Avx2::SynetAdd16bUniform(p)
        {
             _uniform = GetAdd16bUniform(p.aType, p.bType, p.dType);
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
