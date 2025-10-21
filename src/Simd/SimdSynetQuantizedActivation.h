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
#ifndef __SimdSynetQuantizedActivation_h__
#define __SimdSynetQuantizedActivation_h__

#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetActivation.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void QuantizedPrelu(const uint8_t& src, int sBias, float sNorm, float slope, uint8_t& dst, float dNorm, int dZero)
        {
            float _src = DequantizeLinear(src, sBias, sNorm);
            float _dst = Simd::Max(0.0f, _src) + slope * Simd::Min(_src, 0.0f);
            dst = (uint8_t)QuantizeLinear(_dst, dNorm, dZero, 0, 255);
        }

        //--------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> SIMD_INLINE int QuantizeActivateSum(int sum, int sBias, float sNorm,
            int iZero, float iScale, const float* params, size_t offset, float dNorm, int dZero, int min, int max)
        {
            int iInt = QuantizeSumLinear(sum, sBias, sNorm, iZero, min, max);
            float fInt = float(iInt - iZero) * iScale;
            float fDst = Activate<type>(fInt, params, offset);
            return QuantizeLinear(fDst, dNorm, dZero, min, max);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE int QuantizeActivateSum(int sum, int sBias, float sNorm,
            float iScale, const float* params, size_t offset, float dNorm, int dZero, int min, int max)
        {
            int iInt = NearByInt(float(sum + sBias) * sNorm);
            float fInt = float(iInt) * iScale;
            float fDst = Activate<type>(fInt, params, offset);
            return QuantizeLinear(fDst, dNorm, dZero, min, max);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE int QuantizeActivateSum(int sum, int sBias, float sNorm,
            int iLo, int iHi, float iScale, const float* params, size_t offset, float dNorm, int dZero, int min, int max)
        {
            int iInt = RestrictRange(NearByInt(float(sum + sBias) * sNorm), iLo, iHi);
            float fInt = float(iInt) * iScale;
            float fDst = Activate<type>(fInt, params, offset);
            return QuantizeLinear(fDst, dNorm, dZero, min, max);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <Term8iType term> struct QuntizedTerm8i
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero);
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail);

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
                const __m128i* bias, const __m128* norm, const __m128i& zero);
        };

        template <> struct QuntizedTerm8i<Term8iLast8u>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero)
            {
                __m128i i32 = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(sum, bias[index])), norm[index])), zero);
                ((int32_t*)dst)[index] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO));
            }

            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail)
            {
                uint8_t tmp[F];
                QuntizedTerm8i::Save<index>(tmp - index * F, buf, sum, bias, norm, zero);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
                const __m128i* bias, const __m128* norm, const __m128i& zero)
            {
                __m128i d0 = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(sum0, bias[0])), norm[0])), zero);
                __m128i d1 = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(sum1, bias[1])), norm[1])), zero);
                _mm_storel_epi64((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), K_ZERO));
            }
        };

        template <> struct QuntizedTerm8i<Term8iInterim>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero)
            {
                _mm_storeu_si128((__m128i*)buf + index, sum);
            }

            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail)
            {
                int32_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
                const __m128i* bias, const __m128* norm, const __m128i& zero)
            {
                _mm_storeu_si128((__m128i*)buf + 0, sum0);
                _mm_storeu_si128((__m128i*)buf + 1, sum1);
            }
        };

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum,
            const __m128i* bias, const __m128* norm, const __m128i& zero)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero);
        }

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum,
            const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero, tail);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
            const __m128i* bias, const __m128* norm, const __m128i& zero)
        {
            QuntizedTerm8i<term>::Save(dst, buf, sum0, sum1, bias, norm, zero);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
            const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum0, bias, norm, zero);
            QuntizedTerm8i<term>::template Save<1>(dst, buf, sum1, bias, norm, zero, tail);
        }

        //--------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int index> SIMD_INLINE __m128i ToSave32i(__m128i sum, const __m128i* sBias, const __m128* sNorm, 
            const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero)
        {
            if (type == SimdConvolutionActivationIdentity)
            {
                return _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(sum, sBias[index])), sNorm[index])), dZero);
            }
            else
            {
                __m128i i32 = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(sum, sBias[index])), sNorm[index]));
                __m128 f32 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_min_epi32(_mm_max_epi32(iLo, i32), iHi)), iScale);
                return QuantizeLinear(Activate<type>(f32, params, index), dNorm, dZero);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int index> SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
            const __m128i* sBias, const __m128* sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero)
        {
            if (term == Term8iInterim)
            {
                _mm_storeu_si128((__m128i*)buf + index, sum);
            }
            else if(term == Term8iLast8u)
            {
                __m128i d0 = ToSave32i<type, index>(sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                ((int32_t*)dst)[index] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO));
            }
            else
            {
                assert(0);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
            const __m128i* sBias, const __m128* sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero, size_t tail)
        {
            if (term == Term8iInterim)
            {
                int32_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
            else if (term == Term8iLast8u)
            {
                uint8_t tmp[F];
                Save<term, type, index>(tmp - index * F, buf, sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];                
            }
            else
            {
                assert(0);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> static SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
            const __m128i* sBias, const __m128* sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero)
        {
            if (term == Term8iInterim)
            {
                _mm_storeu_si128((__m128i*)buf + 0, sum0);
                _mm_storeu_si128((__m128i*)buf + 1, sum1);
            }
            else if (term == Term8iLast8u)
            {
                __m128i d0 = ToSave32i<type, 0>(sum0, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                __m128i d1 = ToSave32i<type, 1>(sum1, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                _mm_storel_epi64((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), K_ZERO));                
            }
            else
            {
                assert(0);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
            const __m128i* sBias, const __m128* sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero, size_t tail)
        {
            Save<term, type, 0>(dst, buf, sum0, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
            Save<term, type, 1>(dst, buf, sum1, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum,
            const __m128i* sBias, const __m128* sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero)
        {
            Save<term, type, 0>(dst, buf, sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum,
            const __m128i* sBias, const __m128* sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero, size_t tail)
        {
            Save<term, type, 0>(dst, buf, sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <Term8iType term> struct QuntizedTerm8i
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero);
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail);

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
                const __m256i* bias, const __m256* norm, const __m256i& zero);
        };

        template <> struct QuntizedTerm8i<Term8iLast8u>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero)
            {
                __m256i i32 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum, bias[index])), norm[index])), zero);
                _mm_storel_epi64((__m128i*)(dst + index * F), _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO)));
            }

            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail)
            {
                uint8_t tmp[F];
                QuntizedTerm8i::Save<index>(tmp - index * F, buf, sum, bias, norm, zero);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
                const __m256i* bias, const __m256* norm, const __m256i& zero)
            {
                __m256i d0 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum0, bias[0])), norm[0])), zero);
                __m256i d1 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum1, bias[1])), norm[1])), zero);
                _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO)));
            }
        };

        template <> struct QuntizedTerm8i<Term8iInterim>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero)
            {
                _mm256_storeu_si256((__m256i*)buf + index, sum);
            }

            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail)
            {
                int32_t tmp[F];
                _mm256_storeu_si256((__m256i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
                const __m256i* bias, const __m256* norm, const __m256i& zero)
            {
                _mm256_storeu_si256((__m256i*)buf + 0, sum0);
                _mm256_storeu_si256((__m256i*)buf + 1, sum1);
            }
        };

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum,
            const __m256i* bias, const __m256* norm, const __m256i& zero)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero);
        }

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum,
            const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero, tail);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
            const __m256i* bias, const __m256* norm, const __m256i& zero)
        {
            QuntizedTerm8i<term>::Save(dst, buf, sum0, sum1, bias, norm, zero);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
            const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum0, bias, norm, zero);
            QuntizedTerm8i<term>::template Save<1>(dst, buf, sum1, bias, norm, zero, tail);
        }

        //--------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int index> SIMD_INLINE __m256i ToSave32i(__m256i sum,
            const __m256i* sBias, const __m256* sNorm, const __m256i& iLo, const __m256i& iHi, const __m256& iScale, const __m256* params, const __m256& dNorm, const __m256i& dZero)
        {
            if (type == SimdConvolutionActivationIdentity)
            {
                return _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum, sBias[index])), sNorm[index])), dZero);
            }
            else
            {
                __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum, sBias[index])), sNorm[index]));
                __m256 f32 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_min_epi32(_mm256_max_epi32(iLo, i32), iHi)), iScale);
                return QuantizeLinear(Activate<type>(f32, params, index), dNorm, dZero);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int index> SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
            const __m256i* sBias, const __m256* sNorm, const __m256i& iLo, const __m256i& iHi, const __m256& iScale, const __m256* params, const __m256& dNorm, const __m256i& dZero)
        {
            if (term == Term8iInterim)
            {
                _mm256_storeu_si256((__m256i*)buf + index, sum);
            }
            else if (term == Term8iLast8u)
            {
                __m256i d0 = ToSave32i<type, index>(sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                _mm_storel_epi64((__m128i*)(dst + index * F), _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
            }
            else
            {
                assert(0);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
            const __m256i* sBias, const __m256* sNorm, const __m256i& iLo, const __m256i& iHi, const __m256& iScale, const __m256* params, const __m256& dNorm, const __m256i& dZero, size_t tail)
        {
            if (term == Term8iInterim)
            {
                int32_t tmp[F];
                _mm256_storeu_si256((__m256i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
            else if (term == Term8iLast8u)
            {
                uint8_t tmp[F];
                Save<term, type, index>(tmp - index * F, buf, sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }
            else
            {
                assert(0);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> static SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
            const __m256i* sBias, const __m256* sNorm, const __m256i& iLo, const __m256i& iHi, const __m256& iScale, const __m256* params, const __m256& dNorm, const __m256i& dZero)
        {
            if (term == Term8iInterim)
            {
                _mm256_storeu_si256((__m256i*)buf + 0, sum0);
                _mm256_storeu_si256((__m256i*)buf + 1, sum1);
            }
            else if (term == Term8iLast8u)
            {
                __m256i d0 = ToSave32i<type, 0>(sum0, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                __m256i d1 = ToSave32i<type, 1>(sum1, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO)));
            }
            else
            {
                assert(0);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
            const __m256i* sBias, const __m256* sNorm, const __m256i& iLo, const __m256i& iHi, const __m256& iScale, const __m256* params, const __m256& dNorm, const __m256i& dZero, size_t tail)
        {
            Save<term, type, 0>(dst, buf, sum0, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
            Save<term, type, 1>(dst, buf, sum1, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum,
            const __m256i* sBias, const __m256* sNorm, const __m256i& iLo, const __m256i& iHi, const __m256& iScale, const __m256* params, const __m256& dNorm, const __m256i& dZero)
        {
            Save<term, type, 0>(dst, buf, sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum,
            const __m256i* sBias, const __m256* sNorm, const __m256i& iLo, const __m256i& iHi, const __m256& iScale, const __m256* params, const __m256& dNorm, const __m256i& dZero, size_t tail)
        {
            Save<term, type, 0>(dst, buf, sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <Term8iType term> struct QuntizedTerm8i
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1);
        };

        template <> struct QuntizedTerm8i<Term8iLast8u>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
            {
                __m512i i32 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(sum, bias[index])), norm[index])), zero);
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO)));
            }
        };

        template <> struct QuntizedTerm8i<Term8iInterim>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
            {
                _mm512_mask_storeu_epi32(buf + index * F, tail, sum);
            }
        };

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m512i sum,
            const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero, tail);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m512i sum0, __m512i sum1,
            const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum0, bias, norm, zero);
            QuntizedTerm8i<term>::template Save<1>(dst, buf, sum1, bias, norm, zero, tail);
        }

        //--------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int index> SIMD_INLINE __m512i ToSave32i(__m512i sum, const __m512i* sBias, const __m512* sNorm, 
            const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero)
        {
            if (type == SimdConvolutionActivationIdentity)
            {
                return _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(sum, sBias[index])), sNorm[index])), dZero);
            }
            else
            {
                __m512i i32 = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(sum, sBias[index])), sNorm[index]));
                __m512 f32 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_min_epi32(_mm512_max_epi32(iLo, i32), iHi)), iScale);
                return QuantizeLinear(Activate<type>(f32, params, index), dNorm, dZero);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int index> SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum, const __m512i* sBias, 
            const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            if (term == Term8iInterim)
            {
                _mm512_mask_storeu_epi32(buf + index * F, tail, sum);
            }
            else if (term == Term8iLast8u)
            {
                __m512i d0 = ToSave32i<type, index>(sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
            }
            else
            {
                assert(0);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> static SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m512i sum0, __m512i sum1,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            Save<term, type, 0>(dst, buf, sum0, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
            Save<term, type, 1>(dst, buf, sum1, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m512i sum,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            Save<term, type, 0>(dst, buf, sum, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))    
    namespace AmxBf16
    {
        template<Term8iType term, SimdConvolutionActivationType type, int index> SIMD_INLINE void Apply(uint8_t* dst, int32_t* buf, const __m512i* sBias, 
            const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            if (term == Term8iLast8u)
            {
                __m512i d0 = Avx512bw::ToSave32i<type, index>(_mm512_loadu_si512(buf + index * F), sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
                _mm_prefetch((const char*)(dst + index * A), _MM_HINT_NTA);
                _mm_prefetch((const char*)(buf + index * F), _MM_HINT_NTA);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply1(uint8_t* dst, int32_t* buf, const __m512i* sBias, const __m512* sNorm, 
            const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            Apply<term, type, 0>(dst, buf, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply1x8(uint8_t* ptr, int dP, int32_t* buf, int dB, const __m512i* sBias, const __m512* sNorm, 
            const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            Apply1<term, type>(ptr + 0 * dP, buf + 0 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply1<term, type>(ptr + 1 * dP, buf + 1 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply1<term, type>(ptr + 2 * dP, buf + 2 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply1<term, type>(ptr + 3 * dP, buf + 3 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply1<term, type>(ptr + 4 * dP, buf + 4 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply1<term, type>(ptr + 5 * dP, buf + 5 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply1<term, type>(ptr + 6 * dP, buf + 6 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply1<term, type>(ptr + 7 * dP, buf + 7 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply2(uint8_t* dst, int32_t* buf, const __m512i* sBias, const __m512* sNorm, 
            const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            Apply<term, type, 0>(dst, buf, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
            Apply<term, type, 1>(dst, buf, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply2x8(uint8_t* ptr, int dP, int32_t* buf, int dB, const __m512i* sBias, const __m512* sNorm, 
            const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            Apply2<term, type>(ptr + 0 * dP, buf + 0 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply2<term, type>(ptr + 1 * dP, buf + 1 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply2<term, type>(ptr + 2 * dP, buf + 2 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply2<term, type>(ptr + 3 * dP, buf + 3 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply2<term, type>(ptr + 4 * dP, buf + 4 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply2<term, type>(ptr + 5 * dP, buf + 5 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply2<term, type>(ptr + 6 * dP, buf + 6 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply2<term, type>(ptr + 7 * dP, buf + 7 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply8u2(uint8_t* dst, int32_t* buf, const __m512i* sBias, const __m512* sNorm, 
            const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask32 tail = -1)
        {
            __m512i d0 = Avx512bw::ToSave32i<type, 0>(_mm512_loadu_si512(buf + 0), sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
            __m512i d1 = Avx512bw::ToSave32i<type, 1>(_mm512_loadu_si512(buf + F), sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
            _mm256_mask_storeu_epi8(dst, tail, _mm512_castsi512_si256(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO)));
            _mm_prefetch((const char*)(dst), _MM_HINT_NTA);
            _mm_prefetch((const char*)(buf + 0), _MM_HINT_NTA);
            _mm_prefetch((const char*)(buf + F), _MM_HINT_NTA);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply8u2x8(uint8_t* ptr, int dP, int32_t* buf, int dB, const __m512i* sBias, const __m512* sNorm, 
            const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, __mmask32 tail = -1)
        {
            Apply8u2<type>(ptr + 0 * dP, buf + 0 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply8u2<type>(ptr + 1 * dP, buf + 1 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply8u2<type>(ptr + 2 * dP, buf + 2 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply8u2<type>(ptr + 3 * dP, buf + 3 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply8u2<type>(ptr + 4 * dP, buf + 4 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply8u2<type>(ptr + 5 * dP, buf + 5 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply8u2<type>(ptr + 6 * dP, buf + 6 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
            Apply8u2<type>(ptr + 7 * dP, buf + 7 * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tail);
        }
    }
#endif
}

#endif
