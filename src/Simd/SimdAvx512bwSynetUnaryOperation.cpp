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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdErf.h"
#include "Simd/SimdInterleave.h"
#include "Simd/SimdDeinterleave.h"
#include "Simd/SimdTrigonometric.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)     
    namespace Avx512bw
    {
        template<SimdSynetUnaryOperation32fType type> __m512 SynetUnaryOperation32f(__m512 value);

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(__m512 value)
        {
            return AndNot(_mm512_set1_ps(-0.0f), value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fCeil>(__m512 value)
        {
            return _mm512_ceil_ps(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fCos>(__m512 value)
        {
            return Cos(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fErf>(__m512 value)
        {
            return Erf(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(__m512 value)
        {
            return Exponent(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fFloor>(__m512 value)
        {
            return _mm512_floor_ps(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(__m512 value)
        {
            return Logarithm(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(__m512 value)
        {
            return _mm512_sub_ps(_mm512_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fNot>(__m512 value)
        {
            return Not(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fRcp>(__m512 value)
        {
            return Rcp<false>(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fRound>(__m512 value)
        {
            return _mm512_cvtepi32_ps(_mm512_cvtps_epi32(value));
        }
        
        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(__m512 value)
        {
            return _mm512_rsqrt14_ps(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fSign>(__m512 value)
        {
            static const __m512 _0 = _mm512_set1_ps(0.0f), _p = _mm512_set1_ps(1.0f), _n = _mm512_set1_ps(-1.0f);
            return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(value, _0, _CMP_LT_OQ), _mm512_mask_blend_ps(_mm512_cmp_ps_mask(value, _0, _CMP_EQ_OQ), _p, _0), _n);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fSin>(__m512 value)
        {
            return Sin(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(__m512 value)
        {
            return _mm512_sqrt_ps(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(__m512 value)
        {
            return Tanh(value);
        }

        template<> SIMD_INLINE __m512 SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(__m512 value)
        {
            return _mm512_setzero_ps();
        }

        template<SimdSynetUnaryOperation32fType type> void SynetUnaryOperation32f(const float* src, size_t size, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                _mm512_storeu_ps(dst + i + 0 * F, SynetUnaryOperation32f<type>(_mm512_loadu_ps(src + i + 0 * F)));
                _mm512_storeu_ps(dst + i + 1 * F, SynetUnaryOperation32f<type>(_mm512_loadu_ps(src + i + 1 * F)));
                _mm512_storeu_ps(dst + i + 2 * F, SynetUnaryOperation32f<type>(_mm512_loadu_ps(src + i + 2 * F)));
                _mm512_storeu_ps(dst + i + 3 * F, SynetUnaryOperation32f<type>(_mm512_loadu_ps(src + i + 3 * F)));
            }
            for (; i < sizeF; i += F)
                _mm512_storeu_ps(dst + i, SynetUnaryOperation32f<type>(_mm512_loadu_ps(src + i)));
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - sizeF);
                _mm512_mask_storeu_ps(dst + i, tail, SynetUnaryOperation32f<type>(_mm512_maskz_loadu_ps(tail, src + i)));
            }
        }

        void SynetUnaryOperation32f(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            switch (type)
            {
            case SimdSynetUnaryOperation32fAbs: SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(src, size, dst); break;
            case SimdSynetUnaryOperation32fCeil: SynetUnaryOperation32f<SimdSynetUnaryOperation32fCeil>(src, size, dst); break;
            case SimdSynetUnaryOperation32fCos: SynetUnaryOperation32f<SimdSynetUnaryOperation32fCos>(src, size, dst); break;
            case SimdSynetUnaryOperation32fExp: SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(src, size, dst); break;
            case SimdSynetUnaryOperation32fErf: SynetUnaryOperation32f<SimdSynetUnaryOperation32fErf>(src, size, dst); break;
            case SimdSynetUnaryOperation32fFloor: SynetUnaryOperation32f<SimdSynetUnaryOperation32fFloor>(src, size, dst); break;
            case SimdSynetUnaryOperation32fLog: SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNeg: SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNot: SynetUnaryOperation32f<SimdSynetUnaryOperation32fNot>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRcp: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRcp>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRound: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRound>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRsqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSign: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSign>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSin: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSin>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(src, size, dst); break;
            case SimdSynetUnaryOperation32fTanh: SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(src, size, dst); break;
            case SimdSynetUnaryOperation32fZero: SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(src, size, dst); break;
            default:
                Avx2::SynetUnaryOperation32f(src, size, type, dst);
            }
        }
    }
#endif
}
