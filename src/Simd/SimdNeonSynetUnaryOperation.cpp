/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdErf.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Neon
    {
        template<SimdSynetUnaryOperation32fType type> float32x4_t SynetUnaryOperation32f(float32x4_t value);

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(float32x4_t value)
        {
            return vabsq_f32(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fErf>(float32x4_t value)
        {
            return Erf<1>(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(float32x4_t value)
        {
            return Exponent(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(float32x4_t value)
        {
            return Logarithm(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(float32x4_t value)
        {
            return vnegq_f32(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fNot>(float32x4_t value)
        {
            return Not(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fRcp>(float32x4_t value)
        {
            return Reciprocal<1>(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(float32x4_t value)
        {
            return ReciprocalSqrt<1>(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(float32x4_t value)
        {
            return Sqrt<1>(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(float32x4_t value)
        {
            return Tanh<1>(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(float32x4_t value)
        {
            return vdupq_n_f32(0.0f);
        }

        template<SimdSynetUnaryOperation32fType type, bool align> void SynetUnaryOperation32f(const float* src, size_t size, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Neon::Store<align>(dst + i + 0 * F, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i + 0 * F)));
                Neon::Store<align>(dst + i + 1 * F, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i + 1 * F)));
                Neon::Store<align>(dst + i + 2 * F, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i + 2 * F)));
                Neon::Store<align>(dst + i + 3 * F, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i + 3 * F)));
            }
            for (; i < sizeF; i += F)
                Neon::Store<align>(dst + i, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i)));
            for (; i < size; ++i)
                dst[i] = Base::SynetUnaryOperation32f<type>(src[i]);
        }

        template<bool align> void SynetUnaryOperation32f(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            switch (type)
            {
            case SimdSynetUnaryOperation32fAbs: SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fErf: SynetUnaryOperation32f<SimdSynetUnaryOperation32fErf, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fExp: SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fLog: SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNeg: SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNot: SynetUnaryOperation32f<SimdSynetUnaryOperation32fNot, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRcp: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRcp, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRsqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSqrt: SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fTanh: SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fZero: SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero, align>(src, size, dst); break;
            default:
                assert(0);
            }
        }

        void SynetUnaryOperation32f(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetUnaryOperation32f<true>(src, size, type, dst);
            else
                SynetUnaryOperation32f<false>(src, size, type, dst);
        }
    }
#endif// SIMD_NEON_ENABLE
}
