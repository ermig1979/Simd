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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template<bool align> SIMD_INLINE void SynetElu32f(const float * src, const Neon::Exp & exp, float32x4_t alpha, float * dst, size_t offset)
        {
            Store<align>(dst + offset, exp.Elu(Load<align>(src + offset), alpha));
        }

        template<bool align> void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            float32x4_t _alpha = vdupq_n_f32(alpha[0]);
            Neon::Exp exp;
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetElu32f<align>(src, exp, _alpha, dst, i + 0 * F);
                SynetElu32f<align>(src, exp, _alpha, dst, i + 1 * F);
                SynetElu32f<align>(src, exp, _alpha, dst, i + 2 * F);
                SynetElu32f<align>(src, exp, _alpha, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetElu32f<align>(src, exp, _alpha, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetElu32f(src[i], alpha[0]);
        }

        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetElu32f<true>(src, size, alpha, dst);
            else
                SynetElu32f<false>(src, size, alpha, dst);
        }

        //-------------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetHswish32f(const float * src, float32x4_t shift, float32x4_t scale, float * dst, size_t offset)
        {
            float32x4_t _src = Load<align>(src + offset);
            float32x4_t _dst = SynetHswish32f(_src, shift, scale);
            Store<align>(dst + offset, _dst);
        }

        template<bool align> void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            float32x4_t _shift = vdupq_n_f32(shift[0]);
            float32x4_t _scale = vdupq_n_f32(scale[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 0 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 1 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 2 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHswish32f<align>(src, _shift, _scale, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetHswish32f(src[i], shift[0], scale[0]);
        }

        void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetHswish32f<true>(src, size, shift, scale, dst);
            else
                SynetHswish32f<false>(src, size, shift, scale, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            assert(lower[0] <= upper[0]);
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float min = *lower;
            float max = *upper;
            float32x4_t _min = vdupq_n_f32(min);
            float32x4_t _max = vdupq_n_f32(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, vminq_f32(vmaxq_f32(_min, Load<align>(src + i + 0 * F)), _max));
                Store<align>(dst + i + 1 * F, vminq_f32(vmaxq_f32(_min, Load<align>(src + i + 1 * F)), _max));
                Store<align>(dst + i + 2 * F, vminq_f32(vmaxq_f32(_min, Load<align>(src + i + 2 * F)), _max));
                Store<align>(dst + i + 3 * F, vminq_f32(vmaxq_f32(_min, Load<align>(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, vminq_f32(vmaxq_f32(_min, Load<align>(src + i)), _max));
            for (; i < size; ++i)
                dst[i] = Simd::RestrictRange(src[i], min, max);
        }

        void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetRestrictRange32f<true>(src, size, lower, upper, dst);
            else
                SynetRestrictRange32f<false>(src, size, lower, upper, dst);
        }
    }
#endif// SIMD_NEON_ENABLE
}
