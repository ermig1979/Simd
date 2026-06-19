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
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint32_t Float32ToBFloat16(svfloat32_t value, const svbool_t& mask)
        {
            svuint32_t bits = svreinterpret_u32_f32(value);
            svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
            return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
        }

        SIMD_INLINE void Float32ToBFloat16(const float* src, const svbool_t& lo, const svbool_t& hi, const svbool_t& store, uint16_t* dst)
        {
            size_t F = svlen(svfloat32_t());
            svuint16_t _lo = svreinterpret_u16_u32(Float32ToBFloat16(svld1_f32(lo, src + 0), lo));
            svuint16_t _hi = svreinterpret_u16_u32(Float32ToBFloat16(svld1_f32(hi, src + F), hi));
            svst1_u16(store, dst, svuzp1_u16(_lo, _hi));
        }

        void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst)
        {
            size_t A = svlen(svuint16_t()), F = svlen(svfloat32_t()), sizeA = AlignLo(size, A), i = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            for (; i < sizeA; i += A)
                Float32ToBFloat16(src + i, body32, body32, body16, dst + i);
            if (i < size)
            {
                size_t tail = size - i;
                Float32ToBFloat16(src + i, svwhilelt_b32(size_t(0), Simd::Min(tail, F)),
                    svwhilelt_b32(F, tail), svwhilelt_b16(size_t(0), tail), dst + i);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void BFloat16ToFloat32(const uint16_t* src, const svbool_t& load, const svbool_t& lo, const svbool_t& hi, float* dst)
        {
            size_t A = svlen(svuint16_t());
            svuint16_t zero = svdup_n_u16(0);
            svuint16_t _src = svld1_u16(load, src);
            svst1_u16(lo, (uint16_t*)dst + 0 * A, svzip1_u16(zero, _src));
            svst1_u16(hi, (uint16_t*)dst + 1 * A, svzip2_u16(zero, _src));
        }

        void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst)
        {
            size_t A = svlen(svuint16_t()), sizeA = AlignLo(size, A);
            const svbool_t body = svptrue_b16();
            size_t i = 0;
            for (; i < sizeA; i += A)
                BFloat16ToFloat32(src + i, body, body, body, dst + i);
            if (i < size)
            {
                size_t tail = size - i, half = 2 * tail;
                BFloat16ToFloat32(src + i, svwhilelt_b16(size_t(0), tail),
                    svwhilelt_b16(size_t(0), Simd::Min(half, A)), svwhilelt_b16(A, half), dst + i);
            }
        }
    }
#endif
}
