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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE bool InitFillBgrIndex(uint8_t index[3][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t k = 0; k < 3; ++k)
            {
                size_t offset = k * A;
                for (size_t i = 0; i < A; ++i)
                    index[k][i] = (uint8_t)((offset + i) % 3);
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t FILL_BGR_INDEX[3][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool FILL_BGR_INDEX_INITED = InitFillBgrIndex(FILL_BGR_INDEX);

        SIMD_INLINE svuint8_t FillBgr(const svuint8_t& index, const svuint8_t& blue, const svuint8_t& green, const svuint8_t& red, const svbool_t& mask)
        {
            return svsel_u8(svcmpeq_n_u8(mask, index, 0), blue, svsel_u8(svcmpeq_n_u8(mask, index, 1), green, red));
        }

        SIMD_INLINE void FillBgr(uint8_t* dst, size_t A, const svuint8_t& bgr0, const svuint8_t& bgr1, const svuint8_t& bgr2,
            const svbool_t& mask0, const svbool_t& mask1, const svbool_t& mask2)
        {
            svst1_u8(mask0, dst + 0 * A, bgr0);
            svst1_u8(mask1, dst + 1 * A, bgr1);
            svst1_u8(mask2, dst + 2 * A, bgr2);
        }

        void FillBgr(uint8_t* dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            size_t widthA = AlignLo(width, A);
            size_t tail = (width - widthA) * 3;
            const svbool_t body = svptrue_b8();
            const svuint8_t _blue = svdup_n_u8(blue);
            const svuint8_t _green = svdup_n_u8(green);
            const svuint8_t _red = svdup_n_u8(red);
            const svuint8_t index0 = svld1_u8(body, FILL_BGR_INDEX[0]);
            const svuint8_t index1 = svld1_u8(body, FILL_BGR_INDEX[1]);
            const svuint8_t index2 = svld1_u8(body, FILL_BGR_INDEX[2]);
            const svuint8_t bgr0 = FillBgr(index0, _blue, _green, _red, body);
            const svuint8_t bgr1 = FillBgr(index1, _blue, _green, _red, body);
            const svuint8_t bgr2 = FillBgr(index2, _blue, _green, _red, body);
            const svbool_t tail0 = svwhilelt_b8(size_t(0 * A), tail);
            const svbool_t tail1 = svwhilelt_b8(size_t(1 * A), tail);
            const svbool_t tail2 = svwhilelt_b8(size_t(2 * A), tail);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    FillBgr(dst + offset, A, bgr0, bgr1, bgr2, body, body, body);
                if (tail)
                    FillBgr(dst + offset, A, bgr0, bgr1, bgr2, tail0, tail1, tail2);
                dst += stride;
            }
        }

        void FillBgra(uint8_t* dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
        {
#ifdef SIMD_BIG_ENDIAN
            uint32_t bgra = uint32_t(alpha) | (uint32_t(red) << 8) | (uint32_t(green) << 16) | (uint32_t(blue) << 24);
#else
            uint32_t bgra = uint32_t(blue) | (uint32_t(green) << 8) | (uint32_t(red) << 16) | (uint32_t(alpha) << 24);
#endif
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            const svuint32_t _bgra = svdup_n_u32(bgra);
            for (size_t row = 0; row < height; ++row)
            {
                uint32_t* d = (uint32_t*)dst;
                size_t col = 0;
                for (; col + QF <= width; col += QF)
                {
                    svst1_u32(body, d + col + 0 * F, _bgra);
                    svst1_u32(body, d + col + 1 * F, _bgra);
                    svst1_u32(body, d + col + 2 * F, _bgra);
                    svst1_u32(body, d + col + 3 * F, _bgra);
                }
                for (; col + F <= width; col += F)
                    svst1_u32(body, d + col, _bgra);
                if (col < width)
                    svst1_u32(svwhilelt_b32(col, width), d + col, _bgra);
                dst += stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void Fill32f(float* dst, size_t size, const float* value)
        {
            if (value == 0 || value[0] == 0)
                memset(dst, 0, size * sizeof(float));
            else
            {
                size_t F = svcntw(), QF = 4 * F, i = 0;
                const svbool_t body = svptrue_b32();
                const svfloat32_t _value = svdup_n_f32(value[0]);

                for (; i + QF <= size; i += QF)
                {
                    svst1_f32(body, dst + i + 0 * F, _value);
                    svst1_f32(body, dst + i + 1 * F, _value);
                    svst1_f32(body, dst + i + 2 * F, _value);
                    svst1_f32(body, dst + i + 3 * F, _value);
                }
                for (; i + F <= size; i += F)
                    svst1_f32(body, dst + i, _value);
                if (i < size)
                    svst1_f32(svwhilelt_b32(i, size), dst + i, _value);
            }
        }
    }
#endif
}
