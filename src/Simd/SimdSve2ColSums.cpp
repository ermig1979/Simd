/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2018-2018 Radchenko Andrey.
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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t) * width + sizeof(uint32_t) * width);
                    sums16 = (uint16_t*)_p;
                    sums32 = (uint32_t*)(sums16 + width);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t* sums16;
                uint32_t* sums32;
            private:
                void* _p;
            };
        }

        SIMD_INLINE void GetAbsDxColSums(const uint8_t* src, const svbool_t& mask8, const svbool_t& lo16, const svbool_t& hi16, size_t half, uint16_t* sums)
        {
            svuint8_t diff = svabd_u8_x(mask8, svld1_u8(mask8, src), svld1_u8(mask8, src + 1));
            svst1_u16(lo16, sums, svadd_u16_x(lo16, svld1_u16(lo16, sums), svunpklo_u16(diff)));
            svst1_u16(hi16, sums + half, svadd_u16_x(hi16, svld1_u16(hi16, sums + half), svunpkhi_u16(diff)));
        }

        SIMD_INLINE void AddSums16To32(const uint16_t* src, uint32_t* dst, size_t col, size_t width, size_t half)
        {
            svbool_t mask16 = svwhilelt_b16(col, width);
            svbool_t lo32 = svwhilelt_b32(col, width);
            svbool_t hi32 = svwhilelt_b32(col + half, width);
            svuint16_t sums16 = svld1_u16(mask16, src + col);
            svst1_u32(lo32, dst + col, svadd_u32_x(lo32, svld1_u32(lo32, dst + col), svunpklo_u32(sums16)));
            svst1_u32(hi32, dst + col + half, svadd_u32_x(hi32, svld1_u32(hi32, dst + col + half), svunpkhi_u32(sums16)));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void AddColSum16(const svuint8_t& src, const svbool_t& lo16, const svbool_t& hi16, svuint16_t& lo, svuint16_t& hi)
        {
            lo = svadd_u16_x(lo16, lo, svunpklo_u16(src));
            hi = svadd_u16_x(hi16, hi, svunpkhi_u16(src));
        }

        SIMD_INLINE void GetColSum16x1(const uint8_t* src, const svbool_t& mask8, const svbool_t& lo16, const svbool_t& hi16, size_t half, uint16_t* dst)
        {
            svuint16_t lo = svld1_u16(lo16, dst);
            svuint16_t hi = svld1_u16(hi16, dst + half);
            AddColSum16(svld1_u8(mask8, src), lo16, hi16, lo, hi);
            svst1_u16(lo16, dst, lo);
            svst1_u16(hi16, dst + half, hi);
        }

        SIMD_INLINE void GetColSum16x4(const uint8_t* src, size_t stride, const svbool_t& mask8, const svbool_t& lo16, const svbool_t& hi16, size_t half, uint16_t* dst)
        {
            svuint16_t lo = svld1_u16(lo16, dst);
            svuint16_t hi = svld1_u16(hi16, dst + half);
            AddColSum16(svld1_u8(mask8, src + 0 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 1 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 2 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 3 * stride), lo16, hi16, lo, hi);
            svst1_u16(lo16, dst, lo);
            svst1_u16(hi16, dst + half, hi);
        }

        SIMD_INLINE void GetColSum16x8(const uint8_t* src, size_t stride, const svbool_t& mask8, const svbool_t& lo16, const svbool_t& hi16, size_t half, uint16_t* dst)
        {
            svuint16_t lo = svld1_u16(lo16, dst);
            svuint16_t hi = svld1_u16(hi16, dst + half);
            AddColSum16(svld1_u8(mask8, src + 0 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 1 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 2 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 3 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 4 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 5 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 6 * stride), lo16, hi16, lo, hi);
            AddColSum16(svld1_u8(mask8, src + 7 * stride), lo16, hi16, lo, hi);
            svst1_u16(lo16, dst, lo);
            svst1_u16(hi16, dst + half, hi);
        }

        void GetColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            const size_t A = svcntb(), HA = svcnth(), F = svcntw();
            const size_t alignedLoWidth = AlignLo(width, A);
            const size_t alignedHiWidth = AlignHi(width, A);
            const size_t stepSize = 256;
            const size_t stepCount = DivHi(height, stepSize);
            const svbool_t body8 = svptrue_b8();
            const svbool_t body16 = svptrue_b16();

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t) * alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                const size_t rowStart = step * stepSize;
                const size_t rowEnd = Min(rowStart + stepSize, height);
                const size_t rowEnd4 = rowStart + AlignLo(rowEnd - rowStart, 4);
                const size_t rowEnd8 = rowStart + AlignLo(rowEnd - rowStart, 8);

                memset(buffer.sums16, 0, sizeof(uint16_t) * alignedHiWidth);
                const uint8_t* rowSrc = src + rowStart * stride;
                size_t row = rowStart;
                for (; row < rowEnd8; row += 8)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16x8(rowSrc + col, stride, body8, body16, body16, HA, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16x8(rowSrc + col, stride, svwhilelt_b8(col, width), svwhilelt_b16(col, width), svwhilelt_b16(col + HA, width), HA, buffer.sums16 + col);
                    rowSrc += 8 * stride;
                }
                for (; row < rowEnd4; row += 4)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16x4(rowSrc + col, stride, body8, body16, body16, HA, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16x4(rowSrc + col, stride, svwhilelt_b8(col, width), svwhilelt_b16(col, width), svwhilelt_b16(col + HA, width), HA, buffer.sums16 + col);
                    rowSrc += 4 * stride;
                }
                for (; row < rowEnd; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16x1(rowSrc + col, body8, body16, body16, HA, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16x1(rowSrc + col, svwhilelt_b8(col, width), svwhilelt_b16(col, width), svwhilelt_b16(col + HA, width), HA, buffer.sums16 + col);
                    rowSrc += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += HA)
                    AddSums16To32(buffer.sums16, buffer.sums32, col, alignedHiWidth, F);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t) * width);
        }

        void GetAbsDxColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (width < 2)
            {
                if (width)
                    sums[0] = 0;
                return;
            }

            width--;
            const size_t A = svlen(svuint8_t()), HA = svlen(svuint16_t());
            const size_t widthA = AlignLo(width, A);
            const size_t widthB = AlignHi(width, A);
            const svbool_t body8 = svptrue_b8();
            const svbool_t body16 = svptrue_b16();
            const size_t stepSize = SCHAR_MAX + 1;
            const size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Buffer buffer(widthB);
            memset(buffer.sums32, 0, sizeof(uint32_t) * widthB);
            for (size_t step = 0; step < stepCount; ++step)
            {
                const size_t rowStart = step * stepSize;
                const size_t rowEnd = Min(rowStart + stepSize, height);

                memset(buffer.sums16, 0, sizeof(uint16_t) * widthB);
                const uint8_t* rowSrc = src + rowStart * stride;
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    size_t col = 0;
                    for (; col < widthA; col += A)
                        GetAbsDxColSums(rowSrc + col, body8, body16, body16, HA, buffer.sums16 + col);
                    if (col < width)
                        GetAbsDxColSums(rowSrc + col, svwhilelt_b8(col, width), svwhilelt_b16(col, width), svwhilelt_b16(col + HA, width), HA, buffer.sums16 + col);
                    rowSrc += stride;
                }

                for (size_t col = 0; col < width; col += svcntw() * 2)
                    AddSums16To32(buffer.sums16, buffer.sums32, col, width, svcntw());
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t) * width);
            sums[width] = 0;
        }
    }
#endif// SIMD_SVE2_ENABLE
}
