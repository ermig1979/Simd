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

        SIMD_INLINE void AddColSum(const uint8_t* src, const svbool_t& mask, svuint16_t& even, svuint16_t& odd)
        {
            svuint8_t val = svld1_u8(mask, src);
            even = svaddwb_u16(even, val);
            odd = svaddwt_u16(odd, val);
        }

        SIMD_INLINE void AddColSum4(const uint8_t* src, size_t A, const svbool_t& mask,
            svuint16_t& e0, svuint16_t& o0, svuint16_t& e1, svuint16_t& o1,
            svuint16_t& e2, svuint16_t& o2, svuint16_t& e3, svuint16_t& o3)
        {
            AddColSum(src + 0 * A, mask, e0, o0);
            AddColSum(src + 1 * A, mask, e1, o1);
            AddColSum(src + 2 * A, mask, e2, o2);
            AddColSum(src + 3 * A, mask, e3, o3);
        }

        SIMD_INLINE void AddAbsDxColSum(const uint8_t* src, const svbool_t& mask, svuint16_t& even, svuint16_t& odd)
        {
            svuint8_t diff = svabd_u8_x(mask, svld1_u8(mask, src), svld1_u8(mask, src + 1));
            even = svaddwb_u16(even, diff);
            odd = svaddwt_u16(odd, diff);
        }

        SIMD_INLINE void AddAbsDxColSum4(const uint8_t* src, size_t A, const svbool_t& mask,
            svuint16_t& e0, svuint16_t& o0, svuint16_t& e1, svuint16_t& o1,
            svuint16_t& e2, svuint16_t& o2, svuint16_t& e3, svuint16_t& o3)
        {
            AddAbsDxColSum(src + 0 * A, mask, e0, o0);
            AddAbsDxColSum(src + 1 * A, mask, e1, o1);
            AddAbsDxColSum(src + 2 * A, mask, e2, o2);
            AddAbsDxColSum(src + 3 * A, mask, e3, o3);
        }

        SIMD_INLINE void LoadEvenOdd(const uint16_t* dst, size_t half, const svbool_t& mask16, svuint16_t& even, svuint16_t& odd)
        {
            even = svld1_u16(mask16, dst);
            odd = svld1_u16(mask16, dst + half);
        }

        SIMD_INLINE void StoreEvenOdd(uint16_t* dst, size_t half, const svbool_t& mask16, const svuint16_t& even, const svuint16_t& odd)
        {
            svst1_u16(mask16, dst, even);
            svst1_u16(mask16, dst + half, odd);
        }

        SIMD_INLINE void AddEvenOdd16To32(const uint16_t* src, uint32_t* dst, size_t half, size_t quarter)
        {
            const svbool_t mask16 = svptrue_b16();
            const svbool_t mask32 = svptrue_b32();
            svuint16_t even = svld1_u16(mask16, src);
            svuint16_t odd = svld1_u16(mask16, src + half);
            svuint16_t natLo = svzip1_u16(even, odd);
            svuint16_t natHi = svzip2_u16(even, odd);
            svst1_u32(mask32, dst + 0 * quarter, svadd_u32_x(mask32, svld1_u32(mask32, dst + 0 * quarter), svunpklo_u32(natLo)));
            svst1_u32(mask32, dst + 1 * quarter, svadd_u32_x(mask32, svld1_u32(mask32, dst + 1 * quarter), svunpkhi_u32(natLo)));
            svst1_u32(mask32, dst + 2 * quarter, svadd_u32_x(mask32, svld1_u32(mask32, dst + 2 * quarter), svunpklo_u32(natHi)));
            svst1_u32(mask32, dst + 3 * quarter, svadd_u32_x(mask32, svld1_u32(mask32, dst + 3 * quarter), svunpkhi_u32(natHi)));
        }

        SIMD_INLINE void GetColSum16x1(const uint8_t* src, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t even, odd;
            LoadEvenOdd(dst, half, mask16, even, odd);
            AddColSum(src, mask, even, odd);
            StoreEvenOdd(dst, half, mask16, even, odd);
        }

        SIMD_INLINE void GetColSum16x4(const uint8_t* src, size_t stride, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t even, odd;
            LoadEvenOdd(dst, half, mask16, even, odd);
            AddColSum(src + 0 * stride, mask, even, odd);
            AddColSum(src + 1 * stride, mask, even, odd);
            AddColSum(src + 2 * stride, mask, even, odd);
            AddColSum(src + 3 * stride, mask, even, odd);
            StoreEvenOdd(dst, half, mask16, even, odd);
        }

        SIMD_INLINE void GetColSum16x8(const uint8_t* src, size_t stride, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t even, odd;
            LoadEvenOdd(dst, half, mask16, even, odd);
            AddColSum(src + 0 * stride, mask, even, odd);
            AddColSum(src + 1 * stride, mask, even, odd);
            AddColSum(src + 2 * stride, mask, even, odd);
            AddColSum(src + 3 * stride, mask, even, odd);
            AddColSum(src + 4 * stride, mask, even, odd);
            AddColSum(src + 5 * stride, mask, even, odd);
            AddColSum(src + 6 * stride, mask, even, odd);
            AddColSum(src + 7 * stride, mask, even, odd);
            StoreEvenOdd(dst, half, mask16, even, odd);
        }

        SIMD_INLINE void GetColSum16x8x4(const uint8_t* src, size_t stride, size_t A, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t e0, o0, e1, o1, e2, o2, e3, o3;
            LoadEvenOdd(dst + 0 * A, half, mask16, e0, o0);
            LoadEvenOdd(dst + 1 * A, half, mask16, e1, o1);
            LoadEvenOdd(dst + 2 * A, half, mask16, e2, o2);
            LoadEvenOdd(dst + 3 * A, half, mask16, e3, o3);
            AddColSum4(src + 0 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 1 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 2 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 3 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 4 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 5 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 6 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 7 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            StoreEvenOdd(dst + 0 * A, half, mask16, e0, o0);
            StoreEvenOdd(dst + 1 * A, half, mask16, e1, o1);
            StoreEvenOdd(dst + 2 * A, half, mask16, e2, o2);
            StoreEvenOdd(dst + 3 * A, half, mask16, e3, o3);
        }

        SIMD_INLINE void GetColSum16x4x4(const uint8_t* src, size_t stride, size_t A, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t e0, o0, e1, o1, e2, o2, e3, o3;
            LoadEvenOdd(dst + 0 * A, half, mask16, e0, o0);
            LoadEvenOdd(dst + 1 * A, half, mask16, e1, o1);
            LoadEvenOdd(dst + 2 * A, half, mask16, e2, o2);
            LoadEvenOdd(dst + 3 * A, half, mask16, e3, o3);
            AddColSum4(src + 0 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 1 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 2 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddColSum4(src + 3 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            StoreEvenOdd(dst + 0 * A, half, mask16, e0, o0);
            StoreEvenOdd(dst + 1 * A, half, mask16, e1, o1);
            StoreEvenOdd(dst + 2 * A, half, mask16, e2, o2);
            StoreEvenOdd(dst + 3 * A, half, mask16, e3, o3);
        }

        SIMD_INLINE void GetColSum16x1x4(const uint8_t* src, size_t A, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t e0, o0, e1, o1, e2, o2, e3, o3;
            LoadEvenOdd(dst + 0 * A, half, mask16, e0, o0);
            LoadEvenOdd(dst + 1 * A, half, mask16, e1, o1);
            LoadEvenOdd(dst + 2 * A, half, mask16, e2, o2);
            LoadEvenOdd(dst + 3 * A, half, mask16, e3, o3);
            AddColSum4(src, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            StoreEvenOdd(dst + 0 * A, half, mask16, e0, o0);
            StoreEvenOdd(dst + 1 * A, half, mask16, e1, o1);
            StoreEvenOdd(dst + 2 * A, half, mask16, e2, o2);
            StoreEvenOdd(dst + 3 * A, half, mask16, e3, o3);
        }

        void GetColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            const size_t A = svcntb(), HA = svcnth(), F = svcntw();
            const size_t A4 = A * 4;
            const size_t alignedLoWidth = AlignLo(width, A);
            const size_t alignedHiWidth = AlignHi(width, A);
            const size_t alignedLoWidth4 = AlignLo(width, A4);
            const size_t stepSize = 256;
            const size_t stepCount = DivHi(height, stepSize);
            const svbool_t body8 = svptrue_b8();

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
                    for (; col < alignedLoWidth4; col += A4)
                        GetColSum16x8x4(rowSrc + col, stride, A, body8, HA, buffer.sums16 + col);
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16x8(rowSrc + col, stride, body8, HA, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16x8(rowSrc + col, stride, svwhilelt_b8(col, width), HA, buffer.sums16 + col);
                    rowSrc += 8 * stride;
                }
                for (; row < rowEnd4; row += 4)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth4; col += A4)
                        GetColSum16x4x4(rowSrc + col, stride, A, body8, HA, buffer.sums16 + col);
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16x4(rowSrc + col, stride, body8, HA, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16x4(rowSrc + col, stride, svwhilelt_b8(col, width), HA, buffer.sums16 + col);
                    rowSrc += 4 * stride;
                }
                for (; row < rowEnd; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth4; col += A4)
                        GetColSum16x1x4(rowSrc + col, A, body8, HA, buffer.sums16 + col);
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16x1(rowSrc + col, body8, HA, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16x1(rowSrc + col, svwhilelt_b8(col, width), HA, buffer.sums16 + col);
                    rowSrc += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += A)
                    AddEvenOdd16To32(buffer.sums16 + col, buffer.sums32 + col, HA, F);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t) * width);
        }

        SIMD_INLINE void GetAbsDxColSum16x1(const uint8_t* src, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t even, odd;
            LoadEvenOdd(dst, half, mask16, even, odd);
            AddAbsDxColSum(src, mask, even, odd);
            StoreEvenOdd(dst, half, mask16, even, odd);
        }

        SIMD_INLINE void GetAbsDxColSum16x4(const uint8_t* src, size_t stride, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t even, odd;
            LoadEvenOdd(dst, half, mask16, even, odd);
            AddAbsDxColSum(src + 0 * stride, mask, even, odd);
            AddAbsDxColSum(src + 1 * stride, mask, even, odd);
            AddAbsDxColSum(src + 2 * stride, mask, even, odd);
            AddAbsDxColSum(src + 3 * stride, mask, even, odd);
            StoreEvenOdd(dst, half, mask16, even, odd);
        }

        SIMD_INLINE void GetAbsDxColSum16x8(const uint8_t* src, size_t stride, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t even, odd;
            LoadEvenOdd(dst, half, mask16, even, odd);
            AddAbsDxColSum(src + 0 * stride, mask, even, odd);
            AddAbsDxColSum(src + 1 * stride, mask, even, odd);
            AddAbsDxColSum(src + 2 * stride, mask, even, odd);
            AddAbsDxColSum(src + 3 * stride, mask, even, odd);
            AddAbsDxColSum(src + 4 * stride, mask, even, odd);
            AddAbsDxColSum(src + 5 * stride, mask, even, odd);
            AddAbsDxColSum(src + 6 * stride, mask, even, odd);
            AddAbsDxColSum(src + 7 * stride, mask, even, odd);
            StoreEvenOdd(dst, half, mask16, even, odd);
        }

        SIMD_INLINE void GetAbsDxColSum16x8x4(const uint8_t* src, size_t stride, size_t A, const svbool_t& mask, size_t half, uint16_t* dst)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t e0, o0, e1, o1, e2, o2, e3, o3;
            LoadEvenOdd(dst + 0 * A, half, mask16, e0, o0);
            LoadEvenOdd(dst + 1 * A, half, mask16, e1, o1);
            LoadEvenOdd(dst + 2 * A, half, mask16, e2, o2);
            LoadEvenOdd(dst + 3 * A, half, mask16, e3, o3);
            AddAbsDxColSum4(src + 0 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddAbsDxColSum4(src + 1 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddAbsDxColSum4(src + 2 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddAbsDxColSum4(src + 3 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddAbsDxColSum4(src + 4 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddAbsDxColSum4(src + 5 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddAbsDxColSum4(src + 6 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            AddAbsDxColSum4(src + 7 * stride, A, mask, e0, o0, e1, o1, e2, o2, e3, o3);
            StoreEvenOdd(dst + 0 * A, half, mask16, e0, o0);
            StoreEvenOdd(dst + 1 * A, half, mask16, e1, o1);
            StoreEvenOdd(dst + 2 * A, half, mask16, e2, o2);
            StoreEvenOdd(dst + 3 * A, half, mask16, e3, o3);
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
            const size_t A = svcntb(), HA = svcnth(), F = svcntw();
            const size_t A4 = A * 4;
            const size_t alignedLoWidth = AlignLo(width, A);
            const size_t alignedHiWidth = AlignHi(width, A);
            const size_t alignedLoWidth4 = AlignLo(width, A4);
            const svbool_t body8 = svptrue_b8();
            const size_t stepSize = SCHAR_MAX + 1;
            const size_t stepCount = (height + SCHAR_MAX) / stepSize;

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
                    for (; col < alignedLoWidth4; col += A4)
                        GetAbsDxColSum16x8x4(rowSrc + col, stride, A, body8, HA, buffer.sums16 + col);
                    for (; col < alignedLoWidth; col += A)
                        GetAbsDxColSum16x8(rowSrc + col, stride, body8, HA, buffer.sums16 + col);
                    if (col < width)
                        GetAbsDxColSum16x8(rowSrc + col, stride, svwhilelt_b8(col, width), HA, buffer.sums16 + col);
                    rowSrc += 8 * stride;
                }
                for (; row < rowEnd4; row += 4)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetAbsDxColSum16x4(rowSrc + col, stride, body8, HA, buffer.sums16 + col);
                    if (col < width)
                        GetAbsDxColSum16x4(rowSrc + col, stride, svwhilelt_b8(col, width), HA, buffer.sums16 + col);
                    rowSrc += 4 * stride;
                }
                for (; row < rowEnd; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetAbsDxColSum16x1(rowSrc + col, body8, HA, buffer.sums16 + col);
                    if (col < width)
                        GetAbsDxColSum16x1(rowSrc + col, svwhilelt_b8(col, width), HA, buffer.sums16 + col);
                    rowSrc += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += A)
                    AddEvenOdd16To32(buffer.sums16 + col, buffer.sums32 + col, HA, F);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t) * width);
            sums[width] = 0;
        }
    }
#endif// SIMD_SVE2_ENABLE
}
