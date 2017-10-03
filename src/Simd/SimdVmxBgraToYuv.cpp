/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        const v128_u8 K8_PERM_BGRA_TO_B0R0 = SIMD_VEC_SETR_EPI8(0x10, 0x00, 0x10, 0x02, 0x10, 0x04, 0x10, 0x06, 0x10, 0x08, 0x10, 0x0A, 0x10, 0x0C, 0x10, 0x0E);
        const v128_u8 K8_PERM_BGRA_TO_G010 = SIMD_VEC_SETR_EPI8(0x10, 0x01, 0x10, 0x11, 0x10, 0x05, 0x10, 0x11, 0x10, 0x09, 0x10, 0x11, 0x10, 0x0D, 0x10, 0x11);

        template <bool align> SIMD_INLINE void LoadPreparedBgra(const uint8_t * bgra, v128_s16 & b_r, v128_s16 & g_1)
        {
            v128_u8 _bgra = Load<align>(bgra);
            b_r = (v128_s16)vec_perm(_bgra, (v128_u8)K16_0001, K8_PERM_BGRA_TO_B0R0);
            g_1 = (v128_s16)vec_perm(_bgra, (v128_u8)K16_0001, K8_PERM_BGRA_TO_G010);
        }

        const v128_u8 K8_PERM_HADD0 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x03, 0x08, 0x09, 0x0A, 0x0B, 0x10, 0x11, 0x12, 0x13, 0x18, 0x19, 0x1A, 0x1B);
        const v128_u8 K8_PERM_HADD1 = SIMD_VEC_SETR_EPI8(0x04, 0x05, 0x06, 0x07, 0x0C, 0x0D, 0x0E, 0x0F, 0x14, 0x15, 0x16, 0x17, 0x1C, 0x1D, 0x1E, 0x1F);

        SIMD_INLINE v128_s16 HorizontalAdd(v128_s16 a, v128_s16 b)
        {
            return vec_add(vec_perm(a, b, K8_PERM_HADD0), vec_perm(a, b, K8_PERM_HADD1));
        }

        template <bool align> SIMD_INLINE v128_u16 LoadAndConvertY(const uint8_t * bgra, v128_s16 & b_r, v128_s16 & g_1)
        {
            v128_s16 _b_r[2], _g_1[2];
            LoadPreparedBgra<align>(bgra + 0, _b_r[0], _g_1[0]);
            LoadPreparedBgra<align>(bgra + A, _b_r[1], _g_1[1]);
            b_r = HorizontalAdd(_b_r[0], _b_r[1]);
            g_1 = HorizontalAdd(_g_1[0], _g_1[1]);
            return SaturateI16ToU8(vec_add((v128_s16)K16_Y_ADJUST, vec_pack(BgrToY(_b_r[0], _g_1[0]), BgrToY(_b_r[1], _g_1[1]))));
        }

        template <bool align> SIMD_INLINE v128_u8 LoadAndConvertY(const uint8_t * bgra, v128_s16 b_r[2], v128_s16 g_1[2])
        {
            return vec_pack(LoadAndConvertY<align>(bgra, b_r[0], g_1[0]), LoadAndConvertY<align>(bgra + DA, b_r[1], g_1[1]));
        }

        SIMD_INLINE void Average(v128_s16 & a, v128_s16 & b)
        {
            a = (v128_s16)vec_sr(vec_add(vec_add(a, b), (v128_s16)K16_0002), K16_0002);
        }

        SIMD_INLINE v128_u16 ConvertU(v128_s16 b_r[2], v128_s16 g_1[2])
        {
            return SaturateI16ToU8(vec_add((v128_s16)K16_UV_ADJUST, vec_pack(BgrToU(b_r[0], g_1[0]), BgrToU(b_r[1], g_1[1]))));
        }

        SIMD_INLINE v128_u16 ConvertV(v128_s16 b_r[2], v128_s16 g_1[2])
        {
            return SaturateI16ToU8(vec_add((v128_s16)K16_UV_ADJUST, vec_pack(BgrToV(b_r[0], g_1[0]), BgrToV(b_r[1], g_1[1]))));
        }

        template <bool align, bool first> SIMD_INLINE void BgraToYuv420p(const uint8_t * bgra0, size_t bgraStride,
            Storer<align> & y0, Storer<align> & y1, Storer<align> & u, Storer<align> & v)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;

            v128_s16 _b_r[2][2][2], _g_1[2][2][2];
            Store<align, first>(y0, LoadAndConvertY<align>(bgra0, _b_r[0][0], _g_1[0][0]));
            Store<align, false>(y0, LoadAndConvertY<align>(bgra0 + QA, _b_r[0][1], _g_1[0][1]));
            Store<align, first>(y1, LoadAndConvertY<align>(bgra1, _b_r[1][0], _g_1[1][0]));
            Store<align, false>(y1, LoadAndConvertY<align>(bgra1 + QA, _b_r[1][1], _g_1[1][1]));

            Average(_b_r[0][0][0], _b_r[1][0][0]);
            Average(_b_r[0][0][1], _b_r[1][0][1]);
            Average(_b_r[0][1][0], _b_r[1][1][0]);
            Average(_b_r[0][1][1], _b_r[1][1][1]);

            Average(_g_1[0][0][0], _g_1[1][0][0]);
            Average(_g_1[0][0][1], _g_1[1][0][1]);
            Average(_g_1[0][1][0], _g_1[1][1][0]);
            Average(_g_1[0][1][1], _g_1[1][1][1]);

            Store<align, first>(u, vec_pack(ConvertU(_b_r[0][0], _g_1[0][0]), ConvertU(_b_r[0][1], _g_1[0][1])));
            Store<align, first>(v, vec_pack(ConvertV(_b_r[0][0], _g_1[0][0]), ConvertV(_b_r[0][1], _g_1[0][1])));
        }

        template <bool align> void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; row += 2)
            {
                Storer<align> _y0(y), _y1(y + yStride), _u(u), _v(v);
                BgraToYuv420p<align, true>(bgra, bgraStride, _y0, _y1, _u, _v);
                for (size_t col = DA, colBgra = A8; col < alignedWidth; col += DA, colBgra += A8)
                    BgraToYuv420p<align, false>(bgra + colBgra, bgraStride, _y0, _y1, _u, _v);
                Flush(_y0, _y1, _u, _v);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    Storer<false> _y0(y + offset), _y1(y + offset + yStride), _u(u + offset / 2), _v(v + offset / 2);
                    BgraToYuv420p<false, true>(bgra + offset * 4, bgraStride, _y0, _y1, _u, _v);
                    Flush(_y0, _y1, _u, _v);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv420p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv420p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }

        SIMD_INLINE void Average(v128_s16 a[2][2])
        {
            a[0][0] = (v128_s16)vec_sr(vec_add(a[0][0], (v128_s16)K16_0001), K16_0001);
            a[0][1] = (v128_s16)vec_sr(vec_add(a[0][1], (v128_s16)K16_0001), K16_0001);
            a[1][0] = (v128_s16)vec_sr(vec_add(a[1][0], (v128_s16)K16_0001), K16_0001);
            a[1][1] = (v128_s16)vec_sr(vec_add(a[1][1], (v128_s16)K16_0001), K16_0001);
        }

        template <bool align, bool first> SIMD_INLINE void BgraToYuv422p(const uint8_t * bgra, Storer<align> & y, Storer<align> & u, Storer<align> & v)
        {
            v128_s16 _b_r[2][2], _g_1[2][2];
            Store<align, first>(y, LoadAndConvertY<align>(bgra, _b_r[0], _g_1[0]));
            Store<align, false>(y, LoadAndConvertY<align>(bgra + QA, _b_r[1], _g_1[1]));

            Average(_b_r);
            Average(_g_1);

            Store<align, first>(u, vec_pack(ConvertU(_b_r[0], _g_1[0]), ConvertU(_b_r[1], _g_1[1])));
            Store<align, first>(v, vec_pack(ConvertV(_b_r[0], _g_1[0]), ConvertV(_b_r[1], _g_1[1])));
        }

        template <bool align> void BgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _y(y), _u(u), _v(v);
                BgraToYuv422p<align, true>(bgra, _y, _u, _v);
                for (size_t col = DA, colBgra = A8; col < alignedWidth; col += DA, colBgra += A8)
                    BgraToYuv422p<align, false>(bgra + colBgra, _y, _u, _v);
                Flush(_y, _u, _v);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    Storer<false> _y(y + offset), _u(u + offset / 2), _v(v + offset / 2);
                    BgraToYuv422p<false, true>(bgra + offset * 4, _y, _u, _v);
                    Flush(_y, _u, _v);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv422p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv422p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }

        SIMD_INLINE v128_u16 ConvertY(v128_s16 b_r[2], v128_s16 g_1[2])
        {
            return SaturateI16ToU8(vec_add((v128_s16)K16_Y_ADJUST, vec_pack(BgrToY(b_r[0], g_1[0]), BgrToY(b_r[1], g_1[1]))));
        }

        template <bool align, bool first> SIMD_INLINE void BgraToYuv444p(const uint8_t * bgra,
            Storer<align> & y, Storer<align> & u, Storer<align> & v)
        {
            v128_s16 _b_r[2][2], _g_1[2][2];
            LoadPreparedBgra<align>(bgra + 0, _b_r[0][0], _g_1[0][0]);
            LoadPreparedBgra<align>(bgra + A, _b_r[0][1], _g_1[0][1]);
            LoadPreparedBgra<align>(bgra + 2 * A, _b_r[1][0], _g_1[1][0]);
            LoadPreparedBgra<align>(bgra + 3 * A, _b_r[1][1], _g_1[1][1]);

            Store<align, first>(y, vec_pack(ConvertY(_b_r[0], _g_1[0]), ConvertY(_b_r[1], _g_1[1])));
            Store<align, first>(u, vec_pack(ConvertU(_b_r[0], _g_1[0]), ConvertU(_b_r[1], _g_1[1])));
            Store<align, first>(v, vec_pack(ConvertV(_b_r[0], _g_1[0]), ConvertV(_b_r[1], _g_1[1])));
        }

        template <bool align> void BgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _y(y), _u(u), _v(v);
                BgraToYuv444p<align, true>(bgra, _y, _u, _v);
                for (size_t col = A, colBgra = QA; col < alignedWidth; col += A, colBgra += QA)
                    BgraToYuv444p<align, false>(bgra + colBgra, _y, _u, _v);
                Flush(_y, _u, _v);
                if (width != alignedWidth)
                {
                    size_t offset = width - A;
                    Storer<false> _y(y + offset), _u(u + offset), _v(v + offset);
                    BgraToYuv444p<false, true>(bgra + offset * 4, _y, _u, _v);
                    Flush(_y, _u, _v);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv444p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv444p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}
