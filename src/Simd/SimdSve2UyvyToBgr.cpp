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
#include "Simd/SimdPack.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        template<class T> SIMD_INLINE svint16_t AdjustYLo(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(y)), T::Y_LO);
        }

        template<class T> SIMD_INLINE svint16_t AdjustYHi(const svuint8_t& y)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(y)), T::Y_LO);
        }

        template<class T> SIMD_INLINE svint16_t AdjustUVLo(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(uv)), T::UV_Z);
        }

        template<class T> SIMD_INLINE svint16_t AdjustUVHi(const svuint8_t& uv)
        {
            return svsub_n_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(uv)), T::UV_Z);
        }

        template<class T> SIMD_INLINE svint16_t YuvToBlue16(const svint16_t& y, const svint16_t& u)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            lo = svmlalb_n_s32(lo, u, T::U_2_B);
            hi = svmlalt_n_s32(hi, u, T::U_2_B);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, T::F_SHIFT)), svasr_n_s32_x(mask, hi, T::F_SHIFT));
        }

        template<class T> SIMD_INLINE svint16_t YuvToGreen16(const svint16_t& y, const svint16_t& u, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            lo = svmlalb_n_s32(svmlalb_n_s32(lo, u, T::U_2_G), v, T::V_2_G);
            hi = svmlalt_n_s32(svmlalt_n_s32(hi, u, T::U_2_G), v, T::V_2_G);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, T::F_SHIFT)), svasr_n_s32_x(mask, hi, T::F_SHIFT));
        }

        template<class T> SIMD_INLINE svint16_t YuvToRed16(const svint16_t& y, const svint16_t& v)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t lo = svmlalb_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            svint32_t hi = svmlalt_n_s32(svdup_n_s32(T::F_ROUND), y, T::Y_2_A);
            lo = svmlalb_n_s32(lo, v, T::V_2_R);
            hi = svmlalt_n_s32(hi, v, T::V_2_R);
            return svqxtnt_s32(svqxtnb_s32(svasr_n_s32_x(mask, lo, T::F_SHIFT)), svasr_n_s32_x(mask, hi, T::F_SHIFT));
        }

        template<class T> SIMD_INLINE svuint8_t YuvToBlue(const svuint8_t& y, const svuint8_t& u)
        {
            return PackSatIntI16ToU8(
                YuvToBlue16<T>(AdjustYLo<T>(y), AdjustUVLo<T>(u)),
                YuvToBlue16<T>(AdjustYHi<T>(y), AdjustUVHi<T>(u)));
        }

        template<class T> SIMD_INLINE svuint8_t YuvToGreen(const svuint8_t& y, const svuint8_t& u, const svuint8_t& v)
        {
            return PackSatIntI16ToU8(
                YuvToGreen16<T>(AdjustYLo<T>(y), AdjustUVLo<T>(u), AdjustUVLo<T>(v)),
                YuvToGreen16<T>(AdjustYHi<T>(y), AdjustUVHi<T>(u), AdjustUVHi<T>(v)));
        }

        template<class T> SIMD_INLINE svuint8_t YuvToRed(const svuint8_t& y, const svuint8_t& v)
        {
            return PackSatIntI16ToU8(
                YuvToRed16<T>(AdjustYLo<T>(y), AdjustUVLo<T>(v)),
                YuvToRed16<T>(AdjustYHi<T>(y), AdjustUVHi<T>(v)));
        }

        template<class T> SIMD_INLINE void Uyvy422ToBgr(const uint8_t* uyvy, uint8_t* bgr, const svbool_t& mask)
        {
            size_t A = svcntb(), A3 = A * 3;
            svuint8x4_t _uyvy = svld4_u8(mask, uyvy);
            svuint8_t u = svget4(_uyvy, 0);
            svuint8_t y0 = svget4(_uyvy, 1);
            svuint8_t v = svget4(_uyvy, 2);
            svuint8_t y1 = svget4(_uyvy, 3);
            svuint8_t blue0 = YuvToBlue<T>(y0, u);
            svuint8_t blue1 = YuvToBlue<T>(y1, u);
            svuint8_t green0 = YuvToGreen<T>(y0, u, v);
            svuint8_t green1 = YuvToGreen<T>(y1, u, v);
            svuint8_t red0 = YuvToRed<T>(y0, v);
            svuint8_t red1 = YuvToRed<T>(y1, v);
            svst3_u8(mask, bgr, svcreate3_u8(svzip1_u8(blue0, blue1), svzip1_u8(green0, green1), svzip1_u8(red0, red1)));
            svst3_u8(mask, bgr + A3, svcreate3_u8(svzip2_u8(blue0, blue1), svzip2_u8(green0, green1), svzip2_u8(red0, red1)));
        }

        template<class T> SIMD_INLINE void Uyvy422ToBgr(const uint8_t* uyvy, uint8_t* bgr)
        {
            uint8_t u = uyvy[0], v = uyvy[2];
            Base::YuvToBgr<T>(uyvy[1], u, v, bgr + 0);
            Base::YuvToBgr<T>(uyvy[3], u, v, bgr + 3);
        }

        template<class T> void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            size_t A = svcntb(), DA = 2 * A, A4 = 4 * A, A6 = 6 * A, widthDA = AlignLo(width, DA);
            const svbool_t mask = svptrue_b8();
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, colUyvy = 0, colBgr = 0;
                for (; col < widthDA; col += DA, colUyvy += A4, colBgr += A6)
                    Uyvy422ToBgr<T>(uyvy + colUyvy, bgr + colBgr, mask);
                for (; col < width; col += 2, colUyvy += 4, colBgr += 6)
                    Uyvy422ToBgr<T>(uyvy + colUyvy, bgr + colBgr);
                uyvy += uyvyStride;
                bgr += bgrStride;
            }
        }

        void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Uyvy422ToBgr<Base::Bt601>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Uyvy422ToBgr<Base::Bt709>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Uyvy422ToBgr<Base::Bt2020>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Uyvy422ToBgr<Base::Trect871>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
