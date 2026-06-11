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

#include "Simd/SimdBgrToLab.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svint32_t CbrtIndex(const svint32_t& r, const svint32_t& g, const svint32_t& b,
            const svint32_t& c0, const svint32_t& c1, const svint32_t& c2)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t i = svmla_s32_x(mask, svmla_s32_x(mask, svmul_s32_x(mask, r, c0), g, c1), b, c2);
            return svasr_n_s32_x(mask, svadd_n_s32_x(mask, i, Base::LAB_ROUND), Base::LAB_SHIFT);
        }

        SIMD_INLINE svuint8_t PackI32ToU8(const svint32_t& v0, const svint32_t& v1, const svint32_t& v2, const svint32_t& v3)
        {
            svint16_t lo = svqxtnt_s32(svqxtnb_s32(v0), v1);
            svint16_t hi = svqxtnt_s32(svqxtnb_s32(v2), v3);
            return svqxtunt_s16(svqxtunb_s16(lo), hi);
        }

        SIMD_INLINE svint32_t LoadGamma(const svuint32_t& value)
        {
            const svbool_t mask = svptrue_b32();
            return svreinterpret_s32_u32(svld1_gather_u32index_u32(mask, Base::LabGammaTab, value));
        }

        SIMD_INLINE svint32_t LoadCbrt(const svint32_t& index)
        {
            const svbool_t mask = svptrue_b32();
            return svreinterpret_s32_u32(svld1_gather_s32index_u32(mask, Base::LabCbrtTab, index));
        }

        SIMD_INLINE void BgrToLab32(const svuint32_t& blue, const svuint32_t& green, const svuint32_t& red, svint32_t& L, svint32_t& a, svint32_t& b,
            const svint32_t& c0, const svint32_t& c1, const svint32_t& c2, const svint32_t& c3, const svint32_t& c4, const svint32_t& c5,
            const svint32_t& c6, const svint32_t& c7, const svint32_t& c8)
        {
            const svbool_t mask = svptrue_b32();
            svint32_t R = LoadGamma(red);
            svint32_t G = LoadGamma(green);
            svint32_t B = LoadGamma(blue);

            svint32_t fX = LoadCbrt(CbrtIndex(R, G, B, c0, c1, c2));
            svint32_t fY = LoadCbrt(CbrtIndex(R, G, B, c3, c4, c5));
            svint32_t fZ = LoadCbrt(CbrtIndex(R, G, B, c6, c7, c8));

            L = svasr_n_s32_x(mask, svadd_s32_x(mask, svmul_n_s32_x(mask, fY, Base::LAB_L_SCALE), svdup_n_s32(Base::LAB_L_SHIFT)), Base::LAB_SHIFT2);
            a = svasr_n_s32_x(mask, svadd_s32_x(mask, svmul_n_s32_x(mask, svsub_s32_x(mask, fX, fY), Base::LAB_A_SCALE), svdup_n_s32(Base::LAB_AB_SHIFT)), Base::LAB_SHIFT2);
            b = svasr_n_s32_x(mask, svadd_s32_x(mask, svmul_n_s32_x(mask, svsub_s32_x(mask, fY, fZ), Base::LAB_B_SCALE), svdup_n_s32(Base::LAB_AB_SHIFT)), Base::LAB_SHIFT2);
        }

        SIMD_INLINE void BgrToLab(const uint8_t* bgr, uint8_t* lab, const svbool_t& mask,
            const svint32_t& c0, const svint32_t& c1, const svint32_t& c2, const svint32_t& c3, const svint32_t& c4, const svint32_t& c5,
            const svint32_t& c6, const svint32_t& c7, const svint32_t& c8)
        {
            svuint8x3_t _bgr = svld3_u8(mask, bgr);
            svuint16_t blueLo = svmovlb_u16(svget3(_bgr, 0));
            svuint16_t blueHi = svmovlt_u16(svget3(_bgr, 0));
            svuint16_t greenLo = svmovlb_u16(svget3(_bgr, 1));
            svuint16_t greenHi = svmovlt_u16(svget3(_bgr, 1));
            svuint16_t redLo = svmovlb_u16(svget3(_bgr, 2));
            svuint16_t redHi = svmovlt_u16(svget3(_bgr, 2));

            svint32_t L0, L1, L2, L3, a0, a1, a2, a3, b0, b1, b2, b3;
            BgrToLab32(svmovlb_u32(blueLo), svmovlb_u32(greenLo), svmovlb_u32(redLo), L0, a0, b0, c0, c1, c2, c3, c4, c5, c6, c7, c8);
            BgrToLab32(svmovlt_u32(blueLo), svmovlt_u32(greenLo), svmovlt_u32(redLo), L1, a1, b1, c0, c1, c2, c3, c4, c5, c6, c7, c8);
            BgrToLab32(svmovlb_u32(blueHi), svmovlb_u32(greenHi), svmovlb_u32(redHi), L2, a2, b2, c0, c1, c2, c3, c4, c5, c6, c7, c8);
            BgrToLab32(svmovlt_u32(blueHi), svmovlt_u32(greenHi), svmovlt_u32(redHi), L3, a3, b3, c0, c1, c2, c3, c4, c5, c6, c7, c8);

            svst3_u8(mask, lab, svcreate3_u8(PackI32ToU8(L0, L1, L2, L3), PackI32ToU8(a0, a1, a2, a3), PackI32ToU8(b0, b1, b2, b3)));
        }

        void BgrToLab(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* lab, size_t labStride)
        {
            Base::LabInitTabs();
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svint32_t c0 = svdup_n_s32(Base::LabCoeffsTab[0]);
            const svint32_t c1 = svdup_n_s32(Base::LabCoeffsTab[1]);
            const svint32_t c2 = svdup_n_s32(Base::LabCoeffsTab[2]);
            const svint32_t c3 = svdup_n_s32(Base::LabCoeffsTab[3]);
            const svint32_t c4 = svdup_n_s32(Base::LabCoeffsTab[4]);
            const svint32_t c5 = svdup_n_s32(Base::LabCoeffsTab[5]);
            const svint32_t c6 = svdup_n_s32(Base::LabCoeffsTab[6]);
            const svint32_t c7 = svdup_n_s32(Base::LabCoeffsTab[7]);
            const svint32_t c8 = svdup_n_s32(Base::LabCoeffsTab[8]);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A3)
                    BgrToLab(bgr + offset, lab + offset, body, c0, c1, c2, c3, c4, c5, c6, c7, c8);
                if (widthA < width)
                    BgrToLab(bgr + offset, lab + offset, tail, c0, c1, c2, c3, c4, c5, c6, c7, c8);
                bgr += bgrStride;
                lab += labStride;
            }
        }
    }
#endif
}

