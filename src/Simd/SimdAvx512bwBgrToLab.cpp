/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdMath.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i LAB_ROUND = SIMD_MM512_SET1_EPI32(Base::LAB_ROUND);

        const __m512i LAB_L_SCALE = SIMD_MM512_SET1_EPI32(Base::LAB_L_SCALE);
        const __m512i LAB_L_SHIFT = SIMD_MM512_SET1_EPI32(Base::LAB_L_SHIFT);

        const __m512i LAB_A_SCALE = SIMD_MM512_SET1_EPI32(Base::LAB_A_SCALE);
        const __m512i LAB_B_SCALE = SIMD_MM512_SET1_EPI32(Base::LAB_B_SCALE);
        const __m512i LAB_AB_SHIFT = SIMD_MM512_SET1_EPI32(Base::LAB_AB_SHIFT);
        const __m512i LAB_BGRA_TO_BGR_IDX = SIMD_MM512_SETR_EPI8(
            0x0, 0x4, 0x8, 0x1, 0x5, 0x9, 0x2, 0x6, 0xA, 0x3, 0x7, 0xB, -1, -1, -1, -1,
            0x0, 0x4, 0x8, 0x1, 0x5, 0x9, 0x2, 0x6, 0xA, 0x3, 0x7, 0xB, -1, -1, -1, -1,
            0x0, 0x4, 0x8, 0x1, 0x5, 0x9, 0x2, 0x6, 0xA, 0x3, 0x7, 0xB, -1, -1, -1, -1,
            0x0, 0x4, 0x8, 0x1, 0x5, 0x9, 0x2, 0x6, 0xA, 0x3, 0x7, 0xB, -1, -1, -1, -1);
        const __m512i LAB_BGRA_TO_BGR_PERM = SIMD_MM512_SETR_EPI32(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0);

        const __m512i LAB_BGR_TO_B32 = SIMD_MM512_SETR_EPI8(
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1,
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1,
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1,
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1);
        const __m512i LAB_BGR_TO_G32 = SIMD_MM512_SETR_EPI8(
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1,
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1,
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1,
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1);
        const __m512i LAB_BGR_TO_R32 = SIMD_MM512_SETR_EPI8(
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1,
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1,
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1,
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1);

        SIMD_INLINE __m512i CbrtIndex(__m512i r, __m512i g, __m512i b, const __m512i* c)
        {
            __m512i _i = _mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(r, c[0]), _mm512_mullo_epi32(g, c[1])), _mm512_mullo_epi32(b, c[2]));
            return _mm512_srai_epi32(_mm512_add_epi32(_i, LAB_ROUND), Base::LAB_SHIFT);
        }

        SIMD_INLINE void BgrToLab(const uint8_t* bgr, const __m512i* coeffs, uint8_t* lab, __mmask64 mask)
        {
            const int* gamma = (int*)Base::LabGammaTab;
            __m512i _bgr = _mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA, _mm512_maskz_loadu_epi8(mask, bgr));
            __m512i _R = _mm512_i32gather_epi32(_mm512_shuffle_epi8(_bgr, LAB_BGR_TO_R32), gamma, 4);
            __m512i _G = _mm512_i32gather_epi32(_mm512_shuffle_epi8(_bgr, LAB_BGR_TO_G32), gamma, 4);
            __m512i _B = _mm512_i32gather_epi32(_mm512_shuffle_epi8(_bgr, LAB_BGR_TO_B32), gamma, 4);

            const int* cbrt = (int*)Base::LabCbrtTab;
            __m512i _fX = _mm512_i32gather_epi32(CbrtIndex(_R, _G, _B, coeffs + 0), cbrt, 4);
            __m512i _fY = _mm512_i32gather_epi32(CbrtIndex(_R, _G, _B, coeffs + 3), cbrt, 4);
            __m512i _fZ = _mm512_i32gather_epi32(CbrtIndex(_R, _G, _B, coeffs + 6), cbrt, 4);

            __m512i _L = _mm512_srai_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_fY, LAB_L_SCALE), LAB_L_SHIFT), Base::LAB_SHIFT2);
            __m512i _a = _mm512_srai_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_sub_epi32(_fX, _fY), LAB_A_SCALE), LAB_AB_SHIFT), Base::LAB_SHIFT2);
            __m512i _b = _mm512_srai_epi32(_mm512_add_epi32(_mm512_mullo_epi32(_mm512_sub_epi32(_fY, _fZ), LAB_B_SCALE), LAB_AB_SHIFT), Base::LAB_SHIFT2);

            __m512i _Lab = _mm512_packus_epi16(_mm512_packs_epi32(_L, _a), _mm512_packs_epi32(_b, K_ZERO));
            _mm512_mask_storeu_epi8(lab, mask, _mm512_permutexvar_epi32(LAB_BGRA_TO_BGR_PERM, _mm512_shuffle_epi8(_Lab, LAB_BGRA_TO_BGR_IDX)));
        }

        void BgrToLab(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* lab, size_t labStride)
        {
            Base::LabInitTabs();
            size_t widthF = AlignLo(width, F);
            __mmask64 body = 0x0000FFFFFFFFFFFF, tail = TailMask64((width - widthF)*3);
            __m512i coeffs[Base::LabCoeffsTabSize];
            for (size_t i = 0; i < Base::LabCoeffsTabSize; ++i)
                coeffs[i] = _mm512_set1_epi32(Base::LabCoeffsTab[i]);
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* pBgr = bgr + row * bgrStride, * pEnd = pBgr + width * 3, *pEndF = pBgr + widthF * 3;
                uint8_t* pLab = lab + row * labStride;
                for (; pBgr < pEndF; pBgr += 48, pLab += 48)
                    BgrToLab(pBgr, coeffs, pLab, body);
                if(pBgr < pEnd)
                    BgrToLab(pBgr, coeffs, pLab, tail);
            }
        }
    }
#endif
}

