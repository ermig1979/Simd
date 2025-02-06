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
#include "Simd/SimdEnable.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        const __m256i LAB_ROUND = SIMD_MM256_SET1_EPI32(Base::LAB_ROUND);

        const __m256i LAB_L_SCALE = SIMD_MM256_SET1_EPI32(Base::LAB_L_SCALE);
        const __m256i LAB_L_SHIFT = SIMD_MM256_SET1_EPI32(Base::LAB_L_SHIFT);

        const __m256i LAB_A_SCALE = SIMD_MM256_SET1_EPI32(Base::LAB_A_SCALE);
        const __m256i LAB_B_SCALE = SIMD_MM256_SET1_EPI32(Base::LAB_B_SCALE);
        const __m256i LAB_AB_SHIFT = SIMD_MM256_SET1_EPI32(Base::LAB_AB_SHIFT);
        const __m256i LAB_BGRA_TO_BGR_IDX = SIMD_MM256_SETR_EPI8(
            0x0, 0x4, 0x8, 0x1, 0x5, 0x9, 0x2, 0x6, 0xA, 0x3, 0x7, 0xB, -1, -1, -1, -1,
            0x0, 0x4, 0x8, 0x1, 0x5, 0x9, 0x2, 0x6, 0xA, 0x3, 0x7, 0xB, -1, -1, -1, -1);

        const __m256i LAB_BGR_LOAD_PERM = SIMD_MM256_SETR_EPI32(0, 1, 2, 0, 3, 4, 5, 0);
        const __m256i LAB_BGR_TO_B32 = SIMD_MM256_SETR_EPI8(
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1,
            0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1);
        const __m256i LAB_BGR_TO_G32 = SIMD_MM256_SETR_EPI8(
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1,
            0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1);
        const __m256i LAB_BGR_TO_R32 = SIMD_MM256_SETR_EPI8(
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1,
            0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1);

        SIMD_INLINE __m256i CbrtIndex(__m256i r, __m256i g, __m256i b, const __m256i* c)
        {
            __m256i _i = _mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(r, c[0]), _mm256_mullo_epi32(g, c[1])), _mm256_mullo_epi32(b, c[2]));
            return _mm256_srai_epi32(_mm256_add_epi32(_i, LAB_ROUND), Base::LAB_SHIFT);
        }

        SIMD_INLINE void BgrToLabM(const uint8_t* bgr, const __m256i *coeffs, uint8_t* lab)
        {
            const uint32_t* gamma = Base::LabGammaTab;
            SIMD_ALIGNED(32) int R[8], G[8], B[8];
            B[0] = gamma[bgr[0]];
            G[0] = gamma[bgr[1]];
            R[0] = gamma[bgr[2]];

            B[1] = gamma[bgr[3]];
            G[1] = gamma[bgr[4]];
            R[1] = gamma[bgr[5]];

            B[2] = gamma[bgr[6]];
            G[2] = gamma[bgr[7]];
            R[2] = gamma[bgr[8]];

            B[3] = gamma[bgr[9]];
            G[3] = gamma[bgr[10]];
            R[3] = gamma[bgr[11]];

            B[4] = gamma[bgr[12]];
            G[4] = gamma[bgr[13]];
            R[4] = gamma[bgr[14]];

            B[5] = gamma[bgr[15]];
            G[5] = gamma[bgr[16]];
            R[5] = gamma[bgr[17]];

            B[6] = gamma[bgr[18]];
            G[6] = gamma[bgr[19]];
            R[6] = gamma[bgr[20]];

            B[7] = gamma[bgr[21]];
            G[7] = gamma[bgr[22]];
            R[7] = gamma[bgr[23]];

            __m256i _R = _mm256_loadu_si256((__m256i*)R);
            __m256i _G = _mm256_loadu_si256((__m256i*)G);
            __m256i _B = _mm256_loadu_si256((__m256i*)B);

            SIMD_ALIGNED(32) int iX[8], iY[8], iZ[8];
            _mm256_storeu_si256((__m256i*)iX, CbrtIndex(_R, _G, _B, coeffs + 0));
            _mm256_storeu_si256((__m256i*)iY, CbrtIndex(_R, _G, _B, coeffs + 3));
            _mm256_storeu_si256((__m256i*)iZ, CbrtIndex(_R, _G, _B, coeffs + 6));

            const uint32_t* cbrt = Base::LabCbrtTab;
            SIMD_ALIGNED(32) int fX[8], fY[8], fZ[8];
            fX[0] = cbrt[iX[0]];
            fX[1] = cbrt[iX[1]];
            fX[2] = cbrt[iX[2]];
            fX[3] = cbrt[iX[3]];
            fX[4] = cbrt[iX[4]];
            fX[5] = cbrt[iX[5]];
            fX[6] = cbrt[iX[6]];
            fX[7] = cbrt[iX[7]];

            fY[0] = cbrt[iY[0]];
            fY[1] = cbrt[iY[1]];
            fY[2] = cbrt[iY[2]];
            fY[3] = cbrt[iY[3]];
            fY[4] = cbrt[iY[4]];
            fY[5] = cbrt[iY[5]];
            fY[6] = cbrt[iY[6]];
            fY[7] = cbrt[iY[7]];

            fZ[0] = cbrt[iZ[0]];
            fZ[1] = cbrt[iZ[1]];
            fZ[2] = cbrt[iZ[2]];
            fZ[3] = cbrt[iZ[3]];
            fZ[4] = cbrt[iZ[4]];
            fZ[5] = cbrt[iZ[5]];
            fZ[6] = cbrt[iZ[6]];
            fZ[7] = cbrt[iZ[7]];
            __m256i _fX = _mm256_loadu_si256((__m256i*)fX);
            __m256i _fY = _mm256_loadu_si256((__m256i*)fY);
            __m256i _fZ = _mm256_loadu_si256((__m256i*)fZ);

            __m256i _L = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_fY, LAB_L_SCALE), LAB_L_SHIFT), Base::LAB_SHIFT2);
            __m256i _a = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(_fX, _fY), LAB_A_SCALE), LAB_AB_SHIFT), Base::LAB_SHIFT2);
            __m256i _b = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(_fY, _fZ), LAB_B_SCALE), LAB_AB_SHIFT), Base::LAB_SHIFT2);

            __m256i _Lab = _mm256_packus_epi16(_mm256_packs_epi32(_L, _a), _mm256_packs_epi32(_b, K_ZERO));
            Store24<false>(lab, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_Lab, LAB_BGRA_TO_BGR_IDX), K32_PERMUTE_BGRA_TO_BGR));
        }

        SIMD_INLINE void BgrToLabG(const uint8_t* bgr, const __m256i* coeffs, uint8_t* lab)
        {
            const int* gamma = (int*)Base::LabGammaTab;
#if 0
            __m256i _bgr = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)bgr), LAB_BGR_LOAD_PERM);
            __m256i _R = _mm256_i32gather_epi32(gamma, _mm256_shuffle_epi8(_bgr, LAB_BGR_TO_R32), 4);
            __m256i _G = _mm256_i32gather_epi32(gamma, _mm256_shuffle_epi8(_bgr, LAB_BGR_TO_G32), 4);
            __m256i _B = _mm256_i32gather_epi32(gamma, _mm256_shuffle_epi8(_bgr, LAB_BGR_TO_B32), 4);
#else
            SIMD_ALIGNED(32) int R[8], G[8], B[8];
            B[0] = gamma[bgr[0]];
            G[0] = gamma[bgr[1]];
            R[0] = gamma[bgr[2]];

            B[1] = gamma[bgr[3]];
            G[1] = gamma[bgr[4]];
            R[1] = gamma[bgr[5]];

            B[2] = gamma[bgr[6]];
            G[2] = gamma[bgr[7]];
            R[2] = gamma[bgr[8]];

            B[3] = gamma[bgr[9]];
            G[3] = gamma[bgr[10]];
            R[3] = gamma[bgr[11]];

            B[4] = gamma[bgr[12]];
            G[4] = gamma[bgr[13]];
            R[4] = gamma[bgr[14]];

            B[5] = gamma[bgr[15]];
            G[5] = gamma[bgr[16]];
            R[5] = gamma[bgr[17]];

            B[6] = gamma[bgr[18]];
            G[6] = gamma[bgr[19]];
            R[6] = gamma[bgr[20]];

            B[7] = gamma[bgr[21]];
            G[7] = gamma[bgr[22]];
            R[7] = gamma[bgr[23]];

            __m256i _R = _mm256_loadu_si256((__m256i*)R);
            __m256i _G = _mm256_loadu_si256((__m256i*)G);
            __m256i _B = _mm256_loadu_si256((__m256i*)B);
#endif

            const int* cbrt = (int*)Base::LabCbrtTab;
            __m256i _fX = _mm256_i32gather_epi32(cbrt, CbrtIndex(_R, _G, _B, coeffs + 0), 4);
            __m256i _fY = _mm256_i32gather_epi32(cbrt, CbrtIndex(_R, _G, _B, coeffs + 3), 4);
            __m256i _fZ = _mm256_i32gather_epi32(cbrt, CbrtIndex(_R, _G, _B, coeffs + 6), 4);

            __m256i _L = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_fY, LAB_L_SCALE), LAB_L_SHIFT), Base::LAB_SHIFT2);
            __m256i _a = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(_fX, _fY), LAB_A_SCALE), LAB_AB_SHIFT), Base::LAB_SHIFT2);
            __m256i _b = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(_fY, _fZ), LAB_B_SCALE), LAB_AB_SHIFT), Base::LAB_SHIFT2);

            __m256i _Lab = _mm256_packus_epi16(_mm256_packs_epi32(_L, _a), _mm256_packs_epi32(_b, K_ZERO));
            Store24<false>(lab, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_Lab, LAB_BGRA_TO_BGR_IDX), K32_PERMUTE_BGRA_TO_BGR));
        }

        void BgrToLab(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* lab, size_t labStride)
        {
            Base::LabInitTabs();
            size_t widthF = AlignLo(width, F);
            __m256i coeffs[Base::LabCoeffsTabSize];
            for (size_t i = 0; i < Base::LabCoeffsTabSize; ++i)
                coeffs[i] = _mm256_set1_epi32(Base::LabCoeffsTab[i]);
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* pBgr = bgr + row * bgrStride, * pEnd = pBgr + width * 3, *pEndF = pBgr + widthF * 3;
                uint8_t* pLab = lab + row * labStride;
                if (Avx2::SlowGather)
                {
                    for (; pBgr < pEndF; pBgr += 24, pLab += 24)
                        BgrToLabM(pBgr, coeffs, pLab);
                }
                else
                {
                    for (; pBgr < pEndF; pBgr += 24, pLab += 24)
                        BgrToLabG(pBgr, coeffs, pLab);
                }
                for (; pBgr < pEnd; pBgr += 3, pLab += 3)
                    Base::RgbToLab(pBgr[2], pBgr[1], pBgr[0], pLab);
            }
        }
    }
#endif
}

