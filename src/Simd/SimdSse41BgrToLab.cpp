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
#include "Simd/SimdBase.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        const __m128i LAB_ROUND = SIMD_MM_SET1_EPI32(Base::LAB_ROUND);

        const __m128i LAB_L_SCALE = SIMD_MM_SET1_EPI32(Base::LAB_L_SCALE);
        const __m128i LAB_L_SHIFT = SIMD_MM_SET1_EPI32(Base::LAB_L_SHIFT);

        const __m128i LAB_A_SCALE = SIMD_MM_SET1_EPI32(Base::LAB_A_SCALE);
        const __m128i LAB_B_SCALE = SIMD_MM_SET1_EPI32(Base::LAB_B_SCALE);
        const __m128i LAB_AB_SHIFT = SIMD_MM_SET1_EPI32(Base::LAB_AB_SHIFT);
        const __m128i LAB_BGRA_TO_BGR_IDX = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x8, 0x1, 0x5, 0x9, 0x2, 0x6, 0xA, 0x3, 0x7, 0xB, -1, -1, -1, -1);

        SIMD_INLINE void CbrtIndex(__m128i r, __m128i g, __m128i b, const __m128i* c, int * i)
        {
            __m128i _i = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(r, c[0]), _mm_mullo_epi32(g, c[1])), _mm_mullo_epi32(b, c[2]));
            _mm_storeu_si128((__m128i*)i, _mm_srai_epi32(_mm_add_epi32(_i, LAB_ROUND), Base::LAB_SHIFT));
        }

        SIMD_INLINE void BgrToLab(const uint8_t* bgr, const __m128i *coeffs, uint8_t* lab)
        {
            const uint32_t* gamma = Base::LabGammaTab;
            SIMD_ALIGNED(16) int R[4], G[4], B[4];
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

            __m128i _R = _mm_loadu_si128((__m128i*)R);
            __m128i _G = _mm_loadu_si128((__m128i*)G);
            __m128i _B = _mm_loadu_si128((__m128i*)B);

            SIMD_ALIGNED(16) int iX[4], iY[4], iZ[4];
            CbrtIndex(_R, _G, _B, coeffs + 0, iX);
            CbrtIndex(_R, _G, _B, coeffs + 3, iY);
            CbrtIndex(_R, _G, _B, coeffs + 6, iZ);

            const uint32_t* cbrt = Base::LabCbrtTab;
            SIMD_ALIGNED(16) int fX[4], fY[4], fZ[4];
            fX[0] = cbrt[iX[0]];
            fX[1] = cbrt[iX[1]];
            fX[2] = cbrt[iX[2]];
            fX[3] = cbrt[iX[3]];

            fY[0] = cbrt[iY[0]];
            fY[1] = cbrt[iY[1]];
            fY[2] = cbrt[iY[2]];
            fY[3] = cbrt[iY[3]];

            fZ[0] = cbrt[iZ[0]];
            fZ[1] = cbrt[iZ[1]];
            fZ[2] = cbrt[iZ[2]];
            fZ[3] = cbrt[iZ[3]];
            __m128i _fX = _mm_loadu_si128((__m128i*)fX);
            __m128i _fY = _mm_loadu_si128((__m128i*)fY);
            __m128i _fZ = _mm_loadu_si128((__m128i*)fZ);

            __m128i _L = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(_fY, LAB_L_SCALE), LAB_L_SHIFT), Base::LAB_SHIFT2);
            __m128i _a = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(_fX, _fY), LAB_A_SCALE), LAB_AB_SHIFT), Base::LAB_SHIFT2);
            __m128i _b = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(_fY, _fZ), LAB_B_SCALE), LAB_AB_SHIFT), Base::LAB_SHIFT2);

            __m128i _Lab = _mm_packus_epi16(_mm_packs_epi32(_L, _a), _mm_packs_epi32(_b, K_ZERO));
            _mm_storeu_si128((__m128i*)lab, _mm_shuffle_epi8(_Lab, LAB_BGRA_TO_BGR_IDX));
        }

        void BgrToLab(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* lab, size_t labStride)
        {
            Base::LabInitTabs();
            size_t widthF = AlignLo(Max<size_t>(width, 2) - 2, F);
            __m128i coeffs[Base::LabCoeffsTabSize];
            for (size_t i = 0; i < Base::LabCoeffsTabSize; ++i)
                coeffs[i] = _mm_set1_epi32(Base::LabCoeffsTab[i]);
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* pBgr = bgr + row * bgrStride, * pEnd = pBgr + width * 3, *pEndF = pBgr + widthF * 3;
                uint8_t* pLab = lab + row * labStride;
                for (; pBgr < pEndF; pBgr += 12, pLab += 12)
                    BgrToLab(pBgr, coeffs, pLab);
                for (; pBgr < pEnd; pBgr += 3, pLab += 3)
                    Base::RgbToLab(pBgr[2], pBgr[1], pBgr[0], pLab);
            }
        }
    }
#endif
}

