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
        const __m128i LAB_ROUND2 = SIMD_MM_SET1_EPI32(Base::LAB_ROUND2);

        SIMD_INLINE void BgrToLab(const uint8_t* bgr, const __m128 *coeffs, uint8_t* lab)
        {
            const uint16_t* gamma = Base::LabGammaTab;
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

            //int R = LabGammaTab[red];
            //int G = LabGammaTab[green];
            //int B = LabGammaTab[blue];

            //int iX = LabDescale(R * LabCoeffsTab[0] + G * LabCoeffsTab[1] + B * LabCoeffsTab[2]);
            //int iY = LabDescale(R * LabCoeffsTab[3] + G * LabCoeffsTab[4] + B * LabCoeffsTab[5]);
            //int iZ = LabDescale(R * LabCoeffsTab[6] + G * LabCoeffsTab[7] + B * LabCoeffsTab[8]);

            //int fX = LabCbrtTab[iX];
            //int fY = LabCbrtTab[iY];
            //int fZ = LabCbrtTab[iZ];

            //int L = LabDescale2(LAB_L_SCALE * fY + LAB_L_SHIFT);
            //int a = LabDescale2(LAB_A_SCALE * (fX - fY) + LAB_AB_SHIFT);
            //int b = LabDescale2(LAB_B_SCALE * (fY - fZ) + LAB_AB_SHIFT);

            //lab[0] = Base::RestrictRange(L);
            //lab[1] = Base::RestrictRange(a);
            //lab[2] = Base::RestrictRange(b);
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
                //for (; pBgr < pEndF; pBgr += 12, pLab += 12)
                //    Base::RgbToLab(pBgr[2], pBgr[1], pBgr[0], pLab);
                for (; pBgr < pEnd; pBgr += 3, pLab += 3)
                    Base::RgbToLab(pBgr[2], pBgr[1], pBgr[0], pLab);
            }
        }
    }
#endif
}

