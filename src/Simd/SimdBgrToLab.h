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
#ifndef __SimdBgrToLab_h__
#define __SimdBgrToLab_h__

#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        const int LAB_GAMMA_SHIFT = 3;

        const int LabGammaTabSize = 256;
        extern uint16_t LabGammaTab[LabGammaTabSize];

        const int LabCbrtTabSize = 256 * 3 / 2 * (1 << LAB_GAMMA_SHIFT);
        extern uint16_t LabCbrtTab[LabCbrtTabSize];

        const int LabCoeffsTabSize = 9;
        extern uint32_t LabCoeffsTab[LabCoeffsTabSize];

        void LabInitTabs();

        const int LAB_SHIFT = 12;
        const int LAB_ROUND = 1 << (LAB_SHIFT - 1);

        const int LAB_SHIFT2 = LAB_SHIFT + LAB_GAMMA_SHIFT;
        const int LAB_ROUND2 = 1 << (LAB_SHIFT2 - 1);

        const int LAB_L_SCALE = (116 * 255 + 50) / 100;
        const int LAB_L_SHIFT = -((16 * 255 * (1 << LAB_SHIFT2) + 50) / 100);

        const int LAB_A_SCALE = 500;
        const int LAB_B_SCALE = 200;
        const int LAB_AB_SHIFT = 128 * (1 << LAB_SHIFT2);

        SIMD_INLINE int LabDescale(int value)
        {
            return (value + LAB_ROUND) >> LAB_SHIFT;
        }

        SIMD_INLINE int LabDescale2(int value)
        {
            return (value + LAB_ROUND2) >> LAB_SHIFT2;
        }

        SIMD_INLINE void RgbToLab(int red, int green, int blue, uint8_t *lab)
        {
            int R = LabGammaTab[red];
            int G = LabGammaTab[green];
            int B = LabGammaTab[blue];

            int iX = LabDescale(R * LabCoeffsTab[0] + G * LabCoeffsTab[1] + B * LabCoeffsTab[2]);
            int iY = LabDescale(R * LabCoeffsTab[3] + G * LabCoeffsTab[4] + B * LabCoeffsTab[5]);
            int iZ = LabDescale(R * LabCoeffsTab[6] + G * LabCoeffsTab[7] + B * LabCoeffsTab[8]);

            int fX = LabCbrtTab[iX];
            int fY = LabCbrtTab[iY];
            int fZ = LabCbrtTab[iZ];

            int L = LabDescale2(LAB_L_SCALE * fY + LAB_L_SHIFT);
            int a = LabDescale2(LAB_A_SCALE * (fX - fY) + LAB_AB_SHIFT);
            int b = LabDescale2(LAB_B_SCALE * (fY - fZ) + LAB_AB_SHIFT);

            lab[0] = Base::RestrictRange(L);
            lab[1] = Base::RestrictRange(a);
            lab[2] = Base::RestrictRange(b);
        }
    }
}

#endif
