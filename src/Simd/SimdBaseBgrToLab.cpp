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

#define SIMD_BGR_TO_LAB_OPENCV_COMPATIBILITY

namespace Simd
{
    namespace Base
    {
        uint32_t LabGammaTab[LabGammaTabSize];
        uint32_t LabCbrtTab[LabCbrtTabSize];
        uint32_t LabCoeffsTab[9];

        SIMD_INLINE float ApplyGamma(float x)
        {
            return x <= 0.04045f ? x * (1.f / 12.92f) : (float)std::pow((double)(x + 0.055) * (1. / 1.055), 2.4);
        }

        SIMD_INLINE float ApplyCbrt(float x)
        {
            static const float threshold = float(216) / float(24389);
            static const float scale = float(841) / float(108);
            static const float bias = float(16) / float(116);
            return x < threshold ? (x * scale + bias) : cbrt(x);
        }

        static bool LabTabsInited = false;
        void LabInitTabs()
        {
            if (LabTabsInited)
                return;
            const float intScale(255 * (1 << LAB_GAMMA_SHIFT));
            for (int i = 0; i < LabGammaTabSize; i++)
            {
                float x = float(i) / 255.0f;
                LabGammaTab[i] = Round(intScale * ApplyGamma(x));
            }
            const float tabScale(1.0 / (255.0 * (1 << LAB_GAMMA_SHIFT)));
            const float shift2(1 << LAB_SHIFT2);
            for (int i = 0; i < LabCbrtTabSize; i++)
            {
                float x = tabScale * float(i);
                LabCbrtTab[i] = Round(shift2 * ApplyCbrt(x));
            }
#if defined(SIMD_BGR_TO_LAB_OPENCV_COMPATIBILITY)
            if (LabCbrtTab[324] == 17746)
                LabCbrtTab[324] = 17745;
            if (LabCbrtTab[49] == 9455)
                LabCbrtTab[49] = 9454;
#endif
            const double D65[3] = { 0.950456, 1., 1.088754 };
            const double sRGB2XYZ_D65[9] = { 0.412453, 0.357580, 0.180423, 0.212671, 0.715160, 0.072169, 0.019334, 0.119193, 0.950227 };
            const double shift(1 << LAB_SHIFT);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    LabCoeffsTab[i * 3 + j] = Round(shift * sRGB2XYZ_D65[i * 3 + j] / D65[i]);
            LabTabsInited = true;
        }

        //--------------------------------------------------------------------------------------------------

        void BgrToLab(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* lab, size_t labStride)
        {
            LabInitTabs();
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* pBgr = bgr + row * bgrStride, * pEnd = pBgr + width * 3;
                uint8_t* pLab = lab + row * labStride;
                for (; pBgr < pEnd; pBgr += 3, pLab += 3)
                    RgbToLab(pBgr[2], pBgr[1], pBgr[0], pLab);
            }
        }
    }
}

