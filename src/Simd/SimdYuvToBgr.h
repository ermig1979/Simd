/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifndef __SimdYuvToBgr_h__
#define __SimdYuvToBgr_h__

#include "Simd/SimdInit.h"

namespace Simd
{
    namespace Base
    {
        /* Corresponds to BT.601 standard. Uses Kr=0.299, Kb=0.114. Restricts Y to range [16..235], U and V to [16..240]. */
        struct Bt601
        {
            static const int Y_LO = 16;
            static const int Y_HI = 235;
            static const int UV_LO = 16;
            static const int UV_HI = 240;
            static const int UV_Z = 128;

            static const int F_SHIFT = 13;
            static const int F_RANGE = 1 << F_SHIFT;
            static const int F_ROUND = 1 << (F_SHIFT - 1);

            static const int Y_2_BGR = int(1.0f * 255 / (Y_HI - Y_LO) * F_RANGE + 0.5f);
            static const int U_2_B = int(2.0f * (1.0f - 0.114f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int U_2_G = -int(2.0f * 0.114f * (1.0f - 0.114f) / (1.0f - 0.299f - 0.114f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_G = -int(2.0f * 0.299f * (1.0f - 0.299f) / (1.0f - 0.299f - 0.114f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_R = int(2.0f * (1.0f - 0.299f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
        };

        /* Corresponds to BT.709 standard. Uses Kr=0.2126, Kb=0.0722. Restricts Y to range [16..235], U and V to [16..240]. */
        struct Bt709
        {
            static const int Y_LO = 16;
            static const int Y_HI = 235;
            static const int UV_LO = 16;
            static const int UV_HI = 240;
            static const int UV_Z = 128;

            static const int F_SHIFT = 13;
            static const int F_RANGE = 1 << F_SHIFT;
            static const int F_ROUND = 1 << (F_SHIFT - 1);

            static const int Y_2_BGR = int(1.0f * 255 / (Y_HI - Y_LO) * F_RANGE + 0.5f);
            static const int U_2_B = int(2.0f * (1.0f - 0.0722f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int U_2_G = -int(2.0f * 0.0722f * (1.0f - 0.0722f) / (1.0f - 0.2126f - 0.0722f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_G = -int(2.0f * 0.2126f * (1.0f - 0.2126f) / (1.0f - 0.2126f - 0.0722f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_R = int(2.0f * (1.0f - 0.2126f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
        };

        /* Corresponds to BT.2020 standard. Uses Kr=0.2627, Kb=0.0593. Restricts Y to range [16..235], U and V to [16..240]. */
        struct Bt2020
        {
            static const int Y_LO = 16;
            static const int Y_HI = 235;
            static const int UV_LO = 16;
            static const int UV_HI = 240;
            static const int UV_Z = 128;

            static const int F_SHIFT = 13;
            static const int F_RANGE = 1 << F_SHIFT;
            static const int F_ROUND = 1 << (F_SHIFT - 1);

            static const int Y_2_BGR = int(1.0f * 255 / (Y_HI - Y_LO) * F_RANGE + 0.5f);
            static const int U_2_B = int(2.0f * (1.0f - 0.0593f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int U_2_G = -int(2.0f * 0.0593f * (1.0f - 0.0593f) / (1.0f - 0.2627f - 0.0593f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_G = -int(2.0f * 0.2627f * (1.0f - 0.2126f) / (1.0f - 0.2627f - 0.0593f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
            static const int V_2_R = int(2.0f * (1.0f - 0.2627f) * 255 / (UV_HI - UV_LO) * F_RANGE + 0.5f);
        };

        /* Corresponds to T-REC-T.871 standard. Uses Kr=0.299, Kb=0.114. Y, U and V use full range [0..255]. */
        struct Trect871
        {
            static const int Y_LO = 0;
            static const int Y_HI = 255;
            static const int UV_LO = 0;
            static const int UV_HI = 255;
            static const int UV_Z = 128;

            static const int F_SHIFT = 14;
            static const int F_RANGE = 1 << F_SHIFT;
            static const int F_ROUND = 1 << (F_SHIFT - 1);

            static const int Y_2_BGR = int(1.0f * F_RANGE + 0.5f);
            static const int U_2_B = int(2.0f * (1.0f - 0.114f) * F_RANGE + 0.5f);
            static const int U_2_G = -int(2.0f * 0.114f * (1.0f - 0.114f) / (1.0f - 0.299f - 0.114f) * F_RANGE + 0.5f);
            static const int V_2_G = -int(2.0f * 0.299f * (1.0f - 0.299f) / (1.0f - 0.299f - 0.114f) * F_RANGE + 0.5f);
            static const int V_2_R = int(2.0f * (1.0f - 0.299f) * F_RANGE + 0.5f);
        };

        template<class T> SIMD_INLINE int YuvToBlue(int y, int u)
        {
            return RestrictRange((T::Y_2_BGR * (y - T::Y_LO) + T::U_2_B * (u - T::UV_Z) + T::F_ROUND) >> T::F_SHIFT);
        }

        template<class T> SIMD_INLINE int YuvToGreen(int y, int u, int v)
        {
            return RestrictRange((T::Y_2_BGR * (y - T::Y_LO) + T::U_2_G * (u - T::UV_Z) + T::V_2_G * (v - T::UV_Z) + T::F_ROUND) >> T::F_SHIFT);
        }

        template<class T> SIMD_INLINE int YuvToRed(int y, int v)
        {
            return RestrictRange((T::Y_2_BGR * (y - T::Y_LO) + T::V_2_R * (v - T::UV_Z) + T::F_ROUND) >> T::F_SHIFT);
        }

        template<class T> SIMD_INLINE void YuvToBgr(int y, int u, int v, uint8_t* bgr)
        {
            bgr[0] = YuvToBlue<T>(y, u);
            bgr[1] = YuvToGreen<T>(y, u, v);
            bgr[2] = YuvToRed<T>(y, v);
        }

        template<class T> SIMD_INLINE void YuvToBgra(int y, int u, int v, int alpha, uint8_t* bgra)
        {
            bgra[0] = YuvToBlue<T>(y, u);
            bgra[1] = YuvToGreen<T>(y, u, v);
            bgra[2] = YuvToRed<T>(y, v);
            bgra[3] = alpha;
        }
    }
}

#endif//__SimdYuvToBgr_h__
