/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#ifndef __SimdGrayToY_h__
#define __SimdGrayToY_h__

#include "Simd/SimdInit.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdLoad.h"

namespace Simd
{
    namespace Base
    {
        const int G2Y_LO = 16;
        const int G2Y_HI = 235;
        const int Y2G_LO = 0;
        const int Y2G_HI = 255;

        const int G2Y_SHIFT = 13;
        const int G2Y_RANGE = 1 << G2Y_SHIFT;
        const int G2Y_ROUND = 1 << (G2Y_SHIFT - 1);
        const int G2Y_SCALE = int(G2Y_RANGE * (G2Y_HI - G2Y_LO) / (Y2G_HI - Y2G_LO) + 0.5f);

        const int Y2G_SHIFT = 14;
        const int Y2G_RANGE = 1 << Y2G_SHIFT;
        const int Y2G_ROUND = 1 << (Y2G_SHIFT - 1);
        const int Y2G_SCALE = int(Y2G_RANGE * (Y2G_HI - Y2G_LO) / (G2Y_HI - G2Y_LO) + 0.5f);

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int GrayToY(int g)
        {
            int y = ((G2Y_SCALE * g + G2Y_ROUND) >> G2Y_SHIFT) + G2Y_LO;
            return RestrictRange(y, G2Y_LO, G2Y_HI);
        }

        SIMD_INLINE int YToGray(int y)
        {
            y = RestrictRange(y, G2Y_LO, G2Y_HI);
            int g = (Y2G_SCALE * (y - G2Y_LO) + Y2G_ROUND) >> Y2G_SHIFT;
            return RestrictRange(g, Y2G_LO, Y2G_HI);
        }
    }
}

#endif//__SimdGrayToY_h__
