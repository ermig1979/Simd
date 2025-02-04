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

#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        namespace LabV0
        {
#define X(xyz) xyz.data[0]
#define Y(xyz) xyz.data[1]
#define Z(xyz) xyz.data[2]

            typedef union
            {
                double data[3];
                struct { double r, g, b; };
                struct { double L, A, B; };
                struct { double l, c, h; };
            } DoubleTriplet;

            const double eps = (6 * 6 * 6) / (29.0 * 29.0 * 29.0), kap = (29 * 29 * 29) / (3.0 * 3.0 * 3.0);

            const DoubleTriplet xyzReferenceValues = { {0.95047, 1.0, 1.08883} };

            SIMD_INLINE DoubleTriplet xyzFromRgb(DoubleTriplet rgb)
            {
                for (int i = 0; i < 3; ++i)
                {
                    double v = rgb.data[i];
                    rgb.data[i] = (v > 0.04045) ? pow(((v + 0.055) / 1.055), 2.4) : (v / 12.92);
                }
                DoubleTriplet temp = { {
                    rgb.r * 0.4124564 + rgb.g * 0.3575761 + rgb.b * 0.1804375,
                    rgb.r * 0.2126729 + rgb.g * 0.7151522 + rgb.b * 0.0721750,
                    rgb.r * 0.0193339 + rgb.g * 0.1191920 + rgb.b * 0.9503041
                    } };
                return temp;
            }

            SIMD_INLINE DoubleTriplet labFromXyz(DoubleTriplet xyz)
            {
                for (int i = 0; i < 3; ++i)
                {
                    xyz.data[i] /= xyzReferenceValues.data[i];
                    double v = xyz.data[i];
                    xyz.data[i] = (v > eps) ? pow(v, (1 / 3.0)) : ((kap * v + 16) / 116.0);
                }
                DoubleTriplet temp = { {(116 * Y(xyz)) - 16, 500 * (X(xyz) - Y(xyz)), 200 * (Y(xyz) - Z(xyz))} };
                return temp;
            }

            DoubleTriplet labFromRgb(DoubleTriplet rgb) { return labFromXyz(xyzFromRgb(rgb)); }
        }

        SIMD_INLINE void BgrToLab(int blue, int green, int red, uint8_t* lab)
        {
            LabV0::DoubleTriplet _rgb{red / 255.0 , green / 255.0, blue / 255.0 };
            LabV0::DoubleTriplet _lab = labFromRgb(_rgb);
            lab[0] = int(_lab.data[0]);
            lab[1] = int(_lab.data[1]);
            lab[2] = int(_lab.data[2]);
        }
    }
}

#endif
