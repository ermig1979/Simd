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
#include "Simd/SimdMemory.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdCopyPixel.h"

namespace Simd
{
    namespace Base
    {
        ResizerByteBicubic::ResizerByteBicubic(const ResParam & param)
            : Resizer(param)
        {
            float ky = float(BICUBIC_RANGE);
            EstimateIndexAlpha(_param.srcH, _param.dstH, 1, ky, _iy, _ay);
            float kx = float(BICUBIC_LIMIT * BICUBIC_LIMIT) / float(BICUBIC_RANGE);
            EstimateIndexAlpha(_param.srcW, _param.dstW, _param.channels, kx, _ix, _ax);
        }
        
        void ResizerByteBicubic::EstimateIndexAlpha(size_t sizeS, size_t sizeD, size_t N, float range, Array32i& index, Array32i alpha[4])
        {
            index.Resize(sizeD);
            for (int i = 0; i < 4; ++i)
                alpha[i].Resize(sizeD);

            float scale = float(sizeS) / float(sizeD);
            for (size_t i = 0; i < sizeD; ++i)
            {
                float pos = (float)((i + 0.5f) * scale - 0.5f);
                int idx = (int)::floor(pos);
                float d = pos - idx;
                if (idx < 0)
                {
                    idx = 0;
                    d = 0.0f;
                }
                if (idx > (int)sizeS - 2)
                {
                    idx = (int)sizeS - 2;
                    d = 1.0f;
                }
                index[i] = idx * (int)N;
                alpha[0][i] = - int(range * (2.0f - d) * (1.0f - d) * d / 6.0f);
                alpha[1][i] = int(range * (2.0f - d) * (d + 1.0f) * (d - 1.0f) / 2.0f);
                alpha[2][i] = int(range * (2.0f - d) * (d + 1.0f) * d / 2.0f);
                alpha[3][i] = - int(range * (1.0f + d) * (d - 1.0f) * d / 6.0f);
            }
        }        

        void ResizerByteBicubic::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {

        }
    }
}

