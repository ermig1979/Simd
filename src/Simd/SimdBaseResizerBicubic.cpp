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
            for (int i = 0; i < 4; ++i)
                _bx[i].Resize(_param.dstW * _param.channels);
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
            size_t cn = _param.channels;
            size_t rs = _param.dstW * cn;
            int32_t* pbx[4] = { _bx[0].data, _bx[1].data, _bx[2].data, _bx[3].data };
            int32_t prev = -2;
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                //int32_t fy = _ay[dy];
                //int32_t sy = _iy[dy];
                //int32_t k = 0;

                //if (sy == prev)
                //    k = 2;
                //else if (sy == prev + 1)
                //{
                //    Swap(pbx[0], pbx[1]);
                //    k = 1;
                //}

                //prev = sy;

                //for (; k < 2; k++)
                //{
                //    int32_t* pb = pbx[k];
                //    const uint8_t* ps = src + (sy + k) * srcStride;
                //    for (size_t dx = 0; dx < rs; dx++)
                //    {
                //        int32_t sx = _ix[dx];
                //        int32_t fx = _ax[dx];
                //        int32_t t = ps[sx];
                //        pb[dx] = (t << LINEAR_SHIFT) + (ps[sx + cn] - t) * fx;
                //    }
                //}

                //if (fy == 0)
                //    for (size_t dx = 0; dx < rs; dx++)
                //        dst[dx] = ((pbx[0][dx] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
                //else if (fy == FRACTION_RANGE)
                //    for (size_t dx = 0; dx < rs; dx++)
                //        dst[dx] = ((pbx[1][dx] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
                //else
                //{
                //    for (size_t dx = 0; dx < rs; dx++)
                //    {
                //        int32_t t = pbx[0][dx];
                //        dst[dx] = ((t << LINEAR_SHIFT) + (pbx[1][dx] - t) * fy + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
                //    }
                //}
            }
        }
    }
}

