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
        }
        
        void ResizerByteBicubic::EstimateIndexAlpha(size_t sizeS, size_t sizeD, size_t N, Array32i& index, Array32i alpha[4])
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
                alpha[0][i] = Round(BICUBIC_RANGE * (2.0f - d) * (1.0f - d) * d / 6.0f);
                alpha[1][i] = -Round(BICUBIC_RANGE * (2.0f - d) * (d + 1.0f) * (1.0f - d) / 2.0f);
                alpha[2][i] = -Round(BICUBIC_RANGE * (2.0f - d) * (d + 1.0f) * d / 2.0f);
                alpha[3][i] = Round(BICUBIC_RANGE * (1.0f + d) * (1.0f - d) * d / 6.0f);
            }
        } 

        void ResizerByteBicubic::Init()
        {
            if (_iy.data)
                return;
            EstimateIndexAlpha(_param.srcH, _param.dstH, 1, _iy, _ay);
            EstimateIndexAlpha(_param.srcW, _param.dstW, _param.channels, _ix, _ax);
            for (int i = 0; i < 4; ++i)
                _bx[i].Resize(_param.dstW * _param.channels);
        }

        void ResizerByteBicubic::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            Init();
            size_t cn = _param.channels;
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                int32_t ay0 = _ay[0][dy];
                int32_t ay1 = _ay[1][dy];
                int32_t ay2 = _ay[2][dy];
                int32_t ay3 = _ay[3][dy];
                for (size_t dx = 0; dx < _param.dstW; dx++)
                {
                    size_t sx1 = _ix[dx];
                    size_t sx2 = sx1 + cn;
                    size_t sx0 = sx1 ? sx1 - cn : sx1;
                    size_t sx3 = sx1 < _param.srcW - 2 ? sx2 + cn : sx2;
                    int32_t ax0 = _ax[0][dx];
                    int32_t ax1 = _ax[1][dx];
                    int32_t ax2 = _ax[2][dx];
                    int32_t ax3 = _ax[3][dx];
                    for (size_t c = 0; c < cn; ++c)
                    {
                        int32_t rs0 = ax0 * src0[sx0 + c] + ax1 * src0[sx1 + c] + ax2 * src0[sx2 + c] + ax3 * src0[sx3 + c];
                        int32_t rs1 = ax0 * src1[sx0 + c] + ax1 * src1[sx1 + c] + ax2 * src1[sx2 + c] + ax3 * src1[sx3 + c];
                        int32_t rs2 = ax0 * src2[sx0 + c] + ax1 * src2[sx1 + c] + ax2 * src2[sx2 + c] + ax3 * src2[sx3 + c];
                        int32_t rs3 = ax0 * src3[sx0 + c] + ax1 * src3[sx1 + c] + ax2 * src3[sx2 + c] + ax3 * src3[sx3 + c];
                        int32_t fs = ay0 * rs0 + ay1 * rs1 + ay2 * rs2 + ay3 * rs3;
                        dst[dx * cn + c] = RestrictRange((fs + BICUBIC_ROUND) >> BICUBIC_SHIFT, 0, 255);
                    }
                }
            }
        }
    }
}

