/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdCopyPixel.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        ResizerByteBicubic::ResizerByteBicubic(const ResParam& param)
            : Base::ResizerByteBicubic(param)
        {
        }

        void ResizerByteBicubic::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            Init(true);
            size_t cn = _param.channels;
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                const int32_t* ay = _ay.data + dy * 4;
                for (size_t dx = 0; dx < _param.dstW; dx++)
                {
                    size_t sx1 = _ix[dx];
                    size_t sx2 = sx1 + cn;
                    size_t sx0 = sx1 ? sx1 - cn : sx1;
                    size_t sx3 = sx1 < _sxl ? sx2 + cn : sx2;
                    const int32_t* ax = _ax.data + dx * 4;
                    for (size_t c = 0; c < cn; ++c)
                    {
                        int32_t rs0 = ax[0] * src0[sx0 + c] + ax[1] * src0[sx1 + c] + ax[2] * src0[sx2 + c] + ax[3] * src0[sx3 + c];
                        int32_t rs1 = ax[0] * src1[sx0 + c] + ax[1] * src1[sx1 + c] + ax[2] * src1[sx2 + c] + ax[3] * src1[sx3 + c];
                        int32_t rs2 = ax[0] * src2[sx0 + c] + ax[1] * src2[sx1 + c] + ax[2] * src2[sx2 + c] + ax[3] * src2[sx3 + c];
                        int32_t rs3 = ax[0] * src3[sx0 + c] + ax[1] * src3[sx1 + c] + ax[2] * src3[sx2 + c] + ax[3] * src3[sx3 + c];
                        int32_t fs = ay[0] * rs0 + ay[1] * rs1 + ay[2] * rs2 + ay[3] * rs3;
                        dst[dx * cn + c] = Base::RestrictRange((fs + Base::BICUBIC_ROUND) >> Base::BICUBIC_SHIFT, 0, 255);
                    }
                }
            }
        }
    }
#endif//SIMD_SSE41_ENABLE
}

