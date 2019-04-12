/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        ResizerFloatBilinear::ResizerFloatBilinear(const ResParam & param)
            : Base::ResizerFloatBilinear(param)
        {
        }

        void ResizerFloatBilinear::Run(const float * src, size_t srcStride, float * dst, size_t dstStride) 
        {
            size_t cn = _param.channels;
            size_t rs = _param.dstW * cn;
            float * pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rsa = AlignLo(rs, F);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    float * pb = pbx[k];
                    const float * ps = src + (sy + k)*srcStride;
                    size_t dx = 0;
                    if (cn == 1)
                    {
                        float32x4_t _1 = vdupq_n_f32(1.0f);
                        for (; dx < rsa; dx += F)
                        {
                            float32x4_t s01 = Load(ps + _ix[dx + 0], ps + _ix[dx + 1]);
                            float32x4_t s23 = Load(ps + _ix[dx + 2], ps + _ix[dx + 3]);
                            float32x4_t fx1 = Load<true>(_ax.data + dx);
                            float32x4_t fx0 = vsubq_f32(_1, fx1);
                            float32x4x2_t us = vuzpq_f32(s01, s23);
                            Store<true>(pb + dx, vmlaq_f32(vmulq_f32(us.val[0], fx0), us.val[1], fx1));
                        }
                    }
                    if (cn == 3 && rs > 3)
                    {
                        float32x4_t _1 = vdupq_n_f32(1.0f);
                        size_t rs3 = rs - 3;
                        for (; dx < rs3; dx += 3)
                        {
                            float32x4_t s0 = Load<false>(ps + _ix[dx] + 0);
                            float32x4_t s1 = Load<false>(ps + _ix[dx] + 3);
                            float32x4_t fx1 = vdupq_n_f32(_ax.data[dx]);
                            float32x4_t fx0 = vsubq_f32(_1, fx1);
                            Store<false>(pb + dx, vmlaq_f32(vmulq_f32(fx0, s0), fx1, s1));
                        }
                    }
                    for (; dx < rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        float fx = _ax[dx];
                        pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + cn] * fx;
                    }
                }

                size_t dx = 0;
                float32x4_t _fy0 = vdupq_n_f32(fy0);
                float32x4_t _fy1 = vdupq_n_f32(fy1);
                for (; dx < rsa; dx += F)
                    Store<false>(dst + dx, vmlaq_f32(vmulq_f32(Load<true>(pbx[0] + dx), _fy0), Load<true>(pbx[1] + dx), _fy1));
                for (; dx < rs; dx++)
                    dst[dx] = pbx[0][dx] * fy0 + pbx[1][dx] * fy1;
            }
        }

        //---------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
        {
            ResParam param(srcX, srcY, dstX, dstY, channels, type, method, sizeof(float32x4_t));
            if (type == SimdResizeChannelFloat && (method == SimdResizeMethodBilinear || method == SimdResizeMethodCaffeInterp))
                return new ResizerFloatBilinear(param);
            else
                return Base::ResizerInit(srcX, srcY, dstX, dstY, channels, type, method);
        }
    }
#endif// SIMD_NEON_ENABLE
}
