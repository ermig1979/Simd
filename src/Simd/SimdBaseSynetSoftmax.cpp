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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAlignment.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            if (inner == 1)
            {
                if (count == 2)
                {
                    for (size_t o = 0; o < outer; ++o)
                    {
                        float max = Simd::Max(src[0], src[1]);
                        float exp0 = ::exp(src[0] - max);
                        float exp1 = ::exp(src[1] - max);
                        float sum = exp0 + exp1;
                        dst[0] = exp0 / sum;
                        dst[1] = exp1 / sum;
                        src += 2;
                        dst += 2;
                    }
                }
                else if (count == 3)
                {
                    for (size_t o = 0; o < outer; ++o)
                    {
                        float max = Simd::Max(Simd::Max(src[0], src[1]), src[2]);
                        float exp0 = ::exp(src[0] - max);
                        float exp1 = ::exp(src[1] - max);
                        float exp2 = ::exp(src[2] - max);
                        float sum = exp0 + exp1 + exp2;
                        dst[0] = exp0 / sum;
                        dst[1] = exp1 / sum;
                        dst[2] = exp2 / sum;
                        src += 3;
                        dst += 3;
                    }
                }
                else
                {
                    for (size_t o = 0; o < outer; ++o)
                    {
                        float max = src[0];
                        for (size_t c = 1; c < count; ++c)
                            max = Simd::Max(max, src[c]);
                        float sum = 0;
                        for (size_t c = 0; c < count; ++c)
                        {
                            dst[c] = ::exp(src[c] - max);
                            sum += dst[c];                            
                        }
                        float k = 1.0f / sum;
                        for (size_t c = 0; c < count; ++c)
                            dst[c] *= k;
                        src += count;
                        dst += count;
                    }
                }
            }
            else
            {
                Array32f tmp(inner * 2);
                const float * s;
                float * max = tmp.data, *sum = tmp.data + inner, *d;
                for (size_t o = 0; o < outer; ++o)
                {
                    for (size_t i = 0; i < inner; ++i)
                        max[i] = src[i];
                    s = src + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            max[i] = Simd::Max(max[i], s[i]);
                        s += inner;
                    }

                    s = src;
                    d = dst;
                    for (size_t i = 0; i < inner; ++i)
                        sum[i] = 0;
                    for (size_t c = 0; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                        {
                            d[i] = ::exp(s[i] - max[i]);
                            sum[i] += d[i];
                        }
                        s += inner;
                        d += inner;
                    }

                    d = dst;
                    for (size_t c = 0; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            d[i] /= sum[i];
                        d += inner;
                    }
                    src += count * inner;
                    dst += count * inner;
                }
            }
        }
    }
#endif
}
