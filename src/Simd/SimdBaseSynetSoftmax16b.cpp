/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
        void SynetSoftmax16b(const uint16_t* src, size_t outer, size_t count, size_t inner, uint16_t* dst)
        {
            if (inner == 1)
            {
                if (count == 2)
                {
                    for (size_t o = 0; o < outer; ++o)
                    {
                        float src0 = BFloat16ToFloat32(src[0]);
                        float src1 = BFloat16ToFloat32(src[1]);
                        float max = Simd::Max(src0, src1);
                        float exp0 = ::exp(src0 - max);
                        float exp1 = ::exp(src1 - max);
                        float sum = exp0 + exp1;
                        dst[0] = Float32ToBFloat16(exp0 / sum);
                        dst[1] = Float32ToBFloat16(exp1 / sum);
                        src += 2;
                        dst += 2;
                    }
                }
                else if (count == 3)
                {
                    for (size_t o = 0; o < outer; ++o)
                    {
                        float src0 = BFloat16ToFloat32(src[0]);
                        float src1 = BFloat16ToFloat32(src[1]);
                        float src2 = BFloat16ToFloat32(src[2]);
                        float max = Simd::Max(Simd::Max(src0, src1), src2);
                        float exp0 = ::exp(src0 - max);
                        float exp1 = ::exp(src1 - max);
                        float exp2 = ::exp(src2 - max);
                        float sum = exp0 + exp1 + exp2;
                        dst[0] = Float32ToBFloat16(exp0 / sum);
                        dst[1] = Float32ToBFloat16(exp1 / sum);
                        dst[2] = Float32ToBFloat16(exp2 / sum);
                        src += 3;
                        dst += 3;
                    }
                }
                else
                {
                    Array32f buf(count);
                    for (size_t o = 0; o < outer; ++o)
                    {
                        for (size_t c = 0; c < count; ++c)
                            buf[c] = BFloat16ToFloat32(src[c]);

                        float max = buf[0];
                        for (size_t c = 1; c < count; ++c)
                            max = Simd::Max(max, buf[c]);
                        float sum = 0;
                        for (size_t c = 0; c < count; ++c)
                        {
                            buf[c] = ::exp(buf[c] - max);
                            sum += buf[c];
                        }
                        float k = 1.0f / sum;
                        for (size_t c = 0; c < count; ++c)
                            dst[c] = Float32ToBFloat16(buf[c] * k);
                        src += count;
                        dst += count;
                    }
                }
            }
            else
            {
                Array32f _buf(inner * (count + 2));
                float* max = _buf.data, * sum = _buf.data + inner, *buf = sum + inner, *b;
                for (size_t o = 0; o < outer; ++o)
                {
                    for (size_t i = 0, n = count * inner; i < n; ++i)
                        buf[i] = BFloat16ToFloat32(src[i]);

                    for (size_t i = 0; i < inner; ++i)
                        max[i] = buf[i];
                    b = buf + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            max[i] = Simd::Max(max[i], b[i]);
                        b += inner;
                    }

                    b = buf;
                    for (size_t i = 0; i < inner; ++i)
                        sum[i] = 0;
                    for (size_t c = 0; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                        {
                            b[i] = ::exp(b[i] - max[i]);
                            sum[i] += b[i];
                        }
                        b += inner;
                    }

                    b = buf;
                    for (size_t c = 0; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            dst[i] = Float32ToBFloat16(b[i] / sum[i]);
                        b += inner;
                        dst += inner;
                    }
                    src += count * inner;
                }
            }
        }
    }
#endif
}
