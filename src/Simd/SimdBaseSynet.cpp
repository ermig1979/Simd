/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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

namespace Simd
{
    namespace Base
    {
        void SynetAddBias(const float * bias, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            for (size_t i = 0; i < count; ++i)
            {
                float value = bias[i];
                size_t j = 0;
                for (; j < aligned; j += 4)
                {
                    dst[j + 0] += value;
                    dst[j + 1] += value;
                    dst[j + 2] += value;
                    dst[j + 3] += value;
                }
                for (; j < size; ++j)
                    dst[j] += value;
                dst += size;
            }
        }

        void SynetEltwiseLayerForwardProduct(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            for (; j < aligned; j += 4)
            {
                dst[j + 0] = src0[j + 0] * src1[j + 0];
                dst[j + 1] = src0[j + 1] * src1[j + 1];
                dst[j + 2] = src0[j + 2] * src1[j + 2];
                dst[j + 3] = src0[j + 3] * src1[j + 3];
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * src1[j];
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                for (j = 0; j < aligned; j += 4)
                {
                    dst[j + 0] *= srci[j + 0];
                    dst[j + 1] *= srci[j + 1];
                    dst[j + 2] *= srci[j + 2];
                    dst[j + 3] *= srci[j + 3];
                }
                for (; j < size; ++j)
                    dst[j] *= srci[j];
            }
        }

        void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            const float * src0 = src[0];
            const float * src1 = src[1];
            float weight0 = weight[0], weight1 = weight[1];
            size_t j = 0;
            for (; j < aligned; j += 4)
            {
                dst[j + 0] = src0[j + 0] * weight0 + src1[j + 0] * weight1;
                dst[j + 1] = src0[j + 1] * weight0 + src1[j + 1] * weight1;
                dst[j + 2] = src0[j + 2] * weight0 + src1[j + 2] * weight1;
                dst[j + 3] = src0[j + 3] * weight0 + src1[j + 3] * weight1;
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * weight0 + src1[j] * weight1;
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                float weighti = weight[i];
                for (j = 0; j < aligned; j += 4)
                {
                    dst[j + 0] += srci[j + 0] * weighti;
                    dst[j + 1] += srci[j + 1] * weighti;
                    dst[j + 2] += srci[j + 2] * weighti;
                    dst[j + 3] += srci[j + 3] * weighti;
                }
                for (; j < size; ++j)
                    dst[j] += srci[j] * weighti;
            }
        }

        void SynetEltwiseLayerForwardMax(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            for (; j < aligned; j += 4)
            {
                dst[j + 0] = Simd::Max(src0[j + 0], src1[j + 0]);
                dst[j + 1] = Simd::Max(src0[j + 1], src1[j + 1]);
                dst[j + 2] = Simd::Max(src0[j + 2], src1[j + 2]);
                dst[j + 3] = Simd::Max(src0[j + 3], src1[j + 3]);
            }
            for (; j < size; ++j)
                dst[j] = Simd::Max(src0[j], src1[j]);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                for (j = 0; j < aligned; j += 4)
                {
                    dst[j + 0] = Simd::Max(dst[j + 0], srci[j + 0]);
                    dst[j + 1] = Simd::Max(dst[j + 1], srci[j + 1]);
                    dst[j + 2] = Simd::Max(dst[j + 2], srci[j + 2]);
                    dst[j + 3] = Simd::Max(dst[j + 3], srci[j + 3]);
                }
                for (; j < size; ++j)
                    dst[j] = Simd::Max(dst[j], srci[j]);
            }
        }

        void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForwardProduct(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForwardMax(src, count, size, dst);
                break;
            default:
                assert(0);
            }
        }

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t count, size_t size, const float * k, float * dst)
        {
            float k0 = k[0], k1 = k[1], k2 = k[2];
            Array32f sum(size, true), zero(size, true);

            for (size_t i = 0; i < half; ++i)
            {
                const float * pos = src + i * size;
                for (size_t j = 0; j < size; ++j)
                    sum[j] += Simd::Square(pos[j]);
            }

            for (size_t i = 0; i < count; ++i)
            {
                const float * pos = (i < count - half) ? src + half * size : zero.data;
                const float * neg = (i > half) ? src - (half + 1) * size : zero.data;
                for (size_t j = 0; j < size; ++j)
                {
                    sum[j] += Simd::Square(pos[j]);
                    sum[j] -= Simd::Square(neg[j]);
                    dst[j] = src[j] * Pow(k0 + k1 * sum[j], k2);
                }
                src += size;
                dst += size;
            }
        }

        void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            if (bias)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    float s = scale[i];
                    float b = bias[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        dst[j + 0] = src[j + 0] * s + b;
                        dst[j + 1] = src[j + 1] * s + b;
                        dst[j + 2] = src[j + 2] * s + b;
                        dst[j + 3] = src[j + 3] * s + b;
                    }
                    for (; j < size; ++j)
                        dst[j] = src[j] * s + b;
                    src += size;
                    dst += size;
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    float s = scale[i];
                    size_t j = 0;
                    for (; j < aligned; j += 4)
                    {
                        dst[j + 0] = src[j + 0] * s;
                        dst[j + 1] = src[j + 1] * s;
                        dst[j + 2] = src[j + 2] * s;
                        dst[j + 3] = src[j + 3] * s;
                    }
                    for (; j < size; ++j)
                        dst[j] = src[j] * s;
                    src += size;
                    dst += size;
                }
            }
        }
    }
}
