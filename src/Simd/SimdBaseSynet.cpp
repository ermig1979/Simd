/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

namespace Simd
{
    namespace Base
    {
        void SynetAddBias(const float * bias, size_t count, size_t size, float * dst)
        {
            register size_t aligned = Simd::AlignLo(size, 4);
            for (size_t i = 0; i < count; ++i)
            {
                register float value = bias[i];
                register size_t j = 0;
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

        void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst)
        {
            register size_t aligned = Simd::AlignLo(size, 4);
            if (bias)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    register float s = scale[i];
                    register float b = bias[i];
                    register size_t j = 0;
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
                    register float s = scale[i];
                    register size_t j = 0;
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
