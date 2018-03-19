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
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE uint8_t Float32ToUint8(float value, float lower, float upper, float boost)
        {
            return uint8_t((Simd::Min(Simd::Max(value, lower), upper) - lower)*boost);
        }

        void Float32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst)
        {
            float _lower = lower[0], _upper = upper[0], boost = 255.0f / (upper[0] - lower[0]);
            size_t alignedSize = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                dst[i + 0] = Float32ToUint8(src[i + 0], _lower, _upper, boost);
                dst[i + 1] = Float32ToUint8(src[i + 1], _lower, _upper, boost);
                dst[i + 2] = Float32ToUint8(src[i + 2], _lower, _upper, boost);
                dst[i + 3] = Float32ToUint8(src[i + 3], _lower, _upper, boost);
            }
            for (; i < size; ++i)
                dst[i] = Float32ToUint8(src[i], _lower, _upper, boost);
        }

        SIMD_INLINE float Uint8ToFloat32(int value, float lower, float boost)
        {
            return value*boost + lower;
        }

        void Uint8ToFloat32(const uint8_t * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            float _lower = lower[0], boost = (upper[0] - lower[0]) / 255.0f;
            size_t alignedSize = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                dst[i + 0] = Uint8ToFloat32(src[i + 0], _lower, boost);
                dst[i + 1] = Uint8ToFloat32(src[i + 1], _lower, boost);
                dst[i + 2] = Uint8ToFloat32(src[i + 2], _lower, boost);
                dst[i + 3] = Uint8ToFloat32(src[i + 3], _lower, boost);
            }
            for (; i < size; ++i)
                dst[i] = Uint8ToFloat32(src[i], _lower, boost);
        }

        void CosineDistance32f(const float * a, const float * b, size_t size, float * distance)
        {
            float aa = 0, ab = 0, bb = 0;
            for (size_t i = 0; i < size; ++i)
            {
                float _a = a[i];
                float _b = b[i];
                aa += _a * _a;
                ab += _a * _b;
                bb += _b * _b;
            }
            *distance = 1.0f - ab / ::sqrt(aa*bb);
        }
    }
}
