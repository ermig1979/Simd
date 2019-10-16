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
#include "Simd/SimdArray.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
    namespace Base
    {
        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            float _alpha = alpha[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetElu32f(src[i + 0], _alpha);
                dst[i + 1] = SynetElu32f(src[i + 1], _alpha);
                dst[i + 2] = SynetElu32f(src[i + 2], _alpha);
                dst[i + 3] = SynetElu32f(src[i + 3], _alpha);
            }
            for (; i < size; ++i)
                dst[i] = SynetElu32f(src[i], _alpha);
        }

        //---------------------------------------------------------------------

        void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            float _shift = shift[0];
            float _scale = scale[0];
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetHswish32f(src[i + 0], _shift, _scale);
                dst[i + 1] = SynetHswish32f(src[i + 1], _shift, _scale);
                dst[i + 2] = SynetHswish32f(src[i + 2], _shift, _scale);
                dst[i + 3] = SynetHswish32f(src[i + 3], _shift, _scale);
            }
            for (; i < size; ++i)
                dst[i] = SynetHswish32f(src[i], _shift, _scale);
        }

        //---------------------------------------------------------------------

        void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            float min = *lower;
            float max = *upper;
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = Simd::RestrictRange(src[i + 0], min, max);
                dst[i + 1] = Simd::RestrictRange(src[i + 1], min, max);
                dst[i + 2] = Simd::RestrictRange(src[i + 2], min, max);
                dst[i + 3] = Simd::RestrictRange(src[i + 3], min, max);
            }
            for (; i < size; ++i)
                dst[i] = Simd::RestrictRange(src[i], min, max);
        }
    }
}
