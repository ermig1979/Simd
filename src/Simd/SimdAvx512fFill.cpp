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

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        void Fill32f(float * dst, size_t size, const float * value)
        {
            if (value == 0 || value[0] == 0)
                memset(dst, 0, size * sizeof(float));
            else
            {
                float v = value[0];
                const float * nose = (float*)AlignHi(dst, F * sizeof(float));
                for (; dst < nose && size; --size)
                    *dst++ = v;
                const float * end = dst + size;
                const float * endF = dst + AlignLo(size, F);
                const float * endQF = dst + AlignLo(size, QF);
                size_t i = 0;
                __m512 _v = _mm512_set1_ps(v);
                for (; dst < endQF; dst += QF)
                {
                    _mm512_storeu_ps(dst + 0 * F, _v);
                    _mm512_storeu_ps(dst + 1 * F, _v);
                    _mm512_storeu_ps(dst + 2 * F, _v);
                    _mm512_storeu_ps(dst + 3 * F, _v);
                }
                for (; dst < endF; dst += F)
                    _mm512_storeu_ps(dst, _v);
                for (; dst < end;)
                    *dst++ = v;
            }
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
