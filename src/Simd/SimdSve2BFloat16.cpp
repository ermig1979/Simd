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
#include "Simd/SimdMemory.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void BFloat16ToFloat32(const uint16_t* src, const svbool_t& mask, float* dst)
        {
            svuint16_t zero = svdup_n_u16(0);
            svuint16_t _src = svld1_u16(mask, src);
            svst1_u16(mask, (uint16_t*)dst + 0 * svlen(svuint16_t()), svzip1_u16(zero, _src));
            svst1_u16(mask, (uint16_t*)dst + 1 * svlen(svuint16_t()), svzip2_u16(zero, _src));
        }

        void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst)
        {
            size_t A = svlen(svuint16_t()), sizeA = AlignLo(size, A);
            const svbool_t body = svptrue_b16();
            size_t i = 0;
            for (; i < sizeA; i += A)
                BFloat16ToFloat32(src + i, body, dst + i);
            for (; i < size; ++i)
                dst[i] = Base::BFloat16ToFloat32(src[i]);
        }
    }
#endif
}
