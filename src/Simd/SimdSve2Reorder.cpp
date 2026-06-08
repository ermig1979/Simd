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
#include "Simd/SimdReorder.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void Reorder16bit(const uint16_t * src, const svbool_t & mask, uint16_t * dst)
        {
            svst1_u16(mask, dst, svrevb_u16_x(mask, svld1_u16(mask, src)));
        }

        void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 2 == 0);

            size_t A = svlen(svuint16_t()), size16 = size / 2, sizeA = AlignLo(size16, A);
            const uint16_t * s = (const uint16_t*)src;
            uint16_t * d = (uint16_t*)dst;
            const svbool_t body = svptrue_b16();
            size_t i = 0;
            for (; i < sizeA; i += A)
                Reorder16bit(s + i, body, d + i);
            if (i < size16)
                Reorder16bit(s + i, svwhilelt_b16(i, size16), d + i);
        }
    }
#endif
}
