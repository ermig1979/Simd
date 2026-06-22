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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void HogDeinterleave(const float* src, const svuint32_t& offsets, const svbool_t& mask, float* dst)
        {
            svst1_f32(mask, dst, svld1_gather_u32index_f32(mask, src, offsets));
        }

        void HogDeinterleave(const float* src, size_t srcStride, size_t width, size_t height, size_t count, float** dst, size_t dstStride)
        {
            assert(width >= svcntw());

            size_t F = svcntw(), alignedWidth = AlignLo(width, F);
            const svbool_t body = svptrue_b32();
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), (uint32_t)count);

            for (size_t row = 0; row < height; ++row)
            {
                size_t rowOffset = row * dstStride, col = 0;
                for (; col < alignedWidth; col += F)
                {
                    const float* s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < count; ++i)
                        HogDeinterleave(s + i, offsets, body, dst[i] + offset);
                }
                if (col < width)
                {
                    const float* s = src + count * col;
                    size_t offset = rowOffset + col;
                    svbool_t tail = svwhilelt_b32(col, width);
                    for (size_t i = 0; i < count; ++i)
                        HogDeinterleave(s + i, offsets, tail, dst[i] + offset);
                }
                src += srcStride;
            }
        }
    }
#endif
}
