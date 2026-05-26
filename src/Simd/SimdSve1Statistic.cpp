/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar,
*               2018-2018 Radchenko Andrey.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SVE_ENABLE    
    namespace Sve
    {
        SIMD_INLINE void UpdateStatistic(const uint8_t* src, svbool_t mask, svuint8_t _1, svuint8_t& min, svuint8_t& max, svuint32_t& sum)
        {
            svuint8_t val = svld1_u8(mask, src);
            min = svmin_u8_m(mask, min, val);
            max = svmax_u8_m(mask, max, val);
            sum = svdot_u32(sum, val, _1);
        }

        void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t * min, uint8_t * max, uint8_t * average)
        {
            assert(width*height);

            size_t A = svlen(svuint8_t());
            size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b32();
            const svbool_t tail = svwhilelt_b8(widthA, width);

            svuint8_t _1 = svdup_n_u8(1);
            svuint8_t _min = svdup_n_u8(255);
            svuint8_t _max = svdup_n_u8(0);
            uint64_t sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                svuint32_t _sum = svdup_n_u32(0);
                for (; col < widthA; col += A)
                    UpdateStatistic(src + col, body, _1, _min, _max, _sum);
                if (widthA < width)
                    UpdateStatistic(src + col, tail, _1, _min, _max, _sum);
                sum += svaddv_u32(svptrue_b32(), _sum);
                src += stride;
            }

            *min = svminv_u8(svptrue_b32(), _min);
            *max = svmaxv_u8(svptrue_b32(), _max);
            *average = (uint8_t)((sum + width*height / 2) / (width*height));
        }
    }
#endif
}
