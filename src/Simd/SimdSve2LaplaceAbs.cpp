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
        namespace
        {
            SIMD_INLINE int16_t LaplaceAbsScalar(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x1, size_t x2)
            {
                int value = 8 * s1[x1] - (s0[x0] + s0[x1] + s0[x2] + s1[x0] + s1[x2] + s2[x0] + s2[x1] + s2[x2]);
                return (int16_t)Simd::Abs(value);
            }

            SIMD_INLINE svint16_t LaplaceAbsVector(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
            {
                svint16_t center = svlsl_n_s16_x(mask, svld1ub_s16(mask, s1 + 1), 3);
                svint16_t sum0 = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0 + 0), svld1ub_s16(mask, s0 + 1)), svld1ub_s16(mask, s0 + 2));
                svint16_t sum1 = svadd_s16_x(mask, svld1ub_s16(mask, s1 + 0), svld1ub_s16(mask, s1 + 2));
                svint16_t sum2 = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s2 + 0), svld1ub_s16(mask, s2 + 1)), svld1ub_s16(mask, s2 + 2));
                return svabs_s16_x(mask, svsub_s16_x(mask, center, svadd_s16_x(mask, svadd_s16_x(mask, sum0, sum1), sum2)));
            }

            SIMD_INLINE void LaplaceAbsVector(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, int16_t* dst, const svbool_t& mask)
            {
                svst1_s16(mask, dst, LaplaceAbsVector(s0, s1, s2, mask));
            }

            void LaplaceAbs(const uint8_t* src, size_t srcStride, size_t width, size_t height, int16_t* dst, size_t dstStride)
            {
                assert(width > 1);

                const size_t A = svcnth();
                const uint8_t * src0, * src1, * src2;
                for (size_t row = 0; row < height; ++row)
                {
                    src0 = src + srcStride * (row - 1);
                    src1 = src0 + srcStride;
                    src2 = src1 + srcStride;
                    if (row == 0)
                        src0 = src1;
                    if (row == height - 1)
                        src2 = src1;

                    dst[0] = LaplaceAbsScalar(src0, src1, src2, 0, 0, 1);
                    for (size_t col = 1; col < width - 1; col += A)
                        LaplaceAbsVector(src0 + col - 1, src1 + col - 1, src2 + col - 1, dst + col, svwhilelt_b16(col, width - 1));
                    dst[width - 1] = LaplaceAbsScalar(src0, src1, src2, width - 2, width - 1, width - 1);

                    dst += dstStride;
                }
            }
        }

        void LaplaceAbs(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            LaplaceAbs(src, srcStride, width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
        }
    }
#endif
}
