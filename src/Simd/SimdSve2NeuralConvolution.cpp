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
        SIMD_INLINE void AddMultiplied(const svbool_t& mask, const float* src, const svfloat32_t& value, float* dst)
        {
            svst1_f32(mask, dst, svmla_f32_m(mask, svld1_f32(mask, dst), svld1_f32(mask, src), value));
        }

        SIMD_INLINE void AddMultiplied(const float* src, size_t size, const svfloat32_t& value, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();

            for (; i + QF <= size; i += QF)
            {
                AddMultiplied(body, src + i + 0 * F, value, dst + i + 0 * F);
                AddMultiplied(body, src + i + 1 * F, value, dst + i + 1 * F);
                AddMultiplied(body, src + i + 2 * F, value, dst + i + 2 * F);
                AddMultiplied(body, src + i + 3 * F, value, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                AddMultiplied(body, src + i, value, dst + i);
            if (i < size)
                AddMultiplied(svwhilelt_b32(i, size), src + i, value, dst + i);
        }

        void NeuralAddConvolution2x2Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            const svfloat32_t weight0 = svdup_n_f32(weights[0]);
            const svfloat32_t weight1 = svdup_n_f32(weights[1]);
            const svfloat32_t weight2 = svdup_n_f32(weights[2]);
            const svfloat32_t weight3 = svdup_n_f32(weights[3]);

            for (size_t row = 0; row < height; ++row)
            {
                float* dst0 = dst;
                float* dst1 = dst + dstStride;

                AddMultiplied(src, width, weight0, dst0 + 0);
                AddMultiplied(src, width, weight1, dst0 + 1);
                AddMultiplied(src, width, weight2, dst1 + 0);
                AddMultiplied(src, width, weight3, dst1 + 1);

                src += srcStride;
                dst += dstStride;
            }
        }
    }
#endif
}
