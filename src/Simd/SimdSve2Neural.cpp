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
        SIMD_INLINE float Max2(const float* src)
        {
            return Simd::Max(src[0], src[1]);
        }

        SIMD_INLINE float Max2x2(const float* src, size_t stride)
        {
            return Simd::Max(Max2(src), Max2(src + stride));
        }

        SIMD_INLINE float Max2x3(const float* src, size_t stride)
        {
            return Simd::Max(Max2(src), Simd::Max(Max2(src + stride), Max2(src + 2 * stride)));
        }

        SIMD_INLINE svfloat32_t Pooling1x1Max3x1(const svbool_t& mask, const float* src)
        {
            svfloat32_t src0 = svld1_f32(mask, src + 0);
            svfloat32_t src1 = svld1_f32(mask, src + 1);
            svfloat32_t src2 = svld1_f32(mask, src + 2);
            return svmax_f32_x(mask, svmax_f32_x(mask, src0, src1), src2);
        }

        SIMD_INLINE svfloat32_t Pooling1x1Max3x2(const svbool_t& mask, const float* src, size_t stride)
        {
            svfloat32_t src0 = Pooling1x1Max3x1(mask, src);
            svfloat32_t src1 = Pooling1x1Max3x1(mask, src + stride);
            return svmax_f32_x(mask, src0, src1);
        }

        SIMD_INLINE svfloat32_t Pooling1x1Max3x3(const svbool_t& mask, const float* src, size_t stride)
        {
            svfloat32_t src0 = Pooling1x1Max3x1(mask, src);
            svfloat32_t src1 = Pooling1x1Max3x1(mask, src + stride);
            svfloat32_t src2 = Pooling1x1Max3x1(mask, src + 2 * stride);
            return svmax_f32_x(mask, svmax_f32_x(mask, src0, src1), src2);
        }

        SIMD_INLINE void Pooling1x1Max3x2(const float* src, size_t stride, size_t width, float* dst)
        {
            size_t F = svcntw(), bodyWidth = width - 1;
            for (size_t col = 1; col < bodyWidth; col += F)
            {
                svbool_t mask = svwhilelt_b32(col, bodyWidth);
                svst1_f32(mask, dst + col, Pooling1x1Max3x2(mask, src + col - 1, stride));
            }
        }

        SIMD_INLINE void Pooling1x1Max3x3(const float* src, size_t stride, size_t width, float* dst)
        {
            size_t F = svcntw(), bodyWidth = width - 1;
            for (size_t col = 1; col < bodyWidth; col += F)
            {
                svbool_t mask = svwhilelt_b32(col, bodyWidth);
                svst1_f32(mask, dst + col, Pooling1x1Max3x3(mask, src + col - 1, stride));
            }
        }

        void NeuralPooling1x1Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            assert(width > 1 && height > 1);

            dst[0] = Max2x2(src, srcStride);
            Pooling1x1Max3x2(src, srcStride, width, dst);
            dst[width - 1] = Max2x2(src + width - 2, srcStride);
            dst += dstStride;

            for (size_t row = 1; row < height - 1; ++row)
            {
                const float* s = src + (row - 1) * srcStride;
                dst[0] = Max2x3(s, srcStride);
                Pooling1x1Max3x3(s, srcStride, width, dst);
                dst[width - 1] = Max2x3(s + width - 2, srcStride);
                dst += dstStride;
            }

            src += (height - 2) * srcStride;
            dst[0] = Max2x2(src, srcStride);
            Pooling1x1Max3x2(src, srcStride, width, dst);
            dst[width - 1] = Max2x2(src + width - 2, srcStride);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max1x3(const svbool_t& mask, const float* src, const svuint32_t& offsets0, const svuint32_t& offsets1, const svuint32_t& offsets2)
        {
            svfloat32_t src0 = svld1_gather_u32index_f32(mask, src, offsets0);
            svfloat32_t src1 = svld1_gather_u32index_f32(mask, src, offsets1);
            svfloat32_t src2 = svld1_gather_u32index_f32(mask, src, offsets2);
            return svmax_f32_x(mask, svmax_f32_x(mask, src0, src1), src2);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max1x2(const svbool_t& mask, const float* src, const svuint32_t& offsets0, const svuint32_t& offsets1)
        {
            svfloat32_t src0 = svld1_gather_u32index_f32(mask, src, offsets0);
            svfloat32_t src1 = svld1_gather_u32index_f32(mask, src, offsets1);
            return svmax_f32_x(mask, src0, src1);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max2x2(const svbool_t& mask, const float* src, size_t stride, const svuint32_t& offsets0, const svuint32_t& offsets1)
        {
            svfloat32_t src0 = Pooling2x2Max1x2(mask, src, offsets0, offsets1);
            svfloat32_t src1 = Pooling2x2Max1x2(mask, src + stride, offsets0, offsets1);
            return svmax_f32_x(mask, src0, src1);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max3x2(const svbool_t& mask, const float* src, size_t stride, const svuint32_t& offsets0, const svuint32_t& offsets1, const svuint32_t& offsets2)
        {
            svfloat32_t src0 = Pooling2x2Max1x3(mask, src, offsets0, offsets1, offsets2);
            svfloat32_t src1 = Pooling2x2Max1x3(mask, src + stride, offsets0, offsets1, offsets2);
            return svmax_f32_x(mask, src0, src1);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max3x3(const svbool_t& mask, const float* src, size_t stride, const svuint32_t& offsets0, const svuint32_t& offsets1, const svuint32_t& offsets2)
        {
            svfloat32_t src0 = Pooling2x2Max1x3(mask, src, offsets0, offsets1, offsets2);
            svfloat32_t src1 = Pooling2x2Max1x3(mask, src + stride, offsets0, offsets1, offsets2);
            svfloat32_t src2 = Pooling2x2Max1x3(mask, src + 2 * stride, offsets0, offsets1, offsets2);
            return svmax_f32_x(mask, svmax_f32_x(mask, src0, src1), src2);
        }

        void NeuralPooling2x2Max2x2(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            size_t F = svcntw(), heightEven = Simd::AlignLo(height, 2), widthEven = Simd::AlignLo(width, 2), dstWidthEven = widthEven >> 1;
            const svuint32_t offsets0 = svindex_u32(0, 2);
            const svuint32_t offsets1 = svindex_u32(1, 2);

            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < dstWidthEven; col += F)
                {
                    svbool_t mask = svwhilelt_b32(col, dstWidthEven);
                    svst1_f32(mask, dst + col, Pooling2x2Max2x2(mask, src + 2 * col, srcStride, offsets0, offsets1));
                }
                if (width - widthEven)
                    dst[dstWidthEven] = Simd::Max(src[widthEven], src[widthEven + srcStride]);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < dstWidthEven; col += F)
                {
                    svbool_t mask = svwhilelt_b32(col, dstWidthEven);
                    svst1_f32(mask, dst + col, Pooling2x2Max1x2(mask, src + 2 * col, offsets0, offsets1));
                }
                if (width - widthEven)
                    dst[dstWidthEven] = src[widthEven];
            }
        }

        void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            size_t F = svcntw(), heightEven = Simd::AlignLo(height, 2), widthEven = Simd::AlignLo(width, 2), dstWidthEven = widthEven >> 1;
            const svuint32_t offsets0 = svindex_u32(0, 2);
            const svuint32_t offsets1 = svindex_u32(1, 2);
            const svuint32_t offsets2 = svindex_u32(2, 2);

            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < dstWidthEven; col += F)
                {
                    svbool_t mask = svwhilelt_b32(col, dstWidthEven);
                    svst1_f32(mask, dst + col, Pooling2x2Max3x3(mask, src + 2 * col, srcStride, offsets0, offsets1, offsets2));
                }
                if (width - widthEven)
                    dst[dstWidthEven] = Max2x3(src + widthEven, srcStride);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < dstWidthEven; col += F)
                {
                    svbool_t mask = svwhilelt_b32(col, dstWidthEven);
                    svst1_f32(mask, dst + col, Pooling2x2Max3x2(mask, src + 2 * col, srcStride, offsets0, offsets1, offsets2));
                }
                if (width - widthEven)
                    dst[dstWidthEven] = Max2x2(src + widthEven, srcStride);
            }
        }
    }
#endif
}
