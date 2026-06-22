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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        namespace
        {
            struct Buffer
            {
                const int size;
                float* cos;
                float* sin;
                int* index;
                float* value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization / 2)
                {
                    _p = Allocate(width * (sizeof(int) + sizeof(float)) + sizeof(float) * 2 * size);
                    index = (int*)_p;
                    value = (float*)index + width;
                    cos = value + width;
                    sin = cos + size;
                    for (int i = 0; i < size; ++i)
                    {
                        cos[i] = (float)::cos(i * M_PI / size);
                        sin[i] = (float)::sin(i * M_PI / size);
                    }
                }

                ~Buffer()
                {
                    Free(_p);
                }

            private:
                void* _p;
            };
        }

        SIMD_INLINE svbool_t HogTailMask(size_t col, size_t end, const svuint32_t& offsets)
        {
            const svbool_t mask = svptrue_b32();
            return svcmplt_n_u32(mask, svadd_n_u32_x(mask, offsets, (uint32_t)col), (uint32_t)end);
        }

        SIMD_INLINE void HogDirectionHistograms32(const svint32_t& dx, const svint32_t& dy, Buffer& buffer, size_t col, const svuint32_t& offsets, const svbool_t& mask)
        {
            svfloat32_t _dx = svcvt_f32_s32_x(mask, dx);
            svfloat32_t _dy = svcvt_f32_s32_x(mask, dy);
            svfloat32_t bestDot = svdup_n_f32(0.0f);
            svint32_t bestIndex = svdup_n_s32(0);
            for (int i = 0; i < buffer.size; ++i)
            {
                svfloat32_t dot = svmla_f32_x(mask, svmul_n_f32_x(mask, _dx, buffer.cos[i]), _dy, svdup_n_f32(buffer.sin[i]));
                svbool_t positive = svcmpgt_f32(mask, dot, bestDot);
                bestDot = svmax_f32_x(mask, dot, bestDot);
                bestIndex = svsel_s32(positive, svdup_n_s32(i), bestIndex);

                dot = svneg_f32_x(mask, dot);
                svbool_t negative = svcmpgt_f32(mask, dot, bestDot);
                bestDot = svmax_f32_x(mask, dot, bestDot);
                bestIndex = svsel_s32(negative, svdup_n_s32(buffer.size + i), bestIndex);
            }
            svst1_scatter_u32index_s32(mask, buffer.index + col, offsets, bestIndex);
            svst1_scatter_u32index_f32(mask, buffer.value + col, offsets, svsqrt_f32_x(mask, svmla_f32_x(mask, svmul_f32_x(mask, _dx, _dx), _dy, _dy)));
        }

        SIMD_INLINE void HogDirectionHistograms16(const svint16_t& dx, const svint16_t& dy, Buffer& buffer, size_t col, const svuint32_t& loOffsets, const svuint32_t& hiOffsets, const svbool_t& maskLo, const svbool_t& maskHi)
        {
            HogDirectionHistograms32(svmovlb_s32(dx), svmovlb_s32(dy), buffer, col, loOffsets, maskLo);
            HogDirectionHistograms32(svmovlt_s32(dx), svmovlt_s32(dy), buffer, col, hiOffsets, maskHi);
        }

        SIMD_INLINE svint16_t HogDifferenceLo(const svuint8_t& a, const svuint8_t& b)
        {
            svbool_t mask = svptrue_b16();
            return svsub_s16_x(mask, svreinterpret_s16_u16(svmovlb_u16(a)), svreinterpret_s16_u16(svmovlb_u16(b)));
        }

        SIMD_INLINE svint16_t HogDifferenceHi(const svuint8_t& a, const svuint8_t& b)
        {
            svbool_t mask = svptrue_b16();
            return svsub_s16_x(mask, svreinterpret_s16_u16(svmovlt_u16(a)), svreinterpret_s16_u16(svmovlt_u16(b)));
        }

        SIMD_INLINE void HogDirectionHistograms(const uint8_t* src, size_t stride, Buffer& buffer, size_t col, size_t end)
        {
            const uint8_t* s = src + col;
            svbool_t mask = svwhilelt_b8(col, end);
            const svuint32_t offsets0 = svindex_u32(0, 4);
            const svuint32_t offsets1 = svindex_u32(2, 4);
            const svuint32_t offsets2 = svindex_u32(1, 4);
            const svuint32_t offsets3 = svindex_u32(3, 4);
            svuint8_t t = svld1_u8(mask, s - stride);
            svuint8_t l = svld1_u8(mask, s - 1);
            svuint8_t r = svld1_u8(mask, s + 1);
            svuint8_t b = svld1_u8(mask, s + stride);
            HogDirectionHistograms16(HogDifferenceLo(r, l), HogDifferenceLo(b, t), buffer, col, offsets0, offsets1, HogTailMask(col, end, offsets0), HogTailMask(col, end, offsets1));
            HogDirectionHistograms16(HogDifferenceHi(r, l), HogDifferenceHi(b, t), buffer, col, offsets2, offsets3, HogTailMask(col, end, offsets2), HogTailMask(col, end, offsets3));
        }

        void HogDirectionHistograms(const uint8_t* src, size_t stride, size_t width, size_t height,
            size_t cellX, size_t cellY, size_t quantization, float* histograms)
        {
            assert(width % cellX == 0 && height % cellY == 0 && quantization % 2 == 0);

            Buffer buffer(width, quantization);

            memset(histograms, 0, quantization * (width / cellX) * (height / cellY) * sizeof(float));

            size_t A = svcntb(), end = width - 1;
            for (size_t row = 1; row < height - 1; ++row)
            {
                const uint8_t* s = src + stride * row;
                for (size_t col = 1; col < end; col += A)
                    HogDirectionHistograms(s, stride, buffer, col, end);
                Base::AddRowToHistograms(buffer.index, buffer.value, row, width, height, cellX, cellY, quantization, histograms);
            }
        }

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
