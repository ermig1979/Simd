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
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdNeural.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        __m512i K32_PERMUTE_2_0 = SIMD_MM512_SETR_EPI32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        __m512i K32_PERMUTE_2_1 = SIMD_MM512_SETR_EPI32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        __m512i K32_PERMUTE_2_2 = SIMD_MM512_SETR_EPI32(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0);

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max1x3(const float* src, size_t stride)
        {
            return _mm512_max_ps(_mm512_max_ps(Load<align>(src), Load<align>(src + stride)), Load<align>(src + 2 * stride));
        }

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max3x3(const float* src, size_t stride)
        {
            __m512 s0 = Pooling2x2Max1x3<align>(src + 0, stride);
            __m512 sf = Pooling2x2Max1x3<align>(src + F, stride);
            __m512 p0 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_0, sf);
            __m512 p1 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_1, sf);
            __m512 p2 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_2, sf);
            return _mm512_max_ps(_mm512_max_ps(p0, p1), p2);
        }

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max1x2(const float* src, size_t stride)
        {
            return _mm512_max_ps(Load<align>(src), Load<align>(src + stride));
        }

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max3x2(const float* src, size_t stride)
        {
            __m512 s0 = Pooling2x2Max1x2<align>(src + 0, stride);
            __m512 sf = Pooling2x2Max1x2<align>(src + F, stride);
            __m512 p0 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_0, sf);
            __m512 p1 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_1, sf);
            __m512 p2 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_2, sf);
            return _mm512_max_ps(_mm512_max_ps(p0, p1), p2);
        }

        template <bool align> void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            size_t step = DF - 2;
            size_t alignedWidth = width / step * step;
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += step)
                    Store<false, true>(dst + (col >> 1), Pooling2x2Max3x3<false>(src + col, srcStride), __mmask16(0x7FFF));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - step;
                    Store<false, true>(dst + (col >> 1), Pooling2x2Max3x3<false>(src + col, srcStride), __mmask16(0x7FFF));
                }
                if (width - widthEven)
                    Sse41::Max2x3s(src + widthEven, srcStride, dst + (widthEven >> 1));
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += step)
                    Store<false, true>(dst + (col >> 1), Pooling2x2Max3x2<false>(src + col, srcStride), __mmask16(0x7FFF));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - step;
                    Store<false, true>(dst + (col >> 1), Pooling2x2Max3x2<false>(src + col, srcStride), __mmask16(0x7FFF));
                }
                if (width - widthEven)
                    Sse41::Max2x2s(src + widthEven, srcStride, dst + (widthEven >> 1));
            }
        }

        void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling2x2Max3x3<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling2x2Max3x3<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
