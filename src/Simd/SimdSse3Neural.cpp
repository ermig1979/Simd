/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE3_ENABLE    
    namespace Sse3
    {
#if defined(_MSC_VER) && _MSC_VER >= 1800  && _MSC_VER < 1900 // Visual Studio 2013 compiler bug       
        const size_t F = Sse::F;
        using Sse::Load;
        using Sse::RightNotZero;
#endif

        template <bool align> SIMD_INLINE void AddMultiplied3(const float * src, const __m128 & dst, __m128 * sums)
        {
            sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
            sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
            sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
        }

        template <bool align> SIMD_INLINE void AddMultiplied3x3(const float * src, size_t stride, const __m128 & dst, __m128 * sums)
        {
            AddMultiplied3<align>(src + stride*0, dst, sums + 0);
            AddMultiplied3<align>(src + stride*1, dst, sums + 3);
            AddMultiplied3<align>(src + stride*2, dst, sums + 6);
        }

        SIMD_INLINE void Add4ExtractedSums(const __m128 * src, float * dst)
        {
            _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), _mm_hadd_ps(_mm_hadd_ps(src[0], src[1]), _mm_hadd_ps(src[2], src[3]))));
        }

        template <bool align> void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m128 tailMask = RightNotZero(width - alignedWidth);
            __m128 _sums[9];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    AddMultiplied3x3<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m128 _dst = _mm_and_ps(tailMask, Load<false>(dst + col));
                    AddMultiplied3x3<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            Add4ExtractedSums(_sums + 0, sums + 0);
            Add4ExtractedSums(_sums + 4, sums + 4);
            sums[8] += ExtractSum(_sums[8]);
        }

        void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution3x3Sum<true>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolution3x3Sum<false>(src, srcStride, dst, dstStride, width, height, sums);
        }

        template <bool align> SIMD_INLINE void AddMultiplied5(const float * src, const __m128 & dst, __m128 * sums)
        {
            sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
            sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
            sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
            sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(dst, Load<false>(src + 3)));
            sums[4] = _mm_add_ps(sums[4], _mm_mul_ps(dst, Load<false>(src + 4)));
        }

        template <bool align> SIMD_INLINE void AddMultiplied5x5(const float * src, size_t stride, const __m128 & dst, __m128 * sums)
        {
            AddMultiplied5<align>(src + stride * 0, dst, sums + 0);
            AddMultiplied5<align>(src + stride * 1, dst, sums + 5);
            AddMultiplied5<align>(src + stride * 2, dst, sums + 10);
            AddMultiplied5<align>(src + stride * 3, dst, sums + 15);
            AddMultiplied5<align>(src + stride * 4, dst, sums + 20);
        }

        template <bool align> void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m128 tailMask = RightNotZero(width - alignedWidth);
            __m128 _sums[25];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    AddMultiplied5x5<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m128 _dst = _mm_and_ps(tailMask, Load<false>(dst + col));
                    AddMultiplied5x5<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            Add4ExtractedSums(_sums + 0, sums + 0);
            Add4ExtractedSums(_sums + 4, sums + 4);
            Add4ExtractedSums(_sums + 8, sums + 8);
            Add4ExtractedSums(_sums + 12, sums + 12);
            Add4ExtractedSums(_sums + 16, sums + 16);
            Add4ExtractedSums(_sums + 20, sums + 20);
            sums[24] += ExtractSum(_sums[24]);
        }

        void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution5x5Sum<true>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolution5x5Sum<false>(src, srcStride, dst, dstStride, width, height, sums);
        }
    }
#endif// SIMD_SSE3_ENABLE
}
