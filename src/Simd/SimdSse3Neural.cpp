/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"

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

        template<size_t coreX, size_t coreY> struct Convolution
        {
            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m128 & dst, __m128 * sums);
        };

        template<> struct Convolution<2, 2>
        {
            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m128 & dst, __m128 * sums)
            {
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m128 & dst, __m128 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 2);
            }
        };

        template<> struct Convolution<3, 3>
        {
            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m128 & dst, __m128 * sums)
            {
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
                sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m128 & dst, __m128 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 3);
                Sum<align>(src + stride * 2, dst, sums + 6);
            }
        };

        template<> struct Convolution<4, 4>
        {
            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m128 & dst, __m128 * sums)
            {
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
                sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
                sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(dst, Load<false>(src + 3)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m128 & dst, __m128 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 4);
                Sum<align>(src + stride * 2, dst, sums + 8);
                Sum<align>(src + stride * 3, dst, sums + 12);
            }
        };

        template<> struct Convolution<5, 5>
        {
            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m128 & dst, __m128 * sums)
            {
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
                sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
                sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(dst, Load<false>(src + 3)));
                sums[4] = _mm_add_ps(sums[4], _mm_mul_ps(dst, Load<align>(src + 4)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m128 & dst, __m128 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 5);
                Sum<align>(src + stride * 2, dst, sums + 10);
                Sum<align>(src + stride * 3, dst, sums + 15);
                Sum<align>(src + stride * 4, dst, sums + 20);
            }
        };

        SIMD_INLINE void Add4ExtractedSums(const __m128 * src, float * dst)
        {
            _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), _mm_hadd_ps(_mm_hadd_ps(src[0], src[1]), _mm_hadd_ps(src[2], src[3]))));
        }

        template <bool align, size_t coreX, size_t coreY> SIMD_INLINE void NeuralAddConvolutionSum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m128 tailMask = RightNotZero(width - alignedWidth);
            __m128 _sums[coreX*coreY];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    Convolution<coreX, coreY>::template Sum<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m128 _dst = _mm_and_ps(tailMask, Load<false>(dst + col));
                    Convolution<coreX, coreY>::template Sum<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            size_t i = 0, n = Simd::AlignLo(coreX*coreY, F);
            for (; i < n; i += F)
                Add4ExtractedSums(_sums + i, sums + i);
            for (; i < coreX*coreY; ++i)
                sums[i] += ExtractSum(_sums[i]);
        }

        void NeuralAddConvolution2x2Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution4x4Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
        }

        template <bool align> static SIMD_INLINE void AddProductSum4x4(const __m128 & a, size_t K, const float * b, __m128 * sums)
        {
            sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(a, Load<align>(b + 0 * K)));
            sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(a, Load<align>(b + 1 * K)));
            sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(a, Load<align>(b + 2 * K)));
            sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(a, Load<align>(b + 3 * K)));
        }

        template <bool align> static SIMD_INLINE void AddProductSum1x4(const __m128 & a, const float * b, __m128 & sum)
        {
            sum = _mm_add_ps(sum, _mm_mul_ps(a, Load<align>(b)));
        }

        template <bool align> void NeuralConvolutionForwardGemmNT(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
        {
            size_t alignedN = Simd::AlignLo(N, 4);
            size_t alignedK = Simd::AlignLo(K, 4);
            __m128 tailMask = RightNotZero(K - alignedK);
            for (size_t i = 0; i < M; ++i)
            {
                const float * pa = a + i*K;
                float * pc = c + i*N;
                size_t j = 0;
                for (; j < alignedN; j += 4)
                {
                    const float * pb = b + j*K;
                    __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                    for (size_t k = 0; k < alignedK; k += 4)
                    {
                        __m128 _a = Load<false>(pa + k);
                        AddProductSum4x4<align>(_a, K, pb + k, sums);
                    }
                    if (alignedK < K)
                    {
                        size_t k = K - 4;
                        __m128 _a = _mm_and_ps(tailMask, Load<false>(pa + k));
                        AddProductSum4x4<false>(_a, K, pb + k, sums);
                    }
                    Add4ExtractedSums(sums, pc + j);
                }
                for (; j < N; ++j)
                {
                    const float * pb = b + j*K;
                    __m128 sum = _mm_setzero_ps();
                    for (size_t k = 0; k < alignedK; k += 4)
                    {
                        __m128 _a = Load<false>(pa + k);
                        AddProductSum1x4<align>(_a, pb + k, sum);
                    }
                    if (alignedK < K)
                    {
                        size_t k = K - 4;
                        __m128 _a = _mm_and_ps(tailMask, Load<false>(pa + k));
                        AddProductSum1x4<false>(_a, pb + k, sum);
                    }
                    pc[j] += ExtractSum(sum);
                }
            }
        }

        void NeuralConvolutionForward(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
            const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY,
            void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add)
        {
            assert(dstWidth == (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1);
            assert(dstHeight == (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1);

            if (!add)
                memset(dst, 0, dstWidth*dstHeight*dstDepth*sizeof(float));

            float * temporal = NULL;
            void * internal = NULL;

            bool transpose = srcWidth*srcHeight <= 1024;

            if (kernelX == 1 && kernelY == 1 && !transpose)
                temporal = (float*)src;
            else
            {
                size_t required = dstWidth*dstHeight*srcDepth*kernelX*kernelY*sizeof(float);
                if (buffer != AlignHi(buffer, SIMD_ALIGN))
                    required += SIMD_ALIGN;
                if (buffer == NULL || size == NULL || *size < required)
                {
                    internal = Allocate(required);
                    if (size)
                        *size = required;
                    temporal = (float*)internal;
                }
                else
                    temporal = (float*)AlignHi(buffer, SIMD_ALIGN);

                if (transpose)
                    Base::NeuralConvolutionForwardConvertT(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, temporal);
                else
                    Base::NeuralConvolutionForwardConvertN(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, temporal);
            }

            size_t M = dstDepth, N = dstHeight*dstWidth, K = kernelX*kernelY*srcDepth;
            if (transpose)
            {
                if (Aligned(K, F))
                    NeuralConvolutionForwardGemmNT<true>(M, N, K, weight, temporal, dst);
                else
                    NeuralConvolutionForwardGemmNT<false>(M, N, K, weight, temporal, dst);
            }
            else
                Base::NeuralConvolutionForwardGemmNN(M, N, K, weight, temporal, dst);

            if (internal)
                Free(internal);
        }
    }
#endif// SIMD_SSE3_ENABLE
}
