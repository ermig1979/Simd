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

        namespace Ncf
        {
            namespace Ver0
            {
                template <bool align> static SIMD_INLINE void AddProductSum1x4x4(const __m128 & a, size_t K, const float * b, __m128 * sums)
                {
                    sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(a, Load<align>(b + 0 * K)));
                    sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(a, Load<align>(b + 1 * K)));
                    sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(a, Load<align>(b + 2 * K)));
                    sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(a, Load<align>(b + 3 * K)));
                }

                template <bool align> static SIMD_INLINE void AddProductSum1x1x4(const __m128 & a, const float * b, __m128 & sum)
                {
                    sum = _mm_add_ps(sum, _mm_mul_ps(a, Load<align>(b)));
                }

                template <bool align> static SIMD_INLINE void AddProductSum2x4x4(const __m128 & a0, const __m128 & a1, size_t K, const float * b, __m128 * sums)
                {
                    __m128 b0 = Load<align>(b + 0 * K);
                    sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(a0, b0));
                    sums[4] = _mm_add_ps(sums[4], _mm_mul_ps(a1, b0));
                    __m128 b1 = Load<align>(b + 1 * K);
                    sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(a0, b1));
                    sums[5] = _mm_add_ps(sums[5], _mm_mul_ps(a1, b1));
                    __m128 b2 = Load<align>(b + 2 * K);
                    sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(a0, b2));
                    sums[6] = _mm_add_ps(sums[6], _mm_mul_ps(a1, b2));
                    __m128 b3 = Load<align>(b + 3 * K);
                    sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(a0, b3));
                    sums[7] = _mm_add_ps(sums[7], _mm_mul_ps(a1, b3));
                }

                template <bool align> static SIMD_INLINE void AddProductSum2x1x4(const __m128 & a0, const __m128 & a1, const float * b, __m128 * sums)
                {
                    sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(a0, Load<align>(b)));
                    sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(a1, Load<align>(b)));
                }

                template <bool align> void NeuralConvolutionForwardGemmNT(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    size_t M2 = Simd::AlignLo(M, 2);
                    size_t N4 = Simd::AlignLo(N, 4);
                    size_t K4 = Simd::AlignLo(K, 4);
                    __m128 tailMask = RightNotZero(K - K4);
                    size_t i = 0;
                    for (; i < M2; i += 2)
                    {
                        const float * pa0 = a + i*K;
                        const float * pa1 = a + i*K + K;
                        float * pc0 = c + i*N;
                        float * pc1 = c + i*N + N;
                        size_t j = 0;
                        for (; j < N4; j += 4)
                        {
                            const float * pb = b + j*K;
                            __m128 sums[8] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                            size_t k = 0;
                            for (; k < K4; k += 4)
                                AddProductSum2x4x4<align>(Load<false>(pa0 + k), Load<false>(pa1 + k), K, pb + k, sums);
                            if (K4 < K)
                            {
                                size_t k = K - 4;
                                __m128 _a0 = _mm_and_ps(tailMask, Load<false>(pa0 + k));
                                __m128 _a1 = _mm_and_ps(tailMask, Load<false>(pa1 + k));
                                AddProductSum2x4x4<false>(_a0, _a1, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0, pc0 + j);
                            Add4ExtractedSums(sums + 4, pc1 + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float * pb = b + j*K;
                            __m128 sums[2] = { _mm_setzero_ps(), _mm_setzero_ps() };
                            for (size_t k = 0; k < K4; k += 4)
                            {
                                __m128 _a0 = Load<false>(pa0 + k);
                                __m128 _a1 = Load<false>(pa1 + k);
                                AddProductSum2x1x4<align>(_a0, _a1, pb + k, sums);
                            }
                            if (K4 < K)
                            {
                                size_t k = K - 4;
                                __m128 _a0 = _mm_and_ps(tailMask, Load<false>(pa0 + k));
                                __m128 _a1 = _mm_and_ps(tailMask, Load<false>(pa1 + k));
                                AddProductSum2x1x4<false>(_a0, _a1, pb + k, sums);
                            }
                            pc0[j] += ExtractSum(sums[0]);
                            pc1[j] += ExtractSum(sums[1]);
                        }
                    }
                    for (; i < M; ++i)
                    {
                        const float * pa = a + i*K;
                        float * pc = c + i*N;
                        size_t j = 0;
                        for (; j < N4; j += 4)
                        {
                            const float * pb = b + j*K;
                            __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                            for (size_t k = 0; k < K4; k += 4)
                            {
                                __m128 _a = Load<false>(pa + k);
                                AddProductSum1x4x4<align>(_a, K, pb + k, sums);
                            }
                            if (K4 < K)
                            {
                                size_t k = K - 4;
                                __m128 _a = _mm_and_ps(tailMask, Load<false>(pa + k));
                                AddProductSum1x4x4<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0, pc + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float * pb = b + j*K;
                            __m128 sum = _mm_setzero_ps();
                            for (size_t k = 0; k < K4; k += 4)
                            {
                                __m128 _a = Load<false>(pa + k);
                                AddProductSum1x1x4<align>(_a, pb + k, sum);
                            }
                            if (K4 < K)
                            {
                                size_t k = K - 4;
                                __m128 _a = _mm_and_ps(tailMask, Load<false>(pa + k));
                                AddProductSum1x1x4<false>(_a, pb + k, sum);
                            }
                            pc[j] += ExtractSum(sum);
                        }
                    }
                }

                void NeuralConvolutionForwardGemmNT(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    if (Aligned(K, F))
                        NeuralConvolutionForwardGemmNT<true>(M, N, K, a, b, c);
                    else
                        NeuralConvolutionForwardGemmNT<false>(M, N, K, a, b, c);
                }
            }

            struct Manager
            {
                size_t size;

                void (*prepare) (const float * src, ptrdiff_t srcWidth, ptrdiff_t srcHeight, ptrdiff_t srcDepth, ptrdiff_t kernelX, ptrdiff_t kernelY, 
                    ptrdiff_t padX, ptrdiff_t padY, ptrdiff_t strideX, ptrdiff_t strideY, ptrdiff_t dilationX, ptrdiff_t dilationY, float * dst);

                void (*execute)(size_t M, size_t N, size_t K, const float * a, const float * b, float * c);

                Manager(size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                    : size(0)
                    , prepare(0)
                    , execute(0)
                {
                    if (dstWidth*dstHeight / kernelX <= 2000)
                    {
                        size = dstWidth*dstHeight*srcDepth*kernelX*kernelY*sizeof(float);
                        prepare = Base::NeuralConvolutionForwardConvertT;
                        execute = Ver0::NeuralConvolutionForwardGemmNT;
                    }
                    else
                    {
                        if (kernelX > 1 || kernelY > 1)
                        {
                            size = dstWidth*dstHeight*srcDepth*kernelX*kernelY*sizeof(float);
                            prepare = Base::NeuralConvolutionForwardConvertN;
                        }
                        execute = Base::NeuralConvolutionForwardGemmNN;
                    }
                }

            private:
            };
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

            Ncf::Manager manager(srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth);

            if(manager.prepare)
            {
                if (buffer != AlignHi(buffer, SIMD_ALIGN))
                    manager.size += SIMD_ALIGN;
                if (buffer == NULL || size == NULL || *size < manager.size)
                {
                    internal = Allocate(manager.size);
                    if (size)
                        *size = manager.size;
                    temporal = (float*)internal;
                }
                else
                    temporal = (float*)AlignHi(buffer, SIMD_ALIGN);

                manager.prepare(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, temporal);
            }
            else
                temporal = (float*)src;

            manager.execute(dstDepth, dstHeight*dstWidth, kernelX*kernelY*srcDepth, weight, temporal, dst);

            if (internal)
                Free(internal);
        }
    }
#endif// SIMD_SSE3_ENABLE
}
