/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template<size_t kernelX, size_t kernelY> struct Convolution
        {
            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m128 & dst, __m128 * sums);
        };

        template<> struct Convolution<2, 2>
        {
            template <bool align> static SIMD_INLINE __m128 Convolution2(const float * src, const __m128 * weights)
            {
                return _mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]),
                    _mm_mul_ps(Load<false>(src + 1), weights[1]));
            }

            template<bool align> static SIMD_INLINE __m128 Forward(const float * src, size_t stride, const __m128 * weights)
            {
                return _mm_add_ps(Convolution2<align>(src, weights),
                    Convolution2<align>(src + stride, weights + 2));
            }

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
            template <bool align> static SIMD_INLINE __m128 Convolution3(const float * src, const __m128 * weights)
            {
                return _mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]),
                    _mm_add_ps(_mm_mul_ps(Load<false>(src + 1), weights[1]),
                        _mm_mul_ps(Load<false>(src + 2), weights[2])));
            }

            template<bool align> static SIMD_INLINE __m128 Forward(const float * src, size_t stride, const __m128 * weights)
            {
                return _mm_add_ps(Convolution3<align>(src, weights),
                    _mm_add_ps(Convolution3<align>(src + stride, weights + 3),
                        Convolution3<align>(src + 2 * stride, weights + 6)));
            }

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
            template <bool align> static SIMD_INLINE __m128 Convolution4(const float * src, const __m128 * weights)
            {
                return _mm_add_ps(_mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]), _mm_mul_ps(Load<false>(src + 1), weights[1])),
                    _mm_add_ps(_mm_mul_ps(Load<false>(src + 2), weights[2]), _mm_mul_ps(Load<false>(src + 3), weights[3])));
            }

            template<bool align> static SIMD_INLINE __m128 Forward(const float * src, size_t stride, const __m128 * weights)
            {
                return _mm_add_ps(_mm_add_ps(Convolution4<align>(src, weights),
                    Convolution4<align>(src + stride, weights + 4)),
                    _mm_add_ps(Convolution4<align>(src + 2 * stride, weights + 8),
                        Convolution4<align>(src + 3 * stride, weights + 12)));
            }

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
            template <bool align> static SIMD_INLINE __m128 Convolution5(const float * src, const __m128 * weights)
            {
                return _mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]), _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(Load<false>(src + 1), weights[1]), _mm_mul_ps(Load<false>(src + 2), weights[2])),
                    _mm_add_ps(_mm_mul_ps(Load<false>(src + 3), weights[3]), _mm_mul_ps(Load<align>(src + 4), weights[4]))));
            }

            template<bool align> static SIMD_INLINE __m128 Forward(const float * src, size_t stride, const __m128 * weights)
            {
                return _mm_add_ps(Convolution5<align>(src, weights),
                    _mm_add_ps(_mm_add_ps(Convolution5<align>(src + stride, weights + 5),
                        Convolution5<align>(src + 2 * stride, weights + 10)),
                        _mm_add_ps(Convolution5<align>(src + 3 * stride, weights + 15),
                            Convolution5<align>(src + 4 * stride, weights + 20))));
            }

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

        template <bool align, size_t kernelX, size_t kernelY> SIMD_INLINE void NeuralAddConvolutionSum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m128 tailMask = RightNotZero32f(width - alignedWidth);
            __m128 _sums[kernelX*kernelY];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    Convolution<kernelX, kernelY>::template Sum<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m128 _dst = _mm_and_ps(tailMask, Load<false>(dst + col));
                    Convolution<kernelX, kernelY>::template Sum<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            size_t i = 0, n = Simd::AlignLo(kernelX*kernelY, F);
            for (; i < n; i += F)
                Add4ExtractedSums(_sums + i, sums + i);
            for (; i < kernelX*kernelY; ++i)
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
                void PrepareB(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY,
                    size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, float * dst)
                {
                    const size_t K = kernelX*kernelY*srcDepth, N = dstHeight*dstWidth;
                    if (dilationX*dilationY*strideX*strideY != 1)
                    {
                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            size_t srcRow0 = dstRow*strideY - padY;
                            for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                            {
                                size_t srcCol0 = dstCol*strideX - padX;
                                for (size_t channel = 0; channel < srcDepth; ++channel)
                                {
                                    for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                    {
                                        size_t srcRow = srcRow0 + kernelRow*dilationY;
                                        if (srcRow < srcHeight)
                                        {
                                            const float * psrc = src + (channel*srcHeight + srcRow)*srcWidth;
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                            {
                                                size_t srcCol = srcCol0 + kernelCol*dilationX;
                                                if (srcCol < srcWidth)
                                                    *(dst++) = psrc[srcCol];
                                                else
                                                    *(dst++) = 0;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                                *(dst++) = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else if (kernelX*kernelY != 1)
                    {
                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            size_t srcRow0 = dstRow - padY;
                            for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                            {
                                size_t srcCol0 = dstCol - padX;
                                for (size_t channel = 0; channel < srcDepth; ++channel)
                                {
                                    for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                    {
                                        size_t srcRow = srcRow0 + kernelRow;
                                        if (srcRow < srcHeight)
                                        {
                                            const float * psrc = src + (channel*srcHeight + srcRow)*srcWidth;
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                            {
                                                size_t srcCol = srcCol0 + kernelCol;
                                                if (srcCol < srcWidth)
                                                    *(dst++) = psrc[srcCol];
                                                else
                                                    *(dst++) = 0;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                                *(dst++) = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < N; ++i)
                        {
                            for (size_t k = 0; k < K; ++k)
                                *(dst++) = src[k*N + i];
                        }
                    }
                }

                template <bool align> static SIMD_INLINE void Kernel1x4x4(const __m128 & a, size_t K, const float * b, __m128 * sums)
                {
                    sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(a, Load<align>(b + 0 * K)));
                    sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(a, Load<align>(b + 1 * K)));
                    sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(a, Load<align>(b + 2 * K)));
                    sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(a, Load<align>(b + 3 * K)));
                }

                template <bool align> static SIMD_INLINE void Kernel1x1x4(const __m128 & a, const float * b, __m128 & sum)
                {
                    sum = _mm_add_ps(sum, _mm_mul_ps(a, Load<align>(b)));
                }

                template <bool align> static SIMD_INLINE void Kernel3x4x4(const __m128 * a, size_t K, const float * b, __m128 * sums)
                {
                    __m128 _b;
                    _b = Load<align>(b + 0 * K);
                    sums[0x0] = _mm_add_ps(sums[0x0], _mm_mul_ps(a[0], _b));
                    sums[0x4] = _mm_add_ps(sums[0x4], _mm_mul_ps(a[1], _b));
                    sums[0x8] = _mm_add_ps(sums[0x8], _mm_mul_ps(a[2], _b));
                    _b = Load<align>(b + 1 * K);
                    sums[0x1] = _mm_add_ps(sums[0x1], _mm_mul_ps(a[0], _b));
                    sums[0x5] = _mm_add_ps(sums[0x5], _mm_mul_ps(a[1], _b));
                    sums[0x9] = _mm_add_ps(sums[0x9], _mm_mul_ps(a[2], _b));
                    _b = Load<align>(b + 2 * K);
                    sums[0x2] = _mm_add_ps(sums[0x2], _mm_mul_ps(a[0], _b));
                    sums[0x6] = _mm_add_ps(sums[0x6], _mm_mul_ps(a[1], _b));
                    sums[0xA] = _mm_add_ps(sums[0xA], _mm_mul_ps(a[2], _b));
                    _b = Load<align>(b + 3 * K);
                    sums[0x3] = _mm_add_ps(sums[0x3], _mm_mul_ps(a[0], _b));
                    sums[0x7] = _mm_add_ps(sums[0x7], _mm_mul_ps(a[1], _b));
                    sums[0xB] = _mm_add_ps(sums[0xB], _mm_mul_ps(a[2], _b));
                }

                template <bool align> static SIMD_INLINE void Kernel3x1x4(const __m128 * a, const float * b, __m128 * sums)
                {
                    __m128 _b = Load<align>(b);
                    sums[0x0] = _mm_add_ps(sums[0x0], _mm_mul_ps(a[0], _b));
                    sums[0x1] = _mm_add_ps(sums[0x1], _mm_mul_ps(a[1], _b));
                    sums[0x2] = _mm_add_ps(sums[0x2], _mm_mul_ps(a[2], _b));
                }

                template <bool align> void Execute(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    size_t M3 = M / 3 * 3;
                    size_t N4 = Simd::AlignLo(N, 4);
                    size_t K4 = Simd::AlignLo(K, 4);
                    __m128 tailMask = RightNotZero32f(K - K4);
                    size_t i = 0;
                    for (; i < M3; i += 3)
                    {
                        const float * pa = a + i * K;
                        float * pc = c + i * N;
                        size_t j = 0;
                        for (; j < N4; j += 4)
                        {
                            const float * pb = b + j * K;
                            __m128 sums[12] = {
                                _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(),
                                _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(),
                                _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                            __m128 _a[3];
                            for (size_t k = 0; k < K4; k += 4)
                            {
                                _a[0] = Load<false>(pa + k + 0 * K);
                                _a[1] = Load<false>(pa + k + 1 * K);
                                _a[2] = Load<false>(pa + k + 2 * K);
                                Kernel3x4x4<align>(_a, K, pb + k, sums);
                            }
                            if (K4 < K)
                            {
                                size_t k = K - 4;
                                _a[0] = _mm_and_ps(tailMask, Load<false>(pa + k + 0 * K));
                                _a[1] = _mm_and_ps(tailMask, Load<false>(pa + k + 1 * K));
                                _a[2] = _mm_and_ps(tailMask, Load<false>(pa + k + 2 * K));
                                Kernel3x4x4<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0, pc + j + 0 * N);
                            Add4ExtractedSums(sums + 4, pc + j + 1 * N);
                            Add4ExtractedSums(sums + 8, pc + j + 2 * N);
                        }
                        for (; j < N; ++j)
                        {
                            const float * pb = b + j * K;
                            __m128 sums[3] = { _mm_setzero_ps(), _mm_setzero_ps() , _mm_setzero_ps() };
                            __m128 _a[3];
                            for (size_t k = 0; k < K4; k += 4)
                            {
                                _a[0] = Load<false>(pa + k + 0 * K);
                                _a[1] = Load<false>(pa + k + 1 * K);
                                _a[2] = Load<false>(pa + k + 2 * K);
                                Kernel3x1x4<align>(_a, pb + k, sums);
                            }
                            if (K4 < K)
                            {
                                size_t k = K - 4;
                                _a[0] = _mm_and_ps(tailMask, Load<false>(pa + k + 0 * K));
                                _a[1] = _mm_and_ps(tailMask, Load<false>(pa + k + 1 * K));
                                _a[2] = _mm_and_ps(tailMask, Load<false>(pa + k + 2 * K));
                                Kernel3x1x4<false>(_a, pb + k, sums);
                            }
                            pc[j + 0 * N] += ExtractSum(sums[0]);
                            pc[j + 1 * N] += ExtractSum(sums[1]);
                            pc[j + 2 * N] += ExtractSum(sums[2]);
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
                                Kernel1x4x4<align>(_a, K, pb + k, sums);
                            }
                            if (K4 < K)
                            {
                                size_t k = K - 4;
                                __m128 _a = _mm_and_ps(tailMask, Load<false>(pa + k));
                                Kernel1x4x4<false>(_a, K, pb + k, sums);
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
                                Kernel1x1x4<align>(_a, pb + k, sum);
                            }
                            if (K4 < K)
                            {
                                size_t k = K - 4;
                                __m128 _a = _mm_and_ps(tailMask, Load<false>(pa + k));
                                Kernel1x1x4<false>(_a, pb + k, sum);
                            }
                            pc[j] += ExtractSum(sum);
                        }
                    }
                }

                void Execute(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    if (Aligned(K, F))
                        Execute<true>(M, N, K, a, b, c);
                    else
                        Execute<false>(M, N, K, a, b, c);
                }
            }

            namespace Ver1
            {
                void PrepareA(const float * src, size_t M, size_t K, size_t cell, float * dst)
                {
                    size_t K4 = AlignLo(K, 4);
                    for (size_t i = 0; i < M; i += cell)
                    {
                        size_t n = Simd::Min(cell, M - i), k = 0;
                        if (cell == 4 && n == 4)
                        {
                            for (; k < K4; k += 4)
                            {
                                const float * ps = src + k;
                                __m128 s0 = Load<false>(ps + 0 * K);
                                __m128 s1 = Load<false>(ps + 1 * K);
                                __m128 s2 = Load<false>(ps + 2 * K);
                                __m128 s3 = Load<false>(ps + 3 * K);
                                __m128 s00 = _mm_unpacklo_ps(s0, s2);
                                __m128 s01 = _mm_unpacklo_ps(s1, s3);
                                __m128 s10 = _mm_unpackhi_ps(s0, s2);
                                __m128 s11 = _mm_unpackhi_ps(s1, s3);
                                Store<false>(dst + 0, _mm_unpacklo_ps(s00, s01));
                                Store<false>(dst + 4, _mm_unpackhi_ps(s00, s01));
                                Store<false>(dst + 8, _mm_unpacklo_ps(s10, s11));
                                Store<false>(dst + 12, _mm_unpackhi_ps(s10, s11));
                                dst += 16;
                            }
                        }
                        for (; k < K; ++k)
                        {
                            for (size_t c = 0; c < n; ++c)
                                *(dst++) = src[c*K + k];
                        }
                        src += cell*K;
                    }
                }

                void PrepareB(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY,
                    size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t cell, float * tmp, float * dst)
                {
                    const size_t K = kernelX*kernelY*srcDepth, N = dstHeight*dstWidth;
                    if (kernelX*kernelY != 1)
                    {
                        float * dst = tmp;
                        size_t channelSize = srcHeight * srcWidth;
                        if (dilationX*dilationY*strideX*strideY != 1)
                        {
                            for (size_t channel = 0, k = 0; channel < srcDepth; ++channel, src += channelSize)
                            {
                                for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                {
                                    for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol, ++k)
                                    {
                                        size_t srcRow = kernelRow*dilationY - padY;
                                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                                        {
                                            if (srcRow < srcHeight)
                                            {
                                                size_t srcCol = kernelCol*dilationX - padX;
                                                for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = src[srcRow*srcWidth + srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                    srcCol += strideX;
                                                }
                                            }
                                            else
                                            {
                                                for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                                    *(dst++) = 0;
                                            }
                                            srcRow += strideY;
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            const size_t bodySize = dstWidth - padX * 2;
                            for (size_t channel = 0, k = 0; channel < srcDepth; ++channel, src += channelSize)
                            {
                                for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                {
                                    for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol, ++k)
                                    {
                                        size_t srcRow = kernelRow - padY;
                                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow, ++srcRow)
                                        {
                                            if (srcRow < srcHeight)
                                            {
                                                size_t srcCol = kernelCol - padX, dstCol = 0;
                                                const float * psrc = src + srcRow*srcWidth;
                                                for (; dstCol < padX; ++dstCol, ++srcCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = psrc[srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                }
                                                memcpy(dst, psrc + srcCol, bodySize * 4);
                                                dst += bodySize;
                                                dstCol += bodySize;
                                                srcCol += bodySize;
                                                for (; dstCol < dstWidth; ++dstCol, ++srcCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = psrc[srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                }
                                            }
                                            else
                                            {
                                                memset(dst, 0, dstWidth * 4);
                                                dst += dstWidth;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        src = tmp;
                    }
                    if (cell == 8)
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            if (n == cell)
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float * psrc = src + k*N;
                                    Store<false>(dst + 0, Load<false>(psrc + 0));
                                    Store<false>(dst + 4, Load<false>(psrc + 4));
                                    dst += 8;
                                }
                            }
                            else
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float * psrc = src + k*N;
                                    size_t c = 0;
                                    for (; c < n; ++c)
                                        *(dst++) = *(psrc++);
                                    for (; c < cell; ++c)
                                        *(dst++) = 0;
                                }
                            }
                            src += cell;
                        }
                    }
                    else
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            for (size_t k = 0; k < K; ++k)
                            {
                                const float * psrc = src + k*N;
                                size_t c = 0;
                                for (; c < n; ++c)
                                    *(dst++) = *(psrc++);
                                for (; c < cell; ++c)
                                    *(dst++) = 0;
                            }
                            src += cell;
                        }
                    }
                }

                SIMD_INLINE void AddSum(const __m128 & sum, float * dst)
                {
                    Store<false>(dst, _mm_add_ps(Load<false>(dst), sum));
                }

                SIMD_INLINE void AddSums4(const __m128 * sums, size_t size, const float * mask, float * dst, size_t stride)
                {
                    if (mask)
                    {
                        __m128 _mask = _mm_loadu_ps(mask);
                        for (size_t i = 0; i < size; ++i, dst += stride)
                            AddSum(_mm_and_ps(_mask, sums[i]), dst);
                    }
                    else
                    {
                        for (size_t i = 0; i < size; ++i, dst += stride)
                            AddSum(sums[i], dst);
                    }
                }

                template <bool align> SIMD_INLINE void KernelMx4(size_t N, size_t K, const float * a, const float * b, float * c, const float * mask, size_t m)
                {
                    __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m128 b0 = Load<align>(b);
                        for (size_t s = 0; s < m; ++s)
                            sums[s] = _mm_add_ps(sums[s], _mm_mul_ps(_mm_set1_ps(a[s]), b0));
                        b += 4;
                        a += m;
                    }
                    AddSums4(sums, m, mask, c, N);
                }

                template <bool align> SIMD_INLINE void Kernel4x4(size_t N, size_t K, const float * a, const float * b, float * c, const float * mask)
                {
                    __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m128 b0 = Load<align>(b);
                        sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(_mm_set1_ps(a[0]), b0));
                        sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(_mm_set1_ps(a[1]), b0));
                        sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(_mm_set1_ps(a[2]), b0));
                        sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(_mm_set1_ps(a[3]), b0));
                        b += 4;
                        a += 4;
                    }
                    AddSums4(sums, 4, mask, c, N);
                }

                template <bool align> void Execute4x4(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t N4 = Simd::AlignLo(N, 4);
                    const int32_t mask[8] = { -1, -1, -1, -1, 0, 0, 0, 0 };
                    const float * tail = (float*)mask + 4 - N + N4;
                    size_t i = 0;
                    for (; i < M4; i += 4)
                    {
                        size_t j = 0;
                        for (; j < N4; j += 4)
                            Kernel4x4<align>(N, K, a + i*K, b + j*K, c + i*N + j, NULL);
                        if (N4 < N)
                            Kernel4x4<align>(N, K, a + i*K, b + j*K, c + i*N + j, tail);
                    }
                    if (M4 < M)
                    {
                        size_t j = 0;
                        for (; j < N4; j += 4)
                            KernelMx4<align>(N, K, a + i*K, b + j*K, c + i*N + j, NULL, M - M4);
                        if (N4 < N)
                            KernelMx4<align>(N, K, a + i*K, b + j*K, c + i*N + j, tail, M - M4);
                    }
                }

                SIMD_INLINE void AddSums8(const __m128 * sums, size_t size, const float * mask, float * dst, size_t stride)
                {
                    if (mask)
                    {
                        __m128 mask0 = _mm_loadu_ps(mask + 0);
                        __m128 mask1 = _mm_loadu_ps(mask + 4);
                        for (size_t i = 0; i < size; ++i, dst += stride)
                        {
                            AddSum(_mm_and_ps(mask0, sums[i + 0]), dst + 0);
                            AddSum(_mm_and_ps(mask1, sums[i + 4]), dst + 4);
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < size; ++i, dst += stride)
                        {
                            AddSum(sums[i + 0], dst + 0);
                            AddSum(sums[i + 4], dst + 4);
                        }
                    }
                }

                template <bool align> SIMD_INLINE void KernelMx8(size_t N, size_t K, const float * a, const float * b, float * c, const float * mask, size_t m)
                {
                    __m128 sums[8] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m128 b0 = Load<align>(b + 0);
                        __m128 b1 = Load<align>(b + 4);
                        for (size_t s = 0; s < m; ++s)
                        {
                            __m128 a0 = _mm_set1_ps(a[s]);
                            sums[s + 0] = _mm_add_ps(sums[s + 0], _mm_mul_ps(b0, a0));
                            sums[s + 4] = _mm_add_ps(sums[s + 4], _mm_mul_ps(b1, a0));
                        }
                        b += 8;
                        a += m;
                    }
                    AddSums8(sums, m, mask, c, N);
                }

                template <bool align> SIMD_INLINE void Kernel4x8(size_t N, size_t K, const float * a, const float * b, float * c, const float * mask)
                {
                    __m128 sums[8] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m128 b0 = Load<align>(b + 0);
                        __m128 b1 = Load<align>(b + 4);
                        __m128 a0 = _mm_set1_ps(a[0]);
                        sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(b0, a0));
                        sums[4] = _mm_add_ps(sums[4], _mm_mul_ps(b1, a0));
                        __m128 a1 = _mm_set1_ps(a[1]);
                        sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(b0, a1));
                        sums[5] = _mm_add_ps(sums[5], _mm_mul_ps(b1, a1));
                        __m128 a2 = _mm_set1_ps(a[2]);
                        sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(b0, a2));
                        sums[6] = _mm_add_ps(sums[6], _mm_mul_ps(b1, a2));
                        __m128 a3 = _mm_set1_ps(a[3]);
                        sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(b0, a3));
                        sums[7] = _mm_add_ps(sums[7], _mm_mul_ps(b1, a3));
                        b += 8;
                        a += 4;
                    }
                    AddSums8(sums, 4, mask, c, N);
                }

                template <bool align> void Execute4x8(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t N8 = Simd::AlignLo(N, 8);
                    const int32_t mask[16] = { -1, -1, -1, -1,  -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
                    const float * tail = (float*)mask + 8 - N + N8;
                    size_t i = 0;
                    for (; i < M4; i += 4)
                    {
                        size_t j = 0;
                        for (; j < N8; j += 8)
                            Kernel4x8<align>(N, K, a + i*K, b + j*K, c + i*N + j, NULL);
                        if (N8 < N)
                            Kernel4x8<align>(N, K, a + i*K, b + j*K, c + i*N + j, tail);
                    }
                    if (M4 < M)
                    {
                        size_t j = 0;
                        for (; j < N8; j += 8)
                            KernelMx8<align>(N, K, a + i*K, b + j*K, c + i*N + j, NULL, M - M4);
                        if (N8 < N)
                            KernelMx8<align>(N, K, a + i*K, b + j*K, c + i*N + j, tail, M - M4);
                    }
                }

                void Execute(size_t M, size_t N, size_t K, const float * a, const float * b, float * c, size_t cellA, size_t cellB)
                {
                    if (cellA == 4)
                    {
                        if (cellB == 4)
                            Execute4x4<false>(M, N, K, a, b, c);
                        if (cellB == 8)
                            Execute4x8<false>(M, N, K, a, b, c);
                    }
                }
            }

            namespace Ver2
            {
                void PrepareB(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t padX, size_t padY, float * dst, size_t dstWidth, size_t dstHeight)
                {
                    for (size_t channel = 0; channel < srcDepth; ++channel)
                    {
                        const float * s = src;
                        float * d = dst;
                        memset(d, 0, padY*dstWidth * 4);
                        d += padY*dstWidth;
                        for (size_t row = padY; row < dstHeight - padY; ++row)
                        {
                            memset(d, 0, padX * 4);
                            memcpy(d + padX, s, srcWidth * 4);
                            memset(d + padX + srcWidth, 0, padX * 4);
                            d += dstWidth;
                            s += srcWidth;
                        }
                        memset(d, 0, padY*dstWidth * 4);
                        src += srcWidth*srcHeight;
                        dst += dstWidth*dstHeight;
                    }
                }

                template <size_t size> SIMD_INLINE void LoadWeight(const float * src, __m128 * dst)
                {
                    for (size_t i = 0; i < size; ++i)
                        dst[i] = _mm_set1_ps(src[i]);
                }

                template <bool align, size_t kernelX, size_t kernelY> void AddConvolution(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float * weight, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    size_t alignedWidth = AlignLo(dstWidth, F);
                    __m128 tailMask = RightNotZero32f(dstWidth - alignedWidth);
                    __m128 _weight[kernelX*kernelY];
                    for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                    {
                        for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                        {
                            const float * psrc = src + srcWidth*srcHeight*srcChannel;
                            const float * pweight = weight + (dstChannel*srcDepth + srcChannel)*kernelX*kernelY;
                            float * pdst = dst + dstWidth*dstHeight*dstChannel;
                            LoadWeight<kernelX*kernelY>(pweight, _weight);
                            for (size_t row = 0; row < dstHeight; ++row)
                            {
                                size_t col = 0;
                                for (; col < alignedWidth; col += F)
                                {
                                    __m128 _dst = Load<align>(pdst + col);
                                    _dst = _mm_add_ps(_dst, Convolution<kernelX, kernelY>::template Forward<align>(psrc + col, srcWidth, _weight));
                                    Store<align>(pdst + col, _dst);
                                }
                                if (dstWidth - alignedWidth)
                                {
                                    size_t col = dstWidth - F;
                                    __m128 _dst = Load<false>(pdst + col);
                                    _dst = _mm_add_ps(_dst, _mm_and_ps(tailMask, Convolution<kernelX, kernelY>::template Forward<false>(psrc + col, srcWidth, _weight)));
                                    Store<false>(pdst + col, _dst);
                                }
                                psrc += srcWidth;
                                pdst += dstWidth;
                            }
                        }
                    }
                }

                void Execute(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float * weight, size_t kernelX, size_t kernelY, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    assert(kernelX == kernelY);
                    if (kernelX == 2)
                        AddConvolution<false, 2, 2>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 3)
                        AddConvolution<false, 3, 3>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 4)
                        AddConvolution<false, 4, 4>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 5)
                        AddConvolution<false, 5, 5>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else
                        assert(0);
                }

                bool Preferable(size_t srcDepth, size_t kernelX, size_t kernelY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    if (kernelX == kernelY && kernelX >= 2 && kernelX <= 5 && strideX*strideY*dilationX*dilationY == 1 && dstWidth >= F)
                    {
                        if (dstWidth*dstHeight*kernelX*kernelY >= 8 * 8 * 5 * 5)
                            return true;
                    }
                    return false;
                }
            }

            struct Opt
            {
                enum Alg
                {
                    None,
                    Ver0,
                    Ver1,
                    Ver2,
                } alg;

                size_t sizeA;
                size_t sizeB;
                size_t sizeT;

                size_t cellA;
                size_t cellB;

                size_t M, N, K;
                size_t strideB;
                size_t paddedW;
                size_t paddedH;

                Opt(size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    alg = None;
                    sizeA = 0;
                    sizeB = 0;
                    sizeT = 0;
                    cellA = 1;
                    cellB = 1;

                    M = dstDepth;
                    N = dstHeight*dstWidth;
                    K = kernelX*kernelY*srcDepth;

                    if (dstWidth*dstHeight / kernelX <= 2000)
                        alg = Ver0;
                    else
                        alg = Ver1;
                    if (Ver2::Preferable(srcDepth, kernelX, kernelY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth))
                        alg = Ver2;

                    switch (alg)
                    {
                    case Ver0:
                        sizeB = N*K;
                        break;
                    case Ver1:
                        cellA = 4;
                        cellB = 8;
                        sizeA = M*K;
                        strideB = Simd::AlignHi(N, cellB);
                        sizeB = strideB*K;
                        if (kernelX*kernelY > 1)
                            sizeT = sizeB;
                        break;
                    case Ver2:
                        if (padX > 0 || padY > 0)
                        {
                            paddedW = Simd::AlignHi(srcWidth + 2 * padX, F);
                            paddedH = srcHeight + 2 * padY;
                            sizeB = paddedW*paddedH*srcDepth;
                        }
                        else
                        {
                            paddedW = srcWidth;
                            paddedH = srcHeight;
                        }
                        break;
                    default:
                        assert(0);
                        break;
                    }
                }
            };

            struct Data
            {
                float * a;
                float * b;
                float * t;

                Data(size_t sizeA, size_t sizeB, size_t sizeT, void * externalData, size_t * externalSize)
                    : a(0)
                    , b(0)
                    , _data(0)
                {
                    sizeA = AlignHi(sizeA, F);
                    sizeB = AlignHi(sizeB, F);
                    sizeT = AlignHi(sizeT, F);
                    size_t size = (sizeA + sizeB + sizeT) * sizeof(float);
                    if (size == 0)
                        return;
                    if (externalData != AlignHi(externalData, SIMD_ALIGN))
                        size += SIMD_ALIGN;
                    float * data = NULL;
                    if (externalData == NULL || externalSize == NULL || *externalSize < size)
                    {
                        _data = Simd::Allocate(size);
                        if (externalSize)
                            *externalSize = size;
                        data = (float*)_data;
                    }
                    else
                        data = (float*)AlignHi(externalData, SIMD_ALIGN);
                    if (sizeA)
                        a = data;
                    if (sizeB)
                        b = data + sizeA;
                    if (sizeT)
                        t = data + sizeA + sizeB;
                }

                ~Data()
                {
                    if (_data)
                        Simd::Free(_data);
                }

            private:
                void * _data;
            };
        }

        void NeuralConvolutionForward(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
            const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY,
            void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add)
        {
            using namespace Ncf;

            assert(dstWidth == (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1);
            assert(dstHeight == (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1);

            if (!add)
                memset(dst, 0, dstWidth*dstHeight*dstDepth * sizeof(float));

            Opt opt(srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth);

            Data data(opt.sizeA, opt.sizeB, opt.sizeT, buffer, size);

            if (opt.sizeA)
            {
                switch (opt.alg)
                {
                case Opt::Ver1: Ver1::PrepareA(weight, opt.M, opt.K, opt.cellA, data.a);
                default:
                    break;
                }
            }
            else
                data.a = (float*)weight;

            if (opt.sizeB)
            {
                switch (opt.alg)
                {
                case Opt::Ver0: Ver0::PrepareB(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, data.b); break;
                case Opt::Ver1: Ver1::PrepareB(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, opt.cellB, data.t, data.b); break;
                case Opt::Ver2: Ver2::PrepareB(src, srcWidth, srcHeight, srcDepth, padX, padY, data.b, opt.paddedW, opt.paddedH); break;
                default: break;
                }
            }
            else
                data.b = (float*)src;

            switch (opt.alg)
            {
            case Opt::Ver0: Ver0::Execute(opt.M, opt.N, opt.K, data.a, data.b, dst); break;
            case Opt::Ver1: Ver1::Execute(opt.M, opt.N, opt.K, data.a, data.b, dst, opt.cellA, opt.cellB); break;
            case Opt::Ver2: Ver2::Execute(data.b, opt.paddedW, opt.paddedH, srcDepth, weight, kernelX, kernelY, dst, dstWidth, dstHeight, dstDepth); break;
            default: break;
            }
        }
    }
#endif// SIMD_SSE41_ENABLE
}
