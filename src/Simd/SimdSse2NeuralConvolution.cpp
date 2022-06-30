/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdStream.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdNeural.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        namespace
        {
            template<int count> struct Buffer
            {
                Buffer(size_t width)
                {
                    _size = width * sizeof(float);
                    size_t stride = AlignHi(width + 2 * (count - 1), F);
                    size_t full = count * stride * sizeof(float);
                    _ptr = Allocate(full);
                    memset(_ptr, 0, full);
                    rows[0] = (float*)_ptr;
                    for (size_t i = 1; i < count; ++i)
                        rows[i] = rows[i - 1] + stride;
                }

                void Update(const float* src)
                {
                    float* tmp = rows[0];
                    if (src == NULL)
                        memset(tmp + count - 1, 0, _size);
                    else
                        memcpy(tmp + count - 1, src, _size);
                    for (size_t i = 0; i < count - 1; ++i)
                        rows[i] = rows[i + 1];
                    rows[count - 1] = tmp;
                }

                ~Buffer()
                {
                    Free(_ptr);
                }

                float* rows[count];
            private:
                size_t _size;
                void* _ptr;
            };
        }

        template<size_t coreX, size_t coreY> struct Convolution
        {
            template<bool align> static SIMD_INLINE __m128 Forward(const float* src, size_t stride, const __m128* weights);

            template<bool align> static SIMD_INLINE __m128 Backward(const Buffer<coreX>& buffer, size_t offset, const __m128* weights);

            template <bool align> static SIMD_INLINE void Sum(const float* src, const __m128& dst, __m128* sums);
        };

        template<> struct Convolution<2, 2>
        {
            template <bool align> static SIMD_INLINE __m128 Convolution2(const float* src, const __m128* weights)
            {
                return _mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]),
                    _mm_mul_ps(Load<false>(src + 1), weights[1]));
            }

            template<bool align> static SIMD_INLINE __m128 Forward(const float* src, size_t stride, const __m128* weights)
            {
                return _mm_add_ps(Convolution2<align>(src, weights),
                    Convolution2<align>(src + stride, weights + 2));
            }

            template<bool align> static SIMD_INLINE __m128 Backward(const Buffer<2>& buffer, size_t offset, const __m128* weights)
            {
                return _mm_add_ps(Convolution2<align>(buffer.rows[0] + offset, weights),
                    Convolution2<align>(buffer.rows[1] + offset, weights + 2));
            }

            template <bool align> static SIMD_INLINE void Sum(const float* src, const __m128& dst, __m128* sums)
            {
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float* src, size_t stride, const __m128& dst, __m128* sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 2);
            }
        };

        template<> struct Convolution<3, 3>
        {
            template <bool align> static SIMD_INLINE __m128 Convolution3(const float* src, const __m128* weights)
            {
                return _mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]),
                    _mm_add_ps(_mm_mul_ps(Load<false>(src + 1), weights[1]),
                        _mm_mul_ps(Load<false>(src + 2), weights[2])));
            }

            template<bool align> static SIMD_INLINE __m128 Forward(const float* src, size_t stride, const __m128* weights)
            {
                return _mm_add_ps(Convolution3<align>(src, weights),
                    _mm_add_ps(Convolution3<align>(src + stride, weights + 3),
                        Convolution3<align>(src + 2 * stride, weights + 6)));
            }

            template<bool align> static SIMD_INLINE __m128 Backward(const Buffer<3>& buffer, size_t offset, const __m128* weights)
            {
                return _mm_add_ps(Convolution3<align>(buffer.rows[0] + offset, weights),
                    _mm_add_ps(Convolution3<align>(buffer.rows[1] + offset, weights + 3),
                        Convolution3<align>(buffer.rows[2] + offset, weights + 6)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float* src, const __m128& dst, __m128* sums)
            {
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
                sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float* src, size_t stride, const __m128& dst, __m128* sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 3);
                Sum<align>(src + stride * 2, dst, sums + 6);
            }
        };

        template<> struct Convolution<4, 4>
        {
            template <bool align> static SIMD_INLINE __m128 Convolution4(const float* src, const __m128* weights)
            {
                return _mm_add_ps(_mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]), _mm_mul_ps(Load<false>(src + 1), weights[1])),
                    _mm_add_ps(_mm_mul_ps(Load<false>(src + 2), weights[2]), _mm_mul_ps(Load<false>(src + 3), weights[3])));
            }

            template<bool align> static SIMD_INLINE __m128 Forward(const float* src, size_t stride, const __m128* weights)
            {
                return _mm_add_ps(_mm_add_ps(Convolution4<align>(src, weights),
                    Convolution4<align>(src + stride, weights + 4)),
                    _mm_add_ps(Convolution4<align>(src + 2 * stride, weights + 8),
                        Convolution4<align>(src + 3 * stride, weights + 12)));
            }

            template<bool align> static SIMD_INLINE __m128 Backward(const Buffer<4>& buffer, size_t offset, const __m128* weights)
            {
                return _mm_add_ps(_mm_add_ps(Convolution4<align>(buffer.rows[0] + offset, weights),
                    Convolution4<align>(buffer.rows[1] + offset, weights + 4)),
                    _mm_add_ps(Convolution4<align>(buffer.rows[2] + offset, weights + 8),
                        Convolution4<align>(buffer.rows[3] + offset, weights + 12)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float* src, const __m128& dst, __m128* sums)
            {
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
                sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
                sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(dst, Load<false>(src + 3)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float* src, size_t stride, const __m128& dst, __m128* sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 4);
                Sum<align>(src + stride * 2, dst, sums + 8);
                Sum<align>(src + stride * 3, dst, sums + 12);
            }
        };

        template<> struct Convolution<5, 5>
        {
            template <bool align> static SIMD_INLINE __m128 Convolution5(const float* src, const __m128* weights)
            {
                return _mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]), _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(Load<false>(src + 1), weights[1]), _mm_mul_ps(Load<false>(src + 2), weights[2])),
                    _mm_add_ps(_mm_mul_ps(Load<false>(src + 3), weights[3]), _mm_mul_ps(Load<align>(src + 4), weights[4]))));
            }

            template<bool align> static SIMD_INLINE __m128 Forward(const float* src, size_t stride, const __m128* weights)
            {
                return _mm_add_ps(Convolution5<align>(src, weights),
                    _mm_add_ps(_mm_add_ps(Convolution5<align>(src + stride, weights + 5),
                        Convolution5<align>(src + 2 * stride, weights + 10)),
                        _mm_add_ps(Convolution5<align>(src + 3 * stride, weights + 15),
                            Convolution5<align>(src + 4 * stride, weights + 20))));
            }

            template<bool align> static SIMD_INLINE __m128 Backward(const Buffer<5>& buffer, size_t offset, const __m128* weights)
            {
                return _mm_add_ps(_mm_add_ps(Convolution5<align>(buffer.rows[0] + offset, weights),
                    _mm_add_ps(Convolution5<align>(buffer.rows[1] + offset, weights + 5),
                        Convolution5<align>(buffer.rows[2] + offset, weights + 10))),
                    _mm_add_ps(Convolution5<align>(buffer.rows[3] + offset, weights + 15),
                        Convolution5<align>(buffer.rows[4] + offset, weights + 20)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float* src, const __m128& dst, __m128* sums)
            {
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
                sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
                sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(dst, Load<false>(src + 3)));
                sums[4] = _mm_add_ps(sums[4], _mm_mul_ps(dst, Load<align>(src + 4)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float* src, size_t stride, const __m128& dst, __m128* sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 5);
                Sum<align>(src + stride * 2, dst, sums + 10);
                Sum<align>(src + stride * 3, dst, sums + 15);
                Sum<align>(src + stride * 4, dst, sums + 20);
            }
        };

        //---------------------------------------------------------------------

        template <size_t size> SIMD_INLINE void LoadWeightsForward(const float* src, __m128* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm_set1_ps(src[i]);
        }

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionForward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, F);
            __m128 tailMask = RightNotZero32f(width - alignedWidth);
            __m128 _weights[coreX * coreY];
            LoadWeightsForward<coreX* coreY>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    _dst = _mm_add_ps(_dst, Convolution<coreX, coreY>::template Forward<align>(src + col, srcStride, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m128 _dst = Load<false>(dst + col);
                    _dst = _mm_add_ps(_dst, _mm_and_ps(tailMask, Convolution<coreX, coreY>::template Forward<false>(src + col, srcStride, _weights)));
                    Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution2x2Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution4x4Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
        }

        //---------------------------------------------------------------------

        template<bool condition> struct If
        {
            template<bool align> static SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
            {
                Sse2::AddMultiplied<align>(src, aligned, partial, full, value, dst);
            }
        };

        template<> struct If<false>
        {
            template<bool align> static SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
            {
            }
        };

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackwardSmall(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            size_t aligned = AlignLo(width, QF);
            size_t partial = AlignLo(width, F);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < coreY; ++dy)
                {
                    const float* w = weights + dy * coreX;
                    float* d = dst + dy * dstStride;
                    If < 0 < coreX > ::template AddMultiplied<align>(src, aligned, partial, width, w[0], d + 0);
                    If < 1 < coreX > ::template AddMultiplied<false>(src, aligned, partial, width, w[1], d + 1);
                    If < 2 < coreX > ::template AddMultiplied<false>(src, aligned, partial, width, w[2], d + 2);
                    If < 3 < coreX > ::template AddMultiplied<false>(src, aligned, partial, width, w[3], d + 3);
                    If < 4 < coreX > ::template AddMultiplied<align>(src, aligned, partial, width, w[4], d + 4);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <size_t size> SIMD_INLINE void LoadWeightsBackward(const float* src, __m128* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm_set1_ps(src[size - i - 1]);
        }

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackwardLarge(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            Buffer<coreX> buffer(width);
            height += coreY - 1;
            width += coreX - 1;
            size_t alignedWidth = AlignLo(width, F);
            __m128 tailMask = RightNotZero32f(width - alignedWidth);
            __m128 _weights[coreX * coreY];
            LoadWeightsBackward<coreX* coreY>(weights, _weights);

            for (size_t row = 0; row < height; ++row)
            {
                buffer.Update(row <= height - coreY ? src : NULL);
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    _dst = _mm_add_ps(_dst, Convolution<coreX, coreY>::template Backward<true>(buffer, col, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m128 _dst = Load<false>(dst + col);
                    _dst = _mm_add_ps(_dst, _mm_and_ps(tailMask, Convolution<coreX, coreY>::template Backward<false>(buffer, col, _weights)));
                    Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (width * height < 1024)
                NeuralAddConvolutionBackwardSmall<align, coreX, coreY>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackwardLarge<align, coreX, coreY>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution2x2Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution4x4Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
        }

        //---------------------------------------------------------------------

        template <bool align, size_t coreX, size_t coreY> SIMD_INLINE void NeuralAddConvolutionSum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m128 tailMask = RightNotZero32f(width - alignedWidth);
            __m128 _sums[coreX * coreY];
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
            for (size_t i = 0; i < coreX * coreY; ++i)
                sums[i] += ExtractSum(_sums[i]);
        }

        void NeuralAddConvolution2x2Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution3x3Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution4x4Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution5x5Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
