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
#include "Simd/SimdStream.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, const __m128& norm, const __m128& alpha, const __m128& epsilon, float* gradient, float* weight)
        {
            __m128 d = _mm_mul_ps(Load<align>(delta), norm);
            __m128 _gradient = _mm_add_ps(Load<align>(gradient), _mm_mul_ps(d, d));
            Store<align>(gradient, _gradient);
            Store<align>(weight, _mm_sub_ps(Load<align>(weight), _mm_mul_ps(_mm_mul_ps(alpha, d), _mm_rsqrt_ps(_mm_add_ps(_gradient, epsilon)))));
        }

        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, size_t offset, const __m128& norm, const __m128& alpha, const __m128& epsilon, float* gradient, float* weight)
        {
            AdaptiveGradientUpdate<align>(delta + offset, norm, alpha, epsilon, gradient + offset, weight + offset);
        }

        template <bool align> void NeuralAdaptiveGradientUpdate(const float* delta, size_t size, size_t batch, const float* alpha, const float* epsilon, float* gradient, float* weight)
        {
            if (align)
                assert(Aligned(delta) && Aligned(gradient) && Aligned(weight));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            const float norm = (float)(1.0 / batch);
            __m128 _norm = _mm_set1_ps(norm);
            __m128 _alpha = _mm_set1_ps(*alpha);
            __m128 _epsilon = _mm_set1_ps(*epsilon);
            size_t i = 0;
            if (partialAlignedSize)
            {
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        AdaptiveGradientUpdate<align>(delta, i + F * 0, _norm, _alpha, _epsilon, gradient, weight);
                        AdaptiveGradientUpdate<align>(delta, i + F * 1, _norm, _alpha, _epsilon, gradient, weight);
                        AdaptiveGradientUpdate<align>(delta, i + F * 2, _norm, _alpha, _epsilon, gradient, weight);
                        AdaptiveGradientUpdate<align>(delta, i + F * 3, _norm, _alpha, _epsilon, gradient, weight);
                    }
                }
                for (; i < partialAlignedSize; i += F)
                    AdaptiveGradientUpdate<align>(delta, i, _norm, _alpha, _epsilon, gradient, weight);
            }
            for (; i < size; ++i)
                Base::AdaptiveGradientUpdate(delta, i, norm, *alpha, *epsilon, gradient, weight);
        }

        void NeuralAdaptiveGradientUpdate(const float* delta, size_t size, size_t batch, const float* alpha, const float* epsilon, float* gradient, float* weight)
        {
            if (Aligned(delta) && Aligned(gradient) && Aligned(weight))
                NeuralAdaptiveGradientUpdate<true>(delta, size, batch, alpha, epsilon, gradient, weight);
            else
                NeuralAdaptiveGradientUpdate<false>(delta, size, batch, alpha, epsilon, gradient, weight);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, const __m128& value, float* dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), _mm_mul_ps(value, Load<align>(src))));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
        {
            size_t i = 0;
            if (partial)
            {
                __m128 _value = _mm_set1_ps(value);
                for (; i < aligned; i += QF)
                {
                    AddMultiplied<align>(src + i + F * 0, _value, dst + i + F * 0);
                    AddMultiplied<align>(src + i + F * 1, _value, dst + i + F * 1);
                    AddMultiplied<align>(src + i + F * 2, _value, dst + i + F * 2);
                    AddMultiplied<align>(src + i + F * 3, _value, dst + i + F * 3);
                }
                for (; i < partial; i += F)
                    AddMultiplied<align>(src + i, _value, dst + i);
            }
            for (; i < full; ++i)
                dst[i] += src[i] * value;
        }

        void NeuralAddVectorMultipliedByValue(const float* src, size_t size, const float* value, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(src) && Aligned(dst))
                AddMultiplied<true>(src, aligned, partial, size, *value, dst);
            else
                AddMultiplied<false>(src, aligned, partial, size, *value, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AddVector(const float* src, float* dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), Load<align>(src)));
        }

        template <bool align> SIMD_INLINE void AddVector(const float* src, size_t aligned, size_t partial, size_t full, float* dst)
        {
            size_t i = 0;
            for (; i < aligned; i += QF)
            {
                AddVector<align>(src + i + F * 0, dst + i + F * 0);
                AddVector<align>(src + i + F * 1, dst + i + F * 1);
                AddVector<align>(src + i + F * 2, dst + i + F * 2);
                AddVector<align>(src + i + F * 3, dst + i + F * 3);
            }
            for (; i < partial; i += F)
                AddVector<align>(src + i, dst + i);
            for (; i < full; ++i)
                dst[i] += src[i];
        }

        void NeuralAddVector(const float* src, size_t size, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(src) && Aligned(dst))
                AddVector<true>(src, aligned, partial, size, dst);
            else
                AddVector<false>(src, aligned, partial, size, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AddValue(const __m128& value, float* dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), value));
        }

        template <bool align> SIMD_INLINE void AddValue(const float* value, float* dst, size_t aligned, size_t partial, size_t full)
        {
            size_t i = 0;
            if (partial)
            {
                __m128 _value = _mm_set1_ps(value[0]);
                for (; i < aligned; i += QF)
                {
                    AddValue<align>(_value, dst + i + F * 0);
                    AddValue<align>(_value, dst + i + F * 1);
                    AddValue<align>(_value, dst + i + F * 2);
                    AddValue<align>(_value, dst + i + F * 3);
                }
                for (; i < partial; i += F)
                    AddValue<align>(_value, dst + i);
            }
            for (; i < full; ++i)
                dst[i] += value[0];
        }

        void NeuralAddValue(const float* value, float* dst, size_t size)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(dst))
                AddValue<true>(value, dst, aligned, partial, size);
            else
                AddValue<false>(value, dst, aligned, partial, size);
        }

        //---------------------------------------------------------------------

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

        //---------------------------------------------------------------------

        template <bool inversion> __m128i Invert(__m128i value);

        template <> __m128i Invert<true>(__m128i value)
        {
            return _mm_sub_epi8(K_INV_ZERO, value);
        }

        template <> __m128i Invert<false>(__m128i value)
        {
            return value;
        }

        template <bool align, bool stream> void Convert(__m128i src, const __m128 &_1_255, float * dst)
        {
            Stream<align, stream>(dst + 0, _mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(src)), _1_255));
            Stream<align, stream>(dst + 4, _mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(src)), _1_255));
        }

        template <bool inversion, bool align, bool stream> void Convert(const uint8_t * src, const __m128 &_1_255, float * dst)
        {
            __m128i _src = Invert<inversion>(Load<align>((__m128i*)src));
            Convert<align, stream>(UnpackU8<0>(_src), _1_255, dst + 0);
            Convert<align, stream>(UnpackU8<1>(_src), _1_255, dst + 8);
        }

        template <bool inversion, bool align, bool stream> void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, A);
            __m128 _1_255 = _mm_set1_ps(1.0f / 255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    Convert<inversion, align, stream>(src + col, _1_255, dst + col);
                if (width != alignedWidth)
                    Convert<inversion, false, stream>(src + width - A, _1_255, dst + width - A);
                src += srcStride;
                dst += dstStride;
            }
            if (stream)
                _mm_mfence();
        }

        template <bool inversion> void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
            {
                if (width*height * sizeof(float) >= STREAM_SIZE_MIN)
                    NeuralConvert<inversion, true, true>(src, srcStride, width, height, dst, dstStride);
                else
                    NeuralConvert<inversion, true, false>(src, srcStride, width, height, dst, dstStride);
            }
            else
                NeuralConvert<inversion, false, false>(src, srcStride, width, height, dst, dstStride);
        }

        void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride, int inversion)
        {
            if (inversion)
                NeuralConvert<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralConvert<false>(src, srcStride, width, height, dst, dstStride);
        }

        //---------------------------------------------------------------------

        template <bool align> void NeuralDerivativeRelu(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float s = slope[0];
            __m128 _0 = _mm_set1_ps(0.0f);
            __m128 _s = _mm_set1_ps(s);
            __m128 d = _mm_set1_ps(1.0f - s);
            size_t alignedSize = Simd::AlignLo(size, F);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 mask = _mm_cmpgt_ps(Load<align>(src + i), _0);
                __m128 _dst = Load<align>(dst + i);
                Store<align>(dst + i, _mm_mul_ps(_mm_add_ps(_s, _mm_and_ps(mask, d)), _dst));
            }
            for (; i < size; ++i)
                dst[i] *= src[i] > 0 ? 1.0f : s;
        }

        void NeuralDerivativeRelu(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeRelu<true>(src, size, slope, dst);
            else
                NeuralDerivativeRelu<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralDerivativeSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            __m128 _slope = _mm_set1_ps(*slope);
            __m128 _1 = _mm_set1_ps(1.0f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 _src = Load<align>(src + i);
                __m128 _dst = Load<align>(dst + i);
                Store<align>(dst + i, _mm_mul_ps(_mm_mul_ps(_dst, _slope), _mm_mul_ps(_mm_sub_ps(_1, _src), _src)));
            }
            for (; i < size; ++i)
                dst[i] *= slope[0] * Base::DerivativeSigmoid(src[i]);
        }

        void NeuralDerivativeSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeSigmoid<true>(src, size, slope, dst);
            else
                NeuralDerivativeSigmoid<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralDerivativeTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            __m128 _slope = _mm_set1_ps(*slope);
            __m128 _1 = _mm_set1_ps(1.0f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 _src = Load<align>(src + i);
                __m128 _dst = Load<align>(dst + i);
                Store<align>(dst + i, _mm_mul_ps(_mm_mul_ps(_dst, _slope), _mm_sub_ps(_1, _mm_mul_ps(_src, _src))));
            }
            for (; i < size; ++i)
                dst[i] *= slope[0] * Base::DerivativeTanh(src[i]);
        }

        void NeuralDerivativeTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeTanh<true>(src, size, slope, dst);
            else
                NeuralDerivativeTanh<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE __m128 Pooling1x1Max3x1Body(const float* src)
        {
            return _mm_max_ps(_mm_max_ps(Load<false>(src - 1), Load<align>(src)), Load<false>(src + 1));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Body(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Body<align>(src - stride);
            __m128 src1 = Pooling1x1Max3x1Body<align>(src);
            __m128 src2 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, _mm_max_ps(_mm_max_ps(src0, src1), src2));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Body(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Body<align>(src);
            __m128 src1 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, _mm_max_ps(src0, src1));
        }

        template <bool align> SIMD_INLINE __m128 Pooling1x1Max3x1Nose(const float* src)
        {
            __m128 src1 = Load<align>(src);
            __m128 src0 = _mm_shuffle_ps(src1, src1, 0x90);
            __m128 src2 = Load<false>(src + 1);
            return _mm_max_ps(_mm_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Nose(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Nose<align>(src - stride);
            __m128 src1 = Pooling1x1Max3x1Nose<align>(src);
            __m128 src2 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, _mm_max_ps(_mm_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Nose(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Nose<align>(src);
            __m128 src1 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, _mm_max_ps(src0, src1));
        }

        template <bool align> SIMD_INLINE __m128 Pooling1x1Max3x1Tail(const float* src)
        {
            __m128 src0 = Load<false>(src - 1);
            __m128 src1 = Load<align>(src);
            __m128 src2 = _mm_shuffle_ps(src1, src1, 0xF9);
            return _mm_max_ps(_mm_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Tail(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Tail<align>(src - stride);
            __m128 src1 = Pooling1x1Max3x1Tail<align>(src);
            __m128 src2 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, _mm_max_ps(_mm_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Tail(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Tail<align>(src);
            __m128 src1 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, _mm_max_ps(src0, src1));
        }

        template <bool align> void NeuralPooling1x1Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            assert(width > F && height > 1);

            size_t alignedWidth = AlignHi(width, F) - F;
            height -= 1;

            Pooling1x1Max3x2Nose<align>(src, srcStride, dst);
            for (size_t col = F; col < alignedWidth; col += F)
                Pooling1x1Max3x2Body<align>(src + col, srcStride, dst + col);
            Pooling1x1Max3x2Tail<false>(src + width - F, srcStride, dst + width - F);

            for (size_t row = 1; row < height; ++row)
            {
                src += srcStride;
                dst += dstStride;
                Pooling1x1Max3x3Nose<align>(src, srcStride, dst);
                for (size_t col = F; col < alignedWidth; col += F)
                    Pooling1x1Max3x3Body<align>(src + col, srcStride, dst + col);
                Pooling1x1Max3x3Tail<false>(src + width - F, srcStride, dst + width - F);
            }

            dst += dstStride;
            Pooling1x1Max3x2Nose<align>(src, srcStride, dst);
            for (size_t col = F; col < alignedWidth; col += F)
                Pooling1x1Max3x2Body<align>(src + col, srcStride, dst + col);
            Pooling1x1Max3x2Tail<false>(src + width - F, srcStride, dst + width - F);
        }

        void NeuralPooling1x1Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling1x1Max3x3<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling1x1Max3x3<false>(src, srcStride, width, height, dst, dstStride);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max2x2(const float* src, size_t stride)
        {
            __m128 _src0 = _mm_max_ps(Load<align>(src + 0), Load<align>(src + stride + 0));
            __m128 _src1 = _mm_max_ps(Load<align>(src + F), Load<align>(src + stride + F));
            return _mm_max_ps(_mm_shuffle_ps(_src0, _src1, 0x88), _mm_shuffle_ps(_src0, _src1, 0xDD));
        }

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max2(const float* src)
        {
            __m128 _src0 = Load<align>(src + 0);
            __m128 _src1 = Load<align>(src + F);
            return _mm_max_ps(_mm_shuffle_ps(_src0, _src1, 0x88), _mm_shuffle_ps(_src0, _src1, 0xDD));
        }

        template <bool align> void NeuralPooling2x2Max2x2(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            size_t alignedWidth = AlignLo(width, DF);
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<align>(dst + (col >> 1), Pooling2x2Max2x2<align>(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), Pooling2x2Max2x2<false>(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Simd::Max(src[widthEven], src[widthEven + srcStride]);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<align>(dst + (col >> 1), Pooling2x2Max2<align>(src + col));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), Pooling2x2Max2<false>(src + col));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = src[widthEven];
            }
        }

        void NeuralPooling2x2Max2x2(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling2x2Max2x2<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling2x2Max2x2<false>(src, srcStride, width, height, dst, dstStride);
        }

        //---------------------------------------------------------------------

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

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max1x3(const float* src, size_t stride)
        {
            return _mm_max_ps(_mm_max_ps(Load<align>(src), Load<align>(src + stride)), Load<align>(src + 2 * stride));
        }

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max3x3(const float* src, size_t stride)
        {
            __m128 _0123 = Pooling2x2Max1x3<align>(src, stride);
            __m128 _4567 = Pooling2x2Max1x3<align>(src + F, stride);
            __m128 _5678 = Pooling2x2Max1x3<false>(src + F + 1, stride);
            __m128 _0246 = _mm_shuffle_ps(_0123, _4567, 0x88);
            __m128 _1357 = _mm_shuffle_ps(_0123, _4567, 0xDD);
            __m128 _2468 = _mm_shuffle_ps(_0246, _5678, 0xD9);
            return _mm_max_ps(_mm_max_ps(_0246, _1357), _2468);
        }

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max1x2(const float* src, size_t stride)
        {
            return _mm_max_ps(Load<align>(src), Load<align>(src + stride));
        }

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max3x2(const float* src, size_t stride)
        {
            __m128 _0123 = Pooling2x2Max1x2<align>(src, stride);
            __m128 _4567 = Pooling2x2Max1x2<align>(src + F, stride);
            __m128 _5678 = Pooling2x2Max1x2<false>(src + F + 1, stride);
            __m128 _0246 = _mm_shuffle_ps(_0123, _4567, 0x88);
            __m128 _1357 = _mm_shuffle_ps(_0123, _4567, 0xDD);
            __m128 _2468 = _mm_shuffle_ps(_0246, _5678, 0xD9);
            return _mm_max_ps(_mm_max_ps(_0246, _1357), _2468);
        }

        template <bool align> void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            size_t alignedWidth = AlignLo(width, DF);
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<align>(dst + (col >> 1), Pooling2x2Max3x3<align>(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), Pooling2x2Max3x3<false>(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Max2x3(src + widthEven, srcStride);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<align>(dst + (col >> 1), Pooling2x2Max3x2<align>(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), Pooling2x2Max3x2<false>(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Max2x2(src + widthEven, srcStride);
            }
        }

        void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling2x2Max3x3<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling2x2Max3x3<false>(src, srcStride, width, height, dst, dstStride);
        }

        //---------------------------------------------------------------------

        template<bool align> void NeuralPow(const float * src, size_t size, const float * exponent, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            float e = exponent[0];
            size_t alignedSize = AlignLo(size, F);
            __m128 _e = _mm_set1_ps(e);
            Pow pow;
            size_t i = 0;
            for (; i < alignedSize; i += F)
                Store<align>(dst + i, pow(Load<align>(src + i), _e));
            for (; i < size; ++i)
                dst[i] = Base::Pow(src[i], e);
        }

        void NeuralPow(const float * src, size_t size, const float * exponent, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralPow<true>(src, size, exponent, dst);
            else
                NeuralPow<false>(src, size, exponent, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralProductSum(const float* a, const float* b, size_t offset, __m128& sum)
        {
            __m128 _a = Load<align>(a + offset);
            __m128 _b = Load<align>(b + offset);
            sum = _mm_add_ps(sum, _mm_mul_ps(_a, _b));
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float* a, const float* b, size_t size, float* sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        NeuralProductSum<align>(a, b, i + F * 0, sums[0]);
                        NeuralProductSum<align>(a, b, i + F * 1, sums[1]);
                        NeuralProductSum<align>(a, b, i + F * 2, sums[2]);
                        NeuralProductSum<align>(a, b, i + F * 3, sums[3]);
                    }
                    sums[0] = _mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += F)
                    NeuralProductSum<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for (; i < size; ++i)
                *sum += a[i] * b[i];
        }

        void NeuralProductSum(const float* a, const float* b, size_t size, float* sum)
        {
            if (Aligned(a) && Aligned(b))
                NeuralProductSum<true>(a, b, size, sum);
            else
                NeuralProductSum<false>(a, b, size, sum);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            __m128 _slope = _mm_set1_ps(*slope);
            __m128 _0 = _mm_set1_ps(-0.0f);
            __m128 _1 = _mm_set1_ps(1.0f);
            __m128 _a = _mm_set1_ps(0.5417f);
            __m128 _b = _mm_set1_ps(0.1460f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 _src = Load<align>(src + i);
                __m128 x = _mm_andnot_ps(_0, _mm_mul_ps(_src, _slope));
                __m128 x2 = _mm_mul_ps(x, x);
                __m128 x4 = _mm_mul_ps(x2, x2);
                __m128 series = _mm_add_ps(_mm_add_ps(_1, x), _mm_add_ps(_mm_mul_ps(x2, _a), _mm_mul_ps(x4, _b)));
                __m128 mask = _mm_cmpgt_ps(_src, _0);
                __m128 exp = _mm_or_ps(_mm_and_ps(_mm_rcp_ps(series), mask), _mm_andnot_ps(mask, series));
                __m128 sigmoid = _mm_rcp_ps(_mm_add_ps(_1, exp));
                Store<align>(dst + i, sigmoid);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughSigmoid(src[i] * slope[0]);
        }

        void NeuralRoughSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float* src, const __m128& k, const __m128& o, const __m128& m, float* dst)
        {
            __m128 _src = Load<align>(src);
            __m128 e1 = _mm_max_ps(m, _mm_sub_ps(o, _mm_mul_ps(_src, k)));
            __m128 e2 = _mm_mul_ps(e1, e1);
            __m128 e4 = _mm_mul_ps(e2, e2);
            __m128 e8 = _mm_mul_ps(e4, e4);
            __m128 e16 = _mm_mul_ps(e8, e8);
            __m128 e32 = _mm_mul_ps(e16, e16);
            __m128 e64 = _mm_mul_ps(e32, e32);
            __m128 sigmoid = _mm_rcp_ps(_mm_add_ps(o, _mm_mul_ps(e64, e64)));
            Store<align>(dst, sigmoid);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            __m128 _k = _mm_set1_ps((*slope) * 0.0078125f);
            __m128 _1 = _mm_set1_ps(1.0f);
            __m128 _05 = _mm_set1_ps(0.5f);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                NeuralRoughSigmoid2<align>(src + i + 0 * F, _k, _1, _05, dst + i + 0 * F);
                NeuralRoughSigmoid2<align>(src + i + 1 * F, _k, _1, _05, dst + i + 1 * F);
                NeuralRoughSigmoid2<align>(src + i + 2 * F, _k, _1, _05, dst + i + 2 * F);
                NeuralRoughSigmoid2<align>(src + i + 3 * F, _k, _1, _05, dst + i + 3 * F);
            }
            for (; i < partialAlignedSize; i += F)
                NeuralRoughSigmoid2<align>(src + i, _k, _1, _05, dst + i);
            for (; i < size; ++i)
                dst[i] = Base::RoughSigmoid2(src[i] * slope[0]);
        }

        void NeuralRoughSigmoid2(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid2<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid2<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralRoughTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            __m128 _slope = _mm_set1_ps(*slope);
            __m128 _0 = _mm_set1_ps(-0.0f);
            __m128 _1 = _mm_set1_ps(1.0f);
            __m128 _a = _mm_set1_ps(0.5658f);
            __m128 _b = _mm_set1_ps(0.1430f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 _src = Load<align>(src + i);
                __m128 x = _mm_andnot_ps(_0, _mm_mul_ps(_src, _slope));
                __m128 x2 = _mm_mul_ps(x, x);
                __m128 x4 = _mm_mul_ps(x2, x2);
                __m128 pe = _mm_add_ps(_mm_add_ps(_1, x), _mm_add_ps(_mm_mul_ps(x2, _a), _mm_mul_ps(x4, _b)));
                __m128 ne = _mm_rcp_ps(pe);
                __m128 absTanh = _mm_mul_ps(_mm_sub_ps(pe, ne), _mm_rcp_ps(_mm_add_ps(pe, ne)));
                __m128 tanh = _mm_xor_ps(absTanh, _mm_and_ps(_0, _mm_cmpgt_ps(_0, _src)));
                Store<align>(dst + i, tanh);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughTanh(src[i] * slope[0]);
        }

        void NeuralRoughTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughTanh<true>(src, size, slope, dst);
            else
                NeuralRoughTanh<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void UpdateWeights(const float* x, const __m128& a, const __m128& b, float* d, float* w)
        {
            __m128 _d = _mm_add_ps(_mm_mul_ps(a, Load<align>(d)), _mm_mul_ps(b, Load<align>(x)));
            Store<align>(d, _d);
            Store<align>(w, _mm_add_ps(Load<align>(w), _d));
        }

        template <bool align> SIMD_INLINE void UpdateWeights(const float* x, size_t offset, const __m128& a, const __m128& b, float* d, float* w)
        {
            UpdateWeights<align>(x + offset, a, b, d + offset, w + offset);
        }

        template <bool align> void NeuralUpdateWeights(const float* x, size_t size, const float& a, const float& b, float* d, float* w)
        {
            if (align)
                assert(Aligned(x) && Aligned(d) && Aligned(w));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            __m128 _a = _mm_set1_ps(a);
            __m128 _b = _mm_set1_ps(b);
            size_t i = 0;
            if (partialAlignedSize)
            {
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        UpdateWeights<align>(x, i + F * 0, _a, _b, d, w);
                        UpdateWeights<align>(x, i + F * 1, _a, _b, d, w);
                        UpdateWeights<align>(x, i + F * 2, _a, _b, d, w);
                        UpdateWeights<align>(x, i + F * 3, _a, _b, d, w);
                    }
                }
                for (; i < partialAlignedSize; i += F)
                    UpdateWeights<align>(x, i, _a, _b, d, w);
            }
            for (; i < size; ++i)
                Base::UpdateWeights(x, i, a, b, d, w);
        }

        void NeuralUpdateWeights(const float* x, size_t size, const float* a, const float* b, float* d, float* w)
        {
            if (Aligned(x) && Aligned(d) && Aligned(w))
                NeuralUpdateWeights<true>(x, size, *a, *b, d, w);
            else
                NeuralUpdateWeights<false>(x, size, *a, *b, d, w);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
