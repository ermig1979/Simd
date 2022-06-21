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
#ifndef __SimdNeural_h__
#define __SimdNeural_h__

#include "Simd/SimdLoad.h"

namespace Simd
{
    template<int count> struct ConvolutionBackwardBuffer
    {
        ConvolutionBackwardBuffer(size_t width, size_t align)
        {
            _size = width * sizeof(float);
            size_t stride = AlignHi(width + 2 * (count - 1), align);
            size_t full = count*stride * sizeof(float);
            _ptr = Allocate(full);
            memset(_ptr, 0, full);
            rows[0] = (float*)_ptr;
            for (size_t i = 1; i < count; ++i)
                rows[i] = rows[i - 1] + stride;
        }

        void Update(const float * src)
        {
            float * tmp = rows[0];
            if (src == NULL)
                memset(tmp + count - 1, 0, _size);
            else
                memcpy(tmp + count - 1, src, _size);
            for (size_t i = 0; i < count - 1; ++i)
                rows[i] = rows[i + 1];
            rows[count - 1] = tmp;
        }

        ~ConvolutionBackwardBuffer()
        {
            Free(_ptr);
        }

        float * rows[count];
    private:
        size_t _size;
        void * _ptr;
    };

#ifdef SIMD_SSE2_ENABLE 
    namespace Sse2
    {
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
    }
#endif

#ifdef SIMD_AVX_ENABLE 
    namespace Avx
    {
        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, const __m256& value, float* dst)
        {
            Store<align>(dst, _mm256_add_ps(Load<align>(dst), _mm256_mul_ps(value, Load<align>(src))));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
        {
            size_t i = 0;
            if (partial)
            {
                __m256 _value = _mm256_set1_ps(value);
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
    }
#endif

#ifdef SIMD_AVX2_ENABLE 
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, const __m256& value, float* dst)
        {
            Avx::Store<align>(dst, _mm256_fmadd_ps(value, Load<align>(src), Load<align>(dst)));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
        {
            size_t i = 0;
            if (partial)
            {
                __m256 _value = _mm256_set1_ps(value);
                for (; i < aligned; i += QF)
                {
                    AddMultiplied<align>(src + i + F * 0, _value, dst + i + 0);
                    AddMultiplied<align>(src + i + F * 1, _value, dst + i + 8);
                    AddMultiplied<align>(src + i + F * 2, _value, dst + i + 16);
                    AddMultiplied<align>(src + i + F * 3, _value, dst + i + 24);
                }
                for (; i < partial; i += F)
                    AddMultiplied<align>(src + i, _value, dst + i);
            }
            for (; i < full; ++i)
                dst[i] += src[i] * value;
        }

        //-----------------------------------------------------------------------------------------

        template <size_t size> SIMD_INLINE void LoadWeightsForward(const float * src, __m256 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm256_set1_ps(src[i]);
        }

        template <size_t size> SIMD_INLINE void LoadWeightsBackward(const float * src, __m256 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm256_set1_ps(src[size - i - 1]);
        }

        template<size_t coreX, size_t coreY> struct Convolution
        {
            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights);

            template<bool align> static SIMD_INLINE __m256 Backward(const ConvolutionBackwardBuffer<coreX> & buffer, size_t offset, const __m256 * weights);

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums);
        };

        template<> struct Convolution<2, 2>
        {
            template <bool align> static SIMD_INLINE __m256 RowConvolution(const float * src, const __m256 * weights)
            {
                return _mm256_fmadd_ps(Avx::Load<align>(src), weights[0],
                    _mm256_mul_ps(Avx::Load<false>(src + 1), weights[1]));
            }

            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights)
            {
                return _mm256_add_ps(RowConvolution<align>(src, weights),
                    RowConvolution<align>(src + stride, weights + 2));
            }

            template<bool align> static SIMD_INLINE __m256 Backward(const ConvolutionBackwardBuffer<2> & buffer, size_t offset, const __m256 * weights)
            {
                return _mm256_add_ps(RowConvolution<align>(buffer.rows[0] + offset, weights),
                    RowConvolution<align>(buffer.rows[1] + offset, weights + 2));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m256 & dst, __m256 * sums)
            {
                sums[0] = _mm256_fmadd_ps(dst, Load<align>(src + 0), sums[0]);
                sums[1] = _mm256_fmadd_ps(dst, Load<false>(src + 1), sums[1]);
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 2);
            }
        };

        template<> struct Convolution<3, 3>
        {
            template <bool align> static SIMD_INLINE __m256 RowConvolution(const float * src, const __m256 * weights)
            {
                return _mm256_fmadd_ps(Avx::Load<align>(src), weights[0],
                    _mm256_fmadd_ps(Avx::Load<false>(src + 1), weights[1],
                        _mm256_mul_ps(Avx::Load<false>(src + 2), weights[2])));
            }

            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights)
            {
                return _mm256_add_ps(RowConvolution<align>(src, weights),
                    _mm256_add_ps(RowConvolution<align>(src + stride, weights + 3),
                        RowConvolution<align>(src + 2 * stride, weights + 6)));
            }

            template<bool align> static SIMD_INLINE __m256 Backward(const ConvolutionBackwardBuffer<3> & buffer, size_t offset, const __m256 * weights)
            {
                return _mm256_add_ps(RowConvolution<align>(buffer.rows[0] + offset, weights),
                    _mm256_add_ps(RowConvolution<align>(buffer.rows[1] + offset, weights + 3),
                        RowConvolution<align>(buffer.rows[2] + offset, weights + 6)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m256 & dst, __m256 * sums)
            {
                __m256 s0 = Load<align>(src + 0);
                __m256 s4 = Load<false>(src + 4);
                sums[0] = _mm256_fmadd_ps(dst, s0, sums[0]);
                sums[1] = _mm256_fmadd_ps(dst, Alignr<1>(s0, s4), sums[1]);
                sums[2] = _mm256_fmadd_ps(dst, Alignr<2>(s0, s4), sums[2]);
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 3);
                Sum<align>(src + stride * 2, dst, sums + 6);
            }
        };

        template<> struct Convolution<4, 4>
        {
            template <bool align> static SIMD_INLINE __m256 RowConvolution(const float * src, const __m256 * weights)
            {
                return _mm256_add_ps(
                    _mm256_fmadd_ps(Load<align>(src + 0), weights[0], _mm256_mul_ps(Load<false>(src + 1), weights[1])),
                    _mm256_fmadd_ps(Load<false>(src + 2), weights[2], _mm256_mul_ps(Load<false>(src + 3), weights[3])));
            }

            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_add_ps(RowConvolution<align>(src, weights),
                    RowConvolution<align>(src + stride, weights + 4)),
                    _mm256_add_ps(RowConvolution<align>(src + 2 * stride, weights + 8),
                        RowConvolution<align>(src + 3 * stride, weights + 12)));
            }

            template<bool align> static SIMD_INLINE __m256 Backward(const ConvolutionBackwardBuffer<4> & buffer, size_t offset, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_add_ps(RowConvolution<align>(buffer.rows[0] + offset, weights),
                    RowConvolution<align>(buffer.rows[1] + offset, weights + 4)),
                    _mm256_add_ps(RowConvolution<align>(buffer.rows[2] + offset, weights + 8),
                        RowConvolution<align>(buffer.rows[3] + offset, weights + 12)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m256 & dst, __m256 * sums)
            {
                __m256 s0 = Load<align>(src + 0);
                __m256 s4 = Load<false>(src + 4);
                sums[0] = _mm256_fmadd_ps(dst, s0, sums[0]);
                sums[1] = _mm256_fmadd_ps(dst, Alignr<1>(s0, s4), sums[1]);
                sums[2] = _mm256_fmadd_ps(dst, Alignr<2>(s0, s4), sums[2]);
                sums[3] = _mm256_fmadd_ps(dst, Alignr<3>(s0, s4), sums[3]);
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 4);
                Sum<align>(src + stride * 2, dst, sums + 8);
                Sum<align>(src + stride * 3, dst, sums + 12);
            }
        };

        template<> struct Convolution<5, 5>
        {
            template <bool align> static SIMD_INLINE __m256 RowConvolution(const float * src, const __m256 * weights)
            {
                __m256 s0 = Load<align>(src + 0);
                __m256 s4 = Load<false>(src + 4);
                return _mm256_fmadd_ps(s0, weights[0], _mm256_add_ps(
                    _mm256_fmadd_ps(Alignr<1>(s0, s4), weights[1], _mm256_mul_ps(Alignr<2>(s0, s4), weights[2])),
                    _mm256_fmadd_ps(s4, weights[4], _mm256_mul_ps(Alignr<3>(s0, s4), weights[3]))));
            }

            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights)
            {
                return _mm256_add_ps(RowConvolution<align>(src, weights),
                    _mm256_add_ps(_mm256_add_ps(RowConvolution<align>(src + stride, weights + 5),
                        RowConvolution<align>(src + 2 * stride, weights + 10)),
                        _mm256_add_ps(RowConvolution<align>(src + 3 * stride, weights + 15),
                            RowConvolution<align>(src + 4 * stride, weights + 20))));
            }

            template<bool align> static SIMD_INLINE __m256 Backward(const ConvolutionBackwardBuffer<5> & buffer, size_t offset, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_add_ps(RowConvolution<align>(buffer.rows[0] + offset, weights),
                    _mm256_add_ps(RowConvolution<align>(buffer.rows[1] + offset, weights + 5),
                        RowConvolution<align>(buffer.rows[2] + offset, weights + 10))),
                    _mm256_add_ps(RowConvolution<align>(buffer.rows[3] + offset, weights + 15),
                        RowConvolution<align>(buffer.rows[4] + offset, weights + 20)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m256 & dst, __m256 * sums)
            {
                __m256 s0 = Load<align>(src + 0);
                __m256 s4 = Load<false>(src + 4);
                sums[0] = _mm256_fmadd_ps(dst, s0, sums[0]);
                sums[1] = _mm256_fmadd_ps(dst, Alignr<1>(s0, s4), sums[1]);
                sums[2] = _mm256_fmadd_ps(dst, Alignr<2>(s0, s4), sums[2]);
                sums[3] = _mm256_fmadd_ps(dst, Alignr<3>(s0, s4), sums[3]);
                sums[4] = _mm256_fmadd_ps(dst, s4, sums[4]);
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 5);
                Sum<align>(src + stride * 2, dst, sums + 10);
                Sum<align>(src + stride * 3, dst, sums + 15);
                Sum<align>(src + stride * 4, dst, sums + 20);
            }
        };
    }
#endif 

#ifdef SIMD_AVX512F_ENABLE 
    namespace Avx512f
    {
        template <size_t size> SIMD_INLINE void LoadWeightsForward(const float* src, __m512* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm512_set1_ps(src[i]);
        }

        template <size_t size> SIMD_INLINE void LoadWeightsBackward(const float* src, __m512* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm512_set1_ps(src[size - i - 1]);
        }

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
            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1);

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<coreX>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1);

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1);

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums, __mmask16 m = -1);

            template <bool align, bool mask> static SIMD_INLINE void Sum1x2(const float* src0, size_t srcStride, const float* dst0, __m512* sums);

            template <bool align, bool mask> static SIMD_INLINE void Sum2x2(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums);
        };

        template<> struct Convolution<2, 2>
        {
            template <bool align, bool mask> static SIMD_INLINE __m512 RowConvolution(const float* src, const __m512* weights, __mmask16 m = -1)
            {
                __m512 src0 = Load<align, mask>(src, m);
                __m512 src1 = Load<false, mask>(src + 1, m);
                return _mm512_fmadd_ps(src0, weights[0], _mm512_mul_ps(src1, weights[1]));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(src, weights, m);
                __m512 row1 = RowConvolution<align, mask>(src + stride, weights + 2, m);
                return _mm512_add_ps(row0, row1);
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<2>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(buffer.rows[0] + offset, weights + 0, m);
                __m512 row1 = RowConvolution<align, mask>(buffer.rows[1] + offset, weights + 2, m);
                return _mm512_add_ps(row0, row1);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                sums[0] = _mm512_fmadd_ps(dst00, (Load<align, mask>(src0 + 0, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Load<false, mask>(src0 + 1, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Load<align, mask>(src1 + 0, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, (Load<false, mask>(src1 + 1, m)), sums[3]);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* dst1 = dst0 + dstStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align, mask>(src0, m);
                __m512 src01 = Load<false, mask>(src0 + 1, m);
                __m512 src10 = Load<align, mask>(src1, m);
                __m512 src11 = Load<false, mask>(src1 + 1, m);
                sums[0] = _mm512_fmadd_ps(dst00, src00, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, src01, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, src10, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, src11, sums[3]);
                __m512 dst10 = Load<align, mask>(dst1, m);
                __m512 src20 = Load<align, mask>(src2, m);
                __m512 src21 = Load<false, mask>(src2 + 1, m);
                sums[0] = _mm512_fmadd_ps(dst10, src10, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst10, src11, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst10, src20, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst10, src21, sums[3]);
            }

            template <bool align> static SIMD_INLINE void Sum1x2(const float* src0, size_t srcStride, const float* dst0, __m512* sums)
            {
                const float* src1 = src0 + srcStride;
                __m512 dst00 = Load<align>(dst0);
                __m512 src00 = Load<align>(src0);
                __m512 src01 = Load<align>(src0 + F);
                __m512 src10 = Load<align>(src1);
                __m512 src11 = Load<align>(src1 + F);
                sums[0] = _mm512_fmadd_ps(dst00, src00, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, Alignr<1>(src00, src01), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, src10, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, Alignr<1>(src10, src11), sums[3]);
                __m512 dst10 = Load<align>(dst0 + F);
                __m512 src02 = Load<false>(src0 + F + 1);
                __m512 src12 = Load<false>(src1 + F + 1);
                sums[0] = _mm512_fmadd_ps(dst10, src01, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst10, src02, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst10, src11, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst10, src12, sums[3]);
            }

            template <bool align> static SIMD_INLINE void Sum2x2(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* dst1 = dst0 + dstStride;

                __m512 dst00 = Load<align>(dst0);
                __m512 src000 = Load<align>(src0);
                __m512 src010 = Load<align>(src0 + F);
                __m512 src100 = Load<align>(src1);
                __m512 src110 = Load<align>(src1 + F);
                __m512 src101 = Alignr<1>(src100, src110);
                sums[0] = _mm512_fmadd_ps(dst00, src000, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, Alignr<1>(src000, src010), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, src100, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, src101, sums[3]);

                __m512 dst01 = Load<align>(dst0 + F);
                __m512 src011 = Load<false>(src0 + F + 1);
                __m512 src111 = Load<false>(src1 + F + 1);
                sums[0] = _mm512_fmadd_ps(dst01, src010, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst01, src011, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst01, src110, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst01, src111, sums[3]);

                __m512 dst10 = Load<align>(dst1);
                __m512 src200 = Load<align>(src2);
                __m512 src210 = Load<align>(src2 + F);
                sums[0] = _mm512_fmadd_ps(dst10, src100, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst10, src101, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst10, src200, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst10, Alignr<1>(src200, src210), sums[3]);

                __m512 dst11 = Load<align>(dst1 + F);
                __m512 src211 = Load<false>(src2 + F + 1);
                sums[0] = _mm512_fmadd_ps(dst11, src110, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst11, src111, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst11, src210, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst11, src211, sums[3]);
            }
        };

        template<> struct Convolution<3, 3>
        {
            template <bool align, bool mask> static SIMD_INLINE __m512 RowConvolution(const float* src, const __m512* weights, __mmask16 m = -1)
            {
                __m512 src0 = Load<align, mask>(src, m);
                __m512 src1 = Load<false, mask>(src + 1, m);
                __m512 src2 = Load<false, mask>(src + 2, m);
                return _mm512_fmadd_ps(src0, weights[0], _mm512_fmadd_ps(src1, weights[1], _mm512_mul_ps(src2, weights[2])));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(src, weights, m);
                __m512 row1 = RowConvolution<align, mask>(src + stride, weights + 3, m);
                __m512 row2 = RowConvolution<align, mask>(src + 2 * stride, weights + 6, m);
                return _mm512_add_ps(_mm512_add_ps(row0, row1), row2);
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<3>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(buffer.rows[0] + offset, weights + 0, m);
                __m512 row1 = RowConvolution<align, mask>(buffer.rows[1] + offset, weights + 3, m);
                __m512 row2 = RowConvolution<align, mask>(buffer.rows[2] + offset, weights + 6, m);
                return _mm512_add_ps(_mm512_add_ps(row0, row1), row2);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[3] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src10, src1f, m)), sums[3]);
                sums[4] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src10, src1f, m)), sums[4]);
                sums[5] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src10, src1f, m)), sums[5]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[6] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src20, src2f, m)), sums[6]);
                sums[7] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src20, src2f, m)), sums[7]);
                sums[8] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src20, src2f, m)), sums[8]);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums, __mmask16 m = -1)
            {
                const float* dst1 = dst0 + dstStride;
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* src3 = src2 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                __m512 dst10 = Load<align, mask>(dst1, m);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[0] = _mm512_fmadd_ps(dst10, Mask<mask>(src10, m), sums[0]);
                sums[3] = _mm512_fmadd_ps(dst00, Mask<mask>(src10, m), sums[3]);
                __m512 src11 = Alignr<1, mask>(src10, src1f, m);
                sums[1] = _mm512_fmadd_ps(dst10, src11, sums[1]);
                sums[4] = _mm512_fmadd_ps(dst00, src11, sums[4]);
                __m512 src12 = Alignr<2, mask>(src10, src1f, m);
                sums[2] = _mm512_fmadd_ps(dst10, src12, sums[2]);
                sums[5] = _mm512_fmadd_ps(dst00, src12, sums[5]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[3] = _mm512_fmadd_ps(dst10, Mask<mask>(src20, m), sums[3]);
                sums[6] = _mm512_fmadd_ps(dst00, Mask<mask>(src20, m), sums[6]);
                __m512 src21 = Alignr<1, mask>(src20, src2f, m);
                sums[4] = _mm512_fmadd_ps(dst10, src21, sums[4]);
                sums[7] = _mm512_fmadd_ps(dst00, src21, sums[7]);
                __m512 src22 = Alignr<2, mask>(src20, src2f, m);
                sums[5] = _mm512_fmadd_ps(dst10, src22, sums[5]);
                sums[8] = _mm512_fmadd_ps(dst00, src22, sums[8]);
                __m512 src30 = Load<align>(src3);
                __m512 src3f = Load<align>(src3 + F);
                sums[6] = _mm512_fmadd_ps(dst10, (Alignr<0, mask>(src30, src3f, m)), sums[6]);
                sums[7] = _mm512_fmadd_ps(dst10, (Alignr<1, mask>(src30, src3f, m)), sums[7]);
                sums[8] = _mm512_fmadd_ps(dst10, (Alignr<2, mask>(src30, src3f, m)), sums[8]);
            }
        };

        template<> struct Convolution<4, 4>
        {
            template <bool align, bool mask> static SIMD_INLINE __m512 RowConvolution(const float* src, const __m512* weights, __mmask16 m = -1)
            {
                __m512 src0 = Load<align>(src);
                __m512 srcf = Load<align>(src + F);
                __m512 sum0 = _mm512_fmadd_ps(Alignr<0>(src0, srcf), weights[0], _mm512_mul_ps(Alignr<1>(src0, srcf), weights[1]));
                __m512 sum1 = _mm512_fmadd_ps(Alignr<2>(src0, srcf), weights[2], _mm512_mul_ps(Alignr<3>(src0, srcf), weights[3]));
                return _mm512_add_ps(sum0, sum1);
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(src, weights, m);
                __m512 row1 = RowConvolution<align, mask>(src + stride, weights + 4, m);
                __m512 row2 = RowConvolution<align, mask>(src + 2 * stride, weights + 8, m);
                __m512 row3 = RowConvolution<align, mask>(src + 3 * stride, weights + 12, m);
                return _mm512_add_ps(_mm512_add_ps(row0, row1), _mm512_add_ps(row2, row3));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<4>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(buffer.rows[0] + offset, weights + 0, m);
                __m512 row1 = RowConvolution<align, mask>(buffer.rows[1] + offset, weights + 4, m);
                __m512 row2 = RowConvolution<align, mask>(buffer.rows[2] + offset, weights + 8, m);
                __m512 row3 = RowConvolution<align, mask>(buffer.rows[3] + offset, weights + 12, m);
                return _mm512_add_ps(_mm512_add_ps(row0, row1), _mm512_add_ps(row2, row3));
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* src3 = src2 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src00, src0f, m)), sums[3]);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[4] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src10, src1f, m)), sums[4]);
                sums[5] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src10, src1f, m)), sums[5]);
                sums[6] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src10, src1f, m)), sums[6]);
                sums[7] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src10, src1f, m)), sums[7]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[8] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src20, src2f, m)), sums[8]);
                sums[9] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src20, src2f, m)), sums[9]);
                sums[10] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src20, src2f, m)), sums[10]);
                sums[11] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src20, src2f, m)), sums[11]);
                __m512 src30 = Load<align>(src3);
                __m512 src3f = Load<align>(src3 + F);
                sums[12] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src30, src3f, m)), sums[12]);
                sums[13] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src30, src3f, m)), sums[13]);
                sums[14] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src30, src3f, m)), sums[14]);
                sums[15] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src30, src3f, m)), sums[15]);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums, __mmask16 m = -1)
            {
                const float* dst1 = dst0 + dstStride;
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* src3 = src2 + srcStride;
                const float* src4 = src3 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src00, src0f, m)), sums[3]);
                __m512 dst10 = Load<align, mask>(dst1, m);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[0] = _mm512_fmadd_ps(dst10, Mask<mask>(src10, m), sums[0]);
                sums[4] = _mm512_fmadd_ps(dst00, Mask<mask>(src10, m), sums[4]);
                __m512 src11 = Alignr<1, mask>(src10, src1f, m);
                sums[1] = _mm512_fmadd_ps(dst10, src11, sums[1]);
                sums[5] = _mm512_fmadd_ps(dst00, src11, sums[5]);
                __m512 src12 = Alignr<2, mask>(src10, src1f, m);
                sums[2] = _mm512_fmadd_ps(dst10, src12, sums[2]);
                sums[6] = _mm512_fmadd_ps(dst00, src12, sums[6]);
                __m512 src13 = Alignr<3, mask>(src10, src1f, m);
                sums[3] = _mm512_fmadd_ps(dst10, src13, sums[3]);
                sums[7] = _mm512_fmadd_ps(dst00, src13, sums[7]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[4] = _mm512_fmadd_ps(dst10, Mask<mask>(src20, m), sums[4]);
                sums[8] = _mm512_fmadd_ps(dst00, Mask<mask>(src20, m), sums[8]);
                __m512 src21 = Alignr<1, mask>(src20, src2f, m);
                sums[5] = _mm512_fmadd_ps(dst10, src21, sums[5]);
                sums[9] = _mm512_fmadd_ps(dst00, src21, sums[9]);
                __m512 src22 = Alignr<2, mask>(src20, src2f, m);
                sums[6] = _mm512_fmadd_ps(dst10, src22, sums[6]);
                sums[10] = _mm512_fmadd_ps(dst00, src22, sums[10]);
                __m512 src23 = Alignr<3, mask>(src20, src2f, m);
                sums[7] = _mm512_fmadd_ps(dst10, src23, sums[7]);
                sums[11] = _mm512_fmadd_ps(dst00, src23, sums[11]);
                __m512 src30 = Load<align>(src3);
                __m512 src3f = Load<align>(src3 + F);
                sums[8] = _mm512_fmadd_ps(dst10, Mask<mask>(src30, m), sums[8]);
                sums[12] = _mm512_fmadd_ps(dst00, Mask<mask>(src30, m), sums[12]);
                __m512 src31 = Alignr<1, mask>(src30, src3f, m);
                sums[9] = _mm512_fmadd_ps(dst10, src31, sums[9]);
                sums[13] = _mm512_fmadd_ps(dst00, src31, sums[13]);
                __m512 src32 = Alignr<2, mask>(src30, src3f, m);
                sums[10] = _mm512_fmadd_ps(dst10, src32, sums[10]);
                sums[14] = _mm512_fmadd_ps(dst00, src32, sums[14]);
                __m512 src33 = Alignr<3, mask>(src30, src3f, m);
                sums[11] = _mm512_fmadd_ps(dst10, src33, sums[11]);
                sums[15] = _mm512_fmadd_ps(dst00, src33, sums[15]);
                __m512 src40 = Load<align>(src4);
                __m512 src4f = Load<align>(src4 + F);
                sums[12] = _mm512_fmadd_ps(dst10, (Alignr<0, mask>(src40, src4f, m)), sums[12]);
                sums[13] = _mm512_fmadd_ps(dst10, (Alignr<1, mask>(src40, src4f, m)), sums[13]);
                sums[14] = _mm512_fmadd_ps(dst10, (Alignr<2, mask>(src40, src4f, m)), sums[14]);
                sums[15] = _mm512_fmadd_ps(dst10, (Alignr<3, mask>(src40, src4f, m)), sums[15]);
            }
        };

        template<> struct Convolution<5, 5>
        {
            template <bool align, bool mask> static SIMD_INLINE __m512 RowConvolution(const float* src, const __m512* weights, __mmask16 m = -1)
            {
                __m512 src0 = Load<align>(src);
                __m512 srcf = Load<align>(src + F);
                __m512 sum0 = _mm512_fmadd_ps(Alignr<0>(src0, srcf), weights[0], _mm512_mul_ps(Alignr<1>(src0, srcf), weights[1]));
                __m512 sum1 = _mm512_fmadd_ps(Alignr<2>(src0, srcf), weights[2], _mm512_mul_ps(Alignr<3>(src0, srcf), weights[3]));
                return _mm512_fmadd_ps(Alignr<4>(src0, srcf), weights[4], _mm512_add_ps(sum0, sum1));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1)
            {
                return _mm512_add_ps((RowConvolution<align, mask>(src, weights, m)),
                    _mm512_add_ps(_mm512_add_ps((RowConvolution<align, mask>(src + stride, weights + 5, m)),
                        (RowConvolution<align, mask>(src + 2 * stride, weights + 10, m))),
                        _mm512_add_ps((RowConvolution<align, mask>(src + 3 * stride, weights + 15, m)),
                            (RowConvolution<align, mask>(src + 4 * stride, weights + 20, m)))));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<5>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1)
            {
                return _mm512_add_ps((RowConvolution<align, mask>(buffer.rows[0] + offset, weights, m)),
                    _mm512_add_ps(_mm512_add_ps((RowConvolution<align, mask>(buffer.rows[1] + offset, weights + 5, m)),
                        (RowConvolution<align, mask>(buffer.rows[2] + offset, weights + 10, m))),
                        _mm512_add_ps((RowConvolution<align, mask>(buffer.rows[3] + offset, weights + 15, m)),
                            (RowConvolution<align, mask>(buffer.rows[4] + offset, weights + 20, m)))));
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* src3 = src2 + srcStride;
                const float* src4 = src3 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src00, src0f, m)), sums[3]);
                sums[4] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src00, src0f, m)), sums[4]);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[5] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src10, src1f, m)), sums[5]);
                sums[6] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src10, src1f, m)), sums[6]);
                sums[7] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src10, src1f, m)), sums[7]);
                sums[8] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src10, src1f, m)), sums[8]);
                sums[9] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src10, src1f, m)), sums[9]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[10] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src20, src2f, m)), sums[10]);
                sums[11] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src20, src2f, m)), sums[11]);
                sums[12] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src20, src2f, m)), sums[12]);
                sums[13] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src20, src2f, m)), sums[13]);
                sums[14] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src20, src2f, m)), sums[14]);
                __m512 src30 = Load<align>(src3);
                __m512 src3f = Load<align>(src3 + F);
                sums[15] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src30, src3f, m)), sums[15]);
                sums[16] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src30, src3f, m)), sums[16]);
                sums[17] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src30, src3f, m)), sums[17]);
                sums[18] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src30, src3f, m)), sums[18]);
                sums[19] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src30, src3f, m)), sums[19]);
                __m512 src40 = Load<align>(src4);
                __m512 src4f = Load<align>(src4 + F);
                sums[20] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src40, src4f, m)), sums[20]);
                sums[21] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src40, src4f, m)), sums[21]);
                sums[22] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src40, src4f, m)), sums[22]);
                sums[23] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src40, src4f, m)), sums[23]);
                sums[24] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src40, src4f, m)), sums[24]);
            }

            template <bool align, bool mask> static SIMD_INLINE void SumRow1(const float* src, const __m512& dst, __m512* sums, __mmask16 m)
            {
                __m512 src0 = Load<align>(src + 0);
                __m512 srcf = Load<align>(src + F);
                sums[0] = _mm512_fmadd_ps(dst, (Alignr<0, mask>(src0, srcf, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst, (Alignr<1, mask>(src0, srcf, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst, (Alignr<2, mask>(src0, srcf, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst, (Alignr<3, mask>(src0, srcf, m)), sums[3]);
                sums[4] = _mm512_fmadd_ps(dst, (Alignr<4, mask>(src0, srcf, m)), sums[4]);
            }

            template <bool align, bool mask> static SIMD_INLINE void SumRow2(const float* src, const __m512& dst0, const __m512& dst1, __m512* sums, __mmask16 m)
            {
                __m512 src0 = Load<align>(src + 0);
                __m512 srcf = Load<align>(src + F);
                sums[0] = _mm512_fmadd_ps(dst1, Mask<mask>(src0, m), sums[0]);
                sums[5] = _mm512_fmadd_ps(dst0, Mask<mask>(src0, m), sums[5]);
                __m512 src1 = Alignr<1, mask>(src0, srcf, m);
                sums[1] = _mm512_fmadd_ps(dst1, src1, sums[1]);
                sums[6] = _mm512_fmadd_ps(dst0, src1, sums[6]);
                __m512 src2 = Alignr<2, mask>(src0, srcf, m);
                sums[2] = _mm512_fmadd_ps(dst1, src2, sums[2]);
                sums[7] = _mm512_fmadd_ps(dst0, src2, sums[7]);
                __m512 src3 = Alignr<3, mask>(src0, srcf, m);
                sums[3] = _mm512_fmadd_ps(dst1, src3, sums[3]);
                sums[8] = _mm512_fmadd_ps(dst0, src3, sums[8]);
                __m512 src4 = Alignr<4, mask>(src0, srcf, m);
                sums[4] = _mm512_fmadd_ps(dst1, src4, sums[4]);
                sums[9] = _mm512_fmadd_ps(dst0, src4, sums[9]);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src, size_t srcStride, const float* dst, size_t dstStride, __m512* sums, __mmask16 m = -1)
            {
                __m512 dst0 = Load<align, mask>(dst, m);
                SumRow1<align, mask>(src, dst0, sums + 0, m);
                __m512 dst1 = Load<align, mask>(dst + dstStride, m);
                SumRow2<align, mask>(src + srcStride, dst0, dst1, sums + 0, m);
                SumRow2<align, mask>(src + 2 * srcStride, dst0, dst1, sums + 5, m);
                SumRow2<align, mask>(src + 3 * srcStride, dst0, dst1, sums + 10, m);
                SumRow2<align, mask>(src + 4 * srcStride, dst0, dst1, sums + 15, m);
                SumRow1<align, mask>(src + 5 * srcStride, dst1, sums + 20, m);
            }
        };
    }
#endif

#ifdef SIMD_NEON_ENABLE 
    namespace Neon
    {
        SIMD_INLINE void Add4ExtractedSums(const float32x4_t* src, float* dst)
        {
            float32x2_t sm0 = vadd_f32(vget_high_f32(src[0]), vget_low_f32(src[0]));
            float32x2_t sm1 = vadd_f32(vget_high_f32(src[1]), vget_low_f32(src[1]));
            float32x2_t sm2 = vadd_f32(vget_high_f32(src[2]), vget_low_f32(src[2]));
            float32x2_t sm3 = vadd_f32(vget_high_f32(src[3]), vget_low_f32(src[3]));
            float32x2_t sm01 = vpadd_f32(sm0, sm1);
            float32x2_t sm23 = vpadd_f32(sm2, sm3);
            float32x4_t sm0123 = vcombine_f32(sm01, sm23);
            vst1q_f32(dst, vaddq_f32(vld1q_f32(dst), sm0123));
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, const float32x4_t& value, float* dst)
        {
            Store<align>(dst, vmlaq_f32(Load<align>(dst), value, Load<align>(src)));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(aligned, QF) && Aligned(partial, F));
            size_t i = 0;
            if (partial)
            {
                float32x4_t _value = vdupq_n_f32(value);
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
    }
#endif
}
#endif//__SimdNeural_h__
