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

#ifdef SIMD_AVX2_ENABLE 
    namespace Avx2
    {
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
#endif//SIMD_AVX2_ENABLE 
}
#endif//__SimdNeural_h__
