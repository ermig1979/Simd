/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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

#ifdef SIMD_SSE41_ENABLE 
    namespace Sse41
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

#ifdef SIMD_AVX2_ENABLE 
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, const __m256& value, float* dst)
        {
            Store<align>(dst, _mm256_fmadd_ps(value, Load<align>(src), Load<align>(dst)));
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
                return _mm256_fmadd_ps(Load<align>(src), weights[0],
                    _mm256_mul_ps(Load<false>(src + 1), weights[1]));
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
                return _mm256_fmadd_ps(Load<align>(src), weights[0],
                    _mm256_fmadd_ps(Load<false>(src + 1), weights[1],
                        _mm256_mul_ps(Load<false>(src + 2), weights[2])));
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

#ifdef SIMD_AVX512BW_ENABLE 
    namespace Avx512bw
    {
        template <bool align, bool mask> SIMD_INLINE void AddMultiplied(const float* src, const __m512& value, float* dst, __mmask16 m = -1)
        {
            __m512 _src = Load<align, mask>(src, m);
            __m512 _dst = Load<align, mask>(dst, m);
            Store<align, mask>(dst, _mm512_fmadd_ps(value, _src, _dst), m);
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
        {
            size_t i = 0;
            __m512 _value = _mm512_set1_ps(value);
            for (; i < aligned; i += QF)
            {
                AddMultiplied<align, false>(src + i + F * 0, _value, dst + i + F * 0);
                AddMultiplied<align, false>(src + i + F * 1, _value, dst + i + F * 1);
                AddMultiplied<align, false>(src + i + F * 2, _value, dst + i + F * 2);
                AddMultiplied<align, false>(src + i + F * 3, _value, dst + i + F * 3);
            }
            for (; i < partial; i += F)
                AddMultiplied<align, false>(src + i, _value, dst + i);
            if (i < full)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - full);
                AddMultiplied<align, true>(src + i, _value, dst + i, tailMask);
            }
        }
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
