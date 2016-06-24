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
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        template <bool align> SIMD_INLINE void AnnProductSum(const float * a, const float * b, size_t offset, __m256 & sum)
        {
            __m256 _a = Load<align>(a + offset);
            __m256 _b = Load<align>(b + offset);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_a, _b));
        }

        template <bool align> SIMD_INLINE void AnnProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, 8);
            size_t fullAlignedSize = AlignLo(size, 32);
            size_t i = 0;
            if(partialAlignedSize)
            {
                __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
                if(fullAlignedSize)
                {
                    for(; i < fullAlignedSize; i += 32)
                    {
						AnnProductSum<align>(a, b, i, sums[0]);
						AnnProductSum<align>(a, b, i + 8, sums[1]);
						AnnProductSum<align>(a, b, i + 16, sums[2]);
						AnnProductSum<align>(a, b, i + 24, sums[3]);
                    }
                    sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
                }
                for(; i < partialAlignedSize; i += 8)
					AnnProductSum<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for(; i < size; ++i)
                *sum += a[i]*b[i];
        }

        void AnnProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(Aligned(a) && Aligned(b))
				AnnProductSum<true>(a, b, size, sum);
            else
				AnnProductSum<false>(a, b, size, sum);
        }

		template <bool align> SIMD_INLINE void AnnRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			size_t alignedSize = Simd::AlignLo(size, 8);
			__m256 _slope = _mm256_set1_ps(*slope);
			__m256 _0 = _mm256_set1_ps(-0.0f);
			__m256 _1 = _mm256_set1_ps(1.0f);
			__m256 _0555 = _mm256_set1_ps(0.555f);
			__m256 _0143 = _mm256_set1_ps(0.143f);
			size_t i = 0;
			for (; i < alignedSize; i += 8)
			{
				__m256 _src = Load<align>(src + i);
				__m256 x = _mm256_andnot_ps(_0, _mm256_mul_ps(_src, _slope));
				__m256 x2 = _mm256_mul_ps(x, x);
				__m256 x4 = _mm256_mul_ps(x2, x2);
				__m256 series = _mm256_add_ps(_mm256_add_ps(_1, x), _mm256_add_ps(_mm256_mul_ps(x2, _0555), _mm256_mul_ps(x4, _0143)));
				__m256 mask = _mm256_cmp_ps(_src, _0, _CMP_GT_OS);
				__m256 exp = _mm256_or_ps(_mm256_and_ps(_mm256_rcp_ps(series), mask), _mm256_andnot_ps(mask, series));
				__m256 sigmoid = _mm256_rcp_ps(_mm256_add_ps(_1, exp));
				Store<align>(dst + i, sigmoid);
			}
			for (; i < size; ++i)
				dst[i] = Base::RoughSigmoid(src[i] * slope[0]);
		}

		void AnnRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			if (Aligned(src) && Aligned(dst))
				AnnRoughSigmoid<true>(src, size, slope, dst);
			else
				AnnRoughSigmoid<false>(src, size, slope, dst);
		}

        template <bool align> SIMD_INLINE void AnnDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 8);
            __m256 _slope = _mm256_set1_ps(*slope);
            __m256 _1 = _mm256_set1_ps(1.0f);
            size_t i = 0;
            for (; i < alignedSize; i += 8)
            {
                __m256 _src = Load<align>(src + i);
                Store<align>(dst + i, _mm256_mul_ps(_slope, _mm256_mul_ps(_mm256_sub_ps(_1, _src), _src)));
            }
            for (; i < size; ++i)
                dst[i] = slope[0] * Base::DerivativeSigmoid(src[i]);
        }

        void AnnDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                AnnDerivativeSigmoid<true>(src, size, slope, dst);
            else
                AnnDerivativeSigmoid<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void AnnRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 8);
            __m256 _slope = _mm256_set1_ps(*slope);
            __m256 _0 = _mm256_set1_ps(-0.0f);
            __m256 _1 = _mm256_set1_ps(1.0f);
            __m256 _0559 = _mm256_set1_ps(0.559f);
            __m256 _0148 = _mm256_set1_ps(0.148f);
            size_t i = 0;
            for (; i < alignedSize; i += 8)
            {
                __m256 _src = Load<align>(src + i);
                __m256 x = _mm256_andnot_ps(_0, _mm256_mul_ps(_src, _slope));
                __m256 x2 = _mm256_mul_ps(x, x);
                __m256 x4 = _mm256_mul_ps(x2, x2);
                __m256 pe = _mm256_add_ps(_mm256_add_ps(_1, x), _mm256_add_ps(_mm256_mul_ps(x2, _0559), _mm256_mul_ps(x4, _0148)));
                __m256 ne = _mm256_rcp_ps(pe);
                __m256 absTanh = _mm256_mul_ps(_mm256_sub_ps(pe, ne), _mm256_rcp_ps(_mm256_add_ps(pe, ne)));
                __m256 tanh = _mm256_xor_ps(absTanh, _mm256_and_ps(_0, _mm256_cmp_ps(_0, _src, _CMP_GT_OS)));
                Store<align>(dst + i, tanh);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughTanh(src[i] * slope[0]);
        }

        void AnnRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                AnnRoughTanh<true>(src, size, slope, dst);
            else
                AnnRoughTanh<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void UpdateWeights(const float * x, const __m256 & a, const __m256 & b, float * d, float * w)
        {
            __m256 _d = _mm256_add_ps(_mm256_mul_ps(a, Load<align>(d)), _mm256_mul_ps(b, Load<align>(x)));
            Store<align>(d, _d);
            Store<align>(w, _mm256_add_ps(Load<align>(w), _d));
        }

        template <bool align> SIMD_INLINE void UpdateWeights(const float * x, size_t offset, const __m256 & a, const __m256 & b, float * d, float * w)
        {
            UpdateWeights<align>(x + offset, a, b, d + offset, w + offset);
        }

        template <bool align> SIMD_INLINE void AnnUpdateWeights(const float * x, size_t size, const float & a, const float & b, float * d, float * w)
        {
            if (align)
                assert(Aligned(x) && Aligned(d) && Aligned(w));

            size_t partialAlignedSize = AlignLo(size, 8);
            size_t fullAlignedSize = AlignLo(size, 32);
            __m256 _a = _mm256_set1_ps(a);
            __m256 _b = _mm256_set1_ps(b);
            size_t i = 0;
            if (partialAlignedSize)
            {
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += 32)
                    {
                        UpdateWeights<align>(x, i + 0, _a, _b, d, w);
                        UpdateWeights<align>(x, i + 8, _a, _b, d, w);
                        UpdateWeights<align>(x, i + 16, _a, _b, d, w);
                        UpdateWeights<align>(x, i + 24, _a, _b, d, w);
                    }
                }
                for (; i < partialAlignedSize; i += 8)
                    UpdateWeights<align>(x, i, _a, _b, d, w);
            }
            for (; i < size; ++i)
                Base::UpdateWeights(x, i, a, b, d, w);
        }

        void AnnUpdateWeights(const float * x, size_t size, const float * a, const float * b, float * d, float * w)
        {
            if (Aligned(x) && Aligned(d) && Aligned(w))
                AnnUpdateWeights<true>(x, size, *a, *b, d, w);
            else
                AnnUpdateWeights<false>(x, size, *a, *b, d, w);
        }

        template <bool align> SIMD_INLINE __m256 Convolution3(const float * src, const __m256 * weights)
        {
            return _mm256_add_ps(_mm256_mul_ps(Load<align>(src), weights[0]),
                _mm256_add_ps(_mm256_mul_ps(Load<false>(src + 1), weights[1]),
                    _mm256_mul_ps(Load<false>(src + 2), weights[2])));
        }

        template <bool align> SIMD_INLINE __m256 Convolution3x3(const float * src, size_t stride, const __m256 * weights)
        {
            return _mm256_add_ps(Convolution3<align>(src, weights),
                _mm256_add_ps(Convolution3<align>(src + stride, weights + 3),
                    Convolution3<align>(src + 2 * stride, weights + 6)));
        }

        template <size_t size> SIMD_INLINE void LoadWeights(const float * src, __m256 dst[size])
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm256_set1_ps(src[i]);
        }

        template <bool align> void AnnAddConvolution3x3(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, 8);
            __m256 tailMask = RightNotZero(width - alignedWidth);
            __m256 _weights[9];
            LoadWeights<9>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += 8)
                {
                    __m256 _dst = Load<align>(dst + col);
                    _dst = _mm256_add_ps(_dst, Convolution3x3<align>(src + col, srcStride, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - 8;
                    __m256 _dst = Load<false>(dst + col);
                    _dst = _mm256_add_ps(_dst, _mm256_and_ps(tailMask, Convolution3x3<false>(src + col, srcStride, _weights)));
                    Store<false>(dst + col, _dst);
                }                
                src += srcStride;
                dst += dstStride;
            }
        }

        void AnnAddConvolution3x3(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AnnAddConvolution3x3<true>(src, srcStride, width, height, weights, dst, dstStride);
            else
                AnnAddConvolution3x3<false>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <bool align> SIMD_INLINE __m256 Convolution5(const float * src, const __m256 * weights)
        {
            return _mm256_add_ps(_mm256_mul_ps(Load<align>(src), weights[0]), _mm256_add_ps(
                _mm256_add_ps(_mm256_mul_ps(Load<false>(src + 1), weights[1]), _mm256_mul_ps(Load<false>(src + 2), weights[2])),
                _mm256_add_ps(_mm256_mul_ps(Load<false>(src + 3), weights[3]), _mm256_mul_ps(Load<false>(src + 4), weights[4]))));
        }

        template <bool align> SIMD_INLINE __m256 Convolution5x5(const float * src, size_t stride, const __m256 * weights)
        {
            return _mm256_add_ps(Convolution5<align>(src, weights), _mm256_add_ps(
                _mm256_add_ps(Convolution5<align>(src + stride, weights + 5), Convolution5<align>(src + 2 * stride, weights + 10)),
                _mm256_add_ps(Convolution5<align>(src + 3 * stride, weights + 15), Convolution5<align>(src + 4 * stride, weights + 20))));
        }

        template <bool align> void AnnAddConvolution5x5(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, 8);
            __m256 tailMask = RightNotZero(width - alignedWidth);
            __m256 _weights[25];
            LoadWeights<25>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += 8)
                {
                    __m256 _dst = Load<align>(dst + col);
                    _dst = _mm256_add_ps(_dst, Convolution5x5<align>(src + col, srcStride, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - 8;
                    __m256 _dst = Load<false>(dst + col);
                    _dst = _mm256_add_ps(_dst, _mm256_and_ps(tailMask, Convolution5x5<false>(src + col, srcStride, _weights)));
                    Store<false>(dst + col, _dst);
                } 
                src += srcStride;
                dst += dstStride;
            }
        }

        void AnnAddConvolution5x5(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AnnAddConvolution5x5<true>(src, srcStride, width, height, weights, dst, dstStride);
            else
                AnnAddConvolution5x5<false>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <bool align> SIMD_INLINE __m256 Max2x2(const float * src, size_t stride)
        {
            __m256 lo = _mm256_max_ps(Load<align>(src + 0), Load<align>(src + stride + 0));
            __m256 hi = _mm256_max_ps(Load<align>(src + 8), Load<align>(src + stride + 8));
            __m256 _lo = _mm256_permute2f128_ps(lo, hi, 0x20);
            __m256 _hi = _mm256_permute2f128_ps(lo, hi, 0x31);
            return _mm256_max_ps(_mm256_shuffle_ps(_lo, _hi, 0x88), _mm256_shuffle_ps(_lo, _hi, 0xDD));
        }

        template <bool align> void AnnMax2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, 16);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += 16)
                {
                    Store<align>(dst + (col >> 1), Max2x2<align>(src + col, srcStride));
                }
                if (width - alignedWidth)
                {
                    size_t col = width - 16;
                    Store<false>(dst + (col >> 1), Max2x2<false>(src + col, srcStride));
                }
                src += 2*srcStride;
                dst += dstStride;
            }
        }

        void AnnMax2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AnnMax2x2<true>(src, srcStride, width, height, dst, dstStride);
            else
                AnnMax2x2<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif// SIMD_AVX_ENABLE
}
