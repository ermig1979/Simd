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
#include "Simd/SimdStream.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
#if defined(_MSC_VER) && _MSC_VER >= 1800  && _MSC_VER < 1900 // Visual Studio 2013 compiler bug       
        const size_t F = Avx::F;
        const size_t QF = Avx::QF;
        using Avx::RightNotZero;
#endif

		template <bool inversion> __m128i Invert(__m128i value);

		template <> __m128i Invert<true>(__m128i value)
		{
			return _mm_sub_epi8(Sse2::K_INV_ZERO, value);
		}

		template <> __m128i Invert<false>(__m128i value)
		{
			return value;
		}

		template <bool inversion, bool align, bool stream> void Convert(const uint8_t * src, const __m256 & _1_255, float * dst)
		{
			__m128i _src = Invert<inversion>(_mm_loadl_epi64((__m128i*)src));
			Avx::Stream<align, stream>(dst, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_src)), _1_255));
		}

		template <bool inversion, bool align, bool stream> void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst)
		{
			assert(width >= 8);
			if (align)
				assert(Aligned(dst) && Aligned(width));

			size_t alignedWidth = AlignLo(width, 8);
			__m256 _1_255 = _mm256_set1_ps(1.0f / 255.0f);

			for (size_t row = 0; row < height; ++row)
			{
				for (size_t col = 0; col < alignedWidth; col += 8)
					Convert<inversion, align, stream>(src + col, _1_255, dst + col);
				if(width != alignedWidth)
					Convert<inversion, false, stream>(src + width - 8, _1_255, dst + width - 8);
				src += stride;
				dst += width;
			}
            if (stream)
                _mm_mfence();
		}

        template <bool inversion> void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst)
        {
            if (Aligned(src) && Aligned(stride) && Aligned(dst) && Aligned(width))
            {
                if (width*height*sizeof(float) >= STREAM_SIZE_MIN)
                    NeuralConvert<inversion, true, true>(src, stride, width, height, dst);
                else
                    NeuralConvert<inversion, true, false>(src, stride, width, height, dst);
            }
            else
                NeuralConvert<inversion, false, false>(src, stride, width, height, dst);
        }

		void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst, int inversion)
		{
			if (inversion)
				NeuralConvert<true>(src, stride, width, height, dst);
			else
				NeuralConvert<false>(src, stride, width, height, dst);
		}

        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t offset, __m256 & sum)
        {
            __m256 _a = Load<align>(a + offset);
            __m256 _b = Load<align>(b + offset);
            sum = _mm256_fmadd_ps(_a, _b, sum);
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        NeuralProductSum<align>(a, b, i + F * 0, sums[0]);
                        NeuralProductSum<align>(a, b, i + F * 1, sums[1]);
                        NeuralProductSum<align>(a, b, i + F * 2, sums[2]);
                        NeuralProductSum<align>(a, b, i + F * 3, sums[3]);
                    }
                    sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += F)
                    NeuralProductSum<align>(a, b, i, sums[0]);
                *sum += Avx::ExtractSum(sums[0]);
            }
            for (; i < size; ++i)
                *sum += a[i] * b[i];
        }

        void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                NeuralProductSum<true>(a, b, size, sum);
            else
                NeuralProductSum<false>(a, b, size, sum);
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, const __m256 & value, float * dst)
        {
            Avx::Store<align>(dst, _mm256_fmadd_ps(value, Load<align>(src), Load<align>(dst)));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
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

        void NeuralAddVectorMultipliedByValue(const float * src, size_t size, const float * value, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(src) && Aligned(dst))
                AddMultiplied<true>(src, aligned, partial, size, *value, dst);
            else
                AddMultiplied<false>(src, aligned, partial, size, *value, dst);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, const __m256 & k, const __m256 & o, const __m256 & m, float * dst)
        {
            __m256 _src = Load<align>(src);
            __m256 e1 = _mm256_max_ps(m, _mm256_fmadd_ps(_src, k, o));
            __m256 e2 = _mm256_mul_ps(e1, e1);
            __m256 e4 = _mm256_mul_ps(e2, e2);
            __m256 e8 = _mm256_mul_ps(e4, e4);
            __m256 e16 = _mm256_mul_ps(e8, e8);
            __m256 e32 = _mm256_mul_ps(e16, e16);
            __m256 e64 = _mm256_mul_ps(e32, e32);
            __m256 sigmoid = _mm256_rcp_ps(_mm256_fmadd_ps(e64, e64, o));
            Avx::Store<align>(dst, sigmoid);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            __m256 _k = _mm256_set1_ps(-(*slope)*0.0078125f);
            __m256 _1 = _mm256_set1_ps(1.0f);
            __m256 _05 = _mm256_set1_ps(0.5f);
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

        void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid2<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid2<false>(src, size, slope, dst);
        }

        template <size_t size> SIMD_INLINE void LoadWeights(const float * src, __m256 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm256_set1_ps(src[i]);
        }

        template<int shift> SIMD_INLINE __m256 Alignr(const __m256 & s0, const __m256 & s4)
        {
            return _mm256_castsi256_ps(_mm256_alignr_epi8(_mm256_castps_si256(s4), _mm256_castps_si256(s0), shift*4));
        }

        template <bool align> SIMD_INLINE __m256 Convolution5(const float * src, const __m256 * weights)
        {
            __m256 s0 = Load<align>(src + 0);
            __m256 s4 = Load<false>(src + 4);
            return _mm256_fmadd_ps(s0, weights[0], _mm256_add_ps(
                _mm256_fmadd_ps(Alignr<1>(s0, s4), weights[1], _mm256_mul_ps(Alignr<2>(s0, s4), weights[2])),
                _mm256_fmadd_ps(s4, weights[4], _mm256_mul_ps(Alignr<3>(s0, s4), weights[3]))));
        }

        template <bool align> SIMD_INLINE __m256 Convolution5x5(const float * src, size_t stride, const __m256 * weights)
        {
            return _mm256_add_ps(Convolution5<align>(src, weights), _mm256_add_ps(
                _mm256_add_ps(Convolution5<align>(src + stride, weights + 5), Convolution5<align>(src + 2 * stride, weights + 10)),
                _mm256_add_ps(Convolution5<align>(src + 3 * stride, weights + 15), Convolution5<align>(src + 4 * stride, weights + 20))));
        }

        template <bool align> void NeuralAddConvolution5x5(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, F);
            __m256 tailMask = RightNotZero(width - alignedWidth);
            __m256 _weights[25];
            LoadWeights<25>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m256 _dst = Load<align>(dst + col);
                    _dst = _mm256_add_ps(_dst, Convolution5x5<align>(src + col, srcStride, _weights));
                    Avx::Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m256 _dst = Load<false>(dst + col);
                    _dst = _mm256_add_ps(_dst, _mm256_and_ps(tailMask, Convolution5x5<false>(src + col, srcStride, _weights)));
                    Avx::Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution5x5(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution5x5<true>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolution5x5<false>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <bool align> SIMD_INLINE void AddMultiplied3(const float * src, const __m256 & dst, __m256 * sums)
        {
            __m256 s0 = Load<align>(src + 0);
            __m256 s4 = Load<false>(src + 4);
            sums[0] = _mm256_fmadd_ps(dst, s0, sums[0]);
            sums[1] = _mm256_fmadd_ps(dst, Alignr<1>(s0, s4), sums[1]);
            sums[2] = _mm256_fmadd_ps(dst, Alignr<2>(s0, s4), sums[2]);
        }

        template <bool align> SIMD_INLINE void AddMultiplied3x3(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
        {
            AddMultiplied3<align>(src + stride * 0, dst, sums + 0);
            AddMultiplied3<align>(src + stride * 1, dst, sums + 3);
            AddMultiplied3<align>(src + stride * 2, dst, sums + 6);
        }

        template <bool align> void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m256 tailMask = RightNotZero(width - alignedWidth);
            __m256 _sums[9];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m256 _dst = Load<align>(dst + col);
                    AddMultiplied3x3<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m256 _dst = _mm256_and_ps(tailMask, Load<false>(dst + col));
                    AddMultiplied3x3<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            Add8ExtractedSums(_sums, sums);
            sums[8] += Avx::ExtractSum(_sums[8]);
        }

        void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution3x3Sum<true>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolution3x3Sum<false>(src, srcStride, dst, dstStride, width, height, sums);
        }

        template <bool align> SIMD_INLINE void AddMultiplied5(const float * src, const __m256 & dst, __m256 * sums)
        {
            __m256 s0 = Load<align>(src + 0);
            __m256 s4 = Load<false>(src + 4);
            sums[0] = _mm256_fmadd_ps(dst, s0, sums[0]);
            sums[1] = _mm256_fmadd_ps(dst, Alignr<1>(s0, s4), sums[1]);
            sums[2] = _mm256_fmadd_ps(dst, Alignr<2>(s0, s4), sums[2]);
            sums[3] = _mm256_fmadd_ps(dst, Alignr<3>(s0, s4), sums[3]);
            sums[4] = _mm256_fmadd_ps(dst, s4, sums[4]);
        }

        template <bool align> SIMD_INLINE void AddMultiplied5x5(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
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
            __m256 tailMask = RightNotZero(width - alignedWidth);
            __m256 _sums[25];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m256 _dst = Load<align>(dst + col);
                    AddMultiplied5x5<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m256 _dst = _mm256_and_ps(tailMask, Load<false>(dst + col));
                    AddMultiplied5x5<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            Add8ExtractedSums(_sums + 0, sums + 0);
            Add8ExtractedSums(_sums + 8, sums + 8);
            Add8ExtractedSums(_sums + 16, sums + 16);
            sums[24] += Avx::ExtractSum(_sums[24]);
        }

        void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution5x5Sum<true>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolution5x5Sum<false>(src, srcStride, dst, dstStride, width, height, sums);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
