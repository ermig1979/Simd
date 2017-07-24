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
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
		template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t offset, __m512 & sum)
		{
			__m512 _a = Load<align>(a + offset);
			__m512 _b = Load<align>(b + offset);
			sum = _mm512_fmadd_ps(_a, _b, sum);
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
				__m512 sums[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
				if (fullAlignedSize)
				{
					for (; i < fullAlignedSize; i += QF)
					{
						NeuralProductSum<align>(a, b, i + F * 0, sums[0]);
						NeuralProductSum<align>(a, b, i + F * 1, sums[1]);
						NeuralProductSum<align>(a, b, i + F * 2, sums[2]);
						NeuralProductSum<align>(a, b, i + F * 3, sums[3]);
					}
					sums[0] = _mm512_add_ps(_mm512_add_ps(sums[0], sums[1]), _mm512_add_ps(sums[2], sums[3]));
				}
				for (; i < partialAlignedSize; i += F)
					NeuralProductSum<align>(a, b, i, sums[0]);
				*sum += ExtractSum(sums[0]);
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

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, const __m512 & value, float * dst)
        {
            Store<align>(dst, _mm512_fmadd_ps(value, Load<align>(src), Load<align>(dst)));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
        {
            size_t i = 0;
            if (partial)
            {
                __m512 _value = _mm512_set1_ps(value);
                for (; i < aligned; i += QF)
                {
                    AddMultiplied<align>(src + i + F*0, _value, dst + i + F*0);
                    AddMultiplied<align>(src + i + F*1, _value, dst + i + F*1);
                    AddMultiplied<align>(src + i + F*2, _value, dst + i + F*2);
                    AddMultiplied<align>(src + i + F*3, _value, dst + i + F*3);
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

		template <bool align> SIMD_INLINE void AddVector(const float * src, float * dst)
		{
			Store<align>(dst, _mm512_add_ps(Load<align>(dst), Load<align>(src)));
		}

		template <bool align> SIMD_INLINE void AddVector(const float * src, size_t aligned, size_t partial, size_t full, float * dst)
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

		void NeuralAddVector(const float * src, size_t size, float * dst)
		{
			size_t aligned = AlignLo(size, QF);
			size_t partial = AlignLo(size, F);
			if (Aligned(src) && Aligned(dst))
				AddVector<true>(src, aligned, partial, size, dst);
			else
				AddVector<false>(src, aligned, partial, size, dst);
		}

		template <bool align> SIMD_INLINE void AddValue(const __m512 & value, float * dst)
		{
			Store<align>(dst, _mm512_add_ps(Load<align>(dst), value));
		}

		template <bool align> SIMD_INLINE void AddValue(const float * value, float * dst, size_t aligned, size_t partial, size_t full)
		{
			size_t i = 0;
			if (partial)
			{
				__m512 _value = _mm512_set1_ps(value[0]);
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

		void NeuralAddValue(const float * value, float * dst, size_t size)
		{
			size_t aligned = AlignLo(size, QF);
			size_t partial = AlignLo(size, F);
			if (Aligned(dst))
				AddValue<true>(value, dst, aligned, partial, size);
			else
				AddValue<false>(value, dst, aligned, partial, size);
		}

		SIMD_INLINE __m512 Rcp14(const __m512 & a)
		{
#if defined(_MSC_VER)
			return _mm512_maskz_rcp14_ps(_MM_K0_REG, a);
#else
			return _mm512_rcp14_ps(a);
#endif
		}

		SIMD_INLINE __m512 AndNot(const __m512 & a, const __m512 & b)
		{
#if defined(__clang__)
			return (__m512)_mm512_andnot_epi32((__m512i)a, (__m512i)b);
#else
			return _mm512_castsi512_ps(_mm512_andnot_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)));
#endif
		}

		template <bool align> SIMD_INLINE void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			size_t alignedSize = Simd::AlignLo(size, F);
			__m512 _slope = _mm512_set1_ps(*slope);
			__m512 _0 = _mm512_set1_ps(-0.0f);
			__m512 _1 = _mm512_set1_ps(1.0f);
			__m512 _a = _mm512_set1_ps(0.5417f);
			__m512 _b = _mm512_set1_ps(0.1460f);
			size_t i = 0;
			for (; i < alignedSize; i += F)
			{
				__m512 _src = Load<align>(src + i);
				__m512 x = AndNot(_0, _mm512_mul_ps(_src, _slope));
				__m512 x2 = _mm512_mul_ps(x, x);
				__m512 x4 = _mm512_mul_ps(x2, x2);
				__m512 series = _mm512_add_ps(_mm512_add_ps(_1, x), _mm512_add_ps(_mm512_mul_ps(x2, _a), _mm512_mul_ps(x4, _b)));
				__mmask16 mask = _mm512_cmp_ps_mask(_src, _0, _CMP_GT_OS);
				__m512 exp = _mm512_mask_blend_ps(mask, series, Rcp14(series));
				__m512 sigmoid = Rcp14(_mm512_add_ps(_1, exp));
				Store<align>(dst + i, sigmoid);
			}
			for (; i < size; ++i)
				dst[i] = Base::RoughSigmoid(src[i] * slope[0]);
		}

		void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			if (Aligned(src) && Aligned(dst))
				NeuralRoughSigmoid<true>(src, size, slope, dst);
			else
				NeuralRoughSigmoid<false>(src, size, slope, dst);
		}
    }
#endif// SIMD_AVX512F_ENABLE
}
