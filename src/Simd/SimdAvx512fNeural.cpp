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
		template <bool align, bool mask> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t offset, __m512 & sum, __mmask16 m = -1)
		{
			__m512 _a = Load<align, mask>(a + offset, m);
			__m512 _b = Load<align, mask>(b + offset, m);
			sum = _mm512_fmadd_ps(_a, _b, sum);
		}

		template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
		{
			if (align)
				assert(Aligned(a) && Aligned(b));

			size_t partialAlignedSize = AlignLo(size, F);
			size_t fullAlignedSize = AlignLo(size, QF);
			size_t i = 0;
			__m512 sum0 = _mm512_setzero_ps();
			if (fullAlignedSize)
			{
				__m512 sum1 = _mm512_setzero_ps();
				__m512 sum2 = _mm512_setzero_ps();
				__m512 sum3 = _mm512_setzero_ps();
				for (; i < fullAlignedSize; i += QF)
				{
					NeuralProductSum<align, false>(a, b, i + F * 0, sum0);
					NeuralProductSum<align, false>(a, b, i + F * 1, sum1);
					NeuralProductSum<align, false>(a, b, i + F * 2, sum2);
					NeuralProductSum<align, false>(a, b, i + F * 3, sum3);
				}
				sum0 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
			}
			for (; i < partialAlignedSize; i += F)
				NeuralProductSum<align, false>(a, b, i, sum0);			
			if (i < size)
			{
				__mmask16 tailMask = __mmask16(-1) >> (F + i - size);
				NeuralProductSum<align, true>(a, b, i, sum0, tailMask);
			}
			*sum = ExtractSum(sum0);
		}

		void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
		{
			if (Aligned(a) && Aligned(b))
				NeuralProductSum<true>(a, b, size, sum);
			else
				NeuralProductSum<false>(a, b, size, sum);
		}

        template <bool align, bool mask> SIMD_INLINE void AddMultiplied(const float * src, const __m512 & value, float * dst, __mmask16 m = -1)
        {
			__m512 _src = Load<align, mask>(src, m);
			__m512 _dst = Load<align, mask>(dst, m);
            Store<align, mask>(dst, _mm512_fmadd_ps(value, _src, _dst), m);
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
        {
            size_t i = 0;
            __m512 _value = _mm512_set1_ps(value);
            for (; i < aligned; i += QF)
            {
                AddMultiplied<align, false>(src + i + F*0, _value, dst + i + F*0);
                AddMultiplied<align, false>(src + i + F*1, _value, dst + i + F*1);
                AddMultiplied<align, false>(src + i + F*2, _value, dst + i + F*2);
                AddMultiplied<align, false>(src + i + F*3, _value, dst + i + F*3);
            }
            for (; i < partial; i += F)
                AddMultiplied<align, false>(src + i, _value, dst + i);
			if (i < full)
			{
				__mmask16 tailMask = __mmask16(-1) >> (F + i - full);
				AddMultiplied<align, true>(src + i, _value, dst + i, tailMask);
			}
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

		template <bool align, bool mask> SIMD_INLINE void AddVector(const float * src, float * dst, __mmask16 m = -1)
		{
			__m512 _src = Load<align, mask>(src, m);
			__m512 _dst = Load<align, mask>(dst, m);
			Store<align, mask>(dst, _mm512_add_ps(_src, _dst), m);
		}

		template <bool align> SIMD_INLINE void AddVector(const float * src, size_t aligned, size_t partial, size_t full, float * dst)
		{
			size_t i = 0;
			for (; i < aligned; i += QF)
			{
				AddVector<align, false>(src + i + F * 0, dst + i + F * 0);
				AddVector<align, false>(src + i + F * 1, dst + i + F * 1);
				AddVector<align, false>(src + i + F * 2, dst + i + F * 2);
				AddVector<align, false>(src + i + F * 3, dst + i + F * 3);
			}
			for (; i < partial; i += F)
				AddVector<align, false>(src + i, dst + i);
			if (i < full)
			{
				__mmask16 tailMask = __mmask16(-1) >> (F + i - full);
				AddVector<align, true>(src + i, dst + i, tailMask);
			}
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

		template <bool align, bool mask> SIMD_INLINE void AddValue(const __m512 & value, float * dst, __mmask16 m = -1)
		{
			__m512 _dst = Load<align, mask>(dst, m);
			Store<align, mask>(dst, _mm512_add_ps(_dst, value), m);
		}

		template <bool align> SIMD_INLINE void AddValue(const float * value, float * dst, size_t aligned, size_t partial, size_t full)
		{
			size_t i = 0;
			__m512 _value = _mm512_set1_ps(value[0]);
			for (; i < aligned; i += QF)
			{
				AddValue<align, false>(_value, dst + i + F * 0);
				AddValue<align, false>(_value, dst + i + F * 1);
				AddValue<align, false>(_value, dst + i + F * 2);
				AddValue<align, false>(_value, dst + i + F * 3);
			}
			for (; i < partial; i += F)
				AddValue<align, false>(_value, dst + i);
			if (i < full)
			{
				__mmask16 tailMask = __mmask16(-1) >> (F + i - full);
				AddValue<align, true>(_value, dst + i, tailMask);
			}
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

		template <bool align, bool mask> SIMD_INLINE void NeuralRoughSigmoid(const float * src, const __m512 & _0, const __m512 & _1, 
			const __m512 & a, const __m512 & b, const __m512 & slope , float * dst, __mmask16 m = -1)
		{
			__m512 _src = Load<align, mask>(src, m);
			__m512 x = AndNot(_0, _mm512_mul_ps(_src, slope));
			__m512 x2 = _mm512_mul_ps(x, x);
			__m512 x4 = _mm512_mul_ps(x2, x2);
			__m512 series = _mm512_add_ps(_mm512_fmadd_ps(x2, a, _1), _mm512_fmadd_ps(x4, b, x));
			__m512 exp = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_src, _0, _CMP_GT_OS), series, Rcp14(series));
			__m512 sigmoid = Rcp14(_mm512_add_ps(_1, exp));
			Store<align, mask>(dst, sigmoid, m);
		}

		template <bool align> SIMD_INLINE void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			__m512 _slope = _mm512_set1_ps(*slope);
			__m512 _0 = _mm512_set1_ps(-0.0f);
			__m512 _1 = _mm512_set1_ps(1.0f);
			__m512 _a = _mm512_set1_ps(0.5417f);
			__m512 _b = _mm512_set1_ps(0.1460f);
			size_t i = 0;
			size_t partialAlignedSize = Simd::AlignLo(size, F);
			size_t fullAlignedSize = Simd::AlignLo(size, QF);
			for (; i < fullAlignedSize; i += QF)
			{
				NeuralRoughSigmoid<align, false>(src + i + 0 * F, _0, _1, _a, _b, _slope, dst + i + 0 * F);
				NeuralRoughSigmoid<align, false>(src + i + 1 * F, _0, _1, _a, _b, _slope, dst + i + 1 * F);
				NeuralRoughSigmoid<align, false>(src + i + 2 * F, _0, _1, _a, _b, _slope, dst + i + 2 * F);
				NeuralRoughSigmoid<align, false>(src + i + 3 * F, _0, _1, _a, _b, _slope, dst + i + 3 * F);
			}
			for (; i < partialAlignedSize; i += F)
				NeuralRoughSigmoid<align, false>(src + i, _0, _1, _a, _b, _slope, dst + i);
			if (i < size)
			{
				__mmask16 tailMask = __mmask16(-1) >> (F + i -size);
				NeuralRoughSigmoid<align, true>(src + i, _0, _1, _a, _b, _slope, dst + i, tailMask);
			}
		}

		void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			if (Aligned(src) && Aligned(dst))
				NeuralRoughSigmoid<true>(src, size, slope, dst);
			else
				NeuralRoughSigmoid<false>(src, size, slope, dst);
		}

		template <bool align, bool mask> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, const __m512 & k, 
			const __m512 & _1, const __m512 & _05, float * dst, __mmask16 m = -1)
		{
			__m512 _src = Load<align, mask>(src, m);
			__m512 e1 = _mm512_max_ps(_05, _mm512_fmadd_ps(_src, k, _1));
			__m512 e2 = _mm512_mul_ps(e1, e1);
			__m512 e4 = _mm512_mul_ps(e2, e2);
			__m512 e8 = _mm512_mul_ps(e4, e4);
			__m512 e16 = _mm512_mul_ps(e8, e8);
			__m512 e32 = _mm512_mul_ps(e16, e16);
			__m512 e64 = _mm512_mul_ps(e32, e32);
			__m512 sigmoid = Rcp14(_mm512_fmadd_ps(e64, e64, _1));
			Store<align, mask>(dst, sigmoid, m);
		}

		template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
		{
			size_t partialAlignedSize = Simd::AlignLo(size, F);
			size_t fullAlignedSize = Simd::AlignLo(size, QF);
			__m512 _k = _mm512_set1_ps(-(*slope)*0.0078125f);
			__m512 _1 = _mm512_set1_ps(1.0f);
			__m512 _05 = _mm512_set1_ps(0.5f);
			size_t i = 0;
			for (; i < fullAlignedSize; i += QF)
			{
				NeuralRoughSigmoid2<align, true>(src + i + 0 * F, _k, _1, _05, dst + i + 0 * F);
				NeuralRoughSigmoid2<align, true>(src + i + 1 * F, _k, _1, _05, dst + i + 1 * F);
				NeuralRoughSigmoid2<align, true>(src + i + 2 * F, _k, _1, _05, dst + i + 2 * F);
				NeuralRoughSigmoid2<align, true>(src + i + 3 * F, _k, _1, _05, dst + i + 3 * F);
			}
			for (; i < partialAlignedSize; i += F)
				NeuralRoughSigmoid2<align, true>(src + i, _k, _1, _05, dst + i);
			if (i < size)
			{
				__mmask16 tailMask = __mmask16(-1) >> (F + i - size);
				NeuralRoughSigmoid2<align, true>(src + i, _k, _1, _05, dst + i, tailMask);
			}
		}

		void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
		{
			if (Aligned(src) && Aligned(dst))
				NeuralRoughSigmoid2<true>(src, size, slope, dst);
			else
				NeuralRoughSigmoid2<false>(src, size, slope, dst);
		}
    }
#endif// SIMD_AVX512F_ENABLE
}
