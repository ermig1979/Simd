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
    }
#endif// SIMD_AVX512F_ENABLE
}
