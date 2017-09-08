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
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
	namespace Avx512bw
	{
        template<bool increment> __m512i InterferenceChange(__m512i statistic, __m512i value, __m512i saturation);

        template<> SIMD_INLINE __m512i InterferenceChange<true>(__m512i statistic, __m512i value, __m512i saturation)
        {
            return _mm512_min_epi16(_mm512_add_epi16(statistic, value), saturation);
        }

        template<> SIMD_INLINE __m512i InterferenceChange<false>(__m512i statistic, __m512i value, __m512i saturation)
        {
            return _mm512_max_epi16(_mm512_sub_epi16(statistic, value), saturation);
        }

        template<bool align, bool increment, bool mask> SIMD_INLINE void InterferenceChange(int16_t * statistic, __m512i value, __m512i saturation, __mmask32 tail = -1)
        {
            Store<align, mask>(statistic, InterferenceChange<increment>(Load<align, mask>(statistic, tail), value, saturation), tail);
        }

		template<bool align, bool increment> SIMD_INLINE void InterferenceChange4(int16_t * statistic, __m512i value, __m512i saturation)
		{
			Store<align>(statistic + 0 * HA, InterferenceChange<increment>(Load<align>(statistic + 0 * HA), value, saturation));
			Store<align>(statistic + 1 * HA, InterferenceChange<increment>(Load<align>(statistic + 1 * HA), value, saturation));
			Store<align>(statistic + 2 * HA, InterferenceChange<increment>(Load<align>(statistic + 2 * HA), value, saturation));
			Store<align>(statistic + 3 * HA, InterferenceChange<increment>(Load<align>(statistic + 3 * HA), value, saturation));
		}

        template <bool align, bool increment> void InterferenceChange(int16_t * statistic, size_t stride, size_t width, size_t height, uint8_t value, int16_t saturation)
        {
            if(align)
                assert(Aligned(statistic) && Aligned(stride, HA));

			size_t alignedWidth = AlignLo(width, HA);
			size_t fullAlignedWidth = AlignLo(width, DA);
			__mmask32 tailMask = TailMask32(width - alignedWidth);

            __m512i _value = _mm512_set1_epi16(value);
            __m512i _saturation = _mm512_set1_epi16(saturation);
            for(size_t row = 0; row < height; ++row)
            {
				size_t col = 0;
				for (; col < fullAlignedWidth; col += DA)
					InterferenceChange4<align, increment>(statistic + col, _value, _saturation);
				for(; col < alignedWidth; col += HA)
                    InterferenceChange<align, increment, false>(statistic + col, _value, _saturation);
                if(col < width)
                    InterferenceChange<false, increment, true>(statistic + col, _value, _saturation, tailMask);
                statistic += stride;
            }
        }

        void InterferenceIncrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t increment, int16_t saturation)
        {
            assert(Aligned(stride, 2));

            if(Aligned(statistic) && Aligned(stride))
                InterferenceChange<true, true>((int16_t*)statistic, stride/2, width, height, increment, saturation);
            else
                InterferenceChange<false, true>((int16_t*)statistic, stride/2, width, height, increment, saturation);
        }
	}
#endif// SIMD_AVX512BW_ENABLE
}
