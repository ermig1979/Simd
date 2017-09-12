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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
		template <bool align, bool mask> SIMD_INLINE void InterleaveUv(const uint8_t * u, const uint8_t * v, uint8_t * uv, const __mmask64 * tails)
		{
			__m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(u, tails[2])));
			__m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(v, tails[2])));
			Store<align, mask>(uv + 0, UnpackU8<0>(_u, _v), tails[0]);
			Store<align, mask>(uv + A, UnpackU8<1>(_u, _v), tails[1]);
		}

		template <bool align> SIMD_INLINE void InterleaveUv2(const uint8_t * u, const uint8_t * v, uint8_t * uv)
		{
			__m512i u0 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<align>(u + 0));
			__m512i v0 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<align>(v + 0));
			Store<align>(uv + 0 * A, UnpackU8<0>(u0, v0));
			Store<align>(uv + 1 * A, UnpackU8<1>(u0, v0));
			__m512i u1 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<align>(u + A));
			__m512i v1 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<align>(v + A));
			Store<align>(uv + 2 * A, UnpackU8<0>(u1, v1));
			Store<align>(uv + 3 * A, UnpackU8<1>(u1, v1));
		}

        template <bool align> void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, 
			size_t width, size_t height, uint8_t * uv, size_t uvStride)
        {
            if(align)
                assert(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));

			size_t alignedWidth = AlignLo(width, A);
			size_t fullAlignedWidth = AlignLo(width, DA);
			__mmask64 tailMasks[3];
			for (size_t c = 0; c < 2; ++c)
				tailMasks[c] = TailMask64((width - alignedWidth) * 2 - A*c);
			tailMasks[2] = TailMask64(width - alignedWidth);
			for (size_t row = 0; row < height; ++row)
			{
				size_t col = 0;
				for (; col < fullAlignedWidth; col += DA)
					InterleaveUv2<align>(u + col, v + col, uv + col *2);
				for (; col < alignedWidth; col += A)
					InterleaveUv<align, false>(u + col, v + col, uv + col * 2, tailMasks);
				if (col < width)
					InterleaveUv<align, true>(u + col, v + col, uv + col * 2, tailMasks);
				uv += uvStride;
				u += uStride;
				v += vStride;
			}
        }

		void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride)
		{
			if (Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
				InterleaveUv<true>(u, uStride, v, vStride, width, height, uv, uvStride);
			else
				InterleaveUv<false>(u, uStride, v, vStride, width, height, uv, uvStride);
		}
    }
#endif// SIMD_AVX512BW_ENABLE
}
