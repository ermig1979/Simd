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
		const __m512i K8_SHUFFLE_DEINTERLEAVE_UV = SIMD_MM512_SETR_EPI8(
			0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
			0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
			0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
			0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF);

		const __m512i K64_PERMUTE_UV_U = SIMD_MM512_SETR_EPI64(0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE);
		const __m512i K64_PERMUTE_UV_V = SIMD_MM512_SETR_EPI64(0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF);

		template <bool align, bool mask> SIMD_INLINE void DeinterleavedUV(const uint8_t * uv, uint8_t * u, uint8_t * v, const __mmask64 * tailMasks)
		{
			const __m512i uv0 = Load<align, mask>(uv + 0, tailMasks[0]);
			const __m512i uv1 = Load<align, mask>(uv + A, tailMasks[1]);
			const __m512i shuffledUV0 = _mm512_shuffle_epi8(uv0, K8_SHUFFLE_DEINTERLEAVE_UV);
			const __m512i shuffledUV1 = _mm512_shuffle_epi8(uv1, K8_SHUFFLE_DEINTERLEAVE_UV);
			Store<align, mask>(u, _mm512_permutex2var_epi64(shuffledUV0, K64_PERMUTE_UV_U, shuffledUV1), tailMasks[2]);
			Store<align, mask>(v, _mm512_permutex2var_epi64(shuffledUV0, K64_PERMUTE_UV_V, shuffledUV1), tailMasks[2]);
		}

		template <bool align> SIMD_INLINE void DeinterleavedUV2(const uint8_t * uv, uint8_t * u, uint8_t * v)
		{ 
			const __m512i uv0 = _mm512_shuffle_epi8(Load<align>(uv + 0 * A), K8_SHUFFLE_DEINTERLEAVE_UV);
			const __m512i uv1 = _mm512_shuffle_epi8(Load<align>(uv + 1 * A), K8_SHUFFLE_DEINTERLEAVE_UV);
			Store<align>(u + 0, _mm512_permutex2var_epi64(uv0, K64_PERMUTE_UV_U, uv1));
			Store<align>(v + 0, _mm512_permutex2var_epi64(uv0, K64_PERMUTE_UV_V, uv1));
			const __m512i uv2 = _mm512_shuffle_epi8(Load<align>(uv + 2 * A), K8_SHUFFLE_DEINTERLEAVE_UV);
			const __m512i uv3 = _mm512_shuffle_epi8(Load<align>(uv + 3 * A), K8_SHUFFLE_DEINTERLEAVE_UV);
			Store<align>(u + A, _mm512_permutex2var_epi64(uv2, K64_PERMUTE_UV_U, uv3));
			Store<align>(v + A, _mm512_permutex2var_epi64(uv2, K64_PERMUTE_UV_V, uv3));
		}

        template <bool align> void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height, 
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if(align)
                assert(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));

			size_t alignedWidth = AlignLo(width, A);
			size_t fullAlignedWidth = AlignLo(width, DA);
			__mmask64 tailMasks[3];
			for (size_t c = 0; c < 2; ++c)
				tailMasks[c] = TailMask64((width - alignedWidth) * 2 - A*c);
			tailMasks[2] = TailMask64(width - alignedWidth);
            for(size_t row = 0; row < height; ++row)
            {
				size_t col = 0;
				for (; col < fullAlignedWidth; col += DA)
					DeinterleavedUV2<align>(uv + col * 2, u + col, v + col);
				for (; col < alignedWidth; col += A)
					DeinterleavedUV<align, false>(uv + col * 2, u + col, v + col, tailMasks);
				if(col < width)
					DeinterleavedUV<align, true>(uv + col * 2, u + col, v + col, tailMasks);
                uv += uvStride;
                u += uStride;
                v += vStride;
            }
        }

        void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height, 
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
                DeinterleaveUv<true>(uv, uvStride, width, height, u, uStride, v, vStride);
            else
                DeinterleaveUv<false>(uv, uvStride, width, height, u, uStride, v, vStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
