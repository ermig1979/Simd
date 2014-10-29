/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Simd/SimdInit.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdSsse3.h"

namespace Simd
{
#ifdef SIMD_SSSE3_ENABLE    
    namespace Ssse3
    {
        const __m128i K8_SHUFFLE_BGRA_TO_B0R0 = SIMD_MM_SETR_EPI8(0x0,  -1, 0x2,  -1, 0x4,  -1, 0x6,  -1, 0x8,  -1, 0xA,  -1, 0xC,  -1, 0xE,  -1);
        const __m128i K8_SHUFFLE_BGRA_TO_G000 = SIMD_MM_SETR_EPI8(0x1,  -1,  -1,  -1, 0x5,  -1,  -1,  -1, 0x9,  -1,  -1,  -1, 0xD,  -1,  -1,  -1);

        template <bool align> SIMD_INLINE void LoadPreparedBgra16(const __m128i * bgra, __m128i & b16_r16, __m128i & g16_1)
        {
            __m128i _bgra = Load<align>(bgra);
            b16_r16 = _mm_shuffle_epi8(_bgra, K8_SHUFFLE_BGRA_TO_B0R0);
            g16_1 = _mm_or_si128(_mm_shuffle_epi8(_bgra, K8_SHUFFLE_BGRA_TO_G000), K32_00010000);
        }

        template <bool align> SIMD_INLINE __m128i LoadAndConvertY16(const __m128i * bgra, __m128i & b16_r16, __m128i & g16_1)
        {
            __m128i _b16_r16[2], _g16_1[2];
            LoadPreparedBgra16<align>(bgra + 0, _b16_r16[0], _g16_1[0]);
            LoadPreparedBgra16<align>(bgra + 1, _b16_r16[1], _g16_1[1]);
            b16_r16 = _mm_hadd_epi32(_b16_r16[0], _b16_r16[1]);
            g16_1 = _mm_hadd_epi32(_g16_1[0], _g16_1[1]);
            return SaturateI16ToU8(_mm_add_epi16(K16_Y_ADJUST, _mm_packs_epi32(BgrToY32(_b16_r16[0], _g16_1[0]), BgrToY32(_b16_r16[1], _g16_1[1]))));
        }

        template <bool align> SIMD_INLINE __m128i LoadAndConvertY8(const __m128i * bgra, __m128i b16_r16[2], __m128i g16_1[2])
        {
            return _mm_packus_epi16(LoadAndConvertY16<align>(bgra + 0, b16_r16[0], g16_1[0]), LoadAndConvertY16<align>(bgra + 2, b16_r16[1], g16_1[1]));
        }

        SIMD_INLINE void Average16(__m128i & a, const __m128i & b)
        {
            a = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(a, b), K16_0002), 2);
        }

        SIMD_INLINE __m128i ConvertU16(__m128i b16_r16[2], __m128i g16_1[2])
        {
            return SaturateI16ToU8(_mm_add_epi16(K16_UV_ADJUST, _mm_packs_epi32(BgrToU32(b16_r16[0], g16_1[0]), BgrToU32(b16_r16[1], g16_1[1]))));
        }

        SIMD_INLINE __m128i ConvertV16(__m128i b16_r16[2], __m128i g16_1[2])
        {
            return SaturateI16ToU8(_mm_add_epi16(K16_UV_ADJUST, _mm_packs_epi32(BgrToV32(b16_r16[0], g16_1[0]), BgrToV32(b16_r16[1], g16_1[1]))));
        }

        template <bool align> SIMD_INLINE void BgraToYuv420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;

            __m128i _b16_r16[2][2][2], _g16_1[2][2][2];
            Store<align>((__m128i*)y0 + 0, LoadAndConvertY8<align>((__m128i*)bgra0 + 0, _b16_r16[0][0], _g16_1[0][0]));
            Store<align>((__m128i*)y0 + 1, LoadAndConvertY8<align>((__m128i*)bgra0 + 4, _b16_r16[0][1], _g16_1[0][1]));
            Store<align>((__m128i*)y1 + 0, LoadAndConvertY8<align>((__m128i*)bgra1 + 0, _b16_r16[1][0], _g16_1[1][0]));
            Store<align>((__m128i*)y1 + 1, LoadAndConvertY8<align>((__m128i*)bgra1 + 4, _b16_r16[1][1], _g16_1[1][1]));

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            Store<align>((__m128i*)u, _mm_packus_epi16(ConvertU16(_b16_r16[0][0], _g16_1[0][0]), ConvertU16(_b16_r16[0][1], _g16_1[0][1])));
            Store<align>((__m128i*)v, _mm_packus_epi16(ConvertV16(_b16_r16[0][0], _g16_1[0][0]), ConvertV16(_b16_r16[0][1], _g16_1[0][1])));
        }

        template <bool align> void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width%2 == 0) && (height%2 == 0) && (width >= DA) && (height >= 2));
            if(align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) &&  Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A8 = A*8;
            for(size_t row = 0; row < height; row += 2)
            {
                for(size_t colUV = 0, colY = 0, colBgra = 0; colY < alignedWidth; colY += DA, colUV += A, colBgra += A8)
                    BgraToYuv420p<align>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                if(width != alignedWidth)
                {
                    size_t offset = width - DA;
                    BgraToYuv420p<false>(bgra + offset*4, bgraStride, y + offset, yStride, u + offset/2, v + offset/2);
                }
                y += 2*yStride;
                u += uStride;
                v += vStride;
                bgra += 2*bgraStride;
            }
        }

        void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) 
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv420p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv420p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }
    }
#endif// SIMD_SSSE3_ENABLE
}