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
#include "Simd/SimdSse1.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        template<bool mask> SIMD_INLINE void Winograd2x3pSetOutputLoad2t(const float * src, size_t srcStride, __m512 * dst, __mmask16 tail)
        {
            __m512 s0 = Load<false, mask>(src + 0 * srcStride, tail);
            __m512 s1 = Load<false, mask>(src + 1 * srcStride, tail);
            __m512 s2 = Load<false, mask>(src + 2 * srcStride, tail);
            __m512 s3 = Load<false, mask>(src + 3 * srcStride, tail);
            dst[0] = _mm512_add_ps(_mm512_add_ps(s0, s1), s2);
            dst[1] = _mm512_sub_ps(_mm512_sub_ps(s1, s2), s3);
        }

        template<bool main, bool mask> SIMD_INLINE void Winograd2x3pSetOutput16(const float * src, size_t srcStride, float * dst, size_t dstStride, const __mmask16 * tails)
        {
            __m512 t[8], d[4];
            Winograd2x3pSetOutputLoad2t<mask>(src + 0 * srcStride, srcStride, t + 0, tails[0]);
            Winograd2x3pSetOutputLoad2t<mask>(src + 4 * srcStride, srcStride, t + 2, tails[0]);
            Winograd2x3pSetOutputLoad2t<mask>(src + 8 * srcStride, srcStride, t + 4, tails[0]);
            Winograd2x3pSetOutputLoad2t<mask>(src + 12 * srcStride, srcStride, t + 6, tails[0]);
            d[0] = _mm512_add_ps(_mm512_add_ps(t[0], t[2]), t[4]);
            d[1] = _mm512_add_ps(_mm512_add_ps(t[1], t[3]), t[5]);
            Store<false, mask>(dst + 0, Interleave<0>(d[0], d[1]), tails[1]);
            Store<false, mask>(dst + F, Interleave<1>(d[0], d[1]), tails[2]);
            if (main)
            {
                dst += dstStride;
                d[2] = _mm512_sub_ps(_mm512_sub_ps(t[2], t[4]), t[6]);
                d[3] = _mm512_sub_ps(_mm512_sub_ps(t[3], t[5]), t[7]);
                Store<false, mask>(dst + 0, Interleave<0>(d[2], d[3]), tails[1]);
                Store<false, mask>(dst + F, Interleave<1>(d[2], d[3]), tails[2]);
            }
        }

        void Winograd2x3pSetOutput(const float * src, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth)
        {
            size_t tileH = (dstHeight + 1) / 2;
            size_t tileW = (dstWidth + 1) / 2;
            size_t srcStride = dstChannels * tileH*tileW;
            size_t dstH2 = AlignLo(dstHeight, 2);
            size_t dstW2 = AlignLo(dstWidth, 2);
            size_t dstW32 = AlignLo(dstWidth, 32);
            __mmask16 tails[3];
            tails[0] = TailMask16(tileW - AlignLo(tileW, F));
            for (size_t c = 0; c < 2; ++c)
                tails[1 + c] = TailMask16(dstWidth - dstW32 - F * c);
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row = 0, tileY = 0;
                for (; row < dstH2; row += 2, tileY += 1)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + tileY * tileW;
                    float * d = dst + row * dstWidth;
                    for (; col < dstW32; col += 32, tileX += 16)
                        Winograd2x3pSetOutput16<true, false>(s + tileX, srcStride, d + col, dstWidth, tails);
                    if (col < dstWidth)
                        Winograd2x3pSetOutput16<true, true>(s + tileX, srcStride, d + col, dstWidth, tails);
                }
                if (row < dstHeight)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + tileY * tileW;
                    float * d = dst + row * dstWidth;
                    for (col = 0; col < dstW32; col += 32, tileX += 16)
                        Winograd2x3pSetOutput16<false, false>(s + tileX, srcStride, d + col, dstWidth, tails);
                    if (col < dstWidth)
                        Winograd2x3pSetOutput16<false, true>(s + tileX, srcStride, d + col, dstWidth, tails);
                }
                src += tileW * tileH;
                dst += dstHeight * dstWidth;
            }
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
