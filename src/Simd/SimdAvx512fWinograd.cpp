/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdSet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        template <bool mask> SIMD_INLINE void Winograd2x3pSetInputLoad16Row(const float * src, __m512 * dst, const __mmask16 * tails)
        {
            __m512 a0 = Load<false, mask>(src + 0, tails[0]);
            __m512 a1 = Load<false, mask>(src + 2, tails[1]);
            __m512 a2 = Load<false, mask>(src + 16, tails[2]);
            __m512 a3 = Load<false, mask>(src + 18, tails[3]);
            dst[0] = Deinterleave<0>(a0, a2);
            dst[1] = Deinterleave<1>(a0, a2);
            dst[2] = Deinterleave<0>(a1, a3);
            dst[3] = Deinterleave<1>(a1, a3);
        }

        SIMD_INLINE void Winograd2x3pSetInputLoad16Zero(__m512 * dst)
        {
            dst[0] = _mm512_setzero_ps();
            dst[1] = _mm512_setzero_ps();
            dst[2] = _mm512_setzero_ps();
            dst[3] = _mm512_setzero_ps();
        }

        template <bool mask> SIMD_INLINE void Winograd2x3pSetInput16Store(const __m512 * t, float * dst, size_t dstStride, const __mmask16 * tails)
        {
            Store<false, mask>(dst + 0 * dstStride, _mm512_sub_ps(_mm512_sub_ps(t[0], t[8]), _mm512_sub_ps(t[2], t[10])), tails[4]);
            Store<false, mask>(dst + 1 * dstStride, _mm512_add_ps(_mm512_sub_ps(t[1], t[9]), _mm512_sub_ps(t[2], t[10])), tails[4]);
            Store<false, mask>(dst + 2 * dstStride, _mm512_sub_ps(_mm512_sub_ps(t[2], t[10]), _mm512_sub_ps(t[1], t[9])), tails[4]);
            Store<false, mask>(dst + 3 * dstStride, _mm512_sub_ps(_mm512_sub_ps(t[1], t[9]), _mm512_sub_ps(t[3], t[11])), tails[4]);
            Store<false, mask>(dst + 4 * dstStride, _mm512_sub_ps(_mm512_add_ps(t[4], t[8]), _mm512_add_ps(t[6], t[10])), tails[4]);
            Store<false, mask>(dst + 5 * dstStride, _mm512_add_ps(_mm512_add_ps(t[5], t[9]), _mm512_add_ps(t[6], t[10])), tails[4]);
            Store<false, mask>(dst + 6 * dstStride, _mm512_sub_ps(_mm512_add_ps(t[6], t[10]), _mm512_add_ps(t[5], t[9])), tails[4]);
            Store<false, mask>(dst + 7 * dstStride, _mm512_sub_ps(_mm512_add_ps(t[5], t[9]), _mm512_add_ps(t[7], t[11])), tails[4]);
            Store<false, mask>(dst + 8 * dstStride, _mm512_sub_ps(_mm512_sub_ps(t[8], t[4]), _mm512_sub_ps(t[10], t[6])), tails[4]);
            Store<false, mask>(dst + 9 * dstStride, _mm512_add_ps(_mm512_sub_ps(t[9], t[5]), _mm512_sub_ps(t[10], t[6])), tails[4]);
            Store<false, mask>(dst + 10 * dstStride, _mm512_sub_ps(_mm512_sub_ps(t[10], t[6]), _mm512_sub_ps(t[9], t[5])), tails[4]);
            Store<false, mask>(dst + 11 * dstStride, _mm512_sub_ps(_mm512_sub_ps(t[9], t[5]), _mm512_sub_ps(t[11], t[7])), tails[4]);
            Store<false, mask>(dst + 12 * dstStride, _mm512_sub_ps(_mm512_sub_ps(t[4], t[12]), _mm512_sub_ps(t[6], t[14])), tails[4]);
            Store<false, mask>(dst + 13 * dstStride, _mm512_add_ps(_mm512_sub_ps(t[5], t[13]), _mm512_sub_ps(t[6], t[14])), tails[4]);
            Store<false, mask>(dst + 14 * dstStride, _mm512_sub_ps(_mm512_sub_ps(t[6], t[14]), _mm512_sub_ps(t[5], t[13])), tails[4]);
            Store<false, mask>(dst + 15 * dstStride, _mm512_sub_ps(_mm512_sub_ps(t[5], t[13]), _mm512_sub_ps(t[7], t[15])), tails[4]);
        }

        SIMD_INLINE void Winograd2x3pSetInput16Body(const float * src, size_t srcStride, float * dst, size_t dstStride, const __mmask16 * tails)
        {
            __m512 t[16];
            Winograd2x3pSetInputLoad16Row<false>(src + 0 * srcStride, t + 0, tails);
            Winograd2x3pSetInputLoad16Row<false>(src + 1 * srcStride, t + 4, tails);
            Winograd2x3pSetInputLoad16Row<false>(src + 2 * srcStride, t + 8, tails);
            Winograd2x3pSetInputLoad16Row<false>(src + 3 * srcStride, t + 12, tails);
            Winograd2x3pSetInput16Store<false>(t, dst, dstStride, tails);
        }

        template<bool mask> SIMD_INLINE void Winograd2x3pSetInput16Edge(const float * src, size_t srcStride, PadType rowPad, float * dst, size_t dstStride, const __mmask16 * tails)
        {
            __m512 t[16];
            if (rowPad == PadNose1)
                Winograd2x3pSetInputLoad16Zero(t + 0);
            else
                Winograd2x3pSetInputLoad16Row<mask>(src + 0 * srcStride, t + 0, tails);
            Winograd2x3pSetInputLoad16Row<mask>(src + 1 * srcStride, t + 4, tails);
            if (rowPad == PadTail2)
                Winograd2x3pSetInputLoad16Zero(t + 8);
            else
                Winograd2x3pSetInputLoad16Row<mask>(src + 2 * srcStride, t + 8, tails);
            if (rowPad >= PadTail1)
                Winograd2x3pSetInputLoad16Zero(t + 12);
            else
                Winograd2x3pSetInputLoad16Row<mask>(src + 3 * srcStride, t + 12, tails);
            Winograd2x3pSetInput16Store<mask>(t, dst, dstStride, tails);
        }

        void Winograd2x3pSetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, int pad)
        {
            if (srcHeight < 4 || srcWidth < 4)
            {
                Base::Winograd2x3pSetInput(src, srcChannels, srcHeight, srcWidth, dst, pad);
                return;
            }
            size_t dstH = pad ? srcHeight : srcHeight - 2;
            size_t dstW = pad ? srcWidth : srcWidth - 2;
            size_t tileH = (dstH + 1) / 2;
            size_t tileW = (dstW + 1) / 2;
            size_t dstStride = srcChannels * tileH*tileW;

            size_t dstH2 = AlignLo(dstH, 2);
            size_t dstW2 = AlignLo(dstW, 2);
            size_t dstW32 = AlignLo(dstW, 32);
            if (pad && dstW32 == dstW)
                dstW32 -= 32;
            PadType rowPad = dstH2 < dstH ? PadTail1 : PadNone;
            size_t tailRow = dstH2 < dstH ? dstH - 1 : dstH - 2;
            bool specialRowTail = dstH2 < dstH || (pad && dstH2);
            bool specialColTail = pad ? dstW32 : (dstW32 < dstW);

            __mmask16 tails[5], noses[5];
            for (size_t c = 0; c < 2; ++c)
            {
                noses[c * 2 + 0] = TailMask16(dstW - F * c - 0 + (pad ? 1 : 2));
                noses[c * 2 + 1] = TailMask16(dstW - F * c - 2 + (pad ? 1 : 2));
                tails[c * 2 + 0] = TailMask16(dstW - dstW32 - F * c - 0 + (pad ? 1 : 2));
                tails[c * 2 + 1] = TailMask16(dstW - dstW32 - F * c - 2 + (pad ? 1 : 2));
            }
            noses[4] = TailMask16(tileW);
            tails[4] = TailMask16(tileW - dstW32/2);

            if (pad)
            {
                src -= srcWidth + 1;
                rowPad = dstH2 < dstH ? PadTail2 : PadTail1;
                noses[0] = noses[0] & (~1);
                if (dstH2 == dstH)
                    dstH2 -= 2;
            }
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, tileY = 0;
                if (pad)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + row * srcWidth;
                    float * d = dst + tileY * tileW;
                    if (pad)
                        Winograd2x3pSetInput16Edge<true>(s + col, srcWidth, PadNose1, d + tileX, dstStride, noses), col += 32, tileX += 16;
                    for (; col < dstW32; col += 32, tileX += 16)
                        Winograd2x3pSetInput16Edge<false>(s + col, srcWidth, PadNose1, d + tileX, dstStride, tails);
                    if (specialColTail)
                        Winograd2x3pSetInput16Edge<true>(s + col, srcWidth, PadNose1, d + tileX, dstStride, tails);
                    row += 2, tileY += 1;
                }
                for (; row < dstH2; row += 2, tileY += 1)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + row * srcWidth;
                    float * d = dst + tileY * tileW;
                    if (pad)
                        Winograd2x3pSetInput16Edge<true>(s + col, srcWidth, PadNone, d + tileX, dstStride, noses), col += 32, tileX += 16;
                    for (; col < dstW32; col += 32, tileX += 16)
                        Winograd2x3pSetInput16Body(s + col, srcWidth, d + tileX, dstStride, tails);
                    if (specialColTail)
                        Winograd2x3pSetInput16Edge<true>(s + col, srcWidth, PadNone, d + tileX, dstStride, tails);
                }
                if (specialRowTail)
                {
                    size_t col = 0, tileX = 0;
                    const float * s = src + tailRow * srcWidth;
                    float * d = dst + (tileH - 1) * tileW;
                    if (pad)
                        Winograd2x3pSetInput16Edge<true>(s + col, srcWidth, rowPad, d + tileX, dstStride, noses), col += 32, tileX += 16;
                    for (; col < dstW32; col += 32, tileX += 16)
                        Winograd2x3pSetInput16Edge<false>(s + col, srcWidth, rowPad, d + tileX, dstStride, tails);
                    if (specialColTail)
                        Winograd2x3pSetInput16Edge<true>(s + col, srcWidth, rowPad, d + tileX, dstStride, tails);
                }
                src += srcWidth * srcHeight;
                dst += tileW * tileH;
            }
        }

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
            d[0] = _mm512_add_ps(_mm512_add_ps(t[0], t[2]), t[4]);
            d[1] = _mm512_add_ps(_mm512_add_ps(t[1], t[3]), t[5]);
            Store<false, mask>(dst + 0, Interleave<0>(d[0], d[1]), tails[1]);
            Store<false, mask>(dst + F, Interleave<1>(d[0], d[1]), tails[2]);
            if (main)
            {
                dst += dstStride;
                Winograd2x3pSetOutputLoad2t<mask>(src + 12 * srcStride, srcStride, t + 6, tails[0]);
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
