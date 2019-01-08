/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdSse1.h"
#include "Simd/SimdAvx1.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        SIMD_INLINE void Winograd2x3SetFilter16t(const float * src, float * dst, size_t stride, __mmask16 tail = -1)
        {
            const __m512 r2 = _mm512_set1_ps(1.0f / 2.0f);
            const __m512 r4 = _mm512_set1_ps(1.0f / 4.0f);

            __m512 s[9];
            s[0] = _mm512_maskz_loadu_ps(tail, src + 0 * stride);
            s[1] = _mm512_maskz_loadu_ps(tail, src + 1 * stride);
            s[2] = _mm512_maskz_loadu_ps(tail, src + 2 * stride);
            s[3] = _mm512_maskz_loadu_ps(tail, src + 3 * stride);
            s[4] = _mm512_maskz_loadu_ps(tail, src + 4 * stride);
            s[5] = _mm512_maskz_loadu_ps(tail, src + 5 * stride);
            s[6] = _mm512_maskz_loadu_ps(tail, src + 6 * stride);
            s[7] = _mm512_maskz_loadu_ps(tail, src + 7 * stride);
            s[8] = _mm512_maskz_loadu_ps(tail, src + 8 * stride);

            _mm512_mask_storeu_ps(dst + 0 * stride, tail, s[0]);
            __m512 _0a2 = _mm512_add_ps(s[0], s[2]);
            _mm512_mask_storeu_ps(dst + 1 * stride, tail, _mm512_mul_ps(_mm512_add_ps(_0a2, s[1]), r2));
            _mm512_mask_storeu_ps(dst + 2 * stride, tail, _mm512_mul_ps(_mm512_sub_ps(_0a2, s[1]), r2));
            _mm512_mask_storeu_ps(dst + 3 * stride, tail, s[2]);

            __m512 _0a6a3 = _mm512_add_ps(_mm512_add_ps(s[0], s[6]), s[3]);
            _mm512_mask_storeu_ps(dst + 4 * stride, tail, _mm512_mul_ps(_0a6a3, r2));
            __m512 _2a8a5 = _mm512_add_ps(_mm512_add_ps(s[2], s[8]), s[5]);
            __m512 _1a7a4 = _mm512_add_ps(_mm512_add_ps(s[1], s[7]), s[4]);
            _mm512_mask_storeu_ps(dst + 5 * stride, tail, _mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(_0a6a3, _2a8a5), _1a7a4), r4));
            _mm512_mask_storeu_ps(dst + 6 * stride, tail, _mm512_mul_ps(_mm512_sub_ps(_mm512_add_ps(_0a6a3, _2a8a5), _1a7a4), r4));
            _mm512_mask_storeu_ps(dst + 7 * stride, tail, _mm512_mul_ps(_2a8a5, r2));

            __m512 _0a6s3 = _mm512_sub_ps(_mm512_add_ps(s[0], s[6]), s[3]);
            _mm512_mask_storeu_ps(dst + 8 * stride, tail, _mm512_mul_ps(_0a6s3, r2));
            __m512 _2a8s5 = _mm512_sub_ps(_mm512_add_ps(s[2], s[8]), s[5]);
            __m512 _1a7s4 = _mm512_sub_ps(_mm512_add_ps(s[1], s[7]), s[4]);
            _mm512_mask_storeu_ps(dst + 9 * stride, tail, _mm512_mul_ps(_mm512_add_ps(_mm512_add_ps(_0a6s3, _2a8s5), _1a7s4), r4));
            _mm512_mask_storeu_ps(dst + 10 * stride, tail, _mm512_mul_ps(_mm512_sub_ps(_mm512_add_ps(_0a6s3, _2a8s5), _1a7s4), r4));
            _mm512_mask_storeu_ps(dst + 11 * stride, tail, _mm512_mul_ps(_2a8s5, r2));

            _mm512_mask_storeu_ps(dst + 12 * stride, tail, s[6]);
            __m512 _6a8 = _mm512_add_ps(s[6], s[8]);
            _mm512_mask_storeu_ps(dst + 13 * stride, tail, _mm512_mul_ps(_mm512_add_ps(_6a8, s[7]), r2));
            _mm512_mask_storeu_ps(dst + 14 * stride, tail, _mm512_mul_ps(_mm512_sub_ps(_6a8, s[7]), r2));
            _mm512_mask_storeu_ps(dst + 15 * stride, tail, s[8]);
        }

        void Winograd2x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            if (trans)
            {
                size_t sizeF = AlignLo(size, F), i = 0;
                for (; i < sizeF; i += F)
                    Winograd2x3SetFilter16t(src + i, dst + i, size);
                if (i < size)
                {
                    __mmask16 tail = TailMask16(size - sizeF); 
                    Winograd2x3SetFilter16t(src + i, dst + i, size, tail);
                }
            }
            else
            {
                Sse::Winograd2x3SetFilter(src, size, dst, trans);
            }
        }

        template <bool mask> SIMD_INLINE void Winograd2x3SetInputLoad16n(const float * src, __m512 * dst, const __mmask16 * tails)
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

        SIMD_INLINE void Winograd2x3SetInputLoad16z(__m512 * dst)
        {
            dst[0] = _mm512_setzero_ps();
            dst[1] = _mm512_setzero_ps();
            dst[2] = _mm512_setzero_ps();
            dst[3] = _mm512_setzero_ps();
        }

        template <bool mask> SIMD_INLINE void Winograd2x3SetInput16Store(const __m512 * src, float * dst, size_t stride, __mmask16 tail = -1)
        {
            Store<false, mask>(dst + 0 * stride, _mm512_sub_ps(_mm512_sub_ps(src[0], src[8]), _mm512_sub_ps(src[2], src[10])), tail);
            Store<false, mask>(dst + 1 * stride, _mm512_add_ps(_mm512_sub_ps(src[1], src[9]), _mm512_sub_ps(src[2], src[10])), tail);
            Store<false, mask>(dst + 2 * stride, _mm512_sub_ps(_mm512_sub_ps(src[2], src[10]), _mm512_sub_ps(src[1], src[9])), tail);
            Store<false, mask>(dst + 3 * stride, _mm512_sub_ps(_mm512_sub_ps(src[1], src[9]), _mm512_sub_ps(src[3], src[11])), tail);
            Store<false, mask>(dst + 4 * stride, _mm512_sub_ps(_mm512_add_ps(src[4], src[8]), _mm512_add_ps(src[6], src[10])), tail);
            Store<false, mask>(dst + 5 * stride, _mm512_add_ps(_mm512_add_ps(src[5], src[9]), _mm512_add_ps(src[6], src[10])), tail);
            Store<false, mask>(dst + 6 * stride, _mm512_sub_ps(_mm512_add_ps(src[6], src[10]), _mm512_add_ps(src[5], src[9])), tail);
            Store<false, mask>(dst + 7 * stride, _mm512_sub_ps(_mm512_add_ps(src[5], src[9]), _mm512_add_ps(src[7], src[11])), tail);
            Store<false, mask>(dst + 8 * stride, _mm512_sub_ps(_mm512_sub_ps(src[8], src[4]), _mm512_sub_ps(src[10], src[6])), tail);
            Store<false, mask>(dst + 9 * stride, _mm512_add_ps(_mm512_sub_ps(src[9], src[5]), _mm512_sub_ps(src[10], src[6])), tail);
            Store<false, mask>(dst + 10 * stride, _mm512_sub_ps(_mm512_sub_ps(src[10], src[6]), _mm512_sub_ps(src[9], src[5])), tail);
            Store<false, mask>(dst + 11 * stride, _mm512_sub_ps(_mm512_sub_ps(src[9], src[5]), _mm512_sub_ps(src[11], src[7])), tail);
            Store<false, mask>(dst + 12 * stride, _mm512_sub_ps(_mm512_sub_ps(src[4], src[12]), _mm512_sub_ps(src[6], src[14])), tail);
            Store<false, mask>(dst + 13 * stride, _mm512_add_ps(_mm512_sub_ps(src[5], src[13]), _mm512_sub_ps(src[6], src[14])), tail);
            Store<false, mask>(dst + 14 * stride, _mm512_sub_ps(_mm512_sub_ps(src[6], src[14]), _mm512_sub_ps(src[5], src[13])), tail);
            Store<false, mask>(dst + 15 * stride, _mm512_sub_ps(_mm512_sub_ps(src[5], src[13]), _mm512_sub_ps(src[7], src[15])), tail);
        }

        SIMD_INLINE void Winograd2x3SetInput16n(const float * src, size_t srcStride, float * dst, size_t dstStride, const __mmask16 * tails)
        {
            __m512 t[16];
            Winograd2x3SetInputLoad16n<false>(src + 0 * srcStride, t + 0, tails);
            Winograd2x3SetInputLoad16n<false>(src + 1 * srcStride, t + 4, tails);
            Winograd2x3SetInputLoad16n<false>(src + 2 * srcStride, t + 8, tails);
            Winograd2x3SetInputLoad16n<false>(src + 3 * srcStride, t + 12, tails);
            Winograd2x3SetInput16Store<false>(t, dst, dstStride, tails[4]);
        }

        template<bool mask> SIMD_INLINE void Winograd2x3SetInput16n(const float * src, size_t srcStride, PadType rowPad, float * dst, size_t dstStride, const __mmask16 * tails)
        {
            __m512 t[16];
            if (rowPad == PadNose1)
                Winograd2x3SetInputLoad16z(t + 0);
            else
                Winograd2x3SetInputLoad16n<mask>(src + 0 * srcStride, t + 0, tails);
            Winograd2x3SetInputLoad16n<mask>(src + 1 * srcStride, t + 4, tails);
            if (rowPad == PadTail2)
                Winograd2x3SetInputLoad16z(t + 8);
            else
                Winograd2x3SetInputLoad16n<mask>(src + 2 * srcStride, t + 8, tails);
            if (rowPad >= PadTail1)
                Winograd2x3SetInputLoad16z(t + 12);
            else
                Winograd2x3SetInputLoad16n<mask>(src + 3 * srcStride, t + 12, tails);
            Winograd2x3SetInput16Store<mask>(t, dst, dstStride, tails[4]);
        }

        SIMD_INLINE void Winograd2x3SetInput16t(const float * src, size_t srcS, size_t srcC, __m512 dst[16], __mmask16 tail = -1)
        {
            dst[0] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 0 * srcC);
            dst[1] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 1 * srcC);
            dst[2] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 2 * srcC);
            dst[3] = _mm512_maskz_loadu_ps(tail, src + 0 * srcS + 3 * srcC);
            dst[4] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 0 * srcC);
            dst[5] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 1 * srcC);
            dst[6] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 2 * srcC);
            dst[7] = _mm512_maskz_loadu_ps(tail, src + 1 * srcS + 3 * srcC);
            dst[8] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 0 * srcC);
            dst[9] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 1 * srcC);
            dst[10] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 2 * srcC);
            dst[11] = _mm512_maskz_loadu_ps(tail, src + 2 * srcS + 3 * srcC);
            dst[12] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 0 * srcC);
            dst[13] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 1 * srcC);
            dst[14] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 2 * srcC);
            dst[15] = _mm512_maskz_loadu_ps(tail, src + 3 * srcS + 3 * srcC);
        }

        SIMD_INLINE void Winograd2x3SetInput16t(const float * src, size_t srcW, size_t srcC, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC, srcCF = AlignLo(srcC, F), c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 tmp[16];
                Winograd2x3SetInput16t(src + c, srcS, srcC, tmp);
                Winograd2x3SetInput16Store<false>(tmp, dst + c, dstStride);
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 tmp[16];
                Winograd2x3SetInput16t(src + c, srcS, srcC, tmp, tail);
                Winograd2x3SetInput16Store<true>(tmp, dst + c, dstStride, tail);
            }
        }

        SIMD_INLINE void Winograd2x3SetInput16t(const float * src, size_t srcS, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, __m512 dst[16], __mmask16 tail = -1)
        {
            for (size_t i = 0; i < 16; ++i)
                dst[i] = _mm512_setzero_ps();
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    dst[row * 4 + col] = _mm512_maskz_loadu_ps(tail, src + row * srcS + col * srcC);
        }

        SIMD_INLINE void Winograd2x3SetInput16t(const float * src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC, srcCF = AlignLo(srcC, F), c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 tmp[16];
                Winograd2x3SetInput16t(src + c, srcS, srcC, rowB, rowE, colB, colE, tmp);
                Winograd2x3SetInput16Store<false>(tmp, dst + c, dstStride);
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 tmp[16];
                Winograd2x3SetInput16t(src + srcC - F, srcS, srcC, rowB, rowE, colB, colE, tmp, tail);
                Winograd2x3SetInput16Store<true>(tmp, dst + srcC - F, dstStride, tail);
            }
        }

        void Winograd2x3SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, SimdBool pad, SimdBool trans)
        {
            if (trans ? (false) : (srcHeight < 4 || srcWidth < 4))
            {
                Avx::Winograd2x3SetInput(src, srcChannels, srcHeight, srcWidth, dst, pad, trans);
                return;
            }
            size_t dstH = pad ? srcHeight : srcHeight - 2;
            size_t dstW = pad ? srcWidth : srcWidth - 2;
            size_t tileH = (dstH + 1) / 2;
            size_t tileW = (dstW + 1) / 2;
            size_t dstStride = srcChannels * tileH*tileW;
            size_t dstH2 = AlignLo(dstH, 2);
            size_t dstW2 = AlignLo(dstW, 2);
            if (trans)
            {
                size_t noseW = Simd::Min<size_t>(4, dstW + 1);
                size_t noseH = Simd::Min<size_t>(4, dstH + 1);
                size_t start = pad ? 2 : 0;
                if (pad)
                {
                    if (dstH == dstH2)
                        dstH2 -= 2;
                    if (dstW == dstW2)
                        dstW2 -= 2;
                    src -= (srcWidth + 1)*srcChannels;
                }
                size_t tailW = dstW - dstW2 + (pad ? 1 : 2);
                size_t tailH = dstH - dstH2 + (pad ? 1 : 2);
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        Winograd2x3SetInput16t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        Winograd2x3SetInput16t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        Winograd2x3SetInput16t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = start; row < dstH2; row += 2)
                {
                    if (pad)
                        Winograd2x3SetInput16t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 4, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        Winograd2x3SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        Winograd2x3SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 4, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstH)
                {
                    if (pad)
                        Winograd2x3SetInput16t(src + row * srcWidth* srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        Winograd2x3SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        Winograd2x3SetInput16t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
            }
            else
            {
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
                tails[4] = TailMask16(tileW - dstW32 / 2);

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
                            Winograd2x3SetInput16n<true>(s + col, srcWidth, PadNose1, d + tileX, dstStride, noses), col += 32, tileX += 16;
                        for (; col < dstW32; col += 32, tileX += 16)
                            Winograd2x3SetInput16n<false>(s + col, srcWidth, PadNose1, d + tileX, dstStride, tails);
                        if (specialColTail)
                            Winograd2x3SetInput16n<true>(s + col, srcWidth, PadNose1, d + tileX, dstStride, tails);
                        row += 2, tileY += 1;
                    }
                    for (; row < dstH2; row += 2, tileY += 1)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + row * srcWidth;
                        float * d = dst + tileY * tileW;
                        if (pad)
                            Winograd2x3SetInput16n<true>(s + col, srcWidth, PadNone, d + tileX, dstStride, noses), col += 32, tileX += 16;
                        for (; col < dstW32; col += 32, tileX += 16)
                            Winograd2x3SetInput16n(s + col, srcWidth, d + tileX, dstStride, tails);
                        if (specialColTail)
                            Winograd2x3SetInput16n<true>(s + col, srcWidth, PadNone, d + tileX, dstStride, tails);
                    }
                    if (specialRowTail)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + tailRow * srcWidth;
                        float * d = dst + (tileH - 1) * tileW;
                        if (pad)
                            Winograd2x3SetInput16n<true>(s + col, srcWidth, rowPad, d + tileX, dstStride, noses), col += 32, tileX += 16;
                        for (; col < dstW32; col += 32, tileX += 16)
                            Winograd2x3SetInput16n<false>(s + col, srcWidth, rowPad, d + tileX, dstStride, tails);
                        if (specialColTail)
                            Winograd2x3SetInput16n<true>(s + col, srcWidth, rowPad, d + tileX, dstStride, tails);
                    }
                    src += srcWidth * srcHeight;
                    dst += tileW * tileH;
                }
            }
        }

        template<bool mask> SIMD_INLINE void Winograd2x3SetOutputLoad4(const float * src, size_t srcStride, __m512 * dst, __mmask16 tail)
        {
            __m512 s0 = Load<false, mask>(src + 0 * srcStride, tail);
            __m512 s1 = Load<false, mask>(src + 1 * srcStride, tail);
            __m512 s2 = Load<false, mask>(src + 2 * srcStride, tail);
            __m512 s3 = Load<false, mask>(src + 3 * srcStride, tail);
            dst[0] = _mm512_add_ps(_mm512_add_ps(s0, s1), s2);
            dst[1] = _mm512_sub_ps(_mm512_sub_ps(s1, s2), s3);
        }

        template<bool main, bool mask> SIMD_INLINE void Winograd2x3SetOutput16n(const float * src, size_t srcStride, float * dst, size_t dstStride, const __mmask16 * tails)
        {
            __m512 t[8], d[4];
            Winograd2x3SetOutputLoad4<mask>(src + 0 * srcStride, srcStride, t + 0, tails[0]);
            Winograd2x3SetOutputLoad4<mask>(src + 4 * srcStride, srcStride, t + 2, tails[0]);
            Winograd2x3SetOutputLoad4<mask>(src + 8 * srcStride, srcStride, t + 4, tails[0]);
            d[0] = _mm512_add_ps(_mm512_add_ps(t[0], t[2]), t[4]);
            d[1] = _mm512_add_ps(_mm512_add_ps(t[1], t[3]), t[5]);
            Store<false, mask>(dst + 0, Interleave<0>(d[0], d[1]), tails[1]);
            Store<false, mask>(dst + F, Interleave<1>(d[0], d[1]), tails[2]);
            if (main)
            {
                dst += dstStride;
                Winograd2x3SetOutputLoad4<mask>(src + 12 * srcStride, srcStride, t + 6, tails[0]);
                d[2] = _mm512_sub_ps(_mm512_sub_ps(t[2], t[4]), t[6]);
                d[3] = _mm512_sub_ps(_mm512_sub_ps(t[3], t[5]), t[7]);
                Store<false, mask>(dst + 0, Interleave<0>(d[2], d[3]), tails[1]);
                Store<false, mask>(dst + F, Interleave<1>(d[2], d[3]), tails[2]);
            }
        }

        SIMD_INLINE void Winograd2x3SetOutputLoad16(const float * src, size_t stride, __m512 * dst, __mmask16 tail = -1)
        {
            __m512 tmp[8];
            Winograd2x3SetOutputLoad4<true>(src + 0 * stride, stride, tmp + 0, tail);
            Winograd2x3SetOutputLoad4<true>(src + 4 * stride, stride, tmp + 2, tail);
            Winograd2x3SetOutputLoad4<true>(src + 8 * stride, stride, tmp + 4, tail);
            Winograd2x3SetOutputLoad4<true>(src + 12 * stride, stride, tmp + 6, tail);
            dst[0] = _mm512_add_ps(_mm512_add_ps(tmp[0], tmp[2]), tmp[4]);
            dst[1] = _mm512_add_ps(_mm512_add_ps(tmp[1], tmp[3]), tmp[5]);
            dst[2] = _mm512_sub_ps(_mm512_sub_ps(tmp[2], tmp[4]), tmp[6]);
            dst[3] = _mm512_sub_ps(_mm512_sub_ps(tmp[3], tmp[5]), tmp[7]);
        }

        SIMD_INLINE void Winograd2x3SetOutputStore4(const __m512 src[4], float * dst, size_t dstS, size_t dstC, __mmask16 tail = -1)
        {
            _mm512_mask_storeu_ps(dst + 0 * dstS + 0 * dstC, tail, src[0]);
            _mm512_mask_storeu_ps(dst + 0 * dstS + 1 * dstC, tail, src[1]);
            _mm512_mask_storeu_ps(dst + 1 * dstS + 0 * dstC, tail, src[2]);
            _mm512_mask_storeu_ps(dst + 1 * dstS + 1 * dstC, tail, src[3]);
        }

        SIMD_INLINE void Winograd2x3SetOutput16t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F), d = 0;
            for (; d < dstCF; d += F)
            {
                __m512 tmp[4];
                Winograd2x3SetOutputLoad16(src + d, srcStride, tmp);
                Winograd2x3SetOutputStore4(tmp, dst + d, dstS, dstC);
            }
            if (d < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF);
                __m512 tmp[4];
                Winograd2x3SetOutputLoad16(src + d, srcStride, tmp, tail);
                Winograd2x3SetOutputStore4(tmp, dst + d, dstS, dstC, tail);
            }
        }

        SIMD_INLINE void Winograd2x3SetOutputStore4(const __m512 src[4], float * dst, size_t dstS, size_t dstC, size_t rowE, size_t colE, __mmask16 tail = -1)
        {
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    _mm512_mask_storeu_ps(dst + row * dstS + col * dstC, tail, src[row * 2 + col]);
        }

        SIMD_INLINE void Winograd2x3SetOutput16t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC, dstCF = AlignLo(dstC, F), d = 0;
            for (; d < dstCF; d += F)
            {
                __m512 tmp[4];
                Winograd2x3SetOutputLoad16(src + d, srcStride, tmp);
                Winograd2x3SetOutputStore4(tmp, dst + d, dstS, dstC, rowE, colE);
            }
            if (d < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF);
                __m512 tmp[4];
                Winograd2x3SetOutputLoad16(src + d, srcStride, tmp, tail);
                Winograd2x3SetOutputStore4(tmp, dst + d, dstS, dstC, rowE, colE, tail);
            }
        }

        void Winograd2x3SetOutput(const float * src, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            size_t tileH = (dstHeight + 1) / 2;
            size_t tileW = (dstWidth + 1) / 2;
            size_t srcStride = dstChannels * tileH*tileW;
            size_t dstH2 = AlignLo(dstHeight, 2);
            size_t dstW2 = AlignLo(dstWidth, 2);
            if (trans)
            {
                size_t row, col;
                for (row = 0; row < dstH2; row += 2)
                {
                    for (col = 0; col < dstW2; col += 2)
                        Winograd2x3SetOutput16t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        Winograd2x3SetOutput16t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, 2, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstW2; col += 2)
                        Winograd2x3SetOutput16t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, 2), src += dstChannels;
                    if (col < dstWidth)
                        Winograd2x3SetOutput16t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
                }
            }
            else
            {
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
                            Winograd2x3SetOutput16n<true, false>(s + tileX, srcStride, d + col, dstWidth, tails);
                        if (col < dstWidth)
                            Winograd2x3SetOutput16n<true, true>(s + tileX, srcStride, d + col, dstWidth, tails);
                    }
                    if (row < dstHeight)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + tileY * tileW;
                        float * d = dst + row * dstWidth;
                        for (col = 0; col < dstW32; col += 32, tileX += 16)
                            Winograd2x3SetOutput16n<false, false>(s + tileX, srcStride, d + col, dstWidth, tails);
                        if (col < dstWidth)
                            Winograd2x3SetOutput16n<false, true>(s + tileX, srcStride, d + col, dstWidth, tails);
                    }
                    src += tileW * tileH;
                    dst += dstHeight * dstWidth;
                }
            }
        }

        SIMD_INLINE void Winograd4x3SetFilter16Row(const __m512 * t, float * dst, size_t stride, __mmask16 tail)
        {
            const __m512 r4 = _mm512_set1_ps(1.0f / 4.0f);
            const __m512 r6 = _mm512_set1_ps(1.0f / 6.0f);
            const __m512 mr6 = _mm512_set1_ps(-1.0f / 6.0f);
            const __m512 r12 = _mm512_set1_ps(1.0f / 12.0f);
            const __m512 r24 = _mm512_set1_ps(1.0f / 24.0f);
            _mm512_mask_storeu_ps(dst + 0 * stride, tail, _mm512_mul_ps(r4, t[0]));
            __m512 t0 = _mm512_add_ps(t[0], t[2]);
            _mm512_mask_storeu_ps(dst + 1 * stride, tail, _mm512_mul_ps(mr6, _mm512_add_ps(t0, t[1])));
            _mm512_mask_storeu_ps(dst + 2 * stride, tail, _mm512_mul_ps(mr6, _mm512_sub_ps(t0, t[1])));
            __m512 t1 = _mm512_add_ps(_mm512_mul_ps(r24, t[0]), _mm512_mul_ps(r6, t[2]));
            __m512 t2 = _mm512_mul_ps(r12, t[1]);
            _mm512_mask_storeu_ps(dst + 3 * stride, tail, _mm512_add_ps(t1, t2));
            _mm512_mask_storeu_ps(dst + 4 * stride, tail, _mm512_sub_ps(t1, t2));
            _mm512_mask_storeu_ps(dst + 5 * stride, tail, t[2]);
        }

        SIMD_INLINE void Winograd4x3SetFilter16All(const __m512 * s, float * dst, size_t stride, __mmask16 tail)
        {
            const __m512 r4 = _mm512_set1_ps(1.0f / 4.0f);
            const __m512 r6 = _mm512_set1_ps(1.0f / 6.0f);
            const __m512 mr6 = _mm512_set1_ps(-1.0f / 6.0f);
            const __m512 r12 = _mm512_set1_ps(1.0f / 12.0f);
            const __m512 r24 = _mm512_set1_ps(1.0f / 24.0f);

            __m512 t[3];
            t[0] = _mm512_mul_ps(r4, s[0]);
            t[1] = _mm512_mul_ps(r4, s[1]);
            t[2] = _mm512_mul_ps(r4, s[2]);
            Winograd4x3SetFilter16Row(t, dst + 0 * stride, stride, tail);

            t[0] = _mm512_mul_ps(mr6, _mm512_add_ps(_mm512_add_ps(s[0], s[3]), s[6]));
            t[1] = _mm512_mul_ps(mr6, _mm512_add_ps(_mm512_add_ps(s[1], s[4]), s[7]));
            t[2] = _mm512_mul_ps(mr6, _mm512_add_ps(_mm512_add_ps(s[2], s[5]), s[8]));
            Winograd4x3SetFilter16Row(t, dst + 6 * stride, stride, tail);

            t[0] = _mm512_mul_ps(mr6, _mm512_add_ps(_mm512_sub_ps(s[0], s[3]), s[6]));
            t[1] = _mm512_mul_ps(mr6, _mm512_add_ps(_mm512_sub_ps(s[1], s[4]), s[7]));
            t[2] = _mm512_mul_ps(mr6, _mm512_add_ps(_mm512_sub_ps(s[2], s[5]), s[8]));
            Winograd4x3SetFilter16Row(t, dst + 12 * stride, stride, tail);

            t[0] = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(r24, s[0]), _mm512_mul_ps(r12, s[3])), _mm512_mul_ps(r6, s[6]));
            t[1] = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(r24, s[1]), _mm512_mul_ps(r12, s[4])), _mm512_mul_ps(r6, s[7]));
            t[2] = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(r24, s[2]), _mm512_mul_ps(r12, s[5])), _mm512_mul_ps(r6, s[8]));
            Winograd4x3SetFilter16Row(t, dst + 18 * stride, stride, tail);

            t[0] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(r24, s[0]), _mm512_mul_ps(r12, s[3])), _mm512_mul_ps(r6, s[6]));
            t[1] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(r24, s[1]), _mm512_mul_ps(r12, s[4])), _mm512_mul_ps(r6, s[7]));
            t[2] = _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(r24, s[2]), _mm512_mul_ps(r12, s[5])), _mm512_mul_ps(r6, s[8]));
            Winograd4x3SetFilter16Row(t, dst + 24 * stride, stride, tail);

            Winograd4x3SetFilter16Row(s + 6, dst + 30 * stride, stride, tail);
        }

        SIMD_INLINE void Winograd4x3SetFilter16t(const float * src, float * dst, size_t stride, __mmask16 tail = -1)
        {
            __m512 s[9];
            s[0] = _mm512_maskz_loadu_ps(tail, src + 0 * stride);
            s[1] = _mm512_maskz_loadu_ps(tail, src + 1 * stride);
            s[2] = _mm512_maskz_loadu_ps(tail, src + 2 * stride);
            s[3] = _mm512_maskz_loadu_ps(tail, src + 3 * stride);
            s[4] = _mm512_maskz_loadu_ps(tail, src + 4 * stride);
            s[5] = _mm512_maskz_loadu_ps(tail, src + 5 * stride);
            s[6] = _mm512_maskz_loadu_ps(tail, src + 6 * stride);
            s[7] = _mm512_maskz_loadu_ps(tail, src + 7 * stride);
            s[8] = _mm512_maskz_loadu_ps(tail, src + 8 * stride);
            Winograd4x3SetFilter16All(s, dst + 0 * stride, stride, tail);
        }

        void Winograd4x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            if (trans)
            {
                size_t sizeF = AlignLo(size, F), i = 0;
                for (; i < sizeF; i += F)
                    Winograd4x3SetFilter16t(src + i, dst + i, size);
                if (i < size)
                {
                    __mmask16 tail = TailMask16(size - sizeF);
                    Winograd4x3SetFilter16t(src + i, dst + i, size, tail);
                }
            }
            else
            {
                Sse::Winograd4x3SetFilter(src, size, dst, trans);
            }
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
