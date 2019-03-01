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
#include "Simd/SimdWinograd.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE  
    namespace Neon
    {
        SIMD_INLINE void Load4(const float * src, size_t step, float32x4_t * dst)
        {
            float32x4_t a0 = Load<false>(src + 0 * step);
            float32x4_t a1 = Load<false>(src + 1 * step);
            float32x4_t a2 = Load<false>(src + 2 * step);
            float32x4_t a3 = Load<false>(src + 3 * step);
            float32x4x2_t b0 = vzipq_f32(a0, a2);
            float32x4x2_t b1 = vzipq_f32(a1, a3);
            *(float32x4x2_t*)(dst + 0) = vzipq_f32(b0.val[0], b1.val[0]);
            *(float32x4x2_t*)(dst + 2) = vzipq_f32(b0.val[1], b1.val[1]);
        }

        SIMD_INLINE void Winograd2x3SetFilter4n(const float * src, float * dst, size_t stride)
        {
            const float32x4_t r2 = vdupq_n_f32(1.0f / 2.0f);
            const float32x4_t r4 = vdupq_n_f32(1.0f / 4.0f);

            float32x4_t s[9];
            Load4(src + 0, 9, s + 0);
            Load4(src + 4, 9, s + 4);
            s[8] = SetF32(src[8], src[17], src[26], src[35]);

            Store<false>(dst + 0 * stride, s[0]);
            float32x4_t _0a2 = vaddq_f32(s[0], s[2]);
            Store<false>(dst + 1 * stride, vmulq_f32(vaddq_f32(_0a2, s[1]), r2));
            Store<false>(dst + 2 * stride, vmulq_f32(vsubq_f32(_0a2, s[1]), r2));
            Store<false>(dst + 3 * stride, s[2]);

            float32x4_t _0a6a3 = vaddq_f32(vaddq_f32(s[0], s[6]), s[3]);
            Store<false>(dst + 4 * stride, vmulq_f32(_0a6a3, r2));
            float32x4_t _2a8a5 = vaddq_f32(vaddq_f32(s[2], s[8]), s[5]);
            float32x4_t _1a7a4 = vaddq_f32(vaddq_f32(s[1], s[7]), s[4]);
            Store<false>(dst + 5 * stride, vmulq_f32(vaddq_f32(vaddq_f32(_0a6a3, _2a8a5), _1a7a4), r4));
            Store<false>(dst + 6 * stride, vmulq_f32(vsubq_f32(vaddq_f32(_0a6a3, _2a8a5), _1a7a4), r4));
            Store<false>(dst + 7 * stride, vmulq_f32(_2a8a5, r2));

            float32x4_t _0a6s3 = vsubq_f32(vaddq_f32(s[0], s[6]), s[3]);
            Store<false>(dst + 8 * stride, vmulq_f32(_0a6s3, r2));
            float32x4_t _2a8s5 = vsubq_f32(vaddq_f32(s[2], s[8]), s[5]);
            float32x4_t _1a7s4 = vsubq_f32(vaddq_f32(s[1], s[7]), s[4]);
            Store<false>(dst + 9 * stride, vmulq_f32(vaddq_f32(vaddq_f32(_0a6s3, _2a8s5), _1a7s4), r4));
            Store<false>(dst + 10 * stride, vmulq_f32(vsubq_f32(vaddq_f32(_0a6s3, _2a8s5), _1a7s4), r4));
            Store<false>(dst + 11 * stride, vmulq_f32(_2a8s5, r2));

            Store<false>(dst + 12 * stride, s[6]);
            float32x4_t _6a8 = vaddq_f32(s[6], s[8]);
            Store<false>(dst + 13 * stride, vmulq_f32(vaddq_f32(_6a8, s[7]), r2));
            Store<false>(dst + 14 * stride, vmulq_f32(vsubq_f32(_6a8, s[7]), r2));
            Store<false>(dst + 15 * stride, s[8]);
        }

        SIMD_INLINE void Winograd2x3SetFilter4t(const float * src, float * dst, size_t stride)
        {
            const float32x4_t r2 = vdupq_n_f32(1.0f / 2.0f);
            const float32x4_t r4 = vdupq_n_f32(1.0f / 4.0f);

            float32x4_t s[9];
            s[0] = Load<false>(src + 0 * stride);
            s[1] = Load<false>(src + 1 * stride);
            s[2] = Load<false>(src + 2 * stride);
            s[3] = Load<false>(src + 3 * stride);
            s[4] = Load<false>(src + 4 * stride);
            s[5] = Load<false>(src + 5 * stride);
            s[6] = Load<false>(src + 6 * stride);
            s[7] = Load<false>(src + 7 * stride);
            s[8] = Load<false>(src + 8 * stride);

            Store<false>(dst + 0 * stride, s[0]);
            float32x4_t _0a2 = vaddq_f32(s[0], s[2]);
            Store<false>(dst + 1 * stride, vmulq_f32(vaddq_f32(_0a2, s[1]), r2));
            Store<false>(dst + 2 * stride, vmulq_f32(vsubq_f32(_0a2, s[1]), r2));
            Store<false>(dst + 3 * stride, s[2]);

            float32x4_t _0a6a3 = vaddq_f32(vaddq_f32(s[0], s[6]), s[3]);
            Store<false>(dst + 4 * stride, vmulq_f32(_0a6a3, r2));
            float32x4_t _2a8a5 = vaddq_f32(vaddq_f32(s[2], s[8]), s[5]);
            float32x4_t _1a7a4 = vaddq_f32(vaddq_f32(s[1], s[7]), s[4]);
            Store<false>(dst + 5 * stride, vmulq_f32(vaddq_f32(vaddq_f32(_0a6a3, _2a8a5), _1a7a4), r4));
            Store<false>(dst + 6 * stride, vmulq_f32(vsubq_f32(vaddq_f32(_0a6a3, _2a8a5), _1a7a4), r4));
            Store<false>(dst + 7 * stride, vmulq_f32(_2a8a5, r2));

            float32x4_t _0a6s3 = vsubq_f32(vaddq_f32(s[0], s[6]), s[3]);
            Store<false>(dst + 8 * stride, vmulq_f32(_0a6s3, r2));
            float32x4_t _2a8s5 = vsubq_f32(vaddq_f32(s[2], s[8]), s[5]);
            float32x4_t _1a7s4 = vsubq_f32(vaddq_f32(s[1], s[7]), s[4]);
            Store<false>(dst + 9 * stride, vmulq_f32(vaddq_f32(vaddq_f32(_0a6s3, _2a8s5), _1a7s4), r4));
            Store<false>(dst + 10 * stride, vmulq_f32(vsubq_f32(vaddq_f32(_0a6s3, _2a8s5), _1a7s4), r4));
            Store<false>(dst + 11 * stride, vmulq_f32(_2a8s5, r2));

            Store<false>(dst + 12 * stride, s[6]);
            float32x4_t _6a8 = vaddq_f32(s[6], s[8]);
            Store<false>(dst + 13 * stride, vmulq_f32(vaddq_f32(_6a8, s[7]), r2));
            Store<false>(dst + 14 * stride, vmulq_f32(vsubq_f32(_6a8, s[7]), r2));
            Store<false>(dst + 15 * stride, s[8]);
}

        void Winograd2x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            size_t size4 = AlignLo(size, 4), i = 0;
            if (trans)
            {
                for (; i < size4; i += 4)
                    Winograd2x3SetFilter4t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::Winograd2x3SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (; i < size4; i += 4, src += 36, dst += 4)
                    Winograd2x3SetFilter4n(src, dst, size);
                for (; i < size; i += 1, src += 9, dst += 1)
                    Base::Winograd2x3SetFilter1n(src, dst, size);
            }
        }

        SIMD_INLINE void Winograd2x3SetInputLoad4n(const float * src, float32x4_t * dst)
        {
            *(float32x4x2_t*)(dst + 0) = Load2<false>(src + 0);
            *(float32x4x2_t*)(dst + 2) = Load2<false>(src + 2);
        }

        SIMD_INLINE void Winograd2x3SetInputLoad4n(const float * src, float32x4_t * dst, PadType pad)
        {
            float32x4_t a0 = (pad == PadNose1 ? LoadPadZeroNose1(src + 0) : Load<false>(src + 0));
            float32x4_t a1 = Load<false>(src + 2);
            float32x4_t a2 = Load<false>(src + 4);
            float32x4_t a3 = (pad == PadTail2 ? LoadPadZeroTail2(src + 6) : (pad == PadTail1 ? LoadPadZeroTail1(src + 6) : Load<false>(src + 6)));
            *(float32x4x2_t*)(dst + 0) = vuzpq_f32(a0, a2);
            *(float32x4x2_t*)(dst + 2) = vuzpq_f32(a1, a3);
        }

        SIMD_INLINE void Winograd2x3SetInputLoad4z(float32x4_t * dst)
        {
            dst[0] = vdupq_n_f32(0.0f);
            dst[1] = vdupq_n_f32(0.0f);
            dst[2] = vdupq_n_f32(0.0f);
            dst[3] = vdupq_n_f32(0.0f);
        }

        SIMD_INLINE void Winograd2x3SetInput4Store(const float32x4_t * src, float * dst, size_t stride)
        {
            Store<false>(dst + 0 * stride, vsubq_f32(vsubq_f32(src[0], src[8]), vsubq_f32(src[2], src[10])));
            Store<false>(dst + 1 * stride, vaddq_f32(vsubq_f32(src[1], src[9]), vsubq_f32(src[2], src[10])));
            Store<false>(dst + 2 * stride, vsubq_f32(vsubq_f32(src[2], src[10]), vsubq_f32(src[1], src[9])));
            Store<false>(dst + 3 * stride, vsubq_f32(vsubq_f32(src[1], src[9]), vsubq_f32(src[3], src[11])));
            Store<false>(dst + 4 * stride, vsubq_f32(vaddq_f32(src[4], src[8]), vaddq_f32(src[6], src[10])));
            Store<false>(dst + 5 * stride, vaddq_f32(vaddq_f32(src[5], src[9]), vaddq_f32(src[6], src[10])));
            Store<false>(dst + 6 * stride, vsubq_f32(vaddq_f32(src[6], src[10]), vaddq_f32(src[5], src[9])));
            Store<false>(dst + 7 * stride, vsubq_f32(vaddq_f32(src[5], src[9]), vaddq_f32(src[7], src[11])));
            Store<false>(dst + 8 * stride, vsubq_f32(vsubq_f32(src[8], src[4]), vsubq_f32(src[10], src[6])));
            Store<false>(dst + 9 * stride, vaddq_f32(vsubq_f32(src[9], src[5]), vsubq_f32(src[10], src[6])));
            Store<false>(dst + 10 * stride, vsubq_f32(vsubq_f32(src[10], src[6]), vsubq_f32(src[9], src[5])));
            Store<false>(dst + 11 * stride, vsubq_f32(vsubq_f32(src[9], src[5]), vsubq_f32(src[11], src[7])));
            Store<false>(dst + 12 * stride, vsubq_f32(vsubq_f32(src[4], src[12]), vsubq_f32(src[6], src[14])));
            Store<false>(dst + 13 * stride, vaddq_f32(vsubq_f32(src[5], src[13]), vsubq_f32(src[6], src[14])));
            Store<false>(dst + 14 * stride, vsubq_f32(vsubq_f32(src[6], src[14]), vsubq_f32(src[5], src[13])));
            Store<false>(dst + 15 * stride, vsubq_f32(vsubq_f32(src[5], src[13]), vsubq_f32(src[7], src[15])));
        }

        SIMD_INLINE void Winograd2x3SetInput4n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float32x4_t t[16];
            Winograd2x3SetInputLoad4n(src + 0 * srcStride, t + 0);
            Winograd2x3SetInputLoad4n(src + 1 * srcStride, t + 4);
            Winograd2x3SetInputLoad4n(src + 2 * srcStride, t + 8);
            Winograd2x3SetInputLoad4n(src + 3 * srcStride, t + 12);
            Winograd2x3SetInput4Store(t, dst, dstStride);
        }

        SIMD_INLINE void Winograd2x3SetInput4n(const float * src, size_t srcStride, PadType rowPad, PadType colPad, float * dst, size_t dstStride)
        {
            float32x4_t t[16];
            if (rowPad == PadNose1)
                Winograd2x3SetInputLoad4z(t + 0);
            else
                Winograd2x3SetInputLoad4n(src + 0 * srcStride, t + 0, colPad);
            Winograd2x3SetInputLoad4n(src + 1 * srcStride, t + 4, colPad);
            if (rowPad == PadTail2)
                Winograd2x3SetInputLoad4z(t + 8);
            else
                Winograd2x3SetInputLoad4n(src + 2 * srcStride, t + 8, colPad);
            if (rowPad >= PadTail1)
                Winograd2x3SetInputLoad4z(t + 12);
            else
                Winograd2x3SetInputLoad4n(src + 3 * srcStride, t + 12, colPad);
            Winograd2x3SetInput4Store(t, dst, dstStride);
        }

        SIMD_INLINE void Winograd2x3SetInput4t(const float * src, size_t srcS, size_t srcC, float32x4_t dst[16])
        {
            dst[0] = Load<false>(src + 0 * srcS + 0 * srcC);
            dst[1] = Load<false>(src + 0 * srcS + 1 * srcC);
            dst[2] = Load<false>(src + 0 * srcS + 2 * srcC);
            dst[3] = Load<false>(src + 0 * srcS + 3 * srcC);
            dst[4] = Load<false>(src + 1 * srcS + 0 * srcC);
            dst[5] = Load<false>(src + 1 * srcS + 1 * srcC);
            dst[6] = Load<false>(src + 1 * srcS + 2 * srcC);
            dst[7] = Load<false>(src + 1 * srcS + 3 * srcC);
            dst[8] = Load<false>(src + 2 * srcS + 0 * srcC);
            dst[9] = Load<false>(src + 2 * srcS + 1 * srcC);
            dst[10] = Load<false>(src + 2 * srcS + 2 * srcC);
            dst[11] = Load<false>(src + 2 * srcS + 3 * srcC);
            dst[12] = Load<false>(src + 3 * srcS + 0 * srcC);
            dst[13] = Load<false>(src + 3 * srcS + 1 * srcC);
            dst[14] = Load<false>(src + 3 * srcS + 2 * srcC);
            dst[15] = Load<false>(src + 3 * srcS + 3 * srcC);
        }

        SIMD_INLINE void Winograd2x3SetInput4t(const float * src, size_t srcW, size_t srcC, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                float32x4_t tmp[16];
                Winograd2x3SetInput4t(src + c, srcS, srcC, tmp);
                Winograd2x3SetInput4Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                float32x4_t tmp[16];
                Winograd2x3SetInput4t(src + srcC - F, srcS, srcC, tmp);
                Winograd2x3SetInput4Store(tmp, dst + srcC - F, dstStride);
            }
        }

        SIMD_INLINE void Winograd2x3SetInput4t(const float * src, size_t srcS, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float32x4_t dst[16])
        {
            for (size_t i = 0; i < 16; ++i)
                dst[i] = vdupq_n_f32(0.0f);
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    dst[row * 4 + col] = Load<false>(src + row * srcS + col * srcC);
        }

        SIMD_INLINE void Winograd2x3SetInput4t(const float * src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            size_t srcCF = AlignLo(srcC, F);
            for (size_t c = 0; c < srcCF; c += F)
            {
                float32x4_t tmp[16];
                Winograd2x3SetInput4t(src + c, srcS, srcC, rowB, rowE, colB, colE, tmp);
                Winograd2x3SetInput4Store(tmp, dst + c, dstStride);
            }
            if (srcCF < srcC)
            {
                float32x4_t tmp[16];
                Winograd2x3SetInput4t(src + srcC - F, srcS, srcC, rowB, rowE, colB, colE, tmp);
                Winograd2x3SetInput4Store(tmp, dst + srcC - F, dstStride);
            }
        }

        void Winograd2x3SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, SimdBool pad, SimdBool trans)
        {
            if (trans ? (srcChannels < 4) : (srcHeight < 4 || srcWidth < 10))
            {
                Base::Winograd2x3SetInput(src, srcChannels, srcHeight, srcWidth, dst, pad, trans);
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
                        Winograd2x3SetInput4t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        Winograd2x3SetInput4t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        Winograd2x3SetInput4t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = start; row < dstH2; row += 2)
                {
                    if (pad)
                        Winograd2x3SetInput4t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 4, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        Winograd2x3SetInput4t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        Winograd2x3SetInput4t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 4, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstH)
                {
                    if (pad)
                        Winograd2x3SetInput4t(src + row * srcWidth* srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstW2; col += 2)
                        Winograd2x3SetInput4t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        Winograd2x3SetInput4t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
            }
            else
            {
                size_t dstW8 = AlignLo(dstW, 8);
                if (pad && dstW8 == dstW)
                    dstW8 -= 8;
                PadType rowPad = dstH2 < dstH ? PadTail1 : PadNone;
                PadType colPad = dstW2 < dstW ? PadTail1 : PadNone;
                size_t tailCol = dstW2 < dstW ? dstW - 7 : dstW - 8;
                size_t tailRow = dstH2 < dstH ? dstH - 1 : dstH - 2;
                bool specialColTail = dstW8 < dstW || pad;
                bool specialRowTail = dstH2 < dstH || pad;
                if (pad)
                {
                    src -= srcWidth + 1;
                    rowPad = dstH2 < dstH ? PadTail2 : PadTail1;
                    colPad = dstW2 < dstW ? PadTail2 : PadTail1;
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
                            Winograd2x3SetInput4n(s + col, srcWidth, PadNose1, PadNose1, d + tileX, dstStride), col += 8, tileX += 4;
                        for (; col < dstW8; col += 8, tileX += 4)
                            Winograd2x3SetInput4n(s + col, srcWidth, PadNose1, PadNone, d + tileX, dstStride);
                        if (specialColTail)
                            Winograd2x3SetInput4n(s + tailCol, srcWidth, PadNose1, colPad, d + tileW - 4, dstStride);
                        row += 2, tileY += 1;
                    }
                    for (; row < dstH2; row += 2, tileY += 1)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + row * srcWidth;
                        float * d = dst + tileY * tileW;
                        if (pad)
                            Winograd2x3SetInput4n(s + col, srcWidth, PadNone, PadNose1, d + tileX, dstStride), col += 8, tileX += 4;
                        for (; col < dstW8; col += 8, tileX += 4)
                            Winograd2x3SetInput4n(s + col, srcWidth, d + tileX, dstStride);
                        if (specialColTail)
                            Winograd2x3SetInput4n(s + tailCol, srcWidth, PadNone, colPad, d + tileW - 4, dstStride);
                    }
                    if (specialRowTail)
                    {
                        size_t col = 0, tileX = 0;
                        const float * s = src + tailRow * srcWidth;
                        float * d = dst + (tileH - 1) * tileW;
                        if (pad)
                            Winograd2x3SetInput4n(s + col, srcWidth, rowPad, PadNose1, d + tileX, dstStride), col += 8, tileX += 4;
                        for (; col < dstW8; col += 8, tileX += 4)
                            Winograd2x3SetInput4n(s + col, srcWidth, rowPad, PadNone, d + tileX, dstStride);
                        if (specialColTail)
                            Winograd2x3SetInput4n(s + tailCol, srcWidth, rowPad, colPad, d + tileW - 4, dstStride);
                    }
                    src += srcWidth * srcHeight;
                    dst += tileW * tileH;
                }
            }
        }

        SIMD_INLINE void Winograd4x3SetFilter4Row(const float32x4_t * t, float * dst, size_t stride)
        {
            const float32x4_t r4 = vdupq_n_f32(1.0f / 4.0f);
            const float32x4_t r6 = vdupq_n_f32(1.0f / 6.0f);
            const float32x4_t mr6 = vdupq_n_f32(-1.0f / 6.0f);
            const float32x4_t r12 = vdupq_n_f32(1.0f / 12.0f);
            const float32x4_t r24 = vdupq_n_f32(1.0f / 24.0f);
            Store<false>(dst + 0 * stride, vmulq_f32(r4, t[0]));
            float32x4_t t0 = vaddq_f32(t[0], t[2]);
            Store<false>(dst + 1 * stride, vmulq_f32(mr6, vaddq_f32(t0, t[1])));
            Store<false>(dst + 2 * stride, vmulq_f32(mr6, vsubq_f32(t0, t[1])));
            float32x4_t t1 = vaddq_f32(vmulq_f32(r24, t[0]), vmulq_f32(r6, t[2]));
            float32x4_t t2 = vmulq_f32(r12, t[1]);
            Store<false>(dst + 3 * stride, vaddq_f32(t1, t2));
            Store<false>(dst + 4 * stride, vsubq_f32(t1, t2));
            Store<false>(dst + 5 * stride, t[2]);
        }

        SIMD_INLINE void Winograd4x3SetFilter4All(const float32x4_t * s, float * dst, size_t stride)
        {
            const float32x4_t r4 = vdupq_n_f32(1.0f / 4.0f);
            const float32x4_t r6 = vdupq_n_f32(1.0f / 6.0f);
            const float32x4_t mr6 = vdupq_n_f32(-1.0f / 6.0f);
            const float32x4_t r12 = vdupq_n_f32(1.0f / 12.0f);
            const float32x4_t r24 = vdupq_n_f32(1.0f / 24.0f);

            float32x4_t t[3];
            t[0] = vmulq_f32(r4, s[0]);
            t[1] = vmulq_f32(r4, s[1]);
            t[2] = vmulq_f32(r4, s[2]);
            Winograd4x3SetFilter4Row(t, dst + 0 * stride, stride);

            t[0] = vmulq_f32(mr6, vaddq_f32(vaddq_f32(s[0], s[3]), s[6]));
            t[1] = vmulq_f32(mr6, vaddq_f32(vaddq_f32(s[1], s[4]), s[7]));
            t[2] = vmulq_f32(mr6, vaddq_f32(vaddq_f32(s[2], s[5]), s[8]));
            Winograd4x3SetFilter4Row(t, dst + 6 * stride, stride);

            t[0] = vmulq_f32(mr6, vaddq_f32(vsubq_f32(s[0], s[3]), s[6]));
            t[1] = vmulq_f32(mr6, vaddq_f32(vsubq_f32(s[1], s[4]), s[7]));
            t[2] = vmulq_f32(mr6, vaddq_f32(vsubq_f32(s[2], s[5]), s[8]));
            Winograd4x3SetFilter4Row(t, dst + 12 * stride, stride);

            t[0] = vaddq_f32(vaddq_f32(vmulq_f32(r24, s[0]), vmulq_f32(r12, s[3])), vmulq_f32(r6, s[6]));
            t[1] = vaddq_f32(vaddq_f32(vmulq_f32(r24, s[1]), vmulq_f32(r12, s[4])), vmulq_f32(r6, s[7]));
            t[2] = vaddq_f32(vaddq_f32(vmulq_f32(r24, s[2]), vmulq_f32(r12, s[5])), vmulq_f32(r6, s[8]));
            Winograd4x3SetFilter4Row(t, dst + 18 * stride, stride);

            t[0] = vaddq_f32(vsubq_f32(vmulq_f32(r24, s[0]), vmulq_f32(r12, s[3])), vmulq_f32(r6, s[6]));
            t[1] = vaddq_f32(vsubq_f32(vmulq_f32(r24, s[1]), vmulq_f32(r12, s[4])), vmulq_f32(r6, s[7]));
            t[2] = vaddq_f32(vsubq_f32(vmulq_f32(r24, s[2]), vmulq_f32(r12, s[5])), vmulq_f32(r6, s[8]));
            Winograd4x3SetFilter4Row(t, dst + 24 * stride, stride);

            Winograd4x3SetFilter4Row(s + 6, dst + 30 * stride, stride);
        }


        SIMD_INLINE void Winograd4x3SetFilter4n(const float * src, float * dst, size_t stride)
        {
            float32x4_t s[9];
            Load4(src + 0, 9, s + 0);
            Load4(src + 4, 9, s + 4);
            s[8] = SetF32(src[8], src[17], src[26], src[35]);
            Winograd4x3SetFilter4All(s, dst + 0 * stride, stride);
        }

        SIMD_INLINE void Winograd4x3SetFilter4t(const float * src, float * dst, size_t stride)
        {
            float32x4_t s[9];
            s[0] = Load<false>(src + 0 * stride);
            s[1] = Load<false>(src + 1 * stride);
            s[2] = Load<false>(src + 2 * stride);
            s[3] = Load<false>(src + 3 * stride);
            s[4] = Load<false>(src + 4 * stride);
            s[5] = Load<false>(src + 5 * stride);
            s[6] = Load<false>(src + 6 * stride);
            s[7] = Load<false>(src + 7 * stride);
            s[8] = Load<false>(src + 8 * stride);
            Winograd4x3SetFilter4All(s, dst + 0 * stride, stride);
        }

        void Winograd4x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            size_t size4 = AlignLo(size, 4), i = 0;
            if (trans)
            {
                for (; i < size4; i += 4)
                    Winograd4x3SetFilter4t(src + i, dst + i, size);
                for (; i < size; i += 1)
                    Base::Winograd4x3SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (; i < size4; i += 4, src += 36, dst += 4)
                    Winograd4x3SetFilter4n(src, dst, size);
                for (; i < size; i += 1, src += 9, dst += 1)
                    Base::Winograd4x3SetFilter1n(src, dst, size);
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
