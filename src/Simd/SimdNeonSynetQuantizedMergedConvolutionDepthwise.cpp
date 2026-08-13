/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSynetQuantizedMergedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
    {
        typedef Base::SynetQuantizedMergedConvolution::AlgParam AlgParam;

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int32x4_t UnpackU8x4(const uint8_t* src)
        {
            uint8x8_t u8 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)src));
            return vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(u8))));
        }

        SIMD_INLINE int32x4_t PackRows(int32x4_t s0, int32x4_t s1)
        {
            return vorrq_s32(s0, vshlq_n_s32(s1, 16));
        }

        SIMD_INLINE int32x4_t ShiftLeft16(int32x4_t value)
        {
            return vshlq_n_s32(value, 16);
        }

        SIMD_INLINE int32x4_t ShiftRight16(int32x4_t value)
        {
            return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(value), 16));
        }

        void QuantizedMergedConvolutionDepthwisePreprocess(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd, uint8_t* dst)
        {
            int16x8_t _zero = vdupq_n_s16(zero[0]);
            size_t byMask = a.dbH - 1, byPad = p.kernelY - 1, byBeg = dyBeg ? dyBeg * p.strideY + byPad : 0, byEnd = dyEnd * p.strideY + byPad;
            if (a.dsB)
            {
                size_t syMask = a.dsH - 1, sC = a.dsH * p.srcW, sR = p.srcW * F;
                size_t bW = a.dbW * 2, bR = a.dbW * a.maC, xPad = p.padX * 2, wPad = p.padW * 2;
                for (size_t c = 0; c < maC; c += F)
                {
                    for (size_t by = byBeg; by < byEnd; by += 2)
                    {
                        int16_t* pd = (int16_t*)dst + (by & byMask) * bR;
                        size_t sy = by - p.padY;
                        const uint8_t* ps0 = (sy + 0) < p.srcH ? src + ((sy + 0) & syMask) * sR : zero;
                        const uint8_t* ps1 = (sy + 1) < p.srcH ? src + ((sy + 1) & syMask) * sR : zero;
                        if (xPad)
                        {
                            for (size_t x = 0; x < xPad; x += 2, pd += DF)
                                vst1q_s16(pd, _zero);
                        }
                        for (size_t sx = 0; sx < sR; sx += F, pd += DF)
                        {
                            int32x4_t s0 = UnpackU8x4(ps0 + sx);
                            int32x4_t s1 = UnpackU8x4(ps1 + sx);
                            vst1q_s32((int32_t*)pd, PackRows(s0, s1));
                        }
                        if (wPad)
                        {
                            for (size_t x = 0; x < wPad; x += 2, pd += DF)
                                vst1q_s16(pd, _zero);
                        }
                    }
                    src += sC * F;
                    dst += bW * DF;
                }
            }
            else
            {
                size_t sR = p.srcW * p.srcC, sC = p.srcC;
                size_t bW = a.dbW * 2, bC = a.maC, xPad = p.padX * 2, wPad = p.padW * 2, bR = a.dbW * a.maC;
                for (size_t by = byBeg; by < byEnd; by += 2)
                {
                    int16_t* pd = (int16_t*)dst + (by & byMask) * bR;
                    size_t sy = by - p.padY;
                    const uint8_t* ps0 = (sy + 0) < p.srcH ? src + (sy + 0) * sR : zero;
                    const uint8_t* ps1 = (sy + 1) < p.srcH ? src + (sy + 1) * sR : zero;
                    if (xPad)
                    {
                        for (size_t x = 0; x < xPad; x += 2, pd += DF)
                            for (size_t c = 0; c < bC; c += F)
                                vst1q_s16(pd + c * bW, _zero);
                    }
                    for (size_t sx = 0; sx < p.srcW; sx++, pd += DF)
                    {
                        for (size_t sc = 0; sc < maC; sc += F)
                        {
                            int32x4_t s0 = UnpackU8x4(ps0 + sc);
                            int32x4_t s1 = UnpackU8x4(ps1 + sc);
                            vst1q_s32((int32_t*)(pd + sc * bW), PackRows(s0, s1));
                        }
                        ps0 += sC;
                        ps1 += sC;
                    }
                    if (wPad)
                    {
                        for (size_t x = 0; x < wPad; x += 2, pd += DF)
                            for (size_t c = 0; c < bC; c += F)
                                vst1q_s16(pd + c * bW, _zero);
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void Madd2(int32x4_t& i32, int32x4_t u8, int32x4_t i8)
        {
            int16x8_t a = vreinterpretq_s16_s32(u8);
            int16x8_t b = vreinterpretq_s16_s32(i8);
            int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
            int32x4_t hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
#if defined(__aarch64__)
            i32 = vaddq_s32(i32, vpaddq_s32(lo, hi));
#else
            i32 = vaddq_s32(i32, vcombine_s32(
                vpadd_s32(vget_low_s32(lo), vget_high_s32(lo)),
                vpadd_s32(vget_low_s32(hi), vget_high_s32(hi))));
#endif
        }

        SIMD_INLINE void Save1(uint8_t* dst, int32x4_t sum, const int32x4_t& bias, const float32x4_t& norm, const int32x4_t& zero)
        {
            QuntizedTerm8i<Term8iLast8u>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero);
        }

        SIMD_INLINE void Save1(uint8_t* dst, int32x4_t sum, const int32x4_t& bias, const float32x4_t& norm, const int32x4_t& zero, size_t tail)
        {
            QuntizedTerm8i<Term8iLast8u>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero, tail);
        }

        SIMD_INLINE int32x4_t LoadI16(const int16_t* src)
        {
            return vld1q_s32((const int32_t*)src);
        }

        //------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionDepthwiseConvolutionAny(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd,
            const int8_t* weight8, const int32_t* bias, const float* norm, int32_t zero, uint8_t* dst)
        {
            const int16_t* src = (int16_t*)src8, *weight = (int16_t*)weight8;
            float32x4_t _norm;
            int32x4_t _zero = vdupq_n_s32(zero), _bias;
            int32x4_t d00, d10, d20, d30, d01, d11, d21, d31, w0, w1, s0;
            size_t sC = maC, sCF = AlignLo(sC, F), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * DF, dW = a.dwStep;
            size_t byMask = a.dbH - 1, bW = a.dbW * 2, bR = a.dbW * a.maC, dstW2 = AlignLo(p.dstW, 2), dstW4 = AlignLo(p.dstW, 4), dD = a.ddB ? a.maC : p.dstC;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.dwSize, dyD = p.dstW * dD;
            if(a.ddB)
                dst += (dyBeg % a.ddStep) * p.dstW * dD;
            else
                dst += dyBeg * p.dstW * dD;
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                size_t sc = 0, sy = dy * sY;
                for (; sc < sCF; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + sc * bW;
                    _bias = vld1q_s32(bias + sc);
                    _norm = vld1q_f32(norm + sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = vdupq_n_s32(0);
                        d10 = vdupq_n_s32(0);
                        d20 = vdupq_n_s32(0);
                        d30 = vdupq_n_s32(0);
                        d01 = vdupq_n_s32(0);
                        d11 = vdupq_n_s32(0);
                        d21 = vdupq_n_s32(0);
                        d31 = vdupq_n_s32(0);
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = LoadI16(pw0);
                                w1 = LoadI16(pw1);
                                s0 = LoadI16(ps + 0 * dX);
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                                s0 = LoadI16(ps + 1 * dX);
                                Madd2(d10, s0, w0);
                                Madd2(d11, s0, w1);
                                s0 = LoadI16(ps + 2 * dX);
                                Madd2(d20, s0, w0);
                                Madd2(d21, s0, w1);
                                s0 = LoadI16(ps + 3 * dX);
                                Madd2(d30, s0, w0);
                                Madd2(d31, s0, w1);
                            }
                        }
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd0 + 1 * dD, d10, _bias, _norm, _zero);
                        Save1(pd0 + 2 * dD, d20, _bias, _norm, _zero);
                        Save1(pd0 + 3 * dD, d30, _bias, _norm, _zero);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero);
                        Save1(pd1 + 1 * dD, d11, _bias, _norm, _zero);
                        Save1(pd1 + 2 * dD, d21, _bias, _norm, _zero);
                        Save1(pd1 + 3 * dD, d31, _bias, _norm, _zero);
                        pd0 += 4 * dD;
                        pd1 += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = vdupq_n_s32(0);
                        d10 = vdupq_n_s32(0);
                        d01 = vdupq_n_s32(0);
                        d11 = vdupq_n_s32(0);
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = LoadI16(pw0);
                                w1 = LoadI16(pw1);
                                s0 = LoadI16(ps + 0 * dX);
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                                s0 = LoadI16(ps + 1 * dX);
                                Madd2(d10, s0, w0);
                                Madd2(d11, s0, w1);
                            }
                        }
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd0 + 1 * dD, d10, _bias, _norm, _zero);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero);
                        Save1(pd1 + 1 * dD, d11, _bias, _norm, _zero);
                        pd0 += 2 * dD;
                        pd1 += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = vdupq_n_s32(0);
                        d01 = vdupq_n_s32(0);
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = LoadI16(pw0);
                                w1 = LoadI16(pw1);
                                s0 = LoadI16(ps + 0 * dX);
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                            }
                        }
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero);
                        pd0 += dD;
                        pd1 += dD;
                    }
                }
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + sc * bW;
                    _bias = vld1q_s32(bias + sc);
                    _norm = vld1q_f32(norm + sc);
                    size_t dx = 0, tail = sC - sCF;
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = vdupq_n_s32(0);
                        d01 = vdupq_n_s32(0);
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = LoadI16(pw0);
                                w1 = LoadI16(pw1);
                                s0 = LoadI16(ps + 0 * dX);
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                            }
                        }
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero, tail);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero, tail);
                        pd0 += dD;
                        pd1 += dD;
                    }
                }
                dst += p.dstW * 2 * dD;
            }
            for (; dy < dyEnd; ++dy)
            {
                size_t sc = 0, sy = dy * sY;
                for (; sc < sCF; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + sc * bW;
                    _bias = vld1q_s32(bias + sc);
                    _norm = vld1q_f32(norm + sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = vdupq_n_s32(0);
                        d10 = vdupq_n_s32(0);
                        d20 = vdupq_n_s32(0);
                        d30 = vdupq_n_s32(0);
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = LoadI16(pw);
                                Madd2(d00, LoadI16(ps + 0 * dX), w0);
                                Madd2(d10, LoadI16(ps + 1 * dX), w0);
                                Madd2(d20, LoadI16(ps + 2 * dX), w0);
                                Madd2(d30, LoadI16(ps + 3 * dX), w0);
                            }
                        }
                        Save1(pd + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd + 1 * dD, d10, _bias, _norm, _zero);
                        Save1(pd + 2 * dD, d20, _bias, _norm, _zero);
                        Save1(pd + 3 * dD, d30, _bias, _norm, _zero);
                        pd += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = vdupq_n_s32(0);
                        d10 = vdupq_n_s32(0);
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = LoadI16(pw);
                                Madd2(d00, LoadI16(ps + 0 * dX), w0);
                                Madd2(d10, LoadI16(ps + 1 * dX), w0);
                            }
                        }
                        Save1(pd + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd + 1 * dD, d10, _bias, _norm, _zero);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = vdupq_n_s32(0);
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = LoadI16(pw);
                                Madd2(d00, LoadI16(ps), w0);
                            }
                        }
                        Save1(pd, d00, _bias, _norm, _zero);
                        pd += dD;
                    }
                }
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + sc * bW;
                    _bias = vld1q_s32(bias + sc);
                    _norm = vld1q_f32(norm + sc);
                    size_t dx = 0, tail = sC - sCF;
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = vdupq_n_s32(0);
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = LoadI16(pw);
                                Madd2(d00, LoadI16(ps), w0);
                            }
                        }
                        Save1(pd, d00, _bias, _norm, _zero, tail);
                        pd += dD;
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionDepthwiseConvolution3x3(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd,
            const int8_t* weight8, const int32_t* bias, const float* norm, int32_t zero, uint8_t* dst)
        {
            const int16_t* src = (int16_t*)src8, * weight = (int16_t*)weight8;
            float32x4_t _norm;
            int32x4_t _zero = vdupq_n_s32(zero), _bias;
            int32x4_t d00, d10, w03, w14, w25, s0;
            size_t sC = maC, sCF = AlignLo(sC, F), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * DF, dW = a.dwStep;
            size_t byMask = a.dbH - 1, bW = a.dbW * 2, bR = a.dbW * a.maC, dstW2 = (sX == 1 ? AlignLo(p.dstW, 2) : 0), dD = a.ddB ? a.maC : p.dstC;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.dwSize, dyD = p.dstW * dD;
            if (a.ddB)
                dst += (dyBeg % a.ddStep) * p.dstW * dD;
            else
                dst += dyBeg * p.dstW * dD;
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                int32x4_t d01, w36, w47, w58;
                size_t sc = 0, sy = dy * sY;
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + ((sy + 0) & byMask) * bR + sc * bW;
                    const int16_t* ps2 = src + ((sy + 2) & byMask) * bR + sc * bW;
                    const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                    _bias = vld1q_s32(bias + sc);
                    _norm = vld1q_f32(norm + sc);
                    w03 = LoadI16(pw0 + 0 * 8);
                    w14 = LoadI16(pw0 + 1 * 8);
                    w25 = LoadI16(pw0 + 2 * 8);
                    w36 = LoadI16(pw1 + 3 * 8);
                    w47 = LoadI16(pw1 + 4 * 8);
                    w58 = LoadI16(pw1 + 5 * 8);
                    if (sc < sCF)
                    {
                        size_t dx = 0;
                        for (; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = vdupq_n_s32(0);
                            d01 = vdupq_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * 8);
                            Madd2(d00, s0, w03);
                            Madd2(d01, s0, ShiftLeft16(w03));
                            s0 = LoadI16(ps0 + 1 * 8);
                            Madd2(d00, s0, w14);
                            Madd2(d01, s0, ShiftLeft16(w14));
                            s0 = LoadI16(ps0 + 2 * 8);
                            Madd2(d00, s0, w25);
                            Madd2(d01, s0, ShiftLeft16(w25));
                            s0 = LoadI16(ps2 + 0 * 8);
                            Madd2(d00, s0, ShiftRight16(w36));
                            Madd2(d01, s0, w36);
                            s0 = LoadI16(ps2 + 1 * 8);
                            Madd2(d00, s0, ShiftRight16(w47));
                            Madd2(d01, s0, w47);
                            s0 = LoadI16(ps2 + 2 * 8);
                            Madd2(d00, s0, ShiftRight16(w58));
                            Madd2(d01, s0, w58);

                            Save1(pd0, d00, _bias, _norm, _zero);
                            Save1(pd1, d01, _bias, _norm, _zero);
                            pd0 += dD;
                            pd1 += dD;
                        }
                    }
                    else
                    {
                        size_t tail = sC - sCF;
                        for (size_t dx = 0; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = vdupq_n_s32(0);
                            d01 = vdupq_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * 8);
                            Madd2(d00, s0, w03);
                            Madd2(d01, s0, ShiftLeft16(w03));
                            s0 = LoadI16(ps0 + 1 * 8);
                            Madd2(d00, s0, w14);
                            Madd2(d01, s0, ShiftLeft16(w14));
                            s0 = LoadI16(ps0 + 2 * 8);
                            Madd2(d00, s0, w25);
                            Madd2(d01, s0, ShiftLeft16(w25));
                            s0 = LoadI16(ps2 + 0 * 8);
                            Madd2(d00, s0, ShiftRight16(w36));
                            Madd2(d01, s0, w36);
                            s0 = LoadI16(ps2 + 1 * 8);
                            Madd2(d00, s0, ShiftRight16(w47));
                            Madd2(d01, s0, w47);
                            s0 = LoadI16(ps2 + 2 * 8);
                            Madd2(d00, s0, ShiftRight16(w58));
                            Madd2(d01, s0, w58);

                            Save1(pd0, d00, _bias, _norm, _zero, tail);
                            Save1(pd1, d01, _bias, _norm, _zero, tail);
                            pd0 += dD;
                            pd1 += dD;
                        }
                    }
                }
                dst += p.dstW * dD * 2;
            }
            for (; dy < dyEnd; ++dy)
            {
                int32x4_t w6, w7, w8;
                size_t sc = 0, sy = dy * sY;
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + ((sy + 0) & byMask) * bR + sc * bW;
                    const int16_t* ps2 = src + ((sy + 2) & byMask) * bR + sc * bW;
                    const int16_t* pw = weight + sc * dW;
                    _bias = vld1q_s32(bias + sc);
                    _norm = vld1q_f32(norm + sc);
                    w03 = LoadI16(pw + 0 * 8);
                    w14 = LoadI16(pw + 1 * 8);
                    w25 = LoadI16(pw + 2 * 8);
                    w6 = LoadI16(pw + 3 * 8);
                    w7 = LoadI16(pw + 4 * 8);
                    w8 = LoadI16(pw + 5 * 8);
                    if (sc < sCF)
                    {
                        size_t dx = 0;
                        for (; dx < dstW2; dx += 2, ps0 += QF, ps2 += QF)
                        {
                            d00 = vdupq_n_s32(0);
                            d10 = vdupq_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * 8);
                            Madd2(d00, s0, w03);
                            s0 = LoadI16(ps0 + 1 * 8);
                            Madd2(d00, s0, w14);
                            Madd2(d10, s0, w03);
                            s0 = LoadI16(ps0 + 2 * 8);
                            Madd2(d00, s0, w25);
                            Madd2(d10, s0, w14);
                            s0 = LoadI16(ps0 + 3 * 8);
                            Madd2(d10, s0, w25);

                            s0 = LoadI16(ps2 + 0 * 8);
                            Madd2(d00, s0, w6);
                            s0 = LoadI16(ps2 + 1 * 8);
                            Madd2(d00, s0, w7);
                            Madd2(d10, s0, w6);
                            s0 = LoadI16(ps2 + 2 * 8);
                            Madd2(d00, s0, w8);
                            Madd2(d10, s0, w7);
                            s0 = LoadI16(ps2 + 3 * 8);
                            Madd2(d10, s0, w8);

                            Save1(pd + 0 * dD, d00, _bias, _norm, _zero);
                            Save1(pd + 1 * dD, d10, _bias, _norm, _zero);
                            pd += 2 * dD;
                        }
                        for (; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = vdupq_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * 8);
                            Madd2(d00, s0, w03);
                            s0 = LoadI16(ps0 + 1 * 8);
                            Madd2(d00, s0, w14);
                            s0 = LoadI16(ps0 + 2 * 8);
                            Madd2(d00, s0, w25);
                            s0 = LoadI16(ps2 + 0 * 8);
                            Madd2(d00, s0, w6);
                            s0 = LoadI16(ps2 + 1 * 8);
                            Madd2(d00, s0, w7);
                            s0 = LoadI16(ps2 + 2 * 8);
                            Madd2(d00, s0, w8);

                            Save1(pd, d00, _bias, _norm, _zero);
                            pd += dD;
                        }
                    }
                    else
                    {
                        size_t tail = sC - sCF;
                        for (size_t dx = 0; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = vdupq_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * 8);
                            Madd2(d00, s0, w03);
                            s0 = LoadI16(ps0 + 1 * 8);
                            Madd2(d00, s0, w14);
                            s0 = LoadI16(ps0 + 2 * 8);
                            Madd2(d00, s0, w25);
                            s0 = LoadI16(ps2 + 0 * 8);
                            Madd2(d00, s0, w6);
                            s0 = LoadI16(ps2 + 1 * 8);
                            Madd2(d00, s0, w7);
                            s0 = LoadI16(ps2 + 2 * 8);
                            Madd2(d00, s0, w8);

                            Save1(pd, d00, _bias, _norm, _zero, tail);
                            pd += dD;
                        }
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SetDepthwisePreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwisePreprocessPtr& func)
        {
            func = QuantizedMergedConvolutionDepthwisePreprocess;
        }

        void SetDepthwiseConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwiseConvolutionPtr& func)
        {
            if(p.IsKernel(3))
                func = QuantizedMergedConvolutionDepthwiseConvolution3x3;
            else
                func = QuantizedMergedConvolutionDepthwiseConvolutionAny;
        }
    }
#endif
}
