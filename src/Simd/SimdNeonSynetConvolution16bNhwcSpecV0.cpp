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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Neon
    {
        typedef Base::SynetConvolution16bNhwcSpecV0::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcSpecV0::PostprocessPtr PostprocessPtr;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void SetZero(uint16_t* dst)
        {
            Store<false>(dst, vdupq_n_u16(0));
        }

        SIMD_INLINE void ConvertDf(const float* src, uint16_t* dst)
        {
            uint32x4_t d0 = Float32ToBFloat16(Load<false>(src + 0));
            uint32x4_t d1 = Float32ToBFloat16(Load<false>(src + F));
            Store<false>(dst, PackU32(d0, d1));
        }

        SIMD_INLINE void ConvertTail(const float* src, size_t size, uint16_t* dst)
        {
            size_t i = 0;
            for (; i < size; ++i)
                dst[i] = Base::Float32ToBFloat16(src[i]);
            for (; i < DF; ++i)
                dst[i] = 0;
        }

        SIMD_INLINE void CopyDf(const uint16_t* src, uint16_t* dst)
        {
            Store<false>(dst, Load<false>(src));
        }

        SIMD_INLINE void CopyTail(const uint16_t* src, size_t size, uint16_t* dst)
        {
            size_t i = 0;
            for (; i < size; ++i)
                dst[i] = src[i];
            for (; i < DF; ++i)
                dst[i] = 0;
        }

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcSpecV0(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            assert(a.microC == DF);
            const float* src = (float*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF), tailC = p.srcC - srcCDF;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        SetZero(dst + c * cD + s * sD);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            SetZero(dst + c * cD + s * sD);
                    dst += a.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        ConvertDf(src + sc, dst + sc * cD);
                    if (tailC)
                        ConvertTail(src + sc, tailC, dst + sc * cD);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        SetZero(dst + c * cD + s * sD);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        SetZero(dst + c * cD + s * sD);
            }
        }

        static void Reorder16bNhwcSpecV0(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            assert(a.microC == DF);
            const uint16_t* src = (uint16_t*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF), tailC = p.srcC - srcCDF;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        SetZero(dst + c * cD + s * sD);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            SetZero(dst + c * cD + s * sD);
                    dst += a.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        CopyDf(src + sc, dst + sc * cD);
                    if (tailC)
                        CopyTail(src + sc, tailC, dst + sc * cD);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        SetZero(dst + c * cD + s * sD);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        SetZero(dst + c * cD + s * sD);
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE float32x4_t BroadcastBf16(uint16_t value)
        {
            return vreinterpretq_f32_u32(vdupq_n_u32(uint32_t(value) << Base::Bf16::SHIFT));
        }

        template<int M> void Convolution16bNhwcSpecV0_2xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, size_t dstC, int zero, const uint16_t* weight0, float* dst)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11;
            size_t dD = a.macroD, dX = a.microC;
            const uint16_t* weight1 = weight0 + a.K * F;
            const uint16_t* src1 = src0 + 1 * dX;
            const uint16_t* src2 = src0 + 2 * dX;
            const uint16_t* src3 = src0 + 3 * dX;
            const uint16_t* src4 = src0 + 4 * dX;
            if (dstC > F)
            {
                if (zero)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0), d01 = Load<false>(dst + 0 * dD + F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0), d11 = Load<false>(dst + 1 * dD + F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0), d21 = Load<false>(dst + 2 * dD + F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0), d31 = Load<false>(dst + 3 * dD + F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0), d41 = Load<false>(dst + 4 * dD + F);
                }
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs = offset[k], end = offs + dX; offs < end; offs += 2)
                    {
                        w01 = Load<false>((float*)weight0);
                        w00 = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_f32(w01), Base::Bf16::SHIFT));
                        w01 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(w01), Bf16::MASK));
                        w11 = Load<false>((float*)weight1);
                        w10 = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_f32(w11), Base::Bf16::SHIFT));
                        w11 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(w11), Bf16::MASK));
                        if (M > 0)
                        {
                            s0 = BroadcastBf16(src0[offs + 0]);
                            d00 = vaddq_f32(vmulq_f32(s0, w00), d00);
                            d01 = vaddq_f32(vmulq_f32(s0, w10), d01);
                            s0 = BroadcastBf16(src0[offs + 1]);
                            d00 = vaddq_f32(vmulq_f32(s0, w01), d00);
                            d01 = vaddq_f32(vmulq_f32(s0, w11), d01);
                        }
                        if (M > 1)
                        {
                            s0 = BroadcastBf16(src1[offs + 0]);
                            d10 = vaddq_f32(vmulq_f32(s0, w00), d10);
                            d11 = vaddq_f32(vmulq_f32(s0, w10), d11);
                            s0 = BroadcastBf16(src1[offs + 1]);
                            d10 = vaddq_f32(vmulq_f32(s0, w01), d10);
                            d11 = vaddq_f32(vmulq_f32(s0, w11), d11);
                        }
                        if (M > 2)
                        {
                            s0 = BroadcastBf16(src2[offs + 0]);
                            d20 = vaddq_f32(vmulq_f32(s0, w00), d20);
                            d21 = vaddq_f32(vmulq_f32(s0, w10), d21);
                            s0 = BroadcastBf16(src2[offs + 1]);
                            d20 = vaddq_f32(vmulq_f32(s0, w01), d20);
                            d21 = vaddq_f32(vmulq_f32(s0, w11), d21);
                        }
                        if (M > 3)
                        {
                            s0 = BroadcastBf16(src3[offs + 0]);
                            d30 = vaddq_f32(vmulq_f32(s0, w00), d30);
                            d31 = vaddq_f32(vmulq_f32(s0, w10), d31);
                            s0 = BroadcastBf16(src3[offs + 1]);
                            d30 = vaddq_f32(vmulq_f32(s0, w01), d30);
                            d31 = vaddq_f32(vmulq_f32(s0, w11), d31);
                        }
                        if (M > 4)
                        {
                            s0 = BroadcastBf16(src4[offs + 0]);
                            d40 = vaddq_f32(vmulq_f32(s0, w00), d40);
                            d41 = vaddq_f32(vmulq_f32(s0, w10), d41);
                            s0 = BroadcastBf16(src4[offs + 1]);
                            d40 = vaddq_f32(vmulq_f32(s0, w01), d40);
                            d41 = vaddq_f32(vmulq_f32(s0, w11), d41);
                        }
                        weight0 += DF;
                        weight1 += DF;
                    }
                }
                if (M > 0) Store<false>(dst + 0 * dD + 0, d00), Store<false>(dst + 0 * dD + F, d01);
                if (M > 1) Store<false>(dst + 1 * dD + 0, d10), Store<false>(dst + 1 * dD + F, d11);
                if (M > 2) Store<false>(dst + 2 * dD + 0, d20), Store<false>(dst + 2 * dD + F, d21);
                if (M > 3) Store<false>(dst + 3 * dD + 0, d30), Store<false>(dst + 3 * dD + F, d31);
                if (M > 4) Store<false>(dst + 4 * dD + 0, d40), Store<false>(dst + 4 * dD + F, d41);
            }
            else
            {
                if (zero)
                {
                    if (M > 0) d00 = vdupq_n_f32(0.0f);
                    if (M > 1) d10 = vdupq_n_f32(0.0f);
                    if (M > 2) d20 = vdupq_n_f32(0.0f);
                    if (M > 3) d30 = vdupq_n_f32(0.0f);
                    if (M > 4) d40 = vdupq_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0);
                }
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs = offset[k], end = offs + dX; offs < end; offs += 2)
                    {
                        w01 = Load<false>((float*)weight0);
                        w00 = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_f32(w01), Base::Bf16::SHIFT));
                        w01 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(w01), Bf16::MASK));
                        if (M > 0)
                        {
                            s0 = BroadcastBf16(src0[offs + 0]);
                            d00 = vaddq_f32(vmulq_f32(s0, w00), d00);
                            s0 = BroadcastBf16(src0[offs + 1]);
                            d00 = vaddq_f32(vmulq_f32(s0, w01), d00);
                        }
                        if (M > 1)
                        {
                            s0 = BroadcastBf16(src1[offs + 0]);
                            d10 = vaddq_f32(vmulq_f32(s0, w00), d10);
                            s0 = BroadcastBf16(src1[offs + 1]);
                            d10 = vaddq_f32(vmulq_f32(s0, w01), d10);
                        }
                        if (M > 2)
                        {
                            s0 = BroadcastBf16(src2[offs + 0]);
                            d20 = vaddq_f32(vmulq_f32(s0, w00), d20);
                            s0 = BroadcastBf16(src2[offs + 1]);
                            d20 = vaddq_f32(vmulq_f32(s0, w01), d20);
                        }
                        if (M > 3)
                        {
                            s0 = BroadcastBf16(src3[offs + 0]);
                            d30 = vaddq_f32(vmulq_f32(s0, w00), d30);
                            s0 = BroadcastBf16(src3[offs + 1]);
                            d30 = vaddq_f32(vmulq_f32(s0, w01), d30);
                        }
                        if (M > 4)
                        {
                            s0 = BroadcastBf16(src4[offs + 0]);
                            d40 = vaddq_f32(vmulq_f32(s0, w00), d40);
                            s0 = BroadcastBf16(src4[offs + 1]);
                            d40 = vaddq_f32(vmulq_f32(s0, w01), d40);
                        }
                        weight0 += DF;
                    }
                }
                if (M > 0) Store<false>(dst + 0 * dD + 0, d00);
                if (M > 1) Store<false>(dst + 1 * dD + 0, d10);
                if (M > 2) Store<false>(dst + 2 * dD + 0, d20);
                if (M > 3) Store<false>(dst + 3 * dD + 0, d30);
                if (M > 4) Store<false>(dst + 4 * dD + 0, d40);
            }
        }

        typedef void(*Convolution16bNhwcSpecV0_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, size_t dstC, int zero, const uint16_t* weight0, float* dst);

        static Convolution16bNhwcSpecV0_2xM_Ptr GetConvolution16bNhwcSpecV0_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Convolution16bNhwcSpecV0_2xM<1>;
            case 2: return Convolution16bNhwcSpecV0_2xM<2>;
            case 3: return Convolution16bNhwcSpecV0_2xM<3>;
            case 4: return Convolution16bNhwcSpecV0_2xM<4>;
            case 5: return Convolution16bNhwcSpecV0_2xM<5>;
            }
            assert(0);
            return NULL;
        }

        static void Convolution16bNhwcSpecV0_2(const uint16_t* src, const ConvParam& p,
            const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t nK, int zero, const uint16_t* weight, float* dst)
        {
            size_t n1 = dstH * a.srcW - a.gapH, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.K * DF;
            size_t dD = a.macroD, dS = a.microC;
            Convolution16bNhwcSpecV0_2xM_Ptr convolution_2xN = GetConvolution16bNhwcSpecV0_2xM(n);
            Convolution16bNhwcSpecV0_2xM_Ptr convolution_2xM = GetConvolution16bNhwcSpecV0_2xM(m);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                for (; i < nn; i += n)
                    convolution_2xN(src + i * dS, p, a, offs, nK, dC, zero, weight, dst + i * dD);
                for (; i < n1; i += m)
                    convolution_2xM(src + i * dS, p, a, offs, nK, dC, zero, weight, dst + i * dD);
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst)
        {
            float32x4_t f32 = Activate<type>(vaddq_f32(Load<false>(src + offset), Load<false>(bias + offset)), params, offset);
            if (term == Term16bLast16b)
                Store<false>((uint16_t*)(dst + offset * 2), vmovn_u32(Float32ToBFloat16(f32)));
            else
                Store<false>((float*)(dst + offset * 4), f32);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
        {
            float32x4_t f32 = Activate<type>(vaddq_f32(Load<false>(src + offset), Load<false>(bias + offset)), params, offset);
            if (term == Term16bLast16b)
            {
                uint16_t tmp[F];
                Store<false>(tmp, vmovn_u32(Float32ToBFloat16(f32)));
                for (size_t i = 0; i < tail; ++i)
                    ((uint16_t*)dst)[offset + i] = tmp[i];
            }
            else
            {
                float tmp[F];
                Store<false>(tmp, f32);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[offset + i] = tmp[i];
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type>  void Postprocess16bNhwcSpecV0(const float* src, const ConvParam& p,
            const AlgParam& a, size_t dstC, size_t dyBeg, size_t dyEnd, const float* bias, const float* params, uint8_t* dst)
        {
            size_t dstCF = AlignLo(dstC, F), tailD = dstC - dstCF;
            size_t rowGap = a.gapH * a.macroD;
            src += dyBeg * a.srcW * a.macroD;
            dst += dyBeg * p.dstW * p.dstC * a.elem;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < dstCF; dc += F)
                        Postprocess<term, type>(src, bias, params, dc, dst);
                    if (tailD)
                        Postprocess<term, type>(src, bias, params, dc, dst, tailD);
                    src += a.macroD;
                    dst += p.dstC * a.elem;
                }
                src += rowGap;
            }
        }

        template<SimdConvolutionActivationType type> void SetPostprocess(const ConvParam& p, const AlgParam& a, PostprocessPtr& postprocess)
        {
            if (p.dstT == SimdTensorData16b)
                postprocess = Postprocess16bNhwcSpecV0<Term16bLast16b, type>;
            else
                postprocess = Postprocess16bNhwcSpecV0<Term16bLast32f, type>;
        }

        //-----------------------------------------------------------------------------------------

        SynetConvolution16bNhwcSpecV0::SynetConvolution16bNhwcSpecV0(const ConvParam& p)
            : Base::SynetConvolution16bNhwcSpecV0(p)
        {
            SetAlgParam(F, F * 2, 5, F * 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
                _preprocess = Reorder16bNhwcSpecV0;
            else
                _preprocess = Convert16bNhwcSpecV0;
            _convolution = Convolution16bNhwcSpecV0_2;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationRelu: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationLeakyRelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationRestrictRange: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationPrelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationElu: SetPostprocess<SimdConvolutionActivationElu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationHswish: SetPostprocess<SimdConvolutionActivationHswish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationMish: SetPostprocess<SimdConvolutionActivationMish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationHardSigmoid: SetPostprocess<SimdConvolutionActivationHardSigmoid>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationSwish: SetPostprocess<SimdConvolutionActivationSwish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationGelu: SetPostprocess<SimdConvolutionActivationGelu>(p, _alg, _postprocess); break;
            default: assert(0);
            }
        }
    }
#endif
}
