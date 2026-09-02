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
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        typedef Base::SynetConvolution16bNhwcSpecV0::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcSpecV0::PostprocessPtr PostprocessPtr;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE svuint32_t Float32ToBFloat16(svfloat32_t value, const svbool_t& mask)
        {
            svuint32_t bits = svreinterpret_u32_f32(value);
            svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
            return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
        }

        SIMD_INLINE svbfloat16_t BroadcastBf16x2(const uint16_t* src)
        {
            return svreinterpret_bf16_u32(svdup_n_u32(uint32_t(src[0]) | (uint32_t(src[1]) << 16)));
        }

        SIMD_INLINE svbfloat16_t LoadBf16x2(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_bf16_u32(svld1_u32(mask, (const uint32_t*)src));
        }

        SIMD_INLINE void SetZero(uint16_t* dst)
        {
            svst1_u16(svptrue_b16(), dst, svdup_n_u16(0));
        }

        SIMD_INLINE void ConvertDf(const float* src, uint16_t* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svst1h_u32(body, dst + 0, Float32ToBFloat16(svld1_f32(body, src + 0), body));
            svst1h_u32(body, dst + F, Float32ToBFloat16(svld1_f32(body, src + F), body));
        }

        SIMD_INLINE void ConvertTail(const float* src, size_t size, uint16_t* dst)
        {
            const size_t DF = svcntw() * 2;
            size_t i = 0;
            for (; i < size; ++i)
                dst[i] = Base::Float32ToBFloat16(src[i]);
            for (; i < DF; ++i)
                dst[i] = 0;
        }

        SIMD_INLINE void CopyDf(const uint16_t* src, uint16_t* dst)
        {
            svst1_u16(svptrue_b16(), dst, svld1_u16(svptrue_b16(), src));
        }

        SIMD_INLINE void CopyTail(const uint16_t* src, size_t size, uint16_t* dst)
        {
            const size_t DF = svcntw() * 2;
            size_t i = 0;
            for (; i < size; ++i)
                dst[i] = src[i];
            for (; i < DF; ++i)
                dst[i] = 0;
        }

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcSpecV0(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            const size_t F = a.F, DF = F * 2;
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
            const size_t F = a.F, DF = F * 2;
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

        template<int M> void Convolution16bNhwcSpecV0_2xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, size_t dstC, int zero, const uint16_t* weight0, float* dst)
        {
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            svbfloat16_t s0, w0, w1;
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
                    if (M > 0) d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
                    if (M > 1) d10 = svdup_n_f32(0.0f), d11 = svdup_n_f32(0.0f);
                    if (M > 2) d20 = svdup_n_f32(0.0f), d21 = svdup_n_f32(0.0f);
                    if (M > 3) d30 = svdup_n_f32(0.0f), d31 = svdup_n_f32(0.0f);
                    if (M > 4) d40 = svdup_n_f32(0.0f), d41 = svdup_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = svld1_f32(body, dst + 0 * dD + 0), d01 = svld1_f32(body, dst + 0 * dD + F);
                    if (M > 1) d10 = svld1_f32(body, dst + 1 * dD + 0), d11 = svld1_f32(body, dst + 1 * dD + F);
                    if (M > 2) d20 = svld1_f32(body, dst + 2 * dD + 0), d21 = svld1_f32(body, dst + 2 * dD + F);
                    if (M > 3) d30 = svld1_f32(body, dst + 3 * dD + 0), d31 = svld1_f32(body, dst + 3 * dD + F);
                    if (M > 4) d40 = svld1_f32(body, dst + 4 * dD + 0), d41 = svld1_f32(body, dst + 4 * dD + F);
                }
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs = offset[k], end = offs + dX; offs < end; offs += 2)
                    {
                        w0 = LoadBf16x2(weight0, body);
                        w1 = LoadBf16x2(weight1, body);
                        if (M > 0)
                        {
                            s0 = BroadcastBf16x2(src0 + offs);
                            d00 = svbfdot_f32(d00, s0, w0);
                            d01 = svbfdot_f32(d01, s0, w1);
                        }
                        if (M > 1)
                        {
                            s0 = BroadcastBf16x2(src1 + offs);
                            d10 = svbfdot_f32(d10, s0, w0);
                            d11 = svbfdot_f32(d11, s0, w1);
                        }
                        if (M > 2)
                        {
                            s0 = BroadcastBf16x2(src2 + offs);
                            d20 = svbfdot_f32(d20, s0, w0);
                            d21 = svbfdot_f32(d21, s0, w1);
                        }
                        if (M > 3)
                        {
                            s0 = BroadcastBf16x2(src3 + offs);
                            d30 = svbfdot_f32(d30, s0, w0);
                            d31 = svbfdot_f32(d31, s0, w1);
                        }
                        if (M > 4)
                        {
                            s0 = BroadcastBf16x2(src4 + offs);
                            d40 = svbfdot_f32(d40, s0, w0);
                            d41 = svbfdot_f32(d41, s0, w1);
                        }
                        weight0 += DF;
                        weight1 += DF;
                    }
                }
                if (M > 0) svst1_f32(body, dst + 0 * dD + 0, d00), svst1_f32(body, dst + 0 * dD + F, d01);
                if (M > 1) svst1_f32(body, dst + 1 * dD + 0, d10), svst1_f32(body, dst + 1 * dD + F, d11);
                if (M > 2) svst1_f32(body, dst + 2 * dD + 0, d20), svst1_f32(body, dst + 2 * dD + F, d21);
                if (M > 3) svst1_f32(body, dst + 3 * dD + 0, d30), svst1_f32(body, dst + 3 * dD + F, d31);
                if (M > 4) svst1_f32(body, dst + 4 * dD + 0, d40), svst1_f32(body, dst + 4 * dD + F, d41);
            }
            else
            {
                if (zero)
                {
                    if (M > 0) d00 = svdup_n_f32(0.0f);
                    if (M > 1) d10 = svdup_n_f32(0.0f);
                    if (M > 2) d20 = svdup_n_f32(0.0f);
                    if (M > 3) d30 = svdup_n_f32(0.0f);
                    if (M > 4) d40 = svdup_n_f32(0.0f);
                }
                else
                {
                    if (M > 0) d00 = svld1_f32(body, dst + 0 * dD + 0);
                    if (M > 1) d10 = svld1_f32(body, dst + 1 * dD + 0);
                    if (M > 2) d20 = svld1_f32(body, dst + 2 * dD + 0);
                    if (M > 3) d30 = svld1_f32(body, dst + 3 * dD + 0);
                    if (M > 4) d40 = svld1_f32(body, dst + 4 * dD + 0);
                }
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs = offset[k], end = offs + dX; offs < end; offs += 2)
                    {
                        w0 = LoadBf16x2(weight0, body);
                        if (M > 0)
                        {
                            s0 = BroadcastBf16x2(src0 + offs);
                            d00 = svbfdot_f32(d00, s0, w0);
                        }
                        if (M > 1)
                        {
                            s0 = BroadcastBf16x2(src1 + offs);
                            d10 = svbfdot_f32(d10, s0, w0);
                        }
                        if (M > 2)
                        {
                            s0 = BroadcastBf16x2(src2 + offs);
                            d20 = svbfdot_f32(d20, s0, w0);
                        }
                        if (M > 3)
                        {
                            s0 = BroadcastBf16x2(src3 + offs);
                            d30 = svbfdot_f32(d30, s0, w0);
                        }
                        if (M > 4)
                        {
                            s0 = BroadcastBf16x2(src4 + offs);
                            d40 = svbfdot_f32(d40, s0, w0);
                        }
                        weight0 += DF;
                    }
                }
                if (M > 0) svst1_f32(body, dst + 0 * dD + 0, d00);
                if (M > 1) svst1_f32(body, dst + 1 * dD + 0, d10);
                if (M > 2) svst1_f32(body, dst + 2 * dD + 0, d20);
                if (M > 3) svst1_f32(body, dst + 3 * dD + 0, d30);
                if (M > 4) svst1_f32(body, dst + 4 * dD + 0, d40);
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
            const size_t F = a.F, DF = F * 2;
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

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* src, const float* bias,
            svfloat32_t param0, svfloat32_t param1, size_t offset, uint8_t* dst, const svbool_t& mask)
        {
            svfloat32_t f32 = Activate<type>(svadd_f32_x(mask, svld1_f32(mask, src + offset), svld1_f32(mask, bias + offset)), param0, param1, 0, mask);
            if (term == Term16bLast16b)
                svst1h_u32(mask, (uint16_t*)(dst + offset * 2), Float32ToBFloat16(f32, mask));
            else
                svst1_f32(mask, (float*)(dst + offset * 4), f32);
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Postprocess16bNhwcSpecV0(const float* src, const ConvParam& p,
            const AlgParam& a, size_t dstC, size_t dyBeg, size_t dyEnd, const float* bias, const float* params, uint8_t* dst)
        {
            const size_t F = a.F;
            const svbool_t body = svptrue_b32();
            size_t dstCF = AlignLo(dstC, F);
            size_t rowGap = a.gapH * a.macroD;
            src += dyBeg * a.srcW * a.macroD;
            dst += dyBeg * p.dstW * p.dstC * a.elem;
            svfloat32_t param0 = svdup_n_f32(params[0]);
            svfloat32_t param1 = svdup_n_f32(params[1]);
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < dstCF; dc += F)
                    {
                        if (type == ::SimdConvolutionActivationPrelu)
                            param0 = svld1_f32(body, params + dc);
                        Postprocess<term, type>(src, bias, param0, param1, dc, dst, body);
                    }
                    if (dc < dstC)
                    {
                        svbool_t tail = svwhilelt_b32(dc, dstC);
                        if (type == ::SimdConvolutionActivationPrelu)
                            param0 = svld1_f32(tail, params + dc);
                        Postprocess<term, type>(src, bias, param0, param1, dc, dst, tail);
                    }
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
            const size_t F = svcntw();
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
