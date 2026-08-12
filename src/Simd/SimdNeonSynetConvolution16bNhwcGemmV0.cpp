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
        typedef Base::SynetConvolution16bNhwcGemmV0::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcGemmV0::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcGemm(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t srcC8 = Simd::AlignLo(p.srcC, 8);
            size_t srcC4 = Simd::AlignLo(p.srcC, 4);
            size_t gap = a.bufK - a.K;
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    uint16_t* row = dst + dr * a.bufK;
                    for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    const float* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    size_t sc = 0;
                                    for (; sc < srcC8; sc += 8)
                                    {
                                        uint32x4_t d0 = Float32ToBFloat16(Load<false>(ps + sc + 0));
                                        uint32x4_t d1 = Float32ToBFloat16(Load<false>(ps + sc + F));
                                        Store<false>(row + sc, PackU32(d0, d1));
                                    }
                                    for (; sc < srcC4; sc += 4)
                                    {
                                        uint32x4_t d0 = Float32ToBFloat16(Load<false>(ps + sc));
                                        Store<false>(row + sc, vmovn_u32(d0));
                                    }
                                    for (; sc < p.srcC; ++sc)
                                        row[sc] = Base::Float32ToBFloat16(ps[sc]);
                                    row += p.srcC;
                                }
                                else
                                {
                                    memset(row, 0, p.srcC * 2);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, 0, p.kernelX * p.srcC * 2);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    for (size_t g = 0; g < gap; ++g)
                        *(row++) = 0;
                }
            }
        }

        static void Reorder16bNhwcGemm(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t gap = a.bufK - a.K;
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    uint16_t* row = dst + dr * a.bufK;
                    for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    const uint16_t* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    memcpy(row, ps, p.srcC * 2);
                                    row += p.srcC;
                                }
                                else
                                {
                                    memset(row, 0, p.srcC * 2);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, 0, p.kernelX * p.srcC * 2);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    for (size_t g = 0; g < gap; ++g)
                        *(row++) = 0;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE float32x4_t BroadcastBf16(uint16_t value)
        {
            return vreinterpretq_f32_u32(vdupq_n_u32(uint32_t(value) << Base::Bf16::SHIFT));
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, float32x4_t val0, const float32x4_t* bias, const float32x4_t* params)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, float32x4_t val0, const float32x4_t* bias, const float32x4_t* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, float32x4_t val0, float32x4_t val1, const float32x4_t* bias, const float32x4_t* params)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, float32x4_t val0, float32x4_t val1, const float32x4_t* bias, const float32x4_t* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, tail);
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Convolution16bNhwcGemm_2xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, const float32x4_t* bias, const float32x4_t* params, float* buf, uint8_t* dst)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            const uint16_t* weight1 = weight0 + a.bufK * F;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
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
                    if (M > 0) d00 = Load<false>(buf + 0 * dB + 0), d01 = Load<false>(buf + 0 * dB + F);
                    if (M > 1) d10 = Load<false>(buf + 1 * dB + 0), d11 = Load<false>(buf + 1 * dB + F);
                    if (M > 2) d20 = Load<false>(buf + 2 * dB + 0), d21 = Load<false>(buf + 2 * dB + F);
                    if (M > 3) d30 = Load<false>(buf + 3 * dB + 0), d31 = Load<false>(buf + 3 * dB + F);
                    if (M > 4) d40 = Load<false>(buf + 4 * dB + 0), d41 = Load<false>(buf + 4 * dB + F);
                }
                for (size_t offs = 0; offs < srcC; offs += 2)
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
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, bias, params), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, bias, params), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, bias, params), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, bias, params), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, bias, params), dst += dD, buf += dB;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, bias, params, dstC), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, bias, params, dstC), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, bias, params, dstC), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, bias, params, dstC), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, bias, params, dstC), dst += dD, buf += dB;
                }
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
                    if (M > 0) d00 = Load<false>(buf + 0 * dB + 0);
                    if (M > 1) d10 = Load<false>(buf + 1 * dB + 0);
                    if (M > 2) d20 = Load<false>(buf + 2 * dB + 0);
                    if (M > 3) d30 = Load<false>(buf + 3 * dB + 0);
                    if (M > 4) d40 = Load<false>(buf + 4 * dB + 0);
                }
                for (size_t offs = 0; offs < srcC; offs += 2)
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
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, bias, params), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, bias, params), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, bias, params), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, bias, params), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, bias, params), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, bias, params, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, bias, params, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, bias, params, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, bias, params, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, bias, params, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*Convolution16bNhwcGemm_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight, const float32x4_t* bias, const float32x4_t* params, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> Convolution16bNhwcGemm_2xM_Ptr GetConvolution16bNhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Convolution16bNhwcGemm_2xM<term, type, 1>;
            case 2: return Convolution16bNhwcGemm_2xM<term, type, 2>;
            case 3: return Convolution16bNhwcGemm_2xM<term, type, 3>;
            case 4: return Convolution16bNhwcGemm_2xM<term, type, 4>;
            case 5: return Convolution16bNhwcGemm_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcGemm_2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n1 = dstH * p.dstW, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            Convolution16bNhwcGemm_2xM_Ptr convolution_2xN = GetConvolution16bNhwcGemm_2xM<term, type>(n);
            Convolution16bNhwcGemm_2xM_Ptr convolution_2xM = GetConvolution16bNhwcGemm_2xM<term, type>(m);

            float32x4_t _params[2], _bias[2];
            _params[0] = vdupq_n_f32(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = vdupq_n_f32(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = Load<false>(bias + dc + 0);
                _bias[1] = Load<false>(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = Load<false>(params + dc + 0);
                    _params[1] = Load<false>(params + dc + F);
                }
                const uint16_t* s = src;
                float* b = buf + dc;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                    convolution_2xN(s, p, a, srcC, dC, zero, weight, _bias, _params, b, d);
                for (; i < n1; i += m, s += m * dS, b += m * dB, d += m * dD)
                    convolution_2xM(s, p, a, srcC, dC, zero, weight, _bias, _params, b, d);
                weight += dW;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = Convolution16bNhwcGemm_2<Term16bInterim, SimdConvolutionActivationIdentity>;
            if (p.dstT == SimdTensorData16b)
                convolutions[1] = Convolution16bNhwcGemm_2<Term16bLast16b, type>;
            else
                convolutions[1] = Convolution16bNhwcGemm_2<Term16bLast32f, type>;
        }

        SynetConvolution16bNhwcGemmV0::SynetConvolution16bNhwcGemmV0(const ConvParam& p)
            : Base::SynetConvolution16bNhwcGemmV0(p)
        {
            SetAlgParam(F, F * 2, 5, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
            {
                AlgParam& a = _alg;
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                    _convert = Reorder16bNhwcGemm;
            }
            else
                _convert = Convert16bNhwcGemm;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationGelu: Set<SimdConvolutionActivationGelu>(p, _alg, _convolutions); break;
            default: assert(0);
            }
        }
    }
#endif
}
