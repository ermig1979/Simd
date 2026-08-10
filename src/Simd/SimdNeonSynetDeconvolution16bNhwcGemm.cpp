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
#include "Simd/SimdSynetDeconvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
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
        typedef Base::SynetDeconvolution16bNhwcGemm::AlgParam AlgParam;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE float32x4_t BroadcastBf16(uint16_t value)
        {
            return vreinterpretq_f32_u32(vdupq_n_u32(uint32_t(value) << Base::Bf16::SHIFT));
        }

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcGemm(const uint8_t* src8, const DeconvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8 + yBeg * p.srcW * p.srcC;
            size_t size = p.srcC, gap = a.bufK - size;
            size_t sizeDF = Simd::AlignLo(size, DF);
            size_t sizeF = Simd::AlignLo(size, F);
            for (size_t sy = yBeg; sy < yEnd; ++sy)
            {
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < sizeDF; sc += DF)
                    {
                        uint32x4_t d0 = Float32ToBFloat16(Load<false>(src + sc + 0));
                        uint32x4_t d1 = Float32ToBFloat16(Load<false>(src + sc + F));
                        Store<false>(dst + sc, PackU32(d0, d1));
                    }
                    for (; sc < sizeF; sc += F)
                    {
                        uint32x4_t d0 = Float32ToBFloat16(Load<false>(src + sc));
                        Store<false>(dst + sc, vmovn_u32(d0));
                    }
                    for (; sc < p.srcC; ++sc)
                        dst[sc] = Base::Float32ToBFloat16(src[sc]);
                    src += size;
                    dst += size;
                    for (size_t g = 0; g < gap; ++g)
                        *(dst++) = 0;
                }
            }
        }

        static void Reorder16bNhwcGemm(const uint8_t* src8, const DeconvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            size_t size = a.K, gap = a.bufK - size;
            const uint16_t* src = (uint16_t*)src8 + yBeg * p.srcW * p.srcC;
            for (size_t sy = yBeg; sy < yEnd; ++sy)
            {
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    memcpy(dst, src, size * 2);
                    src += size;
                    dst += size;
                    for (size_t g = 0; g < gap; ++g)
                        *(dst++) = 0;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Save1(float* dst, float32x4_t val0)
        {
            Store<false>(dst, val0);
        }

        SIMD_INLINE void Save1(float* dst, float32x4_t val0, size_t tail)
        {
            float tmp[F];
            Store<false>(tmp, val0);
            for (size_t i = 0; i < tail; ++i)
                dst[i] = tmp[i];
        }

        SIMD_INLINE void Save2(float* dst, float32x4_t val0, float32x4_t val1)
        {
            Store<false>(dst + 0, val0);
            Store<false>(dst + F, val1);
        }

        SIMD_INLINE void Save2(float* dst, float32x4_t val0, float32x4_t val1, size_t tail)
        {
            Store<false>(dst + 0, val0);
            Save1(dst + F, val1, tail);
        }

        template<int M> void Deconvolution16bNhwcGemm_2xM(const uint16_t* src0, const DeconvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, float* dst)
        {
            float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11;
            size_t dD = a.bufN, dS = a.bufK;
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
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0), d01 = Load<false>(dst + 0 * dD + F);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0), d11 = Load<false>(dst + 1 * dD + F);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0), d21 = Load<false>(dst + 2 * dD + F);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0), d31 = Load<false>(dst + 3 * dD + F);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0), d41 = Load<false>(dst + 4 * dD + F);
                }
                for (size_t offs = 0; offs < srcC; offs += 2)
                {
                    uint32x4_t w0u = Load<false>((uint32_t*)weight0);
                    w00 = vreinterpretq_f32_u32(vshlq_n_u32(w0u, Base::Bf16::SHIFT));
                    w01 = vreinterpretq_f32_u32(vandq_u32(w0u, Bf16::MASK));
                    uint32x4_t w1u = Load<false>((uint32_t*)weight1);
                    w10 = vreinterpretq_f32_u32(vshlq_n_u32(w1u, Base::Bf16::SHIFT));
                    w11 = vreinterpretq_f32_u32(vandq_u32(w1u, Bf16::MASK));
                    if (M > 0)
                    {
                        s0 = BroadcastBf16(src0[offs + 0]);
                        d00 = vmlaq_f32(d00, s0, w00);
                        d01 = vmlaq_f32(d01, s0, w10);
                        s0 = BroadcastBf16(src0[offs + 1]);
                        d00 = vmlaq_f32(d00, s0, w01);
                        d01 = vmlaq_f32(d01, s0, w11);
                    }
                    if (M > 1)
                    {
                        s0 = BroadcastBf16(src1[offs + 0]);
                        d10 = vmlaq_f32(d10, s0, w00);
                        d11 = vmlaq_f32(d11, s0, w10);
                        s0 = BroadcastBf16(src1[offs + 1]);
                        d10 = vmlaq_f32(d10, s0, w01);
                        d11 = vmlaq_f32(d11, s0, w11);
                    }
                    if (M > 2)
                    {
                        s0 = BroadcastBf16(src2[offs + 0]);
                        d20 = vmlaq_f32(d20, s0, w00);
                        d21 = vmlaq_f32(d21, s0, w10);
                        s0 = BroadcastBf16(src2[offs + 1]);
                        d20 = vmlaq_f32(d20, s0, w01);
                        d21 = vmlaq_f32(d21, s0, w11);
                    }
                    if (M > 3)
                    {
                        s0 = BroadcastBf16(src3[offs + 0]);
                        d30 = vmlaq_f32(d30, s0, w00);
                        d31 = vmlaq_f32(d31, s0, w10);
                        s0 = BroadcastBf16(src3[offs + 1]);
                        d30 = vmlaq_f32(d30, s0, w01);
                        d31 = vmlaq_f32(d31, s0, w11);
                    }
                    if (M > 4)
                    {
                        s0 = BroadcastBf16(src4[offs + 0]);
                        d40 = vmlaq_f32(d40, s0, w00);
                        d41 = vmlaq_f32(d41, s0, w10);
                        s0 = BroadcastBf16(src4[offs + 1]);
                        d40 = vmlaq_f32(d40, s0, w01);
                        d41 = vmlaq_f32(d41, s0, w11);
                    }
                    weight0 += DF;
                    weight1 += DF;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2(dst, d00, d01), dst += dD;
                    if (M > 1) Save2(dst, d10, d11), dst += dD;
                    if (M > 2) Save2(dst, d20, d21), dst += dD;
                    if (M > 3) Save2(dst, d30, d31), dst += dD;
                    if (M > 4) Save2(dst, d40, d41), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2(dst, d00, d01, dstC), dst += dD;
                    if (M > 1) Save2(dst, d10, d11, dstC), dst += dD;
                    if (M > 2) Save2(dst, d20, d21, dstC), dst += dD;
                    if (M > 3) Save2(dst, d30, d31, dstC), dst += dD;
                    if (M > 4) Save2(dst, d40, d41, dstC), dst += dD;
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
                    if (M > 0) d00 = Load<false>(dst + 0 * dD + 0);
                    if (M > 1) d10 = Load<false>(dst + 1 * dD + 0);
                    if (M > 2) d20 = Load<false>(dst + 2 * dD + 0);
                    if (M > 3) d30 = Load<false>(dst + 3 * dD + 0);
                    if (M > 4) d40 = Load<false>(dst + 4 * dD + 0);
                }
                for (size_t offs = 0; offs < srcC; offs += 2)
                {
                    uint32x4_t w0u = Load<false>((uint32_t*)weight0);
                    w00 = vreinterpretq_f32_u32(vshlq_n_u32(w0u, Base::Bf16::SHIFT));
                    w01 = vreinterpretq_f32_u32(vandq_u32(w0u, Bf16::MASK));
                    if (M > 0)
                    {
                        s0 = BroadcastBf16(src0[offs + 0]);
                        d00 = vmlaq_f32(d00, s0, w00);
                        s0 = BroadcastBf16(src0[offs + 1]);
                        d00 = vmlaq_f32(d00, s0, w01);
                    }
                    if (M > 1)
                    {
                        s0 = BroadcastBf16(src1[offs + 0]);
                        d10 = vmlaq_f32(d10, s0, w00);
                        s0 = BroadcastBf16(src1[offs + 1]);
                        d10 = vmlaq_f32(d10, s0, w01);
                    }
                    if (M > 2)
                    {
                        s0 = BroadcastBf16(src2[offs + 0]);
                        d20 = vmlaq_f32(d20, s0, w00);
                        s0 = BroadcastBf16(src2[offs + 1]);
                        d20 = vmlaq_f32(d20, s0, w01);
                    }
                    if (M > 3)
                    {
                        s0 = BroadcastBf16(src3[offs + 0]);
                        d30 = vmlaq_f32(d30, s0, w00);
                        s0 = BroadcastBf16(src3[offs + 1]);
                        d30 = vmlaq_f32(d30, s0, w01);
                    }
                    if (M > 4)
                    {
                        s0 = BroadcastBf16(src4[offs + 0]);
                        d40 = vmlaq_f32(d40, s0, w00);
                        s0 = BroadcastBf16(src4[offs + 1]);
                        d40 = vmlaq_f32(d40, s0, w01);
                    }
                    weight0 += DF;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1(dst, d00), dst += dD;
                    if (M > 1) Save1(dst, d10), dst += dD;
                    if (M > 2) Save1(dst, d20), dst += dD;
                    if (M > 3) Save1(dst, d30), dst += dD;
                    if (M > 4) Save1(dst, d40), dst += dD;
                }
                else
                {
                    if (M > 0) Save1(dst, d00, dstC), dst += dD;
                    if (M > 1) Save1(dst, d10, dstC), dst += dD;
                    if (M > 2) Save1(dst, d20, dstC), dst += dD;
                    if (M > 3) Save1(dst, d30, dstC), dst += dD;
                    if (M > 4) Save1(dst, d40, dstC), dst += dD;
                }
            }
        }

        typedef void(*Deconvolution16bNhwcGemm_2xM_Ptr)(const uint16_t* src0, const DeconvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight, float* dst);

        Deconvolution16bNhwcGemm_2xM_Ptr GetDeconvolution16bNhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Deconvolution16bNhwcGemm_2xM<1>;
            case 2: return Deconvolution16bNhwcGemm_2xM<2>;
            case 3: return Deconvolution16bNhwcGemm_2xM<3>;
            case 4: return Deconvolution16bNhwcGemm_2xM<4>;
            case 5: return Deconvolution16bNhwcGemm_2xM<5>;
            }
            assert(0);
            return NULL;
        }

        void Deconvolution16bNhwcGemm_2(const uint16_t* src, const DeconvParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int zero, const uint16_t* wgt, float* dst)
        {
            size_t m1 = M, m = 5, mm = AlignLoAny(m1, m), t = m1 - mm;
            size_t dS = a.bufK, dW = a.bufK * DF, dD = a.bufN;
            Deconvolution16bNhwcGemm_2xM_Ptr deconvolution_2xM = GetDeconvolution16bNhwcGemm_2xM(m);
            Deconvolution16bNhwcGemm_2xM_Ptr deconvolution_2xT = GetDeconvolution16bNhwcGemm_2xM(t);

            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                for (; i < mm; i += m)
                    deconvolution_2xM(src + i * dS, p, a, K, dN, zero, wgt, dst + i * dD);
                for (; i < m1; i += t)
                    deconvolution_2xT(src + i * dS, p, a, K, dN, zero, wgt, dst + i * dD);
                wgt += dW;
                dst += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void RowToImgCommon(const float* src, const DeconvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, float* dst)
        {
            size_t dstCF = AlignLo(p.dstC, F);
            size_t rowSize = p.dstW * p.dstC, gap = a.bufN - a.N;
            size_t dyBeg = yBeg ? yBeg * p.strideY + a.preH : 0;
            size_t dyEnd = Simd::Min(yEnd * p.strideY + a.preH, p.dstH);
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
                memset(dst + dy * rowSize, 0, rowSize * sizeof(float));
            for (size_t sy = yBeg; sy < yEnd; ++sy)
            {
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t dy = sy * p.strideY - p.padY;
                    for (size_t ky = 0; ky < p.kernelY; ky++, dy += p.dilationY)
                    {
                        if (dy < p.dstH)
                        {
                            size_t dx = sx * p.strideX - p.padX;
                            for (size_t kx = 0; kx < p.kernelX; kx++, dx += p.dilationX)
                            {
                                if (dx < p.dstW)
                                {
                                    float* d = dst + (dy * p.dstW + dx) * p.dstC;
                                    size_t dc = 0;
                                    for (; dc < dstCF; dc += F)
                                        Store<false>(d + dc, vaddq_f32(Load<false>(d + dc), Load<false>(src + dc)));
                                    for (; dc < p.dstC; ++dc)
                                        d[dc] += src[dc];
                                }
                                src += p.dstC;
                            }
                        }
                        else
                            src += p.kernelX * p.dstC;
                    }
                    src += gap;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst)
        {
            float32x4_t f32 = Activate<type>(vaddq_f32(Load<false>(src + offset), Load<false>(bias + offset)), params, offset);
            if (term == Term16bLast16b)
                Store<false>((uint16_t*)(dst + offset * 2), vmovn_u32(Float32ToBFloat16(f32)));
            else
                Store<false>((float*)(dst + offset * 4), f32);
        }

        template <Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
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

        template <Term16bType term, SimdConvolutionActivationType type> void BiasActivationCommon(const float* src, const DeconvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, const float* bias, const float* params, uint8_t* dst)
        {
            size_t body = AlignLo(p.dstC, F), tail = p.dstC - body;
            src += yBeg * p.dstW * p.dstC;
            dst += yBeg * p.dstW * p.dstC * a.elem;
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < body; dc += F)
                        Postprocess<term, type>(src, bias, params, dc, dst);
                    if (tail)
                        Postprocess<term, type>(src, bias, params, dc, dst, tail);
                    src += p.dstC;
                    dst += p.dstC * a.elem;
                }
            }
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetBiasAct(const DeconvParam& p, const AlgParam& a, Base::SynetDeconvolution16bNhwcGemm::BiasActPtr& biasAct)
        {
            if (p.dstT == SimdTensorData16b)
                biasAct = BiasActivationCommon<Term16bLast16b, type>;
            else
                biasAct = BiasActivationCommon<Term16bLast32f, type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetDeconvolution16bNhwcGemm::SynetDeconvolution16bNhwcGemm(const DeconvParam& p)
            : Base::SynetDeconvolution16bNhwcGemm(p)
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
            _gemm = Deconvolution16bNhwcGemm_2;
            _toImg = RowToImgCommon;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetBiasAct<SimdConvolutionActivationRestrictRange>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationRelu: SetBiasAct<SimdConvolutionActivationRestrictRange>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationLeakyRelu: SetBiasAct<SimdConvolutionActivationPrelu>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationRestrictRange: SetBiasAct<SimdConvolutionActivationRestrictRange>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationPrelu: SetBiasAct<SimdConvolutionActivationPrelu>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationElu: SetBiasAct<SimdConvolutionActivationElu>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationHswish: SetBiasAct<SimdConvolutionActivationHswish>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationMish: SetBiasAct<SimdConvolutionActivationMish>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationHardSigmoid: SetBiasAct<SimdConvolutionActivationHardSigmoid>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationSwish: SetBiasAct<SimdConvolutionActivationSwish>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationGelu: SetBiasAct<SimdConvolutionActivationGelu>(p, _alg, _biasAct); break;
            default: assert(0);
            }
        }

        bool SynetDeconvolution16bNhwcGemm::Preferable(const DeconvParam& p)
        {
            return p.trans && p.group == 1;
        }
    }
#endif
}
