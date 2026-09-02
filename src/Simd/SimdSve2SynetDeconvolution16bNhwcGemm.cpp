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
        typedef Base::SynetDeconvolution16bNhwcGemm::AlgParam AlgParam;

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

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcGemm(const uint8_t* src8, const DeconvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8 + yBeg * p.srcW * p.srcC;
            size_t size = p.srcC, gap = a.bufK - size;
            for (size_t sy = yBeg; sy < yEnd; ++sy)
            {
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    Float32ToBFloat16(src, size, dst);
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

        template<int M> void Deconvolution16bNhwcGemm_2xM(const uint16_t* src0, const DeconvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, float* dst)
        {
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            svbfloat16_t s0, w0, w1;
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
                for (size_t offs = 0; offs < srcC; offs += 2)
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
                svbool_t mask1 = (dstC == DF) ? body : svwhilelt_b32((size_t)0, dstC - F);
                if (M > 0) svst1_f32(body, dst + 0, d00), svst1_f32(mask1, dst + F, d01), dst += dD;
                if (M > 1) svst1_f32(body, dst + 0, d10), svst1_f32(mask1, dst + F, d11), dst += dD;
                if (M > 2) svst1_f32(body, dst + 0, d20), svst1_f32(mask1, dst + F, d21), dst += dD;
                if (M > 3) svst1_f32(body, dst + 0, d30), svst1_f32(mask1, dst + F, d31), dst += dD;
                if (M > 4) svst1_f32(body, dst + 0, d40), svst1_f32(mask1, dst + F, d41), dst += dD;
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
                for (size_t offs = 0; offs < srcC; offs += 2)
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
                svbool_t mask0 = (dstC == F) ? body : svwhilelt_b32((size_t)0, dstC);
                if (M > 0) svst1_f32(mask0, dst, d00), dst += dD;
                if (M > 1) svst1_f32(mask0, dst, d10), dst += dD;
                if (M > 2) svst1_f32(mask0, dst, d20), dst += dD;
                if (M > 3) svst1_f32(mask0, dst, d30), dst += dD;
                if (M > 4) svst1_f32(mask0, dst, d40), dst += dD;
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
            size_t F = a.F, DF = F * 2;
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
                                    for (size_t dc = 0; dc < p.dstC; dc += svcntw())
                                    {
                                        svbool_t mask = svwhilelt_b32(dc, p.dstC);
                                        svst1_f32(mask, d + dc, svadd_f32_x(mask, svld1_f32(mask, d + dc), svld1_f32(mask, src + dc)));
                                    }
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

        template <Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            size_t index = offset / F;
            svfloat32_t param0 = LoadActParam0<type>(params, index, mask);
            svfloat32_t param1 = LoadActParam1<type>(params, index, mask);
            svfloat32_t f32 = Activate<type>(svadd_f32_x(mask, svld1_f32(mask, src + offset), svld1_f32(mask, bias + offset)), param0, param1, 0, mask);
            if (term == Term16bLast16b)
                svst1h_u32(mask, (uint16_t*)dst + offset, Float32ToBFloat16(f32, mask));
            else
                svst1_f32(mask, (float*)dst + offset, f32);
        }

        template <Term16bType term, SimdConvolutionActivationType type> void BiasActivationCommon(const float* src, const DeconvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, const float* bias, const float* params, uint8_t* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            src += yBeg * p.dstW * p.dstC;
            dst += yBeg * p.dstW * p.dstC * a.elem;
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc + F <= p.dstC; dc += F)
                        Postprocess<term, type>(src, bias, params, dc, dst, body);
                    if (dc < p.dstC)
                        Postprocess<term, type>(src, bias, params, dc, dst, svwhilelt_b32(dc, p.dstC));
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
            const size_t F = svcntw();
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
