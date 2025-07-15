/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Base::SynetDeconvolution16bNhwcGemm::AlgParam AlgParam;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcGemm(const uint8_t* src8, const DeconvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8 + yBeg * p.srcW * p.srcC;
            size_t size = p.srcC, gap = a.bufK - size;
            size_t size32 = AlignLo(size, 32);
            __mmask16 srcMask[2];
            __mmask32 dstMask[1];
            if (size32 < size)
            {
                srcMask[0] = TailMask16(size - size32 - F * 0);
                srcMask[1] = TailMask16(size - size32 - F * 1);
                dstMask[0] = TailMask32(size - size32);
            }
            for (size_t sy = yBeg; sy < yEnd; ++sy)
            {
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < size32; sc += 32)
                        Avx512bw::Float32ToBFloat16<false, false>(src + sc, dst + sc, srcMask, dstMask);
                    if (size32 < size)
                        Avx512bw::Float32ToBFloat16<false, true>(src + sc, dst + sc, srcMask, dstMask);
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

        template<int cfg> void Deconvolution16bNhwcGemm_32x32(const uint16_t* src0, const DeconvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, size_t srcC, int zero, const uint16_t* weight0, float* dst)
        {
            int dD = (int)a.bufN, dS = (int)a.bufK, strideD = dD * 4, strideW = 64, strideS = dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            if(cfg)
                SetTileConf2x2(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_stream_loadd(0, dst + 0, strideD);
                _tile_stream_loadd(1, dst + F, strideD);
                _tile_stream_loadd(2, dst + 16 * dD + 0, strideD);
                _tile_stream_loadd(3, dst + 16 * dD + F, strideD);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(6, weight0 + sc * 16, strideW);
            for (; sc < srcC32;)
            {
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_stream_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                sc += 32;               
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_stream_loadd(5, src1 + sc, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);

            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);
            _tile_stored(3, dst + 16 * dD + F, strideD);
            TileMoveToMemory(dst + 0, dD);
            TileMoveToMemory(dst + F, dD);
            TileMoveToMemory(dst + 16 * dD + 0, dD);
            TileMoveToMemory(dst + 16 * dD + F, dD);
        }

        template<int cfg> void Deconvolution16bNhwcGemm_32x16(const uint16_t* src0, const DeconvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, size_t srcC, int zero, const uint16_t* weight0, float* dst)
		{
			int dD = (int)a.bufN, dS = (int)a.bufK, strideD = dD * 4, strideW = 64, strideS = dS * 2;
			const uint16_t* src1 = src0 + 16 * dS;

            if (cfg)
                SetTileConf2x1(dstS, dstC);
			if (zero)
			{
				_tile_zero(0);
				_tile_zero(2);
			}
			else
			{
				_tile_stream_loadd(0, dst + 0, strideD);
				_tile_stream_loadd(2, dst + 16 * dD + 0, strideD);
			}

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            for (; sc < srcC32;)
            {
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_stream_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_loadd(6, weight0 + sc * 16, strideW);
            _tile_stream_loadd(5, src1 + sc, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(2, 5, 6);

			_tile_stored(0, dst + 0, strideD);
			_tile_stored(2, dst + 16 * dD + 0, strideD);
			TileMoveToMemory(dst + 0, dD);
			TileMoveToMemory(dst + 16 * dD + 0, dD);
		}

        template<int cfg> void Deconvolution16bNhwcGemm_16x32(const uint16_t* src0, const DeconvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, size_t srcC, int zero, const uint16_t* weight0, float* dst)
        {
            int dD = (int)a.bufN, dS = (int)a.bufK, strideD = dD * 4, strideW = 64, strideS = dS * 2;
            const uint16_t* weight1 = weight0 + a.bufK * F;

            if (cfg)
                SetTileConf1x2(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_stream_loadd(0, dst + 0, strideD);
                _tile_stream_loadd(1, dst + F, strideD);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_loadd(6, weight0 + sc * 16, strideW);
            for (; sc < srcC32;)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stream_loadd(4, src0 + sc, strideS);
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);

            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            TileMoveToMemory(dst + 0, dD);
            TileMoveToMemory(dst + F, dD);
        }

        template<int cfg> void Deconvolution16bNhwcGemm_16x16(const uint16_t* src0, const DeconvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, size_t srcC, int zero, const uint16_t* weight0, float* dst)
        {
            int dD = (int)a.bufN, dS = (int)a.bufK, strideD = dD * 4, strideW = 64, strideS = dS * 2;

            if (cfg)
                SetTileConf1x1(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, dst + 0, strideD);
            }

            for (size_t sc = 0; sc < srcC; sc += 32)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }

            _tile_stored(0, dst + 0, strideD);
            TileMoveToMemory(dst + 0, dD);
        }

        typedef void(*Deconvolution16bNhwcGemmPtr)(const uint16_t* src0, const DeconvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, float* dst);

        void Deconvolution16bNhwcGemm_2(const uint16_t* src, const DeconvParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int zero, const uint16_t* wgt, float* dst)
        {
            size_t m = 32, mm = AlignLoAny(M, m), t = M - mm;
            size_t dS = a.bufK, dW = a.bufK * DF, dD = a.bufN;

            if (mm)
            {
                Deconvolution16bNhwcGemmPtr body_2 = Deconvolution16bNhwcGemm_32x32<0>;
                Deconvolution16bNhwcGemmPtr tail_2 = t > 16 ? Deconvolution16bNhwcGemm_32x32<1> : Deconvolution16bNhwcGemm_16x32<1>;
                Deconvolution16bNhwcGemmPtr body_1 = Deconvolution16bNhwcGemm_32x16<1>;
                Deconvolution16bNhwcGemmPtr tail_1 = t > 16 ? Deconvolution16bNhwcGemm_32x16<1> : Deconvolution16bNhwcGemm_16x16<1>;
                SetTileConfFull();
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j), i = 0;
                    if (dN > F)
                    {
                        if(t)
                            SetTileConfFull();
                        for (; i < mm; i += m)
                            body_2(src + i * dS, p, a, m, dN, K, zero, wgt, dst + i * dD);
                        if (t)
                            tail_2(src + i * dS, p, a, t, dN, K, zero, wgt, dst + i * dD);
                    }
                    else
                    {
                        for (; i < mm; i += m)
                            body_1(src + i * dS, p, a, m, dN, K, zero, wgt, dst + i * dD);
                        if (t)
                            tail_1(src + i * dS, p, a, t, dN, K, zero, wgt, dst + i * dD);
                    }
                    wgt += dW;
                    dst += DF;
                }
            }
            else
            {
                Deconvolution16bNhwcGemmPtr tail_2 = t > 16 ? Deconvolution16bNhwcGemm_32x32<0> : Deconvolution16bNhwcGemm_16x32<0>;
                Deconvolution16bNhwcGemmPtr tail_1 = t > 16 ? Deconvolution16bNhwcGemm_32x16<0> : Deconvolution16bNhwcGemm_16x16<0>;
                if (t > 16)
                    SetTileConf2x2(t, 32);
                else
                    SetTileConf1x2(t, 32);
                for (size_t j = 0; j < N; j += DF)
                {
                    size_t dN = Simd::Min(DF, N - j), i = 0;
                    if (dN > F)
                        tail_2(src + i * dS, p, a, t, dN, K, zero, wgt, dst + i * dD);
                    else
                        tail_1(src + i * dS, p, a, t, dN, K, zero, wgt, dst + i * dD);
                    wgt += dW;
                    dst += DF;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void RowToImgCommon(const float* src, const DeconvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, float* dst)
        {
            size_t dstCF = AlignLo(p.dstC, F);
            __mmask16 tail = TailMask16(p.dstC - dstCF);
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
                                        _mm512_storeu_ps(d + dc, _mm512_add_ps(_mm512_loadu_ps(d + dc), _mm512_loadu_ps(src + dc)));
                                    if(tail)
                                        _mm512_mask_storeu_ps(d + dc, tail, _mm512_add_ps(_mm512_maskz_loadu_ps(tail, d + dc), _mm512_maskz_loadu_ps(tail, src + dc)));
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

        template <Term16bType term, SimdConvolutionActivationType type> void BiasActivationCommon(const float* src, const DeconvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, const float* bias, const float* params, uint8_t* dst)
        {
            size_t body = AlignLo(p.dstC, F);
            __mmask16 tail = TailMask16(p.dstC - body);
            src += yBeg * p.dstW * p.dstC;
            dst += yBeg * p.dstW * p.dstC * a.elem;
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < body; dc += F)
                        Postprocess<term, type>(src, bias, params, dc, dst);
                    if(tail)
                        Postprocess<term, type>(src, bias, params, dc, dst, tail);
                    src += p.dstC;
                    dst += p.dstC * a.elem;
                }
            }
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetBiasAct(const DeconvParam& p, const AlgParam & a, Base::SynetDeconvolution16bNhwcGemm::BiasActPtr& biasAct)
        {
            if(p.dstT == SimdTensorData16b)
                biasAct = BiasActivationCommon<Term16bLast16b, type>;
            else
                biasAct = BiasActivationCommon<Term16bLast32f, type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetDeconvolution16bNhwcGemm::SynetDeconvolution16bNhwcGemm(const DeconvParam & p)
            : Avx512bw::SynetDeconvolution16bNhwcGemm(p)
        {
            SetAlgParam(F, F * 2, 32, 32, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
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
    }
#endif
}
