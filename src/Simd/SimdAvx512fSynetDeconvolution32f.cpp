/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdSynetDeconvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512f.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512F_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512f
    {
        SynetDeconvolution32fGemmNN::SynetDeconvolution32fGemmNN(const DeconvParam32f & p)
            : Avx2::SynetDeconvolution32fGemmNN(p)
        {
            _gemm.Init(InitGemmFuncs(Avx512f::Gemm32fNN, "Avx512f", p.gemm, "Ext"));
            if (_param.trans && _param.group == 1)
            {
                if (NHWC_GEMM_RUNTIME)
                {
                    _gemmCb.Init(InitGemmCbFuncs(Avx512f::Gemm32fNNcbBufferSize, Avx512f::Gemm32fNNcbReorderB, Avx512f::Gemm32fNNcbRun, "Avx512f", GemmKernelF2, GemmKernelF3));
                    _nhwcWeight.Resize(_gemmCb.At(0).BufferSize(_M*_merge, _N, _K));
                }
                else
                    _nhwcWeight.Resize(Avx512f::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
                _nhwcRun = Avx512f::Gemm32fNNcbRun;
                _nhwcReorderB = Avx512f::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Avx512f::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        typedef void(*DeconvolutionNhwcDirect2x2_Ptr) (const float * src0, const DeconvParam32f & p, size_t srcC, size_t dstC, 
            const float * weight, const __m512 * bias, const __m512 * params, float * ds, int first);

#if SIMD_ZMM_COUNT == 32
        template<TermType term, SimdConvolutionActivationType type, size_t tail> void DeconvolutionNhwcDirect2x2_M(const float * src0,
            const DeconvParam32f & p, size_t srcC, size_t dstC, const float * weight0, const __m512 * bias, const __m512 * params, float * dst, int first)
        {
            size_t dS = p.srcC, dD = p.dstC;
            const float * weight1 = weight0 + srcC * F, *src1, *src2, *src3, *src4, *src5, *src6;
            if (tail > 1) src1 = src0 + 1 * dS;
            if (tail > 2) src2 = src0 + 2 * dS;
            if (tail > 3) src3 = src0 + 3 * dS;
            if (tail > 4) src4 = src0 + 4 * dS;
            if (tail > 5) src5 = src0 + 5 * dS;
            if (tail > 6) src6 = src0 + 6 * dS;
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, dC0, dC1, dD0, dD1, s0, w0, w1;
            if (first)
            {
                if (tail > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                if (tail > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                if (tail > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                if (tail > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                if (tail > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                if (tail > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                if (tail > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                if (tail > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                if (tail > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                if (tail > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                if (tail > 0xA) dA0 = _mm512_setzero_ps(), dA1 = _mm512_setzero_ps();
                if (tail > 0xB) dB0 = _mm512_setzero_ps(), dB1 = _mm512_setzero_ps();
                if (tail > 0xC) dC0 = _mm512_setzero_ps(), dC1 = _mm512_setzero_ps();
                if (tail > 0xD) dD0 = _mm512_setzero_ps(), dD1 = _mm512_setzero_ps();
            }
            else
            {
                if (tail > 0x0) d00 = _mm512_loadu_ps(dst + 0x00 * dD), d01 = _mm512_loadu_ps(dst + 0x01 * dD);
                if (tail > 0x1) d10 = _mm512_loadu_ps(dst + 0x02 * dD), d11 = _mm512_loadu_ps(dst + 0x03 * dD);
                if (tail > 0x2) d20 = _mm512_loadu_ps(dst + 0x04 * dD), d21 = _mm512_loadu_ps(dst + 0x05 * dD);
                if (tail > 0x3) d30 = _mm512_loadu_ps(dst + 0x06 * dD), d31 = _mm512_loadu_ps(dst + 0x07 * dD);
                if (tail > 0x4) d40 = _mm512_loadu_ps(dst + 0x08 * dD), d41 = _mm512_loadu_ps(dst + 0x09 * dD);
                if (tail > 0x5) d50 = _mm512_loadu_ps(dst + 0x0a * dD), d51 = _mm512_loadu_ps(dst + 0x0b * dD);
                if (tail > 0x6) d60 = _mm512_loadu_ps(dst + 0x0c * dD), d61 = _mm512_loadu_ps(dst + 0x0d * dD);
                if (tail > 0x7) d70 = _mm512_loadu_ps(dst + 0x0e * dD), d71 = _mm512_loadu_ps(dst + 0x0f * dD);
                if (tail > 0x8) d80 = _mm512_loadu_ps(dst + 0x10 * dD), d81 = _mm512_loadu_ps(dst + 0x11 * dD);
                if (tail > 0x9) d90 = _mm512_loadu_ps(dst + 0x12 * dD), d91 = _mm512_loadu_ps(dst + 0x13 * dD);
                if (tail > 0xa) dA0 = _mm512_loadu_ps(dst + 0x14 * dD), dA1 = _mm512_loadu_ps(dst + 0x15 * dD);
                if (tail > 0xb) dB0 = _mm512_loadu_ps(dst + 0x16 * dD), dB1 = _mm512_loadu_ps(dst + 0x17 * dD);
                if (tail > 0xc) dC0 = _mm512_loadu_ps(dst + 0x18 * dD), dC1 = _mm512_loadu_ps(dst + 0x19 * dD);
                if (tail > 0xd) dD0 = _mm512_loadu_ps(dst + 0x1a * dD), dD1 = _mm512_loadu_ps(dst + 0x1b * dD);
            }
            for (size_t sc0 = 0, sc7 = 7 * dS; sc0 < srcC; ++sc0, ++sc7)
            {
                w0 = _mm512_loadu_ps(weight0);
                w1 = _mm512_loadu_ps(weight1);
                if (tail > 0x0) s0 = _mm512_set1_ps(src0[sc0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                if (tail > 0x1) s0 = _mm512_set1_ps(src1[sc0]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                if (tail > 0x2) s0 = _mm512_set1_ps(src2[sc0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                if (tail > 0x3) s0 = _mm512_set1_ps(src3[sc0]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                if (tail > 0x4) s0 = _mm512_set1_ps(src4[sc0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                if (tail > 0x5) s0 = _mm512_set1_ps(src5[sc0]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                if (tail > 0x6) s0 = _mm512_set1_ps(src6[sc0]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
                if (tail > 0x7) s0 = _mm512_set1_ps(src0[sc7]), d70 = _mm512_fmadd_ps(s0, w0, d70), d71 = _mm512_fmadd_ps(s0, w1, d71);
                if (tail > 0x8) s0 = _mm512_set1_ps(src1[sc7]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
                if (tail > 0x9) s0 = _mm512_set1_ps(src2[sc7]), d90 = _mm512_fmadd_ps(s0, w0, d90), d91 = _mm512_fmadd_ps(s0, w1, d91);
                if (tail > 0xA) s0 = _mm512_set1_ps(src3[sc7]), dA0 = _mm512_fmadd_ps(s0, w0, dA0), dA1 = _mm512_fmadd_ps(s0, w1, dA1);
                if (tail > 0xB) s0 = _mm512_set1_ps(src4[sc7]), dB0 = _mm512_fmadd_ps(s0, w0, dB0), dB1 = _mm512_fmadd_ps(s0, w1, dB1);
                if (tail > 0xC) s0 = _mm512_set1_ps(src5[sc7]), dC0 = _mm512_fmadd_ps(s0, w0, dC0), dC1 = _mm512_fmadd_ps(s0, w1, dC1);
                if (tail > 0xD) s0 = _mm512_set1_ps(src6[sc7]), dD0 = _mm512_fmadd_ps(s0, w0, dD0), dD1 = _mm512_fmadd_ps(s0, w1, dD1);
                weight0 += F;
                weight1 += F;
            }
            if (dstC == F)
            {
                if (tail > 0x0) Term<term>::template Save<type, 0>(dst + 0x00 * dD, d00, bias, params), Term<term>::template Save<type, 0>(dst + 0x01 * dD, d01, bias, params);
                if (tail > 0x1) Term<term>::template Save<type, 0>(dst + 0x02 * dD, d10, bias, params), Term<term>::template Save<type, 0>(dst + 0x03 * dD, d11, bias, params);
                if (tail > 0x2) Term<term>::template Save<type, 0>(dst + 0x04 * dD, d20, bias, params), Term<term>::template Save<type, 0>(dst + 0x05 * dD, d21, bias, params);
                if (tail > 0x3) Term<term>::template Save<type, 0>(dst + 0x06 * dD, d30, bias, params), Term<term>::template Save<type, 0>(dst + 0x07 * dD, d31, bias, params);
                if (tail > 0x4) Term<term>::template Save<type, 0>(dst + 0x08 * dD, d40, bias, params), Term<term>::template Save<type, 0>(dst + 0x09 * dD, d41, bias, params);
                if (tail > 0x5) Term<term>::template Save<type, 0>(dst + 0x0A * dD, d50, bias, params), Term<term>::template Save<type, 0>(dst + 0x0B * dD, d51, bias, params);
                if (tail > 0x6) Term<term>::template Save<type, 0>(dst + 0x0C * dD, d60, bias, params), Term<term>::template Save<type, 0>(dst + 0x0D * dD, d61, bias, params);
                if (tail > 0x7) Term<term>::template Save<type, 0>(dst + 0x0E * dD, d70, bias, params), Term<term>::template Save<type, 0>(dst + 0x0F * dD, d71, bias, params);
                if (tail > 0x8) Term<term>::template Save<type, 0>(dst + 0x10 * dD, d80, bias, params), Term<term>::template Save<type, 0>(dst + 0x11 * dD, d81, bias, params);
                if (tail > 0x9) Term<term>::template Save<type, 0>(dst + 0x12 * dD, d90, bias, params), Term<term>::template Save<type, 0>(dst + 0x13 * dD, d91, bias, params);
                if (tail > 0xA) Term<term>::template Save<type, 0>(dst + 0x14 * dD, dA0, bias, params), Term<term>::template Save<type, 0>(dst + 0x15 * dD, dA1, bias, params);
                if (tail > 0xB) Term<term>::template Save<type, 0>(dst + 0x16 * dD, dB0, bias, params), Term<term>::template Save<type, 0>(dst + 0x17 * dD, dB1, bias, params);
                if (tail > 0xC) Term<term>::template Save<type, 0>(dst + 0x18 * dD, dC0, bias, params), Term<term>::template Save<type, 0>(dst + 0x19 * dD, dC1, bias, params);
                if (tail > 0xD) Term<term>::template Save<type, 0>(dst + 0x1A * dD, dD0, bias, params), Term<term>::template Save<type, 0>(dst + 0x1B * dD, dD1, bias, params);
            }
            else
            {
                __mmask16 mask = __mmask16(-1) >> (16 - dstC);
                if (tail > 0x0) Term<term>::template Save<type, 0>(dst + 0x00 * dD, d00, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x01 * dD, d01, bias, params, mask);
                if (tail > 0x1) Term<term>::template Save<type, 0>(dst + 0x02 * dD, d10, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x03 * dD, d11, bias, params, mask);
                if (tail > 0x2) Term<term>::template Save<type, 0>(dst + 0x04 * dD, d20, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x05 * dD, d21, bias, params, mask);
                if (tail > 0x3) Term<term>::template Save<type, 0>(dst + 0x06 * dD, d30, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x07 * dD, d31, bias, params, mask);
                if (tail > 0x4) Term<term>::template Save<type, 0>(dst + 0x08 * dD, d40, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x09 * dD, d41, bias, params, mask);
                if (tail > 0x5) Term<term>::template Save<type, 0>(dst + 0x0A * dD, d50, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x0B * dD, d51, bias, params, mask);
                if (tail > 0x6) Term<term>::template Save<type, 0>(dst + 0x0C * dD, d60, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x0D * dD, d61, bias, params, mask);
                if (tail > 0x7) Term<term>::template Save<type, 0>(dst + 0x0E * dD, d70, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x0F * dD, d71, bias, params, mask);
                if (tail > 0x8) Term<term>::template Save<type, 0>(dst + 0x10 * dD, d80, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x11 * dD, d81, bias, params, mask);
                if (tail > 0x9) Term<term>::template Save<type, 0>(dst + 0x12 * dD, d90, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x13 * dD, d91, bias, params, mask);
                if (tail > 0xA) Term<term>::template Save<type, 0>(dst + 0x14 * dD, dA0, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x15 * dD, dA1, bias, params, mask);
                if (tail > 0xB) Term<term>::template Save<type, 0>(dst + 0x16 * dD, dB0, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x17 * dD, dB1, bias, params, mask);
                if (tail > 0xC) Term<term>::template Save<type, 0>(dst + 0x18 * dD, dC0, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x19 * dD, dC1, bias, params, mask);
                if (tail > 0xD) Term<term>::template Save<type, 0>(dst + 0x1A * dD, dD0, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x1B * dD, dD1, bias, params, mask);
            }
        }
#else
        template<TermType term, SimdConvolutionActivationType type, size_t tail> void DeconvolutionNhwcDirect2x2_M(const float * src0,
            const DeconvParam32f & p, size_t srcC, size_t dstC, const float * weight0, const __m512 * bias, const __m512 * params, float * dst, int first)
        {
            size_t dS = p.srcC, dD = p.dstC;
            const float * weight1 = weight0 + srcC * F, *src1, *src2, *src3, *src4, *src5;
            if (tail > 1) src1 = src0 + 1 * dS;
            if (tail > 2) src2 = src0 + 2 * dS;
            if (tail > 3) src3 = src0 + 3 * dS;
            if (tail > 4) src4 = src0 + 4 * dS;
            if (tail > 5) src5 = src0 + 5 * dS;
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
            if (first)
            {
                if (tail > 0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                if (tail > 1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                if (tail > 2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                if (tail > 3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                if (tail > 4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                if (tail > 5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
            }
            else
            {
                if (tail > 0) d00 = _mm512_loadu_ps(dst + 0x0 * dD), d01 = _mm512_loadu_ps(dst + 0x1 * dD);
                if (tail > 1) d10 = _mm512_loadu_ps(dst + 0x2 * dD), d11 = _mm512_loadu_ps(dst + 0x3 * dD);
                if (tail > 2) d20 = _mm512_loadu_ps(dst + 0x4 * dD), d21 = _mm512_loadu_ps(dst + 0x5 * dD);
                if (tail > 3) d30 = _mm512_loadu_ps(dst + 0x6 * dD), d31 = _mm512_loadu_ps(dst + 0x7 * dD);
                if (tail > 4) d40 = _mm512_loadu_ps(dst + 0x8 * dD), d41 = _mm512_loadu_ps(dst + 0x9 * dD);
                if (tail > 5) d50 = _mm512_loadu_ps(dst + 0xa * dD), d51 = _mm512_loadu_ps(dst + 0xb * dD);
            }
            for (size_t sc = 0; sc < srcC; ++sc)
            {
                w0 = _mm512_loadu_ps(weight0);
                w1 = _mm512_loadu_ps(weight1);
                if (tail > 0) s0 = _mm512_set1_ps(src0[sc]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
                if (tail > 1) s0 = _mm512_set1_ps(src1[sc]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
                if (tail > 2) s0 = _mm512_set1_ps(src2[sc]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
                if (tail > 3) s0 = _mm512_set1_ps(src3[sc]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
                if (tail > 4) s0 = _mm512_set1_ps(src4[sc]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
                if (tail > 5) s0 = _mm512_set1_ps(src5[sc]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
                weight0 += F;
                weight1 += F;
            }
            if (dstC == F)
            {
                if (tail > 0) Term<term>::template Save<type, 0>(dst + 0x0 * dD, d00, bias, params), Term<term>::template Save<type, 0>(dst + 0x1 * dD, d01, bias, params);
                if (tail > 1) Term<term>::template Save<type, 0>(dst + 0x2 * dD, d10, bias, params), Term<term>::template Save<type, 0>(dst + 0x3 * dD, d11, bias, params);
                if (tail > 2) Term<term>::template Save<type, 0>(dst + 0x4 * dD, d20, bias, params), Term<term>::template Save<type, 0>(dst + 0x5 * dD, d21, bias, params);
                if (tail > 3) Term<term>::template Save<type, 0>(dst + 0x6 * dD, d30, bias, params), Term<term>::template Save<type, 0>(dst + 0x7 * dD, d31, bias, params);
                if (tail > 4) Term<term>::template Save<type, 0>(dst + 0x8 * dD, d40, bias, params), Term<term>::template Save<type, 0>(dst + 0x9 * dD, d41, bias, params);
                if (tail > 5) Term<term>::template Save<type, 0>(dst + 0xA * dD, d50, bias, params), Term<term>::template Save<type, 0>(dst + 0xB * dD, d51, bias, params);
            }
            else
            {
                __mmask16 mask = __mmask16(-1) >> (16 - dstC);
                if (tail > 0) Term<term>::template Save<type, 0>(dst + 0x0 * dD, d00, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x1 * dD, d01, bias, params, mask);
                if (tail > 1) Term<term>::template Save<type, 0>(dst + 0x2 * dD, d10, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x3 * dD, d11, bias, params, mask);
                if (tail > 2) Term<term>::template Save<type, 0>(dst + 0x4 * dD, d20, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x5 * dD, d21, bias, params, mask);
                if (tail > 3) Term<term>::template Save<type, 0>(dst + 0x6 * dD, d30, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x7 * dD, d31, bias, params, mask);
                if (tail > 4) Term<term>::template Save<type, 0>(dst + 0x8 * dD, d40, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0x9 * dD, d41, bias, params, mask);
                if (tail > 5) Term<term>::template Save<type, 0>(dst + 0xA * dD, d50, bias, params, mask), Term<term>::template Save<type, 0>(dst + 0xB * dD, d51, bias, params, mask);
            }
        }
#endif

        template <TermType term, SimdConvolutionActivationType type> SIMD_INLINE DeconvolutionNhwcDirect2x2_Ptr GetDeconvolutionNhwcDirect2x2(size_t tail)
        {
            switch (tail)
            {
            case 0: return NULL;
            case 1: return DeconvolutionNhwcDirect2x2_M<term, type, 1>;
            case 2: return DeconvolutionNhwcDirect2x2_M<term, type, 2>;
            case 3: return DeconvolutionNhwcDirect2x2_M<term, type, 3>;
            case 4: return DeconvolutionNhwcDirect2x2_M<term, type, 4>;
            case 5: return DeconvolutionNhwcDirect2x2_M<term, type, 5>;
            case 6: return DeconvolutionNhwcDirect2x2_M<term, type, 6>;
#if SIMD_ZMM_COUNT == 32
            case 7: return DeconvolutionNhwcDirect2x2_M<term, type, 7>;
            case 8: return DeconvolutionNhwcDirect2x2_M<term, type, 8>;
            case 9: return DeconvolutionNhwcDirect2x2_M<term, type, 9>;
            case 10: return DeconvolutionNhwcDirect2x2_M<term, type, 10>;
            case 11: return DeconvolutionNhwcDirect2x2_M<term, type, 11>;
            case 12: return DeconvolutionNhwcDirect2x2_M<term, type, 12>;
            case 13: return DeconvolutionNhwcDirect2x2_M<term, type, 13>;
            case 14: return DeconvolutionNhwcDirect2x2_M<term, type, 14>;
#endif
            default:
                assert(0);
                return NULL;
            }
        }

        template<TermType term, SimdConvolutionActivationType type> void DeconvolutionNhwcDirect2x2(const float * src, const DeconvParam32f & p,
            size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const float * weight, const float * bias, const float * params, float * dst, int first)
        {
#if SIMD_ZMM_COUNT == 32
            size_t step = 14;
#else
            size_t step = 6;
#endif
            size_t body = AlignLoAny(p.srcW, step), tail = p.srcW - body;
            DeconvolutionNhwcDirect2x2_Ptr bodyKernel = GetDeconvolutionNhwcDirect2x2<term, type>(step);
            DeconvolutionNhwcDirect2x2_Ptr tailKernel = GetDeconvolutionNhwcDirect2x2<term, type>(tail);

            __m512 _params[2], _bias[1];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += F)
            {
                size_t dC = Simd::Min(F, dstC - dc);
                _bias[0] = _mm512_loadu_ps(bias + dc);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + dc);
                const float * s = src + yBeg * p.srcW * p.srcC;
                float * d = dst + yBeg * p.strideY * p.dstW * p.dstC;
                const float * w0 = weight + 0 * p.kernelX * srcC * F;
                const float * w1 = weight + 1 * p.kernelX * srcC * F;
                for (size_t sy = yBeg; sy < yEnd; sy += 1, s += p.srcW * p.srcC)
                {
                    for (size_t sx = 0; sx < body; sx += step)
                        bodyKernel(s + sx * p.srcC, p, srcC, dC, w0, _bias, _params, d, first), d += step * p.strideX * p.dstC;
                    if (tail)
                        tailKernel(s + body * p.srcC, p, srcC, dC, w0, _bias, _params, d, first), d += tail * p.strideX * p.dstC;
                    for (size_t sx = 0; sx < body; sx += step)
                        bodyKernel(s + sx * p.srcC, p, srcC, dC, w1, _bias, _params, d, first), d += step * p.strideX * p.dstC;
                    if (tail)
                        tailKernel(s + body * p.srcC, p, srcC, dC, w1, _bias, _params, d, first), d += tail * p.strideX * p.dstC;
                }
                weight += p.kernelY * p.kernelX*srcC*F;
                dst += F;
            }
        }

        template<SimdConvolutionActivationType type> void DeconvolutionNhwcDirect2x2(const float * src, const DeconvParam32f & p,
            const SynetDeconvolution32fNhwcDirect2x2::AlgParam & a, const float * weight, const float * bias, const float * params, float * dst)
        {
            for (size_t dc = 0; dc < p.dstC; dc += a.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + a.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += a.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + a.macroC) - sc;
                    size_t macroK = p.kernelY * p.kernelX * macroC;
                    for (size_t yBeg = 0; yBeg < p.srcH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + a.macroH, p.srcH);
                        if (a.macroC == p.srcC)
                            DeconvolutionNhwcDirect2x2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, 1);
                        else if (sc == 0)
                            DeconvolutionNhwcDirect2x2<TermInterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, 1);
                        else if (sc + macroC == p.srcC)
                            DeconvolutionNhwcDirect2x2<TermLast, type>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, 0);
                        else
                            DeconvolutionNhwcDirect2x2<TermInterim, SimdConvolutionActivationIdentity>(src + sc, p, macroD, yBeg, yEnd, macroC, weight, bias + dc, params, dst + dc, 0);
                        yBeg = yEnd;
                    }
                    weight += AlignHiAny(macroD, a.microD)*macroK;
                }
                if (type == ::SimdConvolutionActivationPrelu)
                    params += macroD;
            }
        }

        SynetDeconvolution32fNhwcDirect2x2::SynetDeconvolution32fNhwcDirect2x2(const DeconvParam32f & p)
            : Avx2::SynetDeconvolution32fNhwcDirect2x2(p)
        {
            if (p.dstC > HF)
            {
                switch (p.activation)
                {
                case SimdConvolutionActivationIdentity: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationRestrictRange>; break;
                case SimdConvolutionActivationRelu: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationRestrictRange>; break;
                case SimdConvolutionActivationLeakyRelu: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationPrelu>; break;
                case SimdConvolutionActivationRestrictRange: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationRestrictRange>; break;
                case SimdConvolutionActivationPrelu: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationPrelu>; break;
                case SimdConvolutionActivationElu: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationElu>; break;
                case SimdConvolutionActivationHswish: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationHswish>; break;
                case SimdConvolutionActivationMish: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationMish>; break;
                case SimdConvolutionActivationHardSigmoid: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationHardSigmoid>; break;
                case SimdConvolutionActivationSwish: _deconvolution = DeconvolutionNhwcDirect2x2<SimdConvolutionActivationSwish>; break;
                default: assert(0);
                }
                SetAlgParam(F, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            }
        }

        //---------------------------------------------------------------------

        void * SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            DeconvParam32f param(batch, conv, gemm);
            if (!param.Valid())
                return NULL;
            if (SynetDeconvolution32fNhwcDirect2x2::Preferable(param))
                return new SynetDeconvolution32fNhwcDirect2x2(param);
            else
                return new SynetDeconvolution32fGemmNN(param);
        }
    }
#endif//SIMD_AVX512F_ENABLE
}
