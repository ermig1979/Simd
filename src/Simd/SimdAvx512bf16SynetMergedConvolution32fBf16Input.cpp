/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAvx512bf16.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_AVX512BF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bf16
    {
        using AlgParam = Base::SynetMergedConvolution32fBf16::AlgParam;
        using InputPtr = Base::SynetMergedConvolution32fBf16::InputConvolutionPtr;

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> void InputConvolution_2x1(const uint16_t* src0,
            const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC, const uint16_t* weight,
            const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            __m512 d00, d01;
            __m512bh s0, w0, w1;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(p.srcC, 2) * QF, sM = a.bufH[0] - 1;
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 2)
                            {
                                w0 = (__m512bh)_mm512_loadu_si512(weight + 0 * DF);
                                w1 = (__m512bh)_mm512_loadu_si512(weight + 1 * DF);
                                s0 = Set2(src0 + offs), d00 = _mm512_dpbf16_ps(d00, s0, w0), d01 = _mm512_dpbf16_ps(d01, s0, w1);
                                weight += QF;
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                SaveInput2<type>(dst0, dst1, d00, d01, bias, params);
            }
            else
            {
                d00 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 2)
                            {
                                w0 = (__m512bh)_mm512_loadu_si512(weight + 0 * DF);
                                s0 = Set2(src0 + offs), d00 = _mm512_dpbf16_ps(d00, s0, w0);
                                weight += QF;
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                SaveInput1<type>(dst0, d00, bias, params);
            }
        }

        template<SimdConvolutionActivationType type, int M> void InputConvolution_2xM(const uint16_t* src0, 
            const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC,
            const uint16_t* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61,
                d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, dc0, dc1, dd0, dd1;
            __m512bh s0, w0, w1;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC, dWz = DivHi(p.srcC, 2) * QF * p.kernelX, sM = a.bufH[0] - 1;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            const uint16_t* src5 = src0 + 5 * dS;
            const uint16_t* src6 = src0 + 6 * dS;
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                if (M > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                if (M > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                if (M > 0xa) da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
                if (M > 0xb) db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
                if (M > 0xc) dc0 = _mm512_setzero_ps(), dc1 = _mm512_setzero_ps();
                if (M > 0xd) dd0 = _mm512_setzero_ps(), dd1 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                            size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs7 = offs0 + 7 * dS;
                            for (; offs0 < end; offs0 += 2, offs7 += 2)
                            {
                                w0 = (__m512bh)_mm512_loadu_si512(weight + 0 * DF);
                                w1 = (__m512bh)_mm512_loadu_si512(weight + 1 * DF);
                                if (M > 0x0) s0 = Set2(src0 + offs0), d00 = _mm512_dpbf16_ps(d00, s0, w0), d01 = _mm512_dpbf16_ps(d01, s0, w1);
                                if (M > 0x1) s0 = Set2(src1 + offs0), d10 = _mm512_dpbf16_ps(d10, s0, w0), d11 = _mm512_dpbf16_ps(d11, s0, w1);
                                if (M > 0x2) s0 = Set2(src2 + offs0), d20 = _mm512_dpbf16_ps(d20, s0, w0), d21 = _mm512_dpbf16_ps(d21, s0, w1);
                                if (M > 0x3) s0 = Set2(src3 + offs0), d30 = _mm512_dpbf16_ps(d30, s0, w0), d31 = _mm512_dpbf16_ps(d31, s0, w1);
                                if (M > 0x4) s0 = Set2(src4 + offs0), d40 = _mm512_dpbf16_ps(d40, s0, w0), d41 = _mm512_dpbf16_ps(d41, s0, w1);
                                if (M > 0x5) s0 = Set2(src5 + offs0), d50 = _mm512_dpbf16_ps(d50, s0, w0), d51 = _mm512_dpbf16_ps(d51, s0, w1);
                                if (M > 0x6) s0 = Set2(src6 + offs0), d60 = _mm512_dpbf16_ps(d60, s0, w0), d61 = _mm512_dpbf16_ps(d61, s0, w1);
                                if (M > 0x7) s0 = Set2(src0 + offs7), d70 = _mm512_dpbf16_ps(d70, s0, w0), d71 = _mm512_dpbf16_ps(d71, s0, w1);
                                if (M > 0x8) s0 = Set2(src1 + offs7), d80 = _mm512_dpbf16_ps(d80, s0, w0), d81 = _mm512_dpbf16_ps(d81, s0, w1);
                                if (M > 0x9) s0 = Set2(src2 + offs7), d90 = _mm512_dpbf16_ps(d90, s0, w0), d91 = _mm512_dpbf16_ps(d91, s0, w1);
                                if (M > 0xa) s0 = Set2(src3 + offs7), da0 = _mm512_dpbf16_ps(da0, s0, w0), da1 = _mm512_dpbf16_ps(da1, s0, w1);
                                if (M > 0xb) s0 = Set2(src4 + offs7), db0 = _mm512_dpbf16_ps(db0, s0, w0), db1 = _mm512_dpbf16_ps(db1, s0, w1);
                                if (M > 0xc) s0 = Set2(src5 + offs7), dc0 = _mm512_dpbf16_ps(dc0, s0, w0), dc1 = _mm512_dpbf16_ps(dc1, s0, w1);
                                if (M > 0xd) s0 = Set2(src6 + offs7), dd0 = _mm512_dpbf16_ps(dd0, s0, w0), dd1 = _mm512_dpbf16_ps(dd1, s0, w1);
                                weight += QF;
                            }
                        }
                    }
                    else
                        weight += dWz;
                }
                if (M > 0x0) SaveInput2<type>(dst0 + 0x0 * F, dst1 + 0x0 * F, d00, d01, bias, params);
                if (M > 0x1) SaveInput2<type>(dst0 + 0x1 * F, dst1 + 0x1 * F, d10, d11, bias, params);
                if (M > 0x2) SaveInput2<type>(dst0 + 0x2 * F, dst1 + 0x2 * F, d20, d21, bias, params);
                if (M > 0x3) SaveInput2<type>(dst0 + 0x3 * F, dst1 + 0x3 * F, d30, d31, bias, params);
                if (M > 0x4) SaveInput2<type>(dst0 + 0x4 * F, dst1 + 0x4 * F, d40, d41, bias, params);
                if (M > 0x5) SaveInput2<type>(dst0 + 0x5 * F, dst1 + 0x5 * F, d50, d51, bias, params);
                if (M > 0x6) SaveInput2<type>(dst0 + 0x6 * F, dst1 + 0x6 * F, d60, d61, bias, params);
                if (M > 0x7) SaveInput2<type>(dst0 + 0x7 * F, dst1 + 0x7 * F, d70, d71, bias, params);
                if (M > 0x8) SaveInput2<type>(dst0 + 0x8 * F, dst1 + 0x8 * F, d80, d81, bias, params);
                if (M > 0x9) SaveInput2<type>(dst0 + 0x9 * F, dst1 + 0x9 * F, d90, d91, bias, params);
                if (M > 0xa) SaveInput2<type>(dst0 + 0xa * F, dst1 + 0xa * F, da0, da1, bias, params);
                if (M > 0xb) SaveInput2<type>(dst0 + 0xb * F, dst1 + 0xb * F, db0, db1, bias, params);
                if (M > 0xc) SaveInput2<type>(dst0 + 0xc * F, dst1 + 0xc * F, dc0, dc1, bias, params);
                if (M > 0xd) SaveInput2<type>(dst0 + 0xd * F, dst1 + 0xd * F, dd0, dd1, bias, params);
            }
            else
            {
                if (M > 0x0) d00 = _mm512_setzero_ps();
                if (M > 0x1) d10 = _mm512_setzero_ps();
                if (M > 0x2) d20 = _mm512_setzero_ps();
                if (M > 0x3) d30 = _mm512_setzero_ps();
                if (M > 0x4) d40 = _mm512_setzero_ps();
                if (M > 0x5) d50 = _mm512_setzero_ps();
                if (M > 0x6) d60 = _mm512_setzero_ps();
                if (M > 0x7) d70 = _mm512_setzero_ps();
                if (M > 0x8) d80 = _mm512_setzero_ps();
                if (M > 0x9) d90 = _mm512_setzero_ps();
                if (M > 0xa) da0 = _mm512_setzero_ps();
                if (M > 0xb) db0 = _mm512_setzero_ps();
                if (M > 0xc) dc0 = _mm512_setzero_ps();
                if (M > 0xd) dd0 = _mm512_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                            size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs7 = offs0 + 7 * dS;
                            for (; offs0 < end; offs0 += 2, offs7 += 2)
                            {
                                w0 = (__m512bh)_mm512_loadu_si512(weight + 0 * DF);
                                if (M > 0x0) s0 = Set2(src0 + offs0), d00 = _mm512_dpbf16_ps(d00, s0, w0);
                                if (M > 0x1) s0 = Set2(src1 + offs0), d10 = _mm512_dpbf16_ps(d10, s0, w0);
                                if (M > 0x2) s0 = Set2(src2 + offs0), d20 = _mm512_dpbf16_ps(d20, s0, w0);
                                if (M > 0x3) s0 = Set2(src3 + offs0), d30 = _mm512_dpbf16_ps(d30, s0, w0);
                                if (M > 0x4) s0 = Set2(src4 + offs0), d40 = _mm512_dpbf16_ps(d40, s0, w0);
                                if (M > 0x5) s0 = Set2(src5 + offs0), d50 = _mm512_dpbf16_ps(d50, s0, w0);
                                if (M > 0x6) s0 = Set2(src6 + offs0), d60 = _mm512_dpbf16_ps(d60, s0, w0);
                                if (M > 0x7) s0 = Set2(src0 + offs7), d70 = _mm512_dpbf16_ps(d70, s0, w0);
                                if (M > 0x8) s0 = Set2(src1 + offs7), d80 = _mm512_dpbf16_ps(d80, s0, w0);
                                if (M > 0x9) s0 = Set2(src2 + offs7), d90 = _mm512_dpbf16_ps(d90, s0, w0);
                                if (M > 0xa) s0 = Set2(src3 + offs7), da0 = _mm512_dpbf16_ps(da0, s0, w0);
                                if (M > 0xb) s0 = Set2(src4 + offs7), db0 = _mm512_dpbf16_ps(db0, s0, w0);
                                if (M > 0xc) s0 = Set2(src5 + offs7), dc0 = _mm512_dpbf16_ps(dc0, s0, w0);
                                if (M > 0xd) s0 = Set2(src6 + offs7), dd0 = _mm512_dpbf16_ps(dd0, s0, w0);
                                weight += QF;
                            }
                        }
                    }
                    else
                        weight += dWz;
                }
                if (M > 0x0) SaveInput1<type>(dst0 + 0x0 * F, d00, bias, params);
                if (M > 0x1) SaveInput1<type>(dst0 + 0x1 * F, d10, bias, params);
                if (M > 0x2) SaveInput1<type>(dst0 + 0x2 * F, d20, bias, params);
                if (M > 0x3) SaveInput1<type>(dst0 + 0x3 * F, d30, bias, params);
                if (M > 0x4) SaveInput1<type>(dst0 + 0x4 * F, d40, bias, params);
                if (M > 0x5) SaveInput1<type>(dst0 + 0x5 * F, d50, bias, params);
                if (M > 0x6) SaveInput1<type>(dst0 + 0x6 * F, d60, bias, params);
                if (M > 0x7) SaveInput1<type>(dst0 + 0x7 * F, d70, bias, params);
                if (M > 0x8) SaveInput1<type>(dst0 + 0x8 * F, d80, bias, params);
                if (M > 0x9) SaveInput1<type>(dst0 + 0x9 * F, d90, bias, params);
                if (M > 0xa) SaveInput1<type>(dst0 + 0xa * F, da0, bias, params);
                if (M > 0xb) SaveInput1<type>(dst0 + 0xb * F, db0, bias, params);
                if (M > 0xc) SaveInput1<type>(dst0 + 0xc * F, dc0, bias, params);
                if (M > 0xd) SaveInput1<type>(dst0 + 0xd * F, dd0, bias, params);
            }
        }

        typedef void(*InputConvolution_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx,
            size_t dstC, const uint16_t* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution_2xM_Ptr GetInputConvolution_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return InputConvolution_2xM<type, 0x1>;
            case 0x2: return InputConvolution_2xM<type, 0x2>;
            case 0x3: return InputConvolution_2xM<type, 0x3>;
            case 0x4: return InputConvolution_2xM<type, 0x4>;
            case 0x5: return InputConvolution_2xM<type, 0x5>;
            case 0x6: return InputConvolution_2xM<type, 0x6>;
            case 0x7: return InputConvolution_2xM<type, 0x7>;
            case 0x8: return InputConvolution_2xM<type, 0x8>;
            case 0x9: return InputConvolution_2xM<type, 0x9>;
            case 0xa: return InputConvolution_2xM<type, 0xa>;
            case 0xb: return InputConvolution_2xM<type, 0xb>;
            case 0xc: return InputConvolution_2xM<type, 0xc>;
            case 0xd: return InputConvolution_2xM<type, 0xd>;
            case 0xe: return InputConvolution_2xM<type, 0xe>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution_2(const uint16_t* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t noseW = NoseW(p), bodyW = BodyW(p), tailW = p.dstW;
            size_t n = 14, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            size_t dstM = (a.bufH[1] - 1), dstS = a.bufH[1] * p.dstW * F;
            InputConvolution_2xM_Ptr inputConvolution_2x1 = InputConvolution_2x1<type>;
            InputConvolution_2xM_Ptr inputConvolution_2xN = GetInputConvolution_2xM<type>(n);
            InputConvolution_2xM_Ptr inputConvolution_2xM = GetInputConvolution_2xM<type>(m);
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    size_t dx = 0;
                    for (; dx < noseW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _bias, _params, dst0, dst1);
                    for (; dx < bodyWn; dx += n, dst0 += F * n, dst1 += F * n)
                        inputConvolution_2xN(src, p, a, dy, dx, dC, weight, _bias, _params, dst0, dst1);
                    for (; dx < bodyW; dx += m, dst0 += F * m, dst1 += F * m)
                        inputConvolution_2xM(src, p, a, dy, dx, dC, weight, _bias, _params, dst0, dst1);
                    for (; dx < tailW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _bias, _params, dst0, dst1);
                }
                dst += a.bufH[1] * p.dstW * DF;
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 2) * QF;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int M> void InputConvolution1x1_2xM(const uint16_t* src0, const ConvParam32f& p,
            const AlgParam& a, size_t dstC, const uint16_t* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61,
                d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, dc0, dc1, dd0, dd1;
            __m512bh s0, w0, w1;
            const uint16_t* src1 = src0 + 1 * p.srcC;
            const uint16_t* src2 = src0 + 2 * p.srcC;
            const uint16_t* src3 = src0 + 3 * p.srcC;
            const uint16_t* src4 = src0 + 4 * p.srcC;
            const uint16_t* src5 = src0 + 5 * p.srcC;
            const uint16_t* src6 = src0 + 6 * p.srcC;
            if (dstC > F)
            {
                if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                if (M > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                if (M > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                if (M > 0xa) da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
                if (M > 0xb) db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
                if (M > 0xc) dc0 = _mm512_setzero_ps(), dc1 = _mm512_setzero_ps();
                if (M > 0xd) dd0 = _mm512_setzero_ps(), dd1 = _mm512_setzero_ps();
                for (size_t offs0 = 0, offs7 = offs0 + 7 * p.srcC, end = p.srcC; offs0 < end; offs0 += 2, offs7 += 2)
                {
                    w0 = (__m512bh)_mm512_loadu_si512(weight + 0 * DF);
                    w1 = (__m512bh)_mm512_loadu_si512(weight + 1 * DF);
                    if (M > 0x0) s0 = Set2(src0 + offs0), d00 = _mm512_dpbf16_ps(d00, s0, w0), d01 = _mm512_dpbf16_ps(d01, s0, w1);
                    if (M > 0x1) s0 = Set2(src1 + offs0), d10 = _mm512_dpbf16_ps(d10, s0, w0), d11 = _mm512_dpbf16_ps(d11, s0, w1);
                    if (M > 0x2) s0 = Set2(src2 + offs0), d20 = _mm512_dpbf16_ps(d20, s0, w0), d21 = _mm512_dpbf16_ps(d21, s0, w1);
                    if (M > 0x3) s0 = Set2(src3 + offs0), d30 = _mm512_dpbf16_ps(d30, s0, w0), d31 = _mm512_dpbf16_ps(d31, s0, w1);
                    if (M > 0x4) s0 = Set2(src4 + offs0), d40 = _mm512_dpbf16_ps(d40, s0, w0), d41 = _mm512_dpbf16_ps(d41, s0, w1);
                    if (M > 0x5) s0 = Set2(src5 + offs0), d50 = _mm512_dpbf16_ps(d50, s0, w0), d51 = _mm512_dpbf16_ps(d51, s0, w1);
                    if (M > 0x6) s0 = Set2(src6 + offs0), d60 = _mm512_dpbf16_ps(d60, s0, w0), d61 = _mm512_dpbf16_ps(d61, s0, w1);
                    if (M > 0x7) s0 = Set2(src0 + offs7), d70 = _mm512_dpbf16_ps(d70, s0, w0), d71 = _mm512_dpbf16_ps(d71, s0, w1);
                    if (M > 0x8) s0 = Set2(src1 + offs7), d80 = _mm512_dpbf16_ps(d80, s0, w0), d81 = _mm512_dpbf16_ps(d81, s0, w1);
                    if (M > 0x9) s0 = Set2(src2 + offs7), d90 = _mm512_dpbf16_ps(d90, s0, w0), d91 = _mm512_dpbf16_ps(d91, s0, w1);
                    if (M > 0xa) s0 = Set2(src3 + offs7), da0 = _mm512_dpbf16_ps(da0, s0, w0), da1 = _mm512_dpbf16_ps(da1, s0, w1);
                    if (M > 0xb) s0 = Set2(src4 + offs7), db0 = _mm512_dpbf16_ps(db0, s0, w0), db1 = _mm512_dpbf16_ps(db1, s0, w1);
                    if (M > 0xc) s0 = Set2(src5 + offs7), dc0 = _mm512_dpbf16_ps(dc0, s0, w0), dc1 = _mm512_dpbf16_ps(dc1, s0, w1);
                    if (M > 0xd) s0 = Set2(src6 + offs7), dd0 = _mm512_dpbf16_ps(dd0, s0, w0), dd1 = _mm512_dpbf16_ps(dd1, s0, w1);
                    weight += QF;
                }
                if (M > 0x0) SaveInput2<type>(dst0 + 0x0 * F, dst1 + 0x0 * F, d00, d01, bias, params);
                if (M > 0x1) SaveInput2<type>(dst0 + 0x1 * F, dst1 + 0x1 * F, d10, d11, bias, params);
                if (M > 0x2) SaveInput2<type>(dst0 + 0x2 * F, dst1 + 0x2 * F, d20, d21, bias, params);
                if (M > 0x3) SaveInput2<type>(dst0 + 0x3 * F, dst1 + 0x3 * F, d30, d31, bias, params);
                if (M > 0x4) SaveInput2<type>(dst0 + 0x4 * F, dst1 + 0x4 * F, d40, d41, bias, params);
                if (M > 0x5) SaveInput2<type>(dst0 + 0x5 * F, dst1 + 0x5 * F, d50, d51, bias, params);
                if (M > 0x6) SaveInput2<type>(dst0 + 0x6 * F, dst1 + 0x6 * F, d60, d61, bias, params);
                if (M > 0x7) SaveInput2<type>(dst0 + 0x7 * F, dst1 + 0x7 * F, d70, d71, bias, params);
                if (M > 0x8) SaveInput2<type>(dst0 + 0x8 * F, dst1 + 0x8 * F, d80, d81, bias, params);
                if (M > 0x9) SaveInput2<type>(dst0 + 0x9 * F, dst1 + 0x9 * F, d90, d91, bias, params);
                if (M > 0xa) SaveInput2<type>(dst0 + 0xa * F, dst1 + 0xa * F, da0, da1, bias, params);
                if (M > 0xb) SaveInput2<type>(dst0 + 0xb * F, dst1 + 0xb * F, db0, db1, bias, params);
                if (M > 0xc) SaveInput2<type>(dst0 + 0xc * F, dst1 + 0xc * F, dc0, dc1, bias, params);
                if (M > 0xd) SaveInput2<type>(dst0 + 0xd * F, dst1 + 0xd * F, dd0, dd1, bias, params);
            }
            else
            {
                if (M > 0x0) d00 = _mm512_setzero_ps();
                if (M > 0x1) d10 = _mm512_setzero_ps();
                if (M > 0x2) d20 = _mm512_setzero_ps();
                if (M > 0x3) d30 = _mm512_setzero_ps();
                if (M > 0x4) d40 = _mm512_setzero_ps();
                if (M > 0x5) d50 = _mm512_setzero_ps();
                if (M > 0x6) d60 = _mm512_setzero_ps();
                if (M > 0x7) d70 = _mm512_setzero_ps();
                if (M > 0x8) d80 = _mm512_setzero_ps();
                if (M > 0x9) d90 = _mm512_setzero_ps();
                if (M > 0xa) da0 = _mm512_setzero_ps();
                if (M > 0xb) db0 = _mm512_setzero_ps();
                if (M > 0xc) dc0 = _mm512_setzero_ps();
                if (M > 0xd) dd0 = _mm512_setzero_ps();
                for (size_t offs0 = 0, offs7 = offs0 + 7 * p.srcC, end = p.srcC; offs0 < end; offs0 += 2, offs7 += 2)
                {
                    w0 = (__m512bh)_mm512_loadu_si512(weight + 0 * DF);
                    if (M > 0x0) s0 = Set2(src0 + offs0), d00 = _mm512_dpbf16_ps(d00, s0, w0);
                    if (M > 0x1) s0 = Set2(src1 + offs0), d10 = _mm512_dpbf16_ps(d10, s0, w0);
                    if (M > 0x2) s0 = Set2(src2 + offs0), d20 = _mm512_dpbf16_ps(d20, s0, w0);
                    if (M > 0x3) s0 = Set2(src3 + offs0), d30 = _mm512_dpbf16_ps(d30, s0, w0);
                    if (M > 0x4) s0 = Set2(src4 + offs0), d40 = _mm512_dpbf16_ps(d40, s0, w0);
                    if (M > 0x5) s0 = Set2(src5 + offs0), d50 = _mm512_dpbf16_ps(d50, s0, w0);
                    if (M > 0x6) s0 = Set2(src6 + offs0), d60 = _mm512_dpbf16_ps(d60, s0, w0);
                    if (M > 0x7) s0 = Set2(src0 + offs7), d70 = _mm512_dpbf16_ps(d70, s0, w0);
                    if (M > 0x8) s0 = Set2(src1 + offs7), d80 = _mm512_dpbf16_ps(d80, s0, w0);
                    if (M > 0x9) s0 = Set2(src2 + offs7), d90 = _mm512_dpbf16_ps(d90, s0, w0);
                    if (M > 0xa) s0 = Set2(src3 + offs7), da0 = _mm512_dpbf16_ps(da0, s0, w0);
                    if (M > 0xb) s0 = Set2(src4 + offs7), db0 = _mm512_dpbf16_ps(db0, s0, w0);
                    if (M > 0xc) s0 = Set2(src5 + offs7), dc0 = _mm512_dpbf16_ps(dc0, s0, w0);
                    if (M > 0xd) s0 = Set2(src6 + offs7), dd0 = _mm512_dpbf16_ps(dd0, s0, w0);
                    weight += QF;
                }
                if (M > 0x0) SaveInput1<type>(dst0 + 0x0 * F, d00, bias, params);
                if (M > 0x1) SaveInput1<type>(dst0 + 0x1 * F, d10, bias, params);
                if (M > 0x2) SaveInput1<type>(dst0 + 0x2 * F, d20, bias, params);
                if (M > 0x3) SaveInput1<type>(dst0 + 0x3 * F, d30, bias, params);
                if (M > 0x4) SaveInput1<type>(dst0 + 0x4 * F, d40, bias, params);
                if (M > 0x5) SaveInput1<type>(dst0 + 0x5 * F, d50, bias, params);
                if (M > 0x6) SaveInput1<type>(dst0 + 0x6 * F, d60, bias, params);
                if (M > 0x7) SaveInput1<type>(dst0 + 0x7 * F, d70, bias, params);
                if (M > 0x8) SaveInput1<type>(dst0 + 0x8 * F, d80, bias, params);
                if (M > 0x9) SaveInput1<type>(dst0 + 0x9 * F, d90, bias, params);
                if (M > 0xa) SaveInput1<type>(dst0 + 0xa * F, da0, bias, params);
                if (M > 0xb) SaveInput1<type>(dst0 + 0xb * F, db0, bias, params);
                if (M > 0xc) SaveInput1<type>(dst0 + 0xc * F, dc0, bias, params);
                if (M > 0xd) SaveInput1<type>(dst0 + 0xd * F, dd0, bias, params);
            }
        }

        typedef void(*InputConvolution1x1_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t dstC,
            const uint16_t* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution1x1_2xM_Ptr GetInputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 0x1: return InputConvolution1x1_2xM<type, 0x1>;
            case 0x2: return InputConvolution1x1_2xM<type, 0x2>;
            case 0x3: return InputConvolution1x1_2xM<type, 0x3>;
            case 0x4: return InputConvolution1x1_2xM<type, 0x4>;
            case 0x5: return InputConvolution1x1_2xM<type, 0x5>;
            case 0x6: return InputConvolution1x1_2xM<type, 0x6>;
            case 0x7: return InputConvolution1x1_2xM<type, 0x7>;
            case 0x8: return InputConvolution1x1_2xM<type, 0x8>;
            case 0x9: return InputConvolution1x1_2xM<type, 0x9>;
            case 0xa: return InputConvolution1x1_2xM<type, 0xa>;
            case 0xb: return InputConvolution1x1_2xM<type, 0xb>;
            case 0xc: return InputConvolution1x1_2xM<type, 0xc>;
            case 0xd: return InputConvolution1x1_2xM<type, 0xd>;
            case 0xe: return InputConvolution1x1_2xM<type, 0xe>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2(const uint16_t* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcM = a.bufH[0] - 1;
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            if (a.bufH[0] == 0 || a.bufH[0] >= p.srcH)
            {
                size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 14;
                size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
                size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xI = GetInputConvolution1x1_2xM<type>(i);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xE = GetInputConvolution1x1_2xM<type>(e);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    if (yInt > yBeg)
                    {
                        const uint16_t* src0 = src + yBeg * p.srcW * p.srcC;
                        float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < in; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                        if (in < i1)
                            inputConvolution1x1_2xI(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                    }
                    if (yEnd > yInt)
                    {
                        const uint16_t* src0 = src + yInt * p.srcW * p.srcC;
                        float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < en; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                        if (en < e1)
                            inputConvolution1x1_2xE(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 2) * QF;
                }
            }
            else
            {
                size_t n = 14, bodyW = p.dstW, bodyWn = AlignLoAny(bodyW, n), m = bodyW - bodyWn;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xM = GetInputConvolution1x1_2xM<type>(m);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    for (size_t dy = yBeg; dy < yEnd; dy++)
                    {
                        const uint16_t* src0 = src + (dy & srcM) * p.srcW * p.srcC;
                        float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        size_t dx = 0;
                        for (; dx < bodyWn; dx += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                        if (dx < bodyW)
                            inputConvolution1x1_2xM(src0, p, a, dC, weight, _bias, _params, dst0, dst1);
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 2) * QF;
                }
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInput(const ConvParam32f& p, InputPtr& input)
        {
            if (Is1x1(p))
                input = InputConvolution1x1_2<type>;
            else
                input = InputConvolution_2<type>;
        }

        void SetInput(const ConvParam32f& p, InputPtr& input)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationRelu: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationLeakyRelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationRestrictRange: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationPrelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationElu: SetInput<SimdConvolutionActivationElu>(p, input); break;
            case SimdConvolutionActivationHswish: SetInput<SimdConvolutionActivationHswish>(p, input); break;
            case SimdConvolutionActivationMish: SetInput<SimdConvolutionActivationMish>(p, input); break;
            case SimdConvolutionActivationHardSigmoid: SetInput<SimdConvolutionActivationHardSigmoid>(p, input); break;
            case SimdConvolutionActivationSwish: SetInput<SimdConvolutionActivationSwish>(p, input); break;
            }
        }
    }
#endif
}
