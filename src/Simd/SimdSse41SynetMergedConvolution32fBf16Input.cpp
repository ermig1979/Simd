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
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        using AlgParam = Base::SynetMergedConvolution32fBf16::AlgParam;
        using InputPtr = Base::SynetMergedConvolution32fBf16::InputConvolutionPtr;

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> 
        SIMD_INLINE void SaveInput1(float* dst, __m128 sum, const __m128* bias, const __m128* params)
        {
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(sum, bias[0]), params, 0));
        }

        template<SimdConvolutionActivationType type>
        SIMD_INLINE void SaveInput2(float* dst0, float* dst1, __m128 sum0, __m128 sum1, const __m128* bias, const __m128* params)
        {
            _mm_storeu_ps(dst0, Activate<type>(_mm_add_ps(sum0, bias[0]), params, 0));
            _mm_storeu_ps(dst1, Activate<type>(_mm_add_ps(sum1, bias[1]), params, 1));
        }

        template<SimdConvolutionActivationType type> void InputConvolution_2x1(const uint16_t* src0,
            const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC, const uint16_t* weight,
            const __m128* bias, const __m128* params, float* dst0, float* dst1)
        {
            __m128 d00, d01, s0, w00, w01, w10, w11, m = _mm_castsi128_ps(Bf16::MASK);
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(p.srcC, 2) * QF, sM = a.bufH[0] - 1;
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 2)
                            {
                                w01 = _mm_loadu_ps((float*)weight + 0);
                                w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                                w01 = _mm_and_ps(w01, m);
                                w11 = _mm_loadu_ps((float*)weight + F);
                                w10 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w11), Base::Bf16::SHIFT));
                                w11 = _mm_and_ps(w11, m);

                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m);
                                d00 = _mm_add_ps(_mm_mul_ps(s0, w00), d00); d01 = _mm_add_ps(_mm_mul_ps(s0, w10), d01);
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m);
                                d00 = _mm_add_ps(_mm_mul_ps(s0, w01), d00); d01 = _mm_add_ps(_mm_mul_ps(s0, w11), d01);

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
                d00 = _mm_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 2)
                            {
                                w01 = _mm_loadu_ps((float*)weight + 0);
                                w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                                w01 = _mm_and_ps(w01, m);

                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m);
                                d00 = _mm_add_ps(_mm_mul_ps(s0, w00), d00);
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m);
                                d00 = _mm_add_ps(_mm_mul_ps(s0, w01), d00);

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
            const uint16_t* weight, const __m128* bias, const __m128* params, float* dst0, float* dst1)
        {
            __m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11, m = _mm_castsi128_ps(Bf16::MASK);
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC, dWz = DivHi(p.srcC, 2) * QF * p.kernelX, sM = a.bufH[0] - 1;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                if (M > 0) d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
                if (M > 1) d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
                if (M > 2) d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
                if (M > 3) d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
                if (M > 4) d40 = _mm_setzero_ps(), d41 = _mm_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 2)
                            {
                                w01 = _mm_loadu_ps((float*)weight + 0);
                                w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                                w01 = _mm_and_ps(w01, m);
                                w11 = _mm_loadu_ps((float*)weight + F);
                                w10 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w11), Base::Bf16::SHIFT));
                                w11 = _mm_and_ps(w11, m);
                                if (M > 0)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m);
                                    d00 = _mm_add_ps(_mm_mul_ps(s0, w00), d00); d01 = _mm_add_ps(_mm_mul_ps(s0, w10), d01);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m);
                                    d00 = _mm_add_ps(_mm_mul_ps(s0, w01), d00); d01 = _mm_add_ps(_mm_mul_ps(s0, w11), d01);
                                }
                                if (M > 1)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 1)), m);
                                    d10 = _mm_add_ps(_mm_mul_ps(s0, w00), d10); d11 = _mm_add_ps(_mm_mul_ps(s0, w10), d11);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 0)), m);
                                    d10 = _mm_add_ps(_mm_mul_ps(s0, w01), d10); d11 = _mm_add_ps(_mm_mul_ps(s0, w11), d11);
                                }
                                if (M > 2)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 1)), m);
                                    d20 = _mm_add_ps(_mm_mul_ps(s0, w00), d20); d21 = _mm_add_ps(_mm_mul_ps(s0, w10), d21);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 0)), m);
                                    d20 = _mm_add_ps(_mm_mul_ps(s0, w01), d20); d21 = _mm_add_ps(_mm_mul_ps(s0, w11), d21);
                                }
                                if (M > 3)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 1)), m);
                                    d30 = _mm_add_ps(_mm_mul_ps(s0, w00), d30); d31 = _mm_add_ps(_mm_mul_ps(s0, w10), d31);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 0)), m);
                                    d30 = _mm_add_ps(_mm_mul_ps(s0, w01), d30); d31 = _mm_add_ps(_mm_mul_ps(s0, w11), d31);
                                }
                                if (M > 4)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 1)), m);
                                    d40 = _mm_add_ps(_mm_mul_ps(s0, w00), d40); d41 = _mm_add_ps(_mm_mul_ps(s0, w10), d41);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 0)), m);
                                    d40 = _mm_add_ps(_mm_mul_ps(s0, w01), d40); d41 = _mm_add_ps(_mm_mul_ps(s0, w11), d41);
                                }
                                weight += QF;
                            }
                        }
                    }
                    else
                        weight += dWz;
                }
                if (M > 0) SaveInput2<type>(dst0 + 0 * F, dst1 + 0 * F, d00, d01, bias, params);
                if (M > 1) SaveInput2<type>(dst0 + 1 * F, dst1 + 1 * F, d10, d11, bias, params);
                if (M > 2) SaveInput2<type>(dst0 + 2 * F, dst1 + 2 * F, d20, d21, bias, params);
                if (M > 3) SaveInput2<type>(dst0 + 3 * F, dst1 + 3 * F, d30, d31, bias, params);
                if (M > 4) SaveInput2<type>(dst0 + 4 * F, dst1 + 4 * F, d40, d41, bias, params);
            }
            else
            {
                if (M > 0) d00 = _mm_setzero_ps();
                if (M > 1) d10 = _mm_setzero_ps();
                if (M > 2) d20 = _mm_setzero_ps();
                if (M > 3) d30 = _mm_setzero_ps();
                if (M > 4) d40 = _mm_setzero_ps();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 2)
                            {
                                w01 = _mm_loadu_ps((float*)weight + 0);
                                w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                                w01 = _mm_and_ps(w01, m);
                                if (M > 0)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m);
                                    d00 = _mm_add_ps(_mm_mul_ps(s0, w00), d00);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m);
                                    d00 = _mm_add_ps(_mm_mul_ps(s0, w01), d00);
                                }
                                if (M > 1)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 1)), m);
                                    d10 = _mm_add_ps(_mm_mul_ps(s0, w00), d10);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 0)), m);
                                    d10 = _mm_add_ps(_mm_mul_ps(s0, w01), d10);
                                }
                                if (M > 2)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 1)), m);
                                    d20 = _mm_add_ps(_mm_mul_ps(s0, w00), d20);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 0)), m);
                                    d20 = _mm_add_ps(_mm_mul_ps(s0, w01), d20);
                                }
                                if (M > 3)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 1)), m);
                                    d30 = _mm_add_ps(_mm_mul_ps(s0, w00), d30);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 0)), m);
                                    d30 = _mm_add_ps(_mm_mul_ps(s0, w01), d30);
                                }
                                if (M > 4)
                                {
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 1)), m);
                                    d40 = _mm_add_ps(_mm_mul_ps(s0, w00), d40);
                                    s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 0)), m);
                                    d40 = _mm_add_ps(_mm_mul_ps(s0, w01), d40);
                                }
                                weight += QF;
                            }
                        }
                    }
                    else
                        weight += dWz;
                }
                if (M > 0) SaveInput1<type>(dst0 + 0 * F, d00, bias, params);
                if (M > 1) SaveInput1<type>(dst0 + 1 * F, d10, bias, params);
                if (M > 2) SaveInput1<type>(dst0 + 2 * F, d20, bias, params);
                if (M > 3) SaveInput1<type>(dst0 + 3 * F, d30, bias, params);
                if (M > 4) SaveInput1<type>(dst0 + 4 * F, d40, bias, params);
            }
        }

        typedef void(*InputConvolution_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t dy, size_t dx,
            size_t dstC, const uint16_t* weight, const __m128* bias, const __m128* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution_2xM_Ptr GetInputConvolution_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return InputConvolution_2xM<type, 1>;
            case 2: return InputConvolution_2xM<type, 2>;
            case 3: return InputConvolution_2xM<type, 3>;
            case 4: return InputConvolution_2xM<type, 4>;
            case 5: return InputConvolution_2xM<type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution_2(const uint16_t* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t noseW = NoseW(p), bodyW = BodyW(p), tailW = p.dstW;
            size_t n = 5, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            size_t dstM = (a.bufH[1] - 1), dstS = a.bufH[1] * p.dstW * F;
            InputConvolution_2xM_Ptr inputConvolution_2x1 = InputConvolution_2x1<type>;
            InputConvolution_2xM_Ptr inputConvolution_2xN = GetInputConvolution_2xM<type>(n);
            InputConvolution_2xM_Ptr inputConvolution_2xM = GetInputConvolution_2xM<type>(m);
            __m128 _bias[2], _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            _params[1] = _mm_set1_ps(params[1]);
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                _bias[0] = _mm_loadu_ps(bias + dc + 0);
                _bias[1] = _mm_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + dc + 0);
                    _params[1] = _mm_loadu_ps(params + dc + F);
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
            const AlgParam& a, size_t dstC, const uint16_t* weight, const __m128* bias, const __m128* params, float* dst0, float* dst1)
        {
            __m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11, m = _mm_castsi128_ps(Bf16::MASK);
            const uint16_t* src1 = src0 + 1 * p.srcC;
            const uint16_t* src2 = src0 + 2 * p.srcC;
            const uint16_t* src3 = src0 + 3 * p.srcC;
            const uint16_t* src4 = src0 + 4 * p.srcC;
            if (dstC > F)
            {
                if (M > 0) d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
                if (M > 1) d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
                if (M > 2) d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
                if (M > 3) d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
                if (M > 4) d40 = _mm_setzero_ps(), d41 = _mm_setzero_ps();
                for (size_t offs = 0, end = p.srcC; offs < end; offs += 2)
                {
                    w01 = _mm_loadu_ps((float*)weight + 0);
                    w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                    w01 = _mm_and_ps(w01, m);
                    w11 = _mm_loadu_ps((float*)weight + F);
                    w10 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w11), Base::Bf16::SHIFT));
                    w11 = _mm_and_ps(w11, m);
                    if (M > 0)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(s0, w00), d00); d01 = _mm_add_ps(_mm_mul_ps(s0, w10), d01);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(s0, w01), d00); d01 = _mm_add_ps(_mm_mul_ps(s0, w11), d01);
                    }
                    if (M > 1)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 1)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(s0, w00), d10); d11 = _mm_add_ps(_mm_mul_ps(s0, w10), d11);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 0)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(s0, w01), d10); d11 = _mm_add_ps(_mm_mul_ps(s0, w11), d11);
                    }
                    if (M > 2)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 1)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(s0, w00), d20); d21 = _mm_add_ps(_mm_mul_ps(s0, w10), d21);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 0)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(s0, w01), d20); d21 = _mm_add_ps(_mm_mul_ps(s0, w11), d21);
                    }
                    if (M > 3)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 1)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(s0, w00), d30); d31 = _mm_add_ps(_mm_mul_ps(s0, w10), d31);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 0)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(s0, w01), d30); d31 = _mm_add_ps(_mm_mul_ps(s0, w11), d31);
                    }
                    if (M > 4)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 1)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(s0, w00), d40); d41 = _mm_add_ps(_mm_mul_ps(s0, w10), d41);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 0)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(s0, w01), d40); d41 = _mm_add_ps(_mm_mul_ps(s0, w11), d41);
                    }
                    weight += QF;
                }
                if (M > 0) SaveInput2<type>(dst0 + 0 * F, dst1 + 0 * F, d00, d01, bias, params);
                if (M > 1) SaveInput2<type>(dst0 + 1 * F, dst1 + 1 * F, d10, d11, bias, params);
                if (M > 2) SaveInput2<type>(dst0 + 2 * F, dst1 + 2 * F, d20, d21, bias, params);
                if (M > 3) SaveInput2<type>(dst0 + 3 * F, dst1 + 3 * F, d30, d31, bias, params);
                if (M > 4) SaveInput2<type>(dst0 + 4 * F, dst1 + 4 * F, d40, d41, bias, params);
            }
            else
            {
                if (M > 0) d00 = _mm_setzero_ps();
                if (M > 1) d10 = _mm_setzero_ps();
                if (M > 2) d20 = _mm_setzero_ps();
                if (M > 3) d30 = _mm_setzero_ps();
                if (M > 4) d40 = _mm_setzero_ps();
                for (size_t offs = 0, end = p.srcC; offs < end; offs += 2)
                {
                    w01 = _mm_loadu_ps((float*)weight + 0);
                    w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                    w01 = _mm_and_ps(w01, m);
                    if (M > 0)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(s0, w00), d00);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(s0, w01), d00);
                    }
                    if (M > 1)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 1)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(s0, w00), d10);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 0)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(s0, w01), d10);
                    }
                    if (M > 2)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 1)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(s0, w00), d20);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 0)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(s0, w01), d20);
                    }
                    if (M > 3)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 1)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(s0, w00), d30);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 0)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(s0, w01), d30);
                    }
                    if (M > 4)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 1)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(s0, w00), d40);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 0)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(s0, w01), d40);
                    }
                    weight += QF;
                }
                if (M > 0) SaveInput1<type>(dst0 + 0 * F, d00, bias, params);
                if (M > 1) SaveInput1<type>(dst0 + 1 * F, d10, bias, params);
                if (M > 2) SaveInput1<type>(dst0 + 2 * F, d20, bias, params);
                if (M > 3) SaveInput1<type>(dst0 + 3 * F, d30, bias, params);
                if (M > 4) SaveInput1<type>(dst0 + 4 * F, d40, bias, params);
            }
        }

        typedef void(*InputConvolution1x1_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, size_t dstC,
            const uint16_t* weight, const __m128* bias, const __m128* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution1x1_2xM_Ptr GetInputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return InputConvolution1x1_2xM<type, 1>;
            case 2: return InputConvolution1x1_2xM<type, 2>;
            case 3: return InputConvolution1x1_2xM<type, 3>;
            case 4: return InputConvolution1x1_2xM<type, 4>;
            case 5: return InputConvolution1x1_2xM<type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2(const uint16_t* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcM = a.bufH[0] - 1;
            __m128 _bias[2], _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            _params[1] = _mm_set1_ps(params[1]);
            if (a.bufH[0] == 0 || a.bufH[0] >= p.srcH)
            {
                size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 5;
                size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
                size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xI = GetInputConvolution1x1_2xM<type>(i);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xE = GetInputConvolution1x1_2xM<type>(e);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _bias[0] = _mm_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm_loadu_ps(params + dc + 0);
                        _params[1] = _mm_loadu_ps(params + dc + F);
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
                size_t n = 5, bodyW = p.dstW, bodyWn = AlignLoAny(bodyW, n), m = bodyW - bodyWn;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xM = GetInputConvolution1x1_2xM<type>(m);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _bias[0] = _mm_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm_loadu_ps(params + dc + 0);
                        _params[1] = _mm_loadu_ps(params + dc + F);
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
