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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512VNNI_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Avx512vnni
    {
        using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
        using InputConvolutionPtr = Base::SynetMergedConvolution8i::InputConvolutionPtr;

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void SaveInput1(float* dst, __m512i sum, const __m512* norm, const __m512* bias, const __m512* params)
        {
            _mm512_storeu_ps((float*)dst, Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), norm[0], bias[0]), params, 0));
        }

        template<SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void SaveInput2(float* dst0, float* dst1, __m512i sum0, __m512i sum1, const __m512* norm, const __m512* bias, const __m512* params)
        {
            _mm512_storeu_ps(dst0, Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum0), norm[0], bias[0]), params, 0));
            _mm512_storeu_ps(dst1, Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum1), norm[1], bias[1]), params, 1));
        }

        template<bool overflow, SimdConvolutionActivationType type, bool nofma> void InputConvolution_2x1(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC, const int8_t* weight,
            const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            __m512i d00, d01, s0, w0, w1;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(p.srcC, 4) * DA, sM = a.bufH[0] - 1;
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                w1 = _mm512_loadu_si512((__m512i*)weight + 1);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d01, s0, w1);
                                weight += DA;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = _mm512_set1_epi32(a.zero);
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    w1 = _mm512_loadu_si512((__m512i*)weight + 1);
                                    Madd4<overflow>(d00, s0, w0);
                                    Madd4<overflow>(d01, s0, w1);
                                    weight += DA;
                                }
                            }
                            else
                                weight += dWz;
                        }
                    }
                }
                SaveInput2<type, nofma>(dst0, dst1, d00, d01, norm, bias, params);
            }
            else
            {
                d00 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                weight += DA;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = _mm512_set1_epi32(a.zero);
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    Madd4<overflow>(d00, s0, w0);
                                    weight += DA;
                                }
                            }
                            else
                                weight += dWz;
                        }
                    }
                }
                SaveInput1<type, nofma>(dst0, d00, norm, bias, params);
            }
        }

        typedef void(*InputConvolution_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx,
            size_t dstC, const int8_t* weight, const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution_2xM_Ptr GetInputConvolution_2x1(const ConvParam8i& p)
        {
            bool nofma = Base::FmaAvoid(p.compatibility);
            if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                return nofma ? InputConvolution_2x1<true, type, true> : InputConvolution_2x1<true, type, false>;
            else
                return nofma ? InputConvolution_2x1<false, type, true> : InputConvolution_2x1<false, type, false>;
        }

        template<SimdConvolutionActivationType type, int M> void InputConvolution_2xM(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC, const int8_t* weight,
            const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC * a.size, dWz = DivHi(p.srcC, 4) * DA * p.kernelX, sM = a.bufH[0] - 1;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            const uint8_t* src5 = src0 + 5 * dS;
            __m512i upper = _mm512_set1_epi32(a.upper);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                if (M > 0x0) d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                if (M > 0x1) d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();
                if (M > 0x2) d20 = _mm512_setzero_si512(), d21 = _mm512_setzero_si512();
                if (M > 0x3) d30 = _mm512_setzero_si512(), d31 = _mm512_setzero_si512();
                if (M > 0x4) d40 = _mm512_setzero_si512(), d41 = _mm512_setzero_si512();
                if (M > 0x5) d50 = _mm512_setzero_si512(), d51 = _mm512_setzero_si512();
                if (M > 0x6) d60 = _mm512_setzero_si512(), d61 = _mm512_setzero_si512();
                if (M > 0x7) d70 = _mm512_setzero_si512(), d71 = _mm512_setzero_si512();
                if (M > 0x8) d80 = _mm512_setzero_si512(), d81 = _mm512_setzero_si512();
                if (M > 0x9) d90 = _mm512_setzero_si512(), d91 = _mm512_setzero_si512();
                if (M > 0xA) dA0 = _mm512_setzero_si512(), dA1 = _mm512_setzero_si512();
                if (M > 0xB) dB0 = _mm512_setzero_si512(), dB1 = _mm512_setzero_si512();
                if (Base::Overflow(p.compatibility))
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs6 = offs0 + 6 * dS;
                                for (; offs0 < end; offs0 += 4, offs6 += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    w1 = _mm512_loadu_si512((__m512i*)weight + 1);
                                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<true>(d50, s0, w0), Madd4<true>(d51, s0, w1);
                                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<true>(d60, s0, w0), Madd4<true>(d61, s0, w1);
                                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<true>(d70, s0, w0), Madd4<true>(d71, s0, w1);
                                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<true>(d80, s0, w0), Madd4<true>(d81, s0, w1);
                                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<true>(d90, s0, w0), Madd4<true>(d91, s0, w1);
                                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<true>(dA0, s0, w0), Madd4<true>(dA1, s0, w1);
                                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<true>(dB0, s0, w0), Madd4<true>(dB1, s0, w1);
                                    weight += DA;
                                }
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm512_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    w1 = _mm512_loadu_si512((__m512i*)weight + 1);
                                    if (M > 0x0) Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                                    if (M > 0x1) Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                                    if (M > 0x2) Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                                    if (M > 0x3) Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                                    if (M > 0x4) Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                                    if (M > 0x5) Madd4<true>(d50, s0, w0), Madd4<true>(d51, s0, w1);
                                    if (M > 0x6) Madd4<true>(d60, s0, w0), Madd4<true>(d61, s0, w1);
                                    if (M > 0x7) Madd4<true>(d70, s0, w0), Madd4<true>(d71, s0, w1);
                                    if (M > 0x8) Madd4<true>(d80, s0, w0), Madd4<true>(d81, s0, w1);
                                    if (M > 0x9) Madd4<true>(d90, s0, w0), Madd4<true>(d91, s0, w1);
                                    if (M > 0xA) Madd4<true>(dA0, s0, w0), Madd4<true>(dA1, s0, w1);
                                    if (M > 0xB) Madd4<true>(dB0, s0, w0), Madd4<true>(dB1, s0, w1);
                                    weight += DA;
                                }
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs6 = offs0 + 6 * dS;
                                for (; offs0 < end; offs0 += 4, offs6 += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    w1 = _mm512_loadu_si512((__m512i*)weight + 1);
                                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<false>(d50, s0, w0), Madd4<false>(d51, s0, w1);
                                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<false>(d60, s0, w0), Madd4<false>(d61, s0, w1);
                                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<false>(d70, s0, w0), Madd4<false>(d71, s0, w1);
                                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<false>(d80, s0, w0), Madd4<false>(d81, s0, w1);
                                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<false>(d90, s0, w0), Madd4<false>(d91, s0, w1);
                                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<false>(dA0, s0, w0), Madd4<false>(dA1, s0, w1);
                                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<false>(dB0, s0, w0), Madd4<false>(dB1, s0, w1);
                                    weight += DA;
                                }
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm512_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    w1 = _mm512_loadu_si512((__m512i*)weight + 1);
                                    if (M > 0x0) Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                                    if (M > 0x1) Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                                    if (M > 0x2) Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                                    if (M > 0x3) Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                                    if (M > 0x4) Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                                    if (M > 0x5) Madd4<false>(d50, s0, w0), Madd4<false>(d51, s0, w1);
                                    if (M > 0x6) Madd4<false>(d60, s0, w0), Madd4<false>(d61, s0, w1);
                                    if (M > 0x7) Madd4<false>(d70, s0, w0), Madd4<false>(d71, s0, w1);
                                    if (M > 0x8) Madd4<false>(d80, s0, w0), Madd4<false>(d81, s0, w1);
                                    if (M > 0x9) Madd4<false>(d90, s0, w0), Madd4<false>(d91, s0, w1);
                                    if (M > 0xA) Madd4<false>(dA0, s0, w0), Madd4<false>(dA1, s0, w1);
                                    if (M > 0xB) Madd4<false>(dB0, s0, w0), Madd4<false>(dB1, s0, w1);
                                    weight += DA;
                                }
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (M > 0x0) SaveInput2<type, true>(dst0 + 0x0 * F, dst1 + 0x0 * F, d00, d01, norm, bias, params);
                    if (M > 0x1) SaveInput2<type, true>(dst0 + 0x1 * F, dst1 + 0x1 * F, d10, d11, norm, bias, params);
                    if (M > 0x2) SaveInput2<type, true>(dst0 + 0x2 * F, dst1 + 0x2 * F, d20, d21, norm, bias, params);
                    if (M > 0x3) SaveInput2<type, true>(dst0 + 0x3 * F, dst1 + 0x3 * F, d30, d31, norm, bias, params);
                    if (M > 0x4) SaveInput2<type, true>(dst0 + 0x4 * F, dst1 + 0x4 * F, d40, d41, norm, bias, params);
                    if (M > 0x5) SaveInput2<type, true>(dst0 + 0x5 * F, dst1 + 0x5 * F, d50, d51, norm, bias, params);
                    if (M > 0x6) SaveInput2<type, true>(dst0 + 0x6 * F, dst1 + 0x6 * F, d60, d61, norm, bias, params);
                    if (M > 0x7) SaveInput2<type, true>(dst0 + 0x7 * F, dst1 + 0x7 * F, d70, d71, norm, bias, params);
                    if (M > 0x8) SaveInput2<type, true>(dst0 + 0x8 * F, dst1 + 0x8 * F, d80, d81, norm, bias, params);
                    if (M > 0x9) SaveInput2<type, true>(dst0 + 0x9 * F, dst1 + 0x9 * F, d90, d91, norm, bias, params);
                    if (M > 0xA) SaveInput2<type, true>(dst0 + 0xA * F, dst1 + 0xA * F, dA0, dA1, norm, bias, params);
                    if (M > 0xB) SaveInput2<type, true>(dst0 + 0xB * F, dst1 + 0xB * F, dB0, dB1, norm, bias, params);
                }
                else
                {
                    if (M > 0x0) SaveInput2<type, false>(dst0 + 0x0 * F, dst1 + 0x0 * F, d00, d01, norm, bias, params);
                    if (M > 0x1) SaveInput2<type, false>(dst0 + 0x1 * F, dst1 + 0x1 * F, d10, d11, norm, bias, params);
                    if (M > 0x2) SaveInput2<type, false>(dst0 + 0x2 * F, dst1 + 0x2 * F, d20, d21, norm, bias, params);
                    if (M > 0x3) SaveInput2<type, false>(dst0 + 0x3 * F, dst1 + 0x3 * F, d30, d31, norm, bias, params);
                    if (M > 0x4) SaveInput2<type, false>(dst0 + 0x4 * F, dst1 + 0x4 * F, d40, d41, norm, bias, params);
                    if (M > 0x5) SaveInput2<type, false>(dst0 + 0x5 * F, dst1 + 0x5 * F, d50, d51, norm, bias, params);
                    if (M > 0x6) SaveInput2<type, false>(dst0 + 0x6 * F, dst1 + 0x6 * F, d60, d61, norm, bias, params);
                    if (M > 0x7) SaveInput2<type, false>(dst0 + 0x7 * F, dst1 + 0x7 * F, d70, d71, norm, bias, params);
                    if (M > 0x8) SaveInput2<type, false>(dst0 + 0x8 * F, dst1 + 0x8 * F, d80, d81, norm, bias, params);
                    if (M > 0x9) SaveInput2<type, false>(dst0 + 0x9 * F, dst1 + 0x9 * F, d90, d91, norm, bias, params);
                    if (M > 0xA) SaveInput2<type, false>(dst0 + 0xA * F, dst1 + 0xA * F, dA0, dA1, norm, bias, params);
                    if (M > 0xB) SaveInput2<type, false>(dst0 + 0xB * F, dst1 + 0xB * F, dB0, dB1, norm, bias, params);
                }
            }
            else
            {
            if (M > 0x0) d00 = _mm512_setzero_si512();
            if (M > 0x1) d10 = _mm512_setzero_si512();
            if (M > 0x2) d20 = _mm512_setzero_si512();
            if (M > 0x3) d30 = _mm512_setzero_si512();
            if (M > 0x4) d40 = _mm512_setzero_si512();
            if (M > 0x5) d50 = _mm512_setzero_si512();
            if (M > 0x6) d60 = _mm512_setzero_si512();
            if (M > 0x7) d70 = _mm512_setzero_si512();
            if (M > 0x8) d80 = _mm512_setzero_si512();
            if (M > 0x9) d90 = _mm512_setzero_si512();
            if (M > 0xA) dA0 = _mm512_setzero_si512();
            if (M > 0xB) dB0 = _mm512_setzero_si512();
            if (Base::Overflow(p.compatibility))
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs6 = offs0 + 6 * dS;
                                for (; offs0 < end; offs0 += 4, offs6 += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<true>(d00, s0, w0);
                                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<true>(d10, s0, w0);
                                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<true>(d20, s0, w0);
                                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<true>(d30, s0, w0);
                                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<true>(d40, s0, w0);
                                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<true>(d50, s0, w0);
                                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<true>(d60, s0, w0);
                                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<true>(d70, s0, w0);
                                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<true>(d80, s0, w0);
                                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<true>(d90, s0, w0);
                                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<true>(dA0, s0, w0);
                                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<true>(dB0, s0, w0);
                                    weight += DA;
                                }
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm512_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    if (M > 0x0) Madd4<true>(d00, s0, w0);
                                    if (M > 0x1) Madd4<true>(d10, s0, w0);
                                    if (M > 0x2) Madd4<true>(d20, s0, w0);
                                    if (M > 0x3) Madd4<true>(d30, s0, w0);
                                    if (M > 0x4) Madd4<true>(d40, s0, w0);
                                    if (M > 0x5) Madd4<true>(d50, s0, w0);
                                    if (M > 0x6) Madd4<true>(d60, s0, w0);
                                    if (M > 0x7) Madd4<true>(d70, s0, w0);
                                    if (M > 0x8) Madd4<true>(d80, s0, w0);
                                    if (M > 0x9) Madd4<true>(d90, s0, w0);
                                    if (M > 0xA) Madd4<true>(dA0, s0, w0);
                                    if (M > 0xB) Madd4<true>(dB0, s0, w0);
                                    weight += DA;
                                }
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs0 = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs0 + p.srcC, offs6 = offs0 + 6 * dS;
                                for (; offs0 < end; offs0 += 4, offs6 += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<false>(d00, s0, w0);
                                    if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<false>(d10, s0, w0);
                                    if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<false>(d20, s0, w0);
                                    if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<false>(d30, s0, w0);
                                    if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<false>(d40, s0, w0);
                                    if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<false>(d50, s0, w0);
                                    if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<false>(d60, s0, w0);
                                    if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<false>(d70, s0, w0);
                                    if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<false>(d80, s0, w0);
                                    if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<false>(d90, s0, w0);
                                    if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<false>(dA0, s0, w0);
                                    if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<false>(dB0, s0, w0);
                                    weight += DA;
                                }
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm512_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                                    if (M > 0x0) Madd4<false>(d00, s0, w0);
                                    if (M > 0x1) Madd4<false>(d10, s0, w0);
                                    if (M > 0x2) Madd4<false>(d20, s0, w0);
                                    if (M > 0x3) Madd4<false>(d30, s0, w0);
                                    if (M > 0x4) Madd4<false>(d40, s0, w0);
                                    if (M > 0x5) Madd4<false>(d50, s0, w0);
                                    if (M > 0x6) Madd4<false>(d60, s0, w0);
                                    if (M > 0x7) Madd4<false>(d70, s0, w0);
                                    if (M > 0x8) Madd4<false>(d80, s0, w0);
                                    if (M > 0x9) Madd4<false>(d90, s0, w0);
                                    if (M > 0xA) Madd4<false>(dA0, s0, w0);
                                    if (M > 0xB) Madd4<false>(dB0, s0, w0);
                                    weight += DA;
                                }
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (M > 0x0) SaveInput1<type, true>(dst0 + 0x0 * F, d00, norm, bias, params);
                    if (M > 0x1) SaveInput1<type, true>(dst0 + 0x1 * F, d10, norm, bias, params);
                    if (M > 0x2) SaveInput1<type, true>(dst0 + 0x2 * F, d20, norm, bias, params);
                    if (M > 0x3) SaveInput1<type, true>(dst0 + 0x3 * F, d30, norm, bias, params);
                    if (M > 0x4) SaveInput1<type, true>(dst0 + 0x4 * F, d40, norm, bias, params);
                    if (M > 0x5) SaveInput1<type, true>(dst0 + 0x5 * F, d50, norm, bias, params);
                    if (M > 0x6) SaveInput1<type, true>(dst0 + 0x6 * F, d60, norm, bias, params);
                    if (M > 0x7) SaveInput1<type, true>(dst0 + 0x7 * F, d70, norm, bias, params);
                    if (M > 0x8) SaveInput1<type, true>(dst0 + 0x8 * F, d80, norm, bias, params);
                    if (M > 0x9) SaveInput1<type, true>(dst0 + 0x9 * F, d90, norm, bias, params);
                    if (M > 0xA) SaveInput1<type, true>(dst0 + 0xA * F, dA0, norm, bias, params);
                    if (M > 0xB) SaveInput1<type, true>(dst0 + 0xB * F, dB0, norm, bias, params);
                }
                else
                {
                    if (M > 0x0) SaveInput1<type, false>(dst0 + 0x0 * F, d00, norm, bias, params);
                    if (M > 0x1) SaveInput1<type, false>(dst0 + 0x1 * F, d10, norm, bias, params);
                    if (M > 0x2) SaveInput1<type, false>(dst0 + 0x2 * F, d20, norm, bias, params);
                    if (M > 0x3) SaveInput1<type, false>(dst0 + 0x3 * F, d30, norm, bias, params);
                    if (M > 0x4) SaveInput1<type, false>(dst0 + 0x4 * F, d40, norm, bias, params);
                    if (M > 0x5) SaveInput1<type, false>(dst0 + 0x5 * F, d50, norm, bias, params);
                    if (M > 0x6) SaveInput1<type, false>(dst0 + 0x6 * F, d60, norm, bias, params);
                    if (M > 0x7) SaveInput1<type, false>(dst0 + 0x7 * F, d70, norm, bias, params);
                    if (M > 0x8) SaveInput1<type, false>(dst0 + 0x8 * F, d80, norm, bias, params);
                    if (M > 0x9) SaveInput1<type, false>(dst0 + 0x9 * F, d90, norm, bias, params);
                    if (M > 0xA) SaveInput1<type, false>(dst0 + 0xA * F, dA0, norm, bias, params);
                    if (M > 0xB) SaveInput1<type, false>(dst0 + 0xB * F, dB0, norm, bias, params);
                }
            }
        }

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
            case 0xA: return InputConvolution_2xM<type, 0xA>;
            case 0xB: return InputConvolution_2xM<type, 0xB>;
            case 0xC: return InputConvolution_2xM<type, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution_2(const uint8_t* src, const ConvParam8i& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight, const float* norm, const float* bias, const float* params, float* dst)
        {
            size_t noseW = p.NoseW(), bodyW = p.BodyW(), tailW = p.dstW;
            size_t n = 12, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            size_t dstM = (a.bufH[1] - 1), dstS = a.bufH[1] * p.dstW * F;
            InputConvolution_2xM_Ptr inputConvolution_2x1 = GetInputConvolution_2x1<type>(p);
            InputConvolution_2xM_Ptr inputConvolution_2xN = GetInputConvolution_2xM<type>(n);
            InputConvolution_2xM_Ptr inputConvolution_2xM = GetInputConvolution_2xM<type>(m);
            __m512 _bias[2], _norm[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                _norm[1] = _mm512_loadu_ps(norm + dc + F);
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
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                    for (; dx < bodyWn; dx += n, dst0 += F * n, dst1 += F * n)
                        inputConvolution_2xN(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                    for (; dx < bodyW; dx += m, dst0 += F * m, dst1 += F * m)
                        inputConvolution_2xM(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                    for (; dx < tailW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                }
                dst += a.bufH[1] * p.dstW * DF;
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int M> void InputConvolution1x1_2xM(const uint8_t* src0, const ConvParam8i& p,
            const AlgParam& a, size_t dstC, const int8_t* weight, const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, s0, w0, w1;
            const uint8_t* src1 = src0 + 1 * p.srcC;
            const uint8_t* src2 = src0 + 2 * p.srcC;
            const uint8_t* src3 = src0 + 3 * p.srcC;
            const uint8_t* src4 = src0 + 4 * p.srcC;
            const uint8_t* src5 = src0 + 5 * p.srcC;
            __m512i upper = _mm512_set1_epi32(a.upper);
            if (dstC > F)
            {
                if (M > 0x0) d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                if (M > 0x1) d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();
                if (M > 0x2) d20 = _mm512_setzero_si512(), d21 = _mm512_setzero_si512();
                if (M > 0x3) d30 = _mm512_setzero_si512(), d31 = _mm512_setzero_si512();
                if (M > 0x4) d40 = _mm512_setzero_si512(), d41 = _mm512_setzero_si512();
                if (M > 0x5) d50 = _mm512_setzero_si512(), d51 = _mm512_setzero_si512();
                if (M > 0x6) d60 = _mm512_setzero_si512(), d61 = _mm512_setzero_si512();
                if (M > 0x7) d70 = _mm512_setzero_si512(), d71 = _mm512_setzero_si512();
                if (M > 0x8) d80 = _mm512_setzero_si512(), d81 = _mm512_setzero_si512();
                if (M > 0x9) d90 = _mm512_setzero_si512(), d91 = _mm512_setzero_si512();
                if (M > 0xA) dA0 = _mm512_setzero_si512(), dA1 = _mm512_setzero_si512();
                if (M > 0xB) dB0 = _mm512_setzero_si512(), dB1 = _mm512_setzero_si512();
                if (Base::Overflow(p.compatibility))
                {
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * p.srcC, end = p.srcC; offs0 < end; offs0 += 4, offs6 += 4)
                    {
                        w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                        w1 = _mm512_loadu_si512((__m512i*)weight + 1);
                        if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                        if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                        if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                        if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                        if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                        if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<true>(d50, s0, w0), Madd4<true>(d51, s0, w1);
                        if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<true>(d60, s0, w0), Madd4<true>(d61, s0, w1);
                        if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<true>(d70, s0, w0), Madd4<true>(d71, s0, w1);
                        if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<true>(d80, s0, w0), Madd4<true>(d81, s0, w1);
                        if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<true>(d90, s0, w0), Madd4<true>(d91, s0, w1);
                        if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<true>(dA0, s0, w0), Madd4<true>(dA1, s0, w1);
                        if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<true>(dB0, s0, w0), Madd4<true>(dB1, s0, w1);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * p.srcC, end = p.srcC; offs0 < end; offs0 += 4, offs6 += 4)
                    {
                        w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                        w1 = _mm512_loadu_si512((__m512i*)weight + 1);
                        if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                        if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                        if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                        if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                        if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                        if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<false>(d50, s0, w0), Madd4<false>(d51, s0, w1);
                        if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<false>(d60, s0, w0), Madd4<false>(d61, s0, w1);
                        if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<false>(d70, s0, w0), Madd4<false>(d71, s0, w1);
                        if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<false>(d80, s0, w0), Madd4<false>(d81, s0, w1);
                        if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<false>(d90, s0, w0), Madd4<false>(d91, s0, w1);
                        if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<false>(dA0, s0, w0), Madd4<false>(dA1, s0, w1);
                        if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<false>(dB0, s0, w0), Madd4<false>(dB1, s0, w1);
                        weight += DA;
                    }
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (M > 0x0) SaveInput2<type, true>(dst0 + 0x0 * F, dst1 + 0x0 * F, d00, d01, norm, bias, params);
                    if (M > 0x1) SaveInput2<type, true>(dst0 + 0x1 * F, dst1 + 0x1 * F, d10, d11, norm, bias, params);
                    if (M > 0x2) SaveInput2<type, true>(dst0 + 0x2 * F, dst1 + 0x2 * F, d20, d21, norm, bias, params);
                    if (M > 0x3) SaveInput2<type, true>(dst0 + 0x3 * F, dst1 + 0x3 * F, d30, d31, norm, bias, params);
                    if (M > 0x4) SaveInput2<type, true>(dst0 + 0x4 * F, dst1 + 0x4 * F, d40, d41, norm, bias, params);
                    if (M > 0x5) SaveInput2<type, true>(dst0 + 0x5 * F, dst1 + 0x5 * F, d50, d51, norm, bias, params);
                    if (M > 0x6) SaveInput2<type, true>(dst0 + 0x6 * F, dst1 + 0x6 * F, d60, d61, norm, bias, params);
                    if (M > 0x7) SaveInput2<type, true>(dst0 + 0x7 * F, dst1 + 0x7 * F, d70, d71, norm, bias, params);
                    if (M > 0x8) SaveInput2<type, true>(dst0 + 0x8 * F, dst1 + 0x8 * F, d80, d81, norm, bias, params);
                    if (M > 0x9) SaveInput2<type, true>(dst0 + 0x9 * F, dst1 + 0x9 * F, d90, d91, norm, bias, params);
                    if (M > 0xA) SaveInput2<type, true>(dst0 + 0xA * F, dst1 + 0xA * F, dA0, dA1, norm, bias, params);
                    if (M > 0xB) SaveInput2<type, true>(dst0 + 0xB * F, dst1 + 0xB * F, dB0, dB1, norm, bias, params);
                }
                else
                {
                    if (M > 0x0) SaveInput2<type, false>(dst0 + 0x0 * F, dst1 + 0x0 * F, d00, d01, norm, bias, params);
                    if (M > 0x1) SaveInput2<type, false>(dst0 + 0x1 * F, dst1 + 0x1 * F, d10, d11, norm, bias, params);
                    if (M > 0x2) SaveInput2<type, false>(dst0 + 0x2 * F, dst1 + 0x2 * F, d20, d21, norm, bias, params);
                    if (M > 0x3) SaveInput2<type, false>(dst0 + 0x3 * F, dst1 + 0x3 * F, d30, d31, norm, bias, params);
                    if (M > 0x4) SaveInput2<type, false>(dst0 + 0x4 * F, dst1 + 0x4 * F, d40, d41, norm, bias, params);
                    if (M > 0x5) SaveInput2<type, false>(dst0 + 0x5 * F, dst1 + 0x5 * F, d50, d51, norm, bias, params);
                    if (M > 0x6) SaveInput2<type, false>(dst0 + 0x6 * F, dst1 + 0x6 * F, d60, d61, norm, bias, params);
                    if (M > 0x7) SaveInput2<type, false>(dst0 + 0x7 * F, dst1 + 0x7 * F, d70, d71, norm, bias, params);
                    if (M > 0x8) SaveInput2<type, false>(dst0 + 0x8 * F, dst1 + 0x8 * F, d80, d81, norm, bias, params);
                    if (M > 0x9) SaveInput2<type, false>(dst0 + 0x9 * F, dst1 + 0x9 * F, d90, d91, norm, bias, params);
                    if (M > 0xA) SaveInput2<type, false>(dst0 + 0xA * F, dst1 + 0xA * F, dA0, dA1, norm, bias, params);
                    if (M > 0xB) SaveInput2<type, false>(dst0 + 0xB * F, dst1 + 0xB * F, dB0, dB1, norm, bias, params);
                }
            }
            else
            {
                if (M > 0x0) d00 = _mm512_setzero_si512();
                if (M > 0x1) d10 = _mm512_setzero_si512();
                if (M > 0x2) d20 = _mm512_setzero_si512();
                if (M > 0x3) d30 = _mm512_setzero_si512();
                if (M > 0x4) d40 = _mm512_setzero_si512();
                if (M > 0x5) d50 = _mm512_setzero_si512();
                if (M > 0x6) d60 = _mm512_setzero_si512();
                if (M > 0x7) d70 = _mm512_setzero_si512();
                if (M > 0x8) d80 = _mm512_setzero_si512();
                if (M > 0x9) d90 = _mm512_setzero_si512();
                if (M > 0xA) dA0 = _mm512_setzero_si512();
                if (M > 0xB) dB0 = _mm512_setzero_si512();
                if (Base::Overflow(p.compatibility))
                {
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * p.srcC, end = p.srcC; offs0 < end; offs0 += 4, offs6 += 4)
                    {
                        w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                        if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<true>(d00, s0, w0);
                        if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<true>(d10, s0, w0);
                        if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<true>(d20, s0, w0);
                        if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<true>(d30, s0, w0);
                        if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<true>(d40, s0, w0);
                        if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<true>(d50, s0, w0);
                        if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<true>(d60, s0, w0);
                        if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<true>(d70, s0, w0);
                        if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<true>(d80, s0, w0);
                        if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<true>(d90, s0, w0);
                        if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<true>(dA0, s0, w0);
                        if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<true>(dB0, s0, w0);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs0 = 0, offs6 = offs0 + 6 * p.srcC, end = p.srcC; offs0 < end; offs0 += 4, offs6 += 4)
                    {
                        w0 = _mm512_loadu_si512((__m512i*)weight + 0);
                        if (M > 0x0) s0 = Set4(src0 + offs0), Madd4<false>(d00, s0, w0);
                        if (M > 0x1) s0 = Set4(src1 + offs0), Madd4<false>(d10, s0, w0);
                        if (M > 0x2) s0 = Set4(src2 + offs0), Madd4<false>(d20, s0, w0);
                        if (M > 0x3) s0 = Set4(src3 + offs0), Madd4<false>(d30, s0, w0);
                        if (M > 0x4) s0 = Set4(src4 + offs0), Madd4<false>(d40, s0, w0);
                        if (M > 0x5) s0 = Set4(src5 + offs0), Madd4<false>(d50, s0, w0);
                        if (M > 0x6) s0 = Set4(src0 + offs6), Madd4<false>(d60, s0, w0);
                        if (M > 0x7) s0 = Set4(src1 + offs6), Madd4<false>(d70, s0, w0);
                        if (M > 0x8) s0 = Set4(src2 + offs6), Madd4<false>(d80, s0, w0);
                        if (M > 0x9) s0 = Set4(src3 + offs6), Madd4<false>(d90, s0, w0);
                        if (M > 0xA) s0 = Set4(src4 + offs6), Madd4<false>(dA0, s0, w0);
                        if (M > 0xB) s0 = Set4(src5 + offs6), Madd4<false>(dB0, s0, w0);
                        weight += DA;
                    }
                }
                if (Base::FmaAvoid(p.compatibility))
                {
                    if (M > 0x0) SaveInput1<type, true>(dst0 + 0x0 * F, d00, norm, bias, params);
                    if (M > 0x1) SaveInput1<type, true>(dst0 + 0x1 * F, d10, norm, bias, params);
                    if (M > 0x2) SaveInput1<type, true>(dst0 + 0x2 * F, d20, norm, bias, params);
                    if (M > 0x3) SaveInput1<type, true>(dst0 + 0x3 * F, d30, norm, bias, params);
                    if (M > 0x4) SaveInput1<type, true>(dst0 + 0x4 * F, d40, norm, bias, params);
                    if (M > 0x5) SaveInput1<type, true>(dst0 + 0x5 * F, d50, norm, bias, params);
                    if (M > 0x6) SaveInput1<type, true>(dst0 + 0x6 * F, d60, norm, bias, params);
                    if (M > 0x7) SaveInput1<type, true>(dst0 + 0x7 * F, d70, norm, bias, params);
                    if (M > 0x8) SaveInput1<type, true>(dst0 + 0x8 * F, d80, norm, bias, params);
                    if (M > 0x9) SaveInput1<type, true>(dst0 + 0x9 * F, d90, norm, bias, params);
                    if (M > 0xA) SaveInput1<type, true>(dst0 + 0xA * F, dA0, norm, bias, params);
                    if (M > 0xB) SaveInput1<type, true>(dst0 + 0xB * F, dB0, norm, bias, params);
                }
                else
                {
                    if (M > 0x0) SaveInput1<type, false>(dst0 + 0x0 * F, d00, norm, bias, params);
                    if (M > 0x1) SaveInput1<type, false>(dst0 + 0x1 * F, d10, norm, bias, params);
                    if (M > 0x2) SaveInput1<type, false>(dst0 + 0x2 * F, d20, norm, bias, params);
                    if (M > 0x3) SaveInput1<type, false>(dst0 + 0x3 * F, d30, norm, bias, params);
                    if (M > 0x4) SaveInput1<type, false>(dst0 + 0x4 * F, d40, norm, bias, params);
                    if (M > 0x5) SaveInput1<type, false>(dst0 + 0x5 * F, d50, norm, bias, params);
                    if (M > 0x6) SaveInput1<type, false>(dst0 + 0x6 * F, d60, norm, bias, params);
                    if (M > 0x7) SaveInput1<type, false>(dst0 + 0x7 * F, d70, norm, bias, params);
                    if (M > 0x8) SaveInput1<type, false>(dst0 + 0x8 * F, d80, norm, bias, params);
                    if (M > 0x9) SaveInput1<type, false>(dst0 + 0x9 * F, d90, norm, bias, params);
                    if (M > 0xA) SaveInput1<type, false>(dst0 + 0xA * F, dA0, norm, bias, params);
                    if (M > 0xB) SaveInput1<type, false>(dst0 + 0xB * F, dB0, norm, bias, params);
                }
            }
        }

        typedef void(*InputConvolution1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dstC,
            const int8_t* weight, const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1);

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
            case 0xA: return InputConvolution1x1_2xM<type, 0xA>;
            case 0xB: return InputConvolution1x1_2xM<type, 0xB>;
            case 0xC: return InputConvolution1x1_2xM<type, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2(const uint8_t* src, const ConvParam8i& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight, const float* norm, const float* bias, const float* params, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcM = a.bufH[0] - 1;
            __m512 _bias[2], _norm[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            if (a.bufH[0] == 0)
            {
                size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 12;
                size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
                size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xI = GetInputConvolution1x1_2xM<type>(i);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xE = GetInputConvolution1x1_2xM<type>(e);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                    _norm[1] = _mm512_loadu_ps(norm + dc + F);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    if (yInt > yBeg)
                    {
                        const uint8_t* src0 = src + yBeg * p.srcW * p.srcC;
                        float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < in; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                        if (in < i1)
                            inputConvolution1x1_2xI(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                    }
                    if (yEnd > yInt)
                    {
                        const uint8_t* src0 = src + yInt * p.srcW * p.srcC;
                        float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < en; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                        if (en < e1)
                            inputConvolution1x1_2xE(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 4) * DA;
                }
            }
            else
            {
                size_t n = 12, bodyW = p.dstW, bodyWn = AlignLoAny(bodyW, n), m = bodyW - bodyWn;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xM = GetInputConvolution1x1_2xM<type>(m);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                    _norm[1] = _mm512_loadu_ps(norm + dc + F);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    for (size_t dy = yBeg; dy < yEnd; dy++)
                    {
                        const uint8_t* src0 = src + (dy & srcM) * p.srcW * p.srcC;
                        float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        size_t dx = 0;
                        for (; dx < bodyWn; dx += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                        if (dx < bodyW)
                            inputConvolution1x1_2xM(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 4) * DA;
                }
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInput(const ConvParam8i& p, InputConvolutionPtr& input)
        {
            if (p.Is1x1())
                input = InputConvolution1x1_2<type>;
            else
                input = InputConvolution_2<type>;
        }

        void SetInput(const ConvParam8i& p, InputConvolutionPtr& input)
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
            case SimdConvolutionActivationSwish: SetInput<SimdConvolutionActivationSwish>(p, input); break;
            }
        }
    }
#endif
}
