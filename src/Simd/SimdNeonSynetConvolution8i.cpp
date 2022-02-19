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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Neon
    {
        using AlgParam = SynetConvolution8iNhwcDirect::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDirect::ConvolutionPtr;

        SIMD_INLINE uint8x16_t Set4(const uint8_t* src)
        {
            return (uint8x16_t)vdupq_n_s32(*(int32_t*)src);
        }

        template<bool overflow> void Madd4(int32x4_t & i32, uint8x16_t u8, int8x16_t i8);

        template<> SIMD_INLINE void Madd4<true>(int32x4_t& i32, uint8x16_t u8, int8x16_t i8)
        {
            int32x4_t lo = vmaxq_s32(vminq_s32(vpaddlq_s16(vmulq_s16(UnpackU8s<0>(u8), UnpackI8<0>(i8))), vdupq_n_s32(SHRT_MAX)), vdupq_n_s32(SHRT_MIN));
            int32x4_t hi = vmaxq_s32(vminq_s32(vpaddlq_s16(vmulq_s16(UnpackU8s<1>(u8), UnpackI8<1>(i8))), vdupq_n_s32(SHRT_MAX)), vdupq_n_s32(SHRT_MIN));
#if defined(__aarch64__)
            int32x4_t sum = vpaddq_s32(lo, hi);
#else
            int32x4_t sum = vcombine_s32(
                vpadd_s32(Half<0>(lo), Half<1>(lo)),
                vpadd_s32(Half<0>(hi), Half<1>(hi)));
#endif
            i32 = vaddq_s32(i32, sum);
        }

        template<> SIMD_INLINE void Madd4<false>(int32x4_t& i32, uint8x16_t u8, int8x16_t i8)
        {
            int32x4_t lo = vpaddlq_s16(vmulq_s16(UnpackU8s<0>(u8), UnpackI8<0>(i8)));
            int32x4_t hi = vpaddlq_s16(vmulq_s16(UnpackU8s<1>(u8), UnpackI8<1>(i8)));
#if defined(__aarch64__)
            int32x4_t sum = vpaddq_s32(lo, hi);
#else
            int32x4_t sum = vcombine_s32(
                vpadd_s32(Half<0>(lo), Half<1>(lo)),
                vpadd_s32(Half<0>(hi), Half<1>(hi)));
#endif
            i32 = vaddq_s32(i32, sum);
        }

        inline void pdpbusd(int32x4_t& sum, uint8x16_t input, int8x16_t weight)
        {
            for (size_t i = 0; i < 4; ++i)
                for (size_t j = 0; j < 4; ++j)
                    sum[i] += int32_t(input[i * 4 + j]) * int32_t(weight[i * 4 + j]);
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x1(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t* weight0, const float32x4_t* norm, 
            const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, int32_t* buf, uint8_t* dst, int first)
        {
            int32x4_t d00, d01;
            uint8x16_t s0;
            int8x16_t w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(srcC, 4) * A;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            uint8x8_t upper = vdup_n_u8(a.upper);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                if (first)
                    d00 = vdupq_n_s32(0), d01 = vdupq_n_s32(0);
                else
                    d00 = Load<false>(buf + 0), d01 = Load<false>(buf + F);
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = Load<false>(weight0);
                                w1 = Load<false>(weight1);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d01, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = (uint8x16_t)vdupq_n_s32(a.zero);
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = Load<false>(weight0);
                                    w1 = Load<false>(weight1);
                                    Madd4<overflow>(d00, s0, w0);
                                    Madd4<overflow>(d01, s0, w1);
                                    weight0 += A, weight1 += A;
                                }
                            }
                            else
                                weight0 += dWz, weight1 += dWz;
                        }
                        weight0 += dW, weight1 += dW;
                    }
                }
                if (dstC == DF)
                    Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper);
                else
                    Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F);
            }
            else
            {
                if (first)
                    d00 = vdupq_n_s32(0);
                else
                    d00 = Load<false>(buf + 0);
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = Load<false>(weight0);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                weight0 += A;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = (uint8x16_t)vdupq_n_s32(a.zero);
                                for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                                {
                                    w0 = Load<false>(weight0);
                                    Madd4<overflow>(d00, s0, w0);
                                    weight0 += A;
                                }
                            }
                            else
                                weight0 += dWz;
                        }
                        weight0 += dW;
                    }
                }
                if (dstC == F)
                    Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper);
                else
                    Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC);
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_2xM(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t* weight0, const float32x4_t* norm,
            const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, int32_t* buf, uint8_t* dst, int first)
        {
            int32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            uint8x16_t s0;
            int8x16_t w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, dWz = (DivHi(srcC, 4) * A + dW) * p.kernelX;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            uint8x8_t upper = vdup_n_u8(a.upper);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_s32(0), d01 = vdupq_n_s32(0);
                    if (M > 1) d10 = vdupq_n_s32(0), d11 = vdupq_n_s32(0);
                    if (M > 2) d20 = vdupq_n_s32(0), d21 = vdupq_n_s32(0);
                    if (M > 3) d30 = vdupq_n_s32(0), d31 = vdupq_n_s32(0);
                    if (M > 4) d40 = vdupq_n_s32(0), d41 = vdupq_n_s32(0);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(buf + 0 * dB + 0), d01 = Load<false>(buf + 0 * dB + F);
                    if (M > 1) d10 = Load<false>(buf + 1 * dB + 0), d11 = Load<false>(buf + 1 * dB + F);
                    if (M > 2) d20 = Load<false>(buf + 2 * dB + 0), d21 = Load<false>(buf + 2 * dB + F);
                    if (M > 3) d30 = Load<false>(buf + 3 * dB + 0), d31 = Load<false>(buf + 3 * dB + F);
                    if (M > 4) d40 = Load<false>(buf + 4 * dB + 0), d41 = Load<false>(buf + 4 * dB + F);
                }
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + M <= p.srcW);
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = Load<false>(weight0);
                                w1 = Load<false>(weight1);
                                if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                                if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                                if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                                if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                                if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else if (a.zero)
                    {
                        s0 = (uint8x16_t)vdupq_n_s32(a.zero);
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                            {
                                w0 = Load<false>(weight0);
                                w1 = Load<false>(weight1);
                                if (M > 0) Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                                if (M > 1) Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                                if (M > 2) Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                                if (M > 3) Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                                if (M > 4) Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                                weight0 += A, weight1 += A;
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                        weight0 += dWz, weight1 += dWz;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                }
            }
            else
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_s32(0);
                    if (M > 1) d10 = vdupq_n_s32(0);
                    if (M > 2) d20 = vdupq_n_s32(0);
                    if (M > 3) d30 = vdupq_n_s32(0);
                    if (M > 4) d40 = vdupq_n_s32(0);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(buf + 0 * dB + 0);
                    if (M > 1) d10 = Load<false>(buf + 1 * dB + 0);
                    if (M > 2) d20 = Load<false>(buf + 2 * dB + 0);
                    if (M > 3) d30 = Load<false>(buf + 3 * dB + 0);
                    if (M > 4) d40 = Load<false>(buf + 4 * dB + 0);
                }
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            assert(sx + kx < p.srcW && sx + kx + M <= p.srcW);
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = Load<false>(weight0);
                                if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0);
                                if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0);
                                if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0);
                                if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0);
                                if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0);
                                weight0 += A;
                            }
                            weight0 += dW;
                        }
                    }
                    else if (a.zero)
                    {
                        s0 = (uint8x16_t)vdupq_n_s32(a.zero);
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                            {
                                w0 = Load<false>(weight0);
                                if (M > 0) Madd4<overflow>(d00, s0, w0);
                                if (M > 1) Madd4<overflow>(d10, s0, w0);
                                if (M > 2) Madd4<overflow>(d20, s0, w0);
                                if (M > 3) Madd4<overflow>(d30, s0, w0);
                                if (M > 4) Madd4<overflow>(d40, s0, w0);
                                weight0 += A;
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += dWz;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*ConvolutionNhwcDirect_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC,
            const int8_t* weight0, const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, int32_t* buf, uint8_t* dst, int first);

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_2xM_Ptr GetConvolutionNhwcDirect_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect_2xM<overflow, term, type, 1>;
            case 2: return ConvolutionNhwcDirect_2xM<overflow, term, type, 2>;
            case 3: return ConvolutionNhwcDirect_2xM<overflow, term, type, 3>;
            case 4: return ConvolutionNhwcDirect_2xM<overflow, term, type, 4>;
            case 5: return ConvolutionNhwcDirect_2xM<overflow, term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const uint8_t* src,
            const ConvParam8i& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            size_t n = 5, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2x5 = GetConvolutionNhwcDirect_2xM<overflow, term, type>(5);
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<overflow, term, type>(m);
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;
            float32x4_t _norm[2], _bias[2], _params[2], _scale[2], _shift[2];
            _params[0] = vdupq_n_f32(params[0]);
            _params[1] = vdupq_n_f32(params[1]);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _norm[0] = Load<false>(norm + dc + 0);
                _norm[1] = Load<false>(norm + dc + F);
                _bias[0] = Load<false>(bias + dc + 0);
                _bias[1] = Load<false>(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = Load<false>(params + dc + 0);
                    _params[1] = Load<false>(params + dc + F);
                }
                _scale[0] = Load<false>(scale + dc + 0);
                _scale[1] = Load<false>(scale + dc + F);
                _shift[0] = Load<false>(shift + dc + 0);
                _shift[1] = Load<false>(shift + dc + F);

                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                    for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                        convolutionNhwcDirect_2x5(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                    for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                }
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0, const float32x4_t* norm,
            const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, int32_t* buf, uint8_t* dst, int first)
        {
            int32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41;
            uint8x16_t s0;
            int8x16_t w0, w1;
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            uint8x8_t upper = vdup_n_u8(a.upper);
            if (dstC > F)
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_s32(0), d01 = vdupq_n_s32(0);
                    if (M > 1) d10 = vdupq_n_s32(0), d11 = vdupq_n_s32(0);
                    if (M > 2) d20 = vdupq_n_s32(0), d21 = vdupq_n_s32(0);
                    if (M > 3) d30 = vdupq_n_s32(0), d31 = vdupq_n_s32(0);
                    if (M > 4) d40 = vdupq_n_s32(0), d41 = vdupq_n_s32(0);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(buf + 0 * dB + 0), d01 = Load<false>(buf + 0 * dB + F);
                    if (M > 1) d10 = Load<false>(buf + 1 * dB + 0), d11 = Load<false>(buf + 1 * dB + F);
                    if (M > 2) d20 = Load<false>(buf + 2 * dB + 0), d21 = Load<false>(buf + 2 * dB + F);
                    if (M > 3) d30 = Load<false>(buf + 3 * dB + 0), d31 = Load<false>(buf + 3 * dB + F);
                    if (M > 4) d40 = Load<false>(buf + 4 * dB + 0), d41 = Load<false>(buf + 4 * dB + F);
                }
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = Load<false>(weight0);
                    w1 = Load<false>(weight1);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0), Madd4<overflow>(d01, s0, w1);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0), Madd4<overflow>(d11, s0, w1);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0), Madd4<overflow>(d21, s0, w1);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0), Madd4<overflow>(d31, s0, w1);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0), Madd4<overflow>(d41, s0, w1);
                    weight0 += A, weight1 += A;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                }
            }
            else
            {
                if (first)
                {
                    if (M > 0) d00 = vdupq_n_s32(0);
                    if (M > 1) d10 = vdupq_n_s32(0);
                    if (M > 2) d20 = vdupq_n_s32(0);
                    if (M > 3) d30 = vdupq_n_s32(0);
                    if (M > 4) d40 = vdupq_n_s32(0);
                }
                else
                {
                    if (M > 0) d00 = Load<false>(buf + 0 * dB + 0);
                    if (M > 1) d10 = Load<false>(buf + 1 * dB + 0);
                    if (M > 2) d20 = Load<false>(buf + 2 * dB + 0);
                    if (M > 3) d30 = Load<false>(buf + 3 * dB + 0);
                    if (M > 4) d40 = Load<false>(buf + 4 * dB + 0);
                }
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = Load<false>(weight0);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<overflow>(d00, s0, w0);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<overflow>(d10, s0, w0);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<overflow>(d20, s0, w0);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<overflow>(d30, s0, w0);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<overflow>(d40, s0, w0);
                    weight0 += A;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0, 
            const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, int32_t* buf, uint8_t* dst, int first);

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 1>;
            case 2: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 2>;
            case 3: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 3>;
            case 4: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 4>;
            case 5: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const uint8_t* src,
            const ConvParam8i& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            size_t n1 = (yEnd - yBeg) * p.dstW, n5 = AlignLoAny(n1, 5), m = n1 - n5;
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2x5 = GetConvolutionNhwcDirect1x1_2xM<overflow, term, type>(5);
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<overflow, term, type>(m);
            float32x4_t _norm[2], _bias[2], _params[2], _scale[2], _shift[2];
            _params[0] = vdupq_n_f32(params[0]);
            _params[1] = vdupq_n_f32(params[1]);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _norm[0] = Load<false>(norm + dc + 0);
                _norm[1] = Load<false>(norm + dc + F);
                _bias[0] = Load<false>(bias + dc + 0);
                _bias[1] = Load<false>(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = Load<false>(params + dc + 0);
                    _params[1] = Load<false>(params + dc + F);
                }
                _scale[0] = Load<false>(scale + dc + 0);
                _scale[1] = Load<false>(scale + dc + F);
                _shift[0] = Load<false>(shift + dc + 0);
                _shift[1] = Load<false>(shift + dc + F);
                const uint8_t* s = src + yBeg * p.srcW * p.srcC;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < n5; i += 5, s += p.srcC * 5, b += p.dstC * 5, d += p.dstC * a.size * 5)
                    convolutionNhwcDirect1x1_2x5(s, p, a, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                for (; i < n1; i += m, s += p.srcC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                    convolutionNhwcDirect1x1_2xM(s, p, a, srcC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                weight += DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template <bool overflow, Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            if (p.Is1x1())
            {
                switch (a.microD)
                {
                case 2 * F: d[term] = ConvolutionNhwcDirect1x1_2<overflow, term, activation>; break;
                default:
                    assert(0);
                }
            }
            else
            {
                switch (a.microD)
                {
                case 2 * F: d[term] = ConvolutionNhwcDirect_2<overflow, term, activation>; break;
                default:
                    assert(0);
                }
            }
        }

        template<Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            if (Base::Overflow(p.compatibility))
                Set<true, term, activation>(p, a, d);
            else
                Set<false, term, activation>(p, a, d);
        }

        template<SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            Set<Term8iLast8u, activation>(p, a, d);
            Set<Term8iLast32f, activation>(p, a, d);
            Set<Term8iInterim, SimdConvolutionActivationIdentity>(p, a, d);
        }

        static void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, a, d); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, a, d); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, a, d); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, a, d); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, a, d); break;
            default: assert(0);
            }
        }

        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam8i& p)
            : Base::SynetConvolution8iNhwcDirect(p)
        {
            SetAlgParam(F, 2 * F, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), 5);
            Set(p, _alg, _convolutions);
            _convertSrc = Neon::SynetConvert32fTo8u;
        }

        bool SynetConvolution8iNhwcDirect::Preferable(const ConvParam8i& p)
        {
            if (p.trans != SimdTrue || p.group != 1)
                return false;
            return true;
        }

        //---------------------------------------------------------------------

        void* SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam8i param(batch, conv, compatibility);
            if (!param.Valid())
                return NULL;
            else if (SynetConvolution8iNhwcDirect::Preferable(param))
                return new SynetConvolution8iNhwcDirect(param);
            else
                return new Base::SynetConvolution8iGemmNN(param);
        }
    }
#endif
}
