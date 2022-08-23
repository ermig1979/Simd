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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx2
    {
        typedef Base::SynetConvolution32fBf16Nhwc::AlgParam AlgParam;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvertPtr Convert;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        void ConvolutionBf16NhwcConvertConv(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, size_t srcC, uint16_t* dst)
        {
            ptrdiff_t beg = yBeg * p.strideY - p.padY;
            ptrdiff_t end = (yEnd - 1) * p.strideY - p.padY + p.kernelY * p.dilationY;
            src += Max<ptrdiff_t>(0, beg) * p.srcW * p.srcC;
            size_t srcC16 = Simd::AlignLo(srcC, 16);
            size_t srcC8 = Simd::AlignLo(srcC, 8);
            size_t srcC4 = Simd::AlignLo(srcC, 4);
            size_t srcW = p.srcW + p.padX + p.padW;
            for (ptrdiff_t sy = beg; sy < end; ++sy)
            {
                if ((size_t)sy >= p.srcH)
                {
                    memset(dst, 0, srcW * srcC * 2);
                    dst += srcW * srcC;
                }
                else
                {
                    if (p.padX)
                    {
                        memset(dst, 0, p.padX * srcC * 2);
                        dst += p.padX * srcC;
                    }
                    if (p.srcC == srcC)
                    {
                        Float32ToBFloat16(src, srcC * p.srcW, dst);
                        src += srcC * p.srcW;
                        dst += srcC * p.srcW;
                    }
                    else
                    {
                        for (size_t sx = 0; sx < p.srcW; ++sx)
                        {
                            size_t sc = 0;
                            for (; sc < srcC16; sc += 16)
                            {
                                __m256i d0 = Float32ToBFloat16(_mm256_loadu_ps(src + sc + 0));
                                __m256i d1 = Float32ToBFloat16(_mm256_loadu_ps(src + sc + 8));
                                _mm256_storeu_si256((__m256i*)(dst + sc), _mm256_permute4x64_epi64(_mm256_packus_epi32(d0, d1), 0xD8));
                            }
                            for (; sc < srcC8; sc += 8)
                            {
                                __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + sc + 0));
                                __m128i d1 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + sc + 4));
                                _mm_storeu_si128((__m128i*)(dst + sc), _mm_packus_epi32(d0, d1));
                            }
                            for (; sc < srcC4; sc += 4)
                            {
                                __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + sc + 0));
                                _mm_storel_epi64((__m128i*)(dst + sc), _mm_packus_epi32(d0, Sse41::K_ZERO));
                            }
                            for (; sc < srcC; ++sc)
                                dst[sc] = Base::Float32ToBFloat16(src[sc]);
                            src += p.srcC;
                            dst += srcC;
                        }
                    }
                    if (p.padW)
                    {
                        memset(dst, 0, p.padW * srcC * 2);
                        dst += p.padW * srcC;
                    }
                }
            }
        }

        void ConvolutionBf16NhwcConvertGemm(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, size_t srcC, uint16_t* dst)
        {
            size_t srcCh2 = AlignHi(srcC, 2);
            size_t srcC16 = Simd::AlignLo(srcC, 16);
            size_t srcC8 = Simd::AlignLo(srcC, 8);
            size_t srcC4 = Simd::AlignLo(srcC, 4);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
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
                                    for (; sc < srcC16; sc += 16)
                                    {
                                        __m256i d0 = Float32ToBFloat16(_mm256_loadu_ps(ps + sc + 0));
                                        __m256i d1 = Float32ToBFloat16(_mm256_loadu_ps(ps + sc + 8));
                                        _mm256_storeu_si256((__m256i*)(dst + sc), _mm256_permute4x64_epi64(_mm256_packus_epi32(d0, d1), 0xD8));
                                    }
                                    for (; sc < srcC8; sc += 8)
                                    {
                                        __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(ps + sc + 0));
                                        __m128i d1 = Sse41::Float32ToBFloat16(_mm_loadu_ps(ps + sc + 4));
                                        _mm_storeu_si128((__m128i*)(dst + sc), _mm_packus_epi32(d0, d1));
                                    }
                                    for (; sc < srcC4; sc += 4)
                                    {
                                        __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(ps + sc + 0));
                                        _mm_storel_epi64((__m128i*)(dst + sc), _mm_packus_epi32(d0, Sse41::K_ZERO));
                                    }
                                    for (; sc < srcC; ++sc)
                                        dst[sc] = Base::Float32ToBFloat16(ps[sc]);
                                    if (sc < srcCh2)
                                        dst[sc] = 0;
                                    dst += srcCh2;
                                }
                                else
                                {
                                    memset(dst, 0, srcCh2 * 2);
                                    dst += srcCh2;
                                }
                            }
                        }
                        else
                        {
                            memset(dst, 0, p.kernelX * srcCh2 * 2);
                            dst += p.kernelX * srcCh2;
                        }
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionBf16NhwcConv_2xM(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11, m = _mm256_castsi256_ps(Bf16::MASK);
            size_t dS = srcC * p.strideX, dY = (p.srcW + p.padX + p.padW) * srcC * p.dilationY, dX = srcC * p.dilationX, dD = p.dstC, kY = p.kernelY, kX = p.kernelX;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            if (dstC > F)
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                    if (M > 4) d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0), d01 = _mm256_loadu_ps(dst + 0 * dD + F);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0), d11 = _mm256_loadu_ps(dst + 1 * dD + F);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0), d21 = _mm256_loadu_ps(dst + 2 * dD + F);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0), d31 = _mm256_loadu_ps(dst + 3 * dD + F);
                    if (M > 4) d40 = _mm256_loadu_ps(dst + 4 * dD + 0), d41 = _mm256_loadu_ps(dst + 4 * dD + F);
                }
                for (size_t ky = 0; ky < kY; ++ky)
                {
                    for (size_t kx = 0; kx < kX; ++kx)
                    {
                        for (size_t offs = ky * dY + kx * dX, end = offs + srcC; offs < end; offs += 2)
                        {
                            w01 = _mm256_loadu_ps((float*)weight + 0);
                            w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                            w01 = _mm256_and_ps(w01, m);
                            w11 = _mm256_loadu_ps((float*)weight + F);
                            w10 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w11), Base::Bf16::SHIFT));
                            w11 = _mm256_and_ps(w11, m);
                            if (M > 0)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 1)), m); 
                                d00 = _mm256_fmadd_ps(s0, w00, d00);
                                d01 = _mm256_fmadd_ps(s0, w10, d01);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 0)), m);
                                d00 = _mm256_fmadd_ps(s0, w01, d00);
                                d01 = _mm256_fmadd_ps(s0, w11, d01);
                            }
                            if (M > 1)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 1)), m);
                                d10 = _mm256_fmadd_ps(s0, w00, d10);
                                d11 = _mm256_fmadd_ps(s0, w10, d11);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 0)), m);
                                d10 = _mm256_fmadd_ps(s0, w01, d10);
                                d11 = _mm256_fmadd_ps(s0, w11, d11);
                            }
                            if (M > 2)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 1)), m);
                                d20 = _mm256_fmadd_ps(s0, w00, d20);
                                d21 = _mm256_fmadd_ps(s0, w10, d21);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 0)), m);
                                d20 = _mm256_fmadd_ps(s0, w01, d20);
                                d21 = _mm256_fmadd_ps(s0, w11, d21);
                            }
                            if (M > 3)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 1)), m);
                                d30 = _mm256_fmadd_ps(s0, w00, d30);
                                d31 = _mm256_fmadd_ps(s0, w10, d31);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 0)), m);
                                d30 = _mm256_fmadd_ps(s0, w01, d30);
                                d31 = _mm256_fmadd_ps(s0, w11, d31);
                            }
                            if (M > 4)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 1)), m);
                                d40 = _mm256_fmadd_ps(s0, w00, d40);
                                d41 = _mm256_fmadd_ps(s0, w10, d41);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 0)), m);
                                d40 = _mm256_fmadd_ps(s0, w01, d40);
                                d41 = _mm256_fmadd_ps(s0, w11, d41);
                            }
                            weight += QF;
                        }
                    }
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                    if (M > 4) Save2<term, type>(dst, d40, d41, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                    if (M > 4) Save2<term, type>(dst, d40, d41, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps();
                    if (M > 4) d40 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0);
                    if (M > 4) d40 = _mm256_loadu_ps(dst + 4 * dD + 0);
                }
                for (size_t ky = 0; ky < kY; ++ky)
                {
                    for (size_t kx = 0; kx < kX; ++kx)
                    {
                        for (size_t offs = ky * dY + kx * dX, end = offs + srcC; offs < end; offs += 2)
                        {
                            w01 = _mm256_loadu_ps((float*)weight + 0);
                            w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                            w01 = _mm256_and_ps(w01, m);
                            if (M > 0)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 1)), m);
                                d00 = _mm256_fmadd_ps(s0, w00, d00);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 0)), m);
                                d00 = _mm256_fmadd_ps(s0, w01, d00);
                            }
                            if (M > 1)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 1)), m);
                                d10 = _mm256_fmadd_ps(s0, w00, d10);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 0)), m);
                                d10 = _mm256_fmadd_ps(s0, w01, d10);
                            }
                            if (M > 2)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 1)), m);
                                d20 = _mm256_fmadd_ps(s0, w00, d20);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 0)), m);
                                d20 = _mm256_fmadd_ps(s0, w01, d20);
                            }
                            if (M > 3)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 1)), m);
                                d30 = _mm256_fmadd_ps(s0, w00, d30);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 0)), m);
                                d30 = _mm256_fmadd_ps(s0, w01, d30);
                            }
                            if (M > 4)
                            {
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 1)), m);
                                d40 = _mm256_fmadd_ps(s0, w00, d40);
                                s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 0)), m);
                                d40 = _mm256_fmadd_ps(s0, w01, d40);
                            }
                            weight += QF;
                        }
                    }
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params), dst += dD;
                    if (M > 4) Save1<term, type>(dst, d40, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                    if (M > 4) Save1<term, type>(dst, d40, bias, params, dstC), dst += dD;
                }
            }
        }

        typedef void(*ConvolutionBf16NhwcConv_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, size_t srcC, 
            size_t dstC, int zero, const uint16_t* weight, const __m256* bias, const __m256* params, float* dst);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionBf16NhwcConv_2xM_Ptr GetConvolutionBf16NhwcConv_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionBf16NhwcConv_2xM<term, type, 1>;
            case 2: return ConvolutionBf16NhwcConv_2xM<term, type, 2>;
            case 3: return ConvolutionBf16NhwcConv_2xM<term, type, 3>;
            case 4: return ConvolutionBf16NhwcConv_2xM<term, type, 4>;
            case 5: return ConvolutionBf16NhwcConv_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionBf16NhwcConv_2(const uint16_t* src, const ConvParam32f& p,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t n = 5, dstWn = AlignLoAny(p.dstW, n), m = p.dstW - dstWn;
            size_t dW = p.kernelY * p.kernelX * AlignHi(srcC, 2) * DF, dD = p.dstW * p.dstC;
            size_t dS = p.strideY * (p.srcW + p.padX + p.padW) * srcC;
            ConvolutionBf16NhwcConv_2xM_Ptr convolution_2xN = GetConvolutionBf16NhwcConv_2xM<term, type>(n);
            ConvolutionBf16NhwcConv_2xM_Ptr convolution_2xM = GetConvolutionBf16NhwcConv_2xM<term, type>(m);

            __m256 _params[2], _bias[2];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm256_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm256_loadu_ps(bias + dc + 0);
                _bias[1] = _mm256_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm256_loadu_ps(params + dc + 0);
                    _params[1] = _mm256_loadu_ps(params + dc + F);
                }
                for (size_t dy = 0; dy < dstH; dy++)
                {
                    float* d = dst + dy * dD;
                    const uint16_t* s = src + dy * dS;
                    size_t dx = 0;
                    for (; dx < dstWn; dx += n, d += n * p.dstC, s += n * p.strideX * srcC)
                        convolution_2xN(s, p, srcC, dC, zero, weight, _bias, _params, d);
                    for (; dx < p.dstW; dx += m, d += m * p.dstC, s += m * p.strideX * srcC)
                        convolution_2xM(s, p, srcC, dC, zero, weight, _bias, _params, d);
                }
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionBf16NhwcGemm_2xM(const uint16_t* src0, 
            const ConvParam32f& p, size_t srcC, size_t dstC, int zero, const uint16_t* weight, const __m256* bias, const __m256* params, float* dst)
        {
            __m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11, m = _mm256_castsi256_ps(Bf16::MASK);
            size_t dD = p.dstC;
            const uint16_t* src1 = src0 + 1 * srcC;
            const uint16_t* src2 = src0 + 2 * srcC;
            const uint16_t* src3 = src0 + 3 * srcC;
            const uint16_t* src4 = src0 + 4 * srcC;
            if (dstC > F)
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
                    if (M > 4) d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0), d01 = _mm256_loadu_ps(dst + 0 * dD + F);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0), d11 = _mm256_loadu_ps(dst + 1 * dD + F);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0), d21 = _mm256_loadu_ps(dst + 2 * dD + F);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0), d31 = _mm256_loadu_ps(dst + 3 * dD + F);
                    if (M > 4) d40 = _mm256_loadu_ps(dst + 4 * dD + 0), d41 = _mm256_loadu_ps(dst + 4 * dD + F);
                }
                for (size_t offs = 0; offs < srcC; offs += 2)
                {
                    w01 = _mm256_loadu_ps((float*)weight + 0);
                    w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                    w01 = _mm256_and_ps(w01, m);
                    w11 = _mm256_loadu_ps((float*)weight + F);
                    w10 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w11), Base::Bf16::SHIFT));
                    w11 = _mm256_and_ps(w11, m);
                    if (M > 0)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 1)), m);
                        d00 = _mm256_fmadd_ps(s0, w00, d00);
                        d01 = _mm256_fmadd_ps(s0, w10, d01);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 0)), m);
                        d00 = _mm256_fmadd_ps(s0, w01, d00);
                        d01 = _mm256_fmadd_ps(s0, w11, d01);
                    }
                    if (M > 1)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 1)), m);
                        d10 = _mm256_fmadd_ps(s0, w00, d10);
                        d11 = _mm256_fmadd_ps(s0, w10, d11);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 0)), m);
                        d10 = _mm256_fmadd_ps(s0, w01, d10);
                        d11 = _mm256_fmadd_ps(s0, w11, d11);
                    }
                    if (M > 2)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 1)), m);
                        d20 = _mm256_fmadd_ps(s0, w00, d20);
                        d21 = _mm256_fmadd_ps(s0, w10, d21);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 0)), m);
                        d20 = _mm256_fmadd_ps(s0, w01, d20);
                        d21 = _mm256_fmadd_ps(s0, w11, d21);
                    }
                    if (M > 3)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 1)), m);
                        d30 = _mm256_fmadd_ps(s0, w00, d30);
                        d31 = _mm256_fmadd_ps(s0, w10, d31);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 0)), m);
                        d30 = _mm256_fmadd_ps(s0, w01, d30);
                        d31 = _mm256_fmadd_ps(s0, w11, d31);
                    }
                    if (M > 4)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 1)), m);
                        d40 = _mm256_fmadd_ps(s0, w00, d40);
                        d41 = _mm256_fmadd_ps(s0, w10, d41);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 0)), m);
                        d40 = _mm256_fmadd_ps(s0, w01, d40);
                        d41 = _mm256_fmadd_ps(s0, w11, d41);
                    }
                    weight += QF;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params), dst += dD;
                    if (M > 4) Save2<term, type>(dst, d40, d41, bias, params), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                    if (M > 4) Save2<term, type>(dst, d40, d41, bias, params, dstC), dst += dD;
                }
            }
            else
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm256_setzero_ps();
                    if (M > 1) d10 = _mm256_setzero_ps();
                    if (M > 2) d20 = _mm256_setzero_ps();
                    if (M > 3) d30 = _mm256_setzero_ps();
                    if (M > 4) d40 = _mm256_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dD + 0);
                    if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dD + 0);
                    if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dD + 0);
                    if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dD + 0);
                    if (M > 4) d40 = _mm256_loadu_ps(dst + 4 * dD + 0);
                }
                for (size_t offs = 0; offs < srcC; offs += 2)
                {
                    w01 = _mm256_loadu_ps((float*)weight + 0);
                    w00 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(w01), Base::Bf16::SHIFT));
                    w01 = _mm256_and_ps(w01, m);
                    if (M > 0)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 1)), m);
                        d00 = _mm256_fmadd_ps(s0, w00, d00);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src0 + offs - 0)), m);
                        d00 = _mm256_fmadd_ps(s0, w01, d00);
                    }
                    if (M > 1)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 1)), m);
                        d10 = _mm256_fmadd_ps(s0, w00, d10);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src1 + offs - 0)), m);
                        d10 = _mm256_fmadd_ps(s0, w01, d10);
                    }
                    if (M > 2)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 1)), m);
                        d20 = _mm256_fmadd_ps(s0, w00, d20);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src2 + offs - 0)), m);
                        d20 = _mm256_fmadd_ps(s0, w01, d20);
                    }
                    if (M > 3)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 1)), m);
                        d30 = _mm256_fmadd_ps(s0, w00, d30);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src3 + offs - 0)), m);
                        d30 = _mm256_fmadd_ps(s0, w01, d30);
                    }
                    if (M > 4)
                    {
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 1)), m);
                        d40 = _mm256_fmadd_ps(s0, w00, d40);
                        s0 = _mm256_and_ps(_mm256_set1_ps(*(float*)(src4 + offs - 0)), m);
                        d40 = _mm256_fmadd_ps(s0, w01, d40);
                    }
                    weight += QF;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params), dst += dD;
                    if (M > 4) Save1<term, type>(dst, d40, bias, params), dst += dD;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                    if (M > 4) Save1<term, type>(dst, d40, bias, params, dstC), dst += dD;
                }
            }
        }

        typedef void(*ConvolutionBf16NhwcGemm_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, size_t srcC, 
            size_t dstC, int zero, const uint16_t* weight, const __m256* bias, const __m256* params, float* dst);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionBf16NhwcGemm_2xM_Ptr GetConvolutionBf16NhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionBf16NhwcGemm_2xM<term, type, 1>;
            case 2: return ConvolutionBf16NhwcGemm_2xM<term, type, 2>;
            case 3: return ConvolutionBf16NhwcGemm_2xM<term, type, 3>;
            case 4: return ConvolutionBf16NhwcGemm_2xM<term, type, 4>;
            case 5: return ConvolutionBf16NhwcGemm_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2(const uint16_t* src, const ConvParam32f& p,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t n1 = dstH * p.dstW, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = AlignHi(srcC, 2) * DF;
            ConvolutionBf16NhwcGemm_2xM_Ptr convolution_2xN = GetConvolutionBf16NhwcGemm_2xM<term, type>(n);
            ConvolutionBf16NhwcGemm_2xM_Ptr convolution_2xM = GetConvolutionBf16NhwcGemm_2xM<term, type>(m);

            __m256 _params[2], _bias[2];
            _params[0] = _mm256_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm256_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm256_loadu_ps(bias + dc + 0);
                _bias[1] = _mm256_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm256_loadu_ps(params + dc + 0);
                    _params[1] = _mm256_loadu_ps(params + dc + F);
                }
                float* d = dst;
                const uint16_t* s = src;
                size_t i = 0;
                for (; i < nn; i += n, s += n * srcC, d += n * p.dstC)
                    convolution_2xN(s, p, srcC, dC, zero, weight, _bias, _params, d);
                for (; i < n1; i += m, s += m * srcC, d += m * p.dstC)
                    convolution_2xM(s, p, srcC, dC, zero, weight, _bias, _params, d);
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam32f& p, const AlgParam& a, Convolution* convolutions)
        {
            if (p.Is1x1() || a.mode)
            {
                convolutions[TermLast] = ConvolutionBf16NhwcGemm_2<TermLast, type>;
                convolutions[TermInterim] = ConvolutionBf16NhwcGemm_2<TermInterim, type>;
            }
            else
            {
                convolutions[TermLast] = ConvolutionBf16NhwcConv_2<TermLast, type>;
                convolutions[TermInterim] = ConvolutionBf16NhwcConv_2<TermInterim, type>;
            }
        }

        SynetConvolution32fBf16Nhwc::SynetConvolution32fBf16Nhwc(const ConvParam32f & p)
            : Sse41::SynetConvolution32fBf16Nhwc(p)
        {
            SetAlgParam(F * 2, 5, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_alg.mode)
                _convert = ConvolutionBf16NhwcConvertGemm;
            else
                _convert = ConvolutionBf16NhwcConvertConv;
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
            default: assert(0);
            }
        }
    }
#endif
}
