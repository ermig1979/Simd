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
#include "Simd/SimdSse41.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse41
    {
        typedef Base::SynetConvolution32fBf16Nhwc::AlgParam AlgParam;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvertPtr Convert;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        void ConvolutionBf16NhwcDirectConvert(const float* src, const ConvParam32f& p, const AlgParam& a, size_t yBeg, size_t yEnd, size_t srcC, uint16_t* dst)
        {
            ptrdiff_t beg = yBeg * p.strideY - p.padY;
            ptrdiff_t end = (yEnd - 1) * p.strideY - p.padY + p.kernelY * p.dilationY;
            src += Max<ptrdiff_t>(0, beg) * p.srcW * p.srcC;
            for (ptrdiff_t sy = beg; sy < end; ++sy)
            {
                if ((size_t)sy >= p.srcH)
                {
                    memset(dst, 0, a.srcW * srcC * 2);
                    dst += a.srcW * srcC;
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
                            Float32ToBFloat16(src, srcC, dst);
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

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m128 SetBf16(uint16_t s)
        {
            return _mm_set1_ps(Base::BFloat16ToFloat32(s));
        }

        SIMD_INLINE void Madd1(__m128 & d, const __m128 & s0, const __m128& w0, const __m128& s1, const __m128& w1)
        {
            d = _mm_add_ps(_mm_mul_ps(s0, w0), d);
            d = _mm_add_ps(_mm_mul_ps(s1, w1), d);
        }

        SIMD_INLINE __m128 Get0(__m128i w)
        {
            return _mm_castsi128_ps(_mm_slli_epi32(w, Base::Bf16::SHIFT));
        }

        SIMD_INLINE __m128 Get1(__m128i w)
        {
            return _mm_castsi128_ps(_mm_and_si128(w, _mm_set1_epi32(Base::Bf16::MASK)));
        }

        SIMD_INLINE void Load1(const uint16_t * p, __m128 &w0, __m128 &w1)
        {
            __m128i w = _mm_loadu_si128((__m128i*)p);
            w0 = Get0(w);
            w1 = Get1(w);
        }

#define BF16_CONV_VER 1

#if BF16_CONV_VER == 2
        const size_t BF16_MICRO_C = 6;
#else
        const size_t BF16_MICRO_C = 5;
#endif

        template<TermType term, SimdConvolutionActivationType type, int M> void ConvolutionBf16NhwcDirect_2xM(const uint16_t* src0, const ConvParam32f& p,
            const AlgParam& a, size_t srcC, size_t dstC, int zero, const uint16_t* weight, const __m128* bias, const __m128* params, float* dst)
        {
#if BF16_CONV_VER == 2
            __m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1, m = _mm_castsi128_ps(Bf16::MASK);
#elif BF16_CONV_VER == 1
            __m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11, m = _mm_castsi128_ps(Bf16::MASK);
#else
            __m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, s1, w00, w01, w10, w11;
#endif
            size_t dS = srcC * p.strideX, dY = a.srcW * srcC * p.dilationY, dX = srcC * p.dilationX, dD = p.dstC, kY = p.kernelY, kX = p.kernelX;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
#if BF16_CONV_VER == 2
            const uint16_t* src5 = src0 + 5 * dS;
#endif
            if (dstC > F)
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
                    if (M > 1) d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
                    if (M > 2) d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
                    if (M > 3) d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
                    if (M > 4) d40 = _mm_setzero_ps(), d41 = _mm_setzero_ps();
#if BF16_CONV_VER == 2
                    if (M > 5) d50 = _mm_setzero_ps(), d51 = _mm_setzero_ps();
#endif
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_ps(dst + 0 * dD + 0), d01 = _mm_loadu_ps(dst + 0 * dD + F);
                    if (M > 1) d10 = _mm_loadu_ps(dst + 1 * dD + 0), d11 = _mm_loadu_ps(dst + 1 * dD + F);
                    if (M > 2) d20 = _mm_loadu_ps(dst + 2 * dD + 0), d21 = _mm_loadu_ps(dst + 2 * dD + F);
                    if (M > 3) d30 = _mm_loadu_ps(dst + 3 * dD + 0), d31 = _mm_loadu_ps(dst + 3 * dD + F);
                    if (M > 4) d40 = _mm_loadu_ps(dst + 4 * dD + 0), d41 = _mm_loadu_ps(dst + 4 * dD + F);
#if BF16_CONV_VER == 2
                    if (M > 5) d50 = _mm_loadu_ps(dst + 5 * dD + 0), d51 = _mm_loadu_ps(dst + 5 * dD + F);
#endif
                }
                for (size_t ky = 0; ky < kY; ++ky)
                {
                    for (size_t kx = 0; kx < kX; ++kx)
                    {
                        for (size_t offs = ky * dY + kx * dX, end = offs + srcC; offs < end; offs += 2)
                        {
#if BF16_CONV_VER == 2
                            w0 = _mm_and_ps(_mm_castsi128_ps(_mm_loadu_si128((__m128i*)(weight - 1) + 0)), m);
                            w1 = _mm_and_ps(_mm_castsi128_ps(_mm_loadu_si128((__m128i*)(weight - 1) + 1)), m);
                            if (M > 0) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m), d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00), d01 = _mm_add_ps(_mm_mul_ps(s0, w1), d01);
                            if (M > 1) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 1)), m), d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10), d11 = _mm_add_ps(_mm_mul_ps(s0, w1), d11);
                            if (M > 2) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 1)), m), d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20), d21 = _mm_add_ps(_mm_mul_ps(s0, w1), d21);
                            if (M > 3) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 1)), m), d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30), d31 = _mm_add_ps(_mm_mul_ps(s0, w1), d31);
                            if (M > 4) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 1)), m), d40 = _mm_add_ps(_mm_mul_ps(s0, w0), d40), d41 = _mm_add_ps(_mm_mul_ps(s0, w1), d41);
                            if (M > 5) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src5 + offs - 1)), m), d50 = _mm_add_ps(_mm_mul_ps(s0, w0), d50), d51 = _mm_add_ps(_mm_mul_ps(s0, w1), d51);
                            w0 = _mm_and_ps(_mm_castsi128_ps(_mm_loadu_si128((__m128i*)(weight - 0) + 0)), m);
                            w1 = _mm_and_ps(_mm_castsi128_ps(_mm_loadu_si128((__m128i*)(weight - 0) + 1)), m);
                            if (M > 0) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m), d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00), d01 = _mm_add_ps(_mm_mul_ps(s0, w1), d01);
                            if (M > 1) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 0)), m), d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10), d11 = _mm_add_ps(_mm_mul_ps(s0, w1), d11);
                            if (M > 2) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 0)), m), d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20), d21 = _mm_add_ps(_mm_mul_ps(s0, w1), d21);
                            if (M > 3) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 0)), m), d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30), d31 = _mm_add_ps(_mm_mul_ps(s0, w1), d31);
                            if (M > 4) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 0)), m), d40 = _mm_add_ps(_mm_mul_ps(s0, w0), d40), d41 = _mm_add_ps(_mm_mul_ps(s0, w1), d41);
                            if (M > 5) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src5 + offs - 0)), m), d50 = _mm_add_ps(_mm_mul_ps(s0, w0), d50), d51 = _mm_add_ps(_mm_mul_ps(s0, w1), d51);
#elif BF16_CONV_VER == 1
                            w01 = _mm_loadu_ps((float*)weight + 0);
                            w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                            w01 = _mm_and_ps(w01, m);
                            w11 = _mm_loadu_ps((float*)weight + F);
                            w10 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w11), Base::Bf16::SHIFT));
                            w11 = _mm_and_ps(w11, m);
                            if (M > 0)
                            {
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m); 
                                d00 = _mm_add_ps(_mm_mul_ps(s0, w00), d00);
                                d01 = _mm_add_ps(_mm_mul_ps(s0, w10), d01);
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m);
                                d00 = _mm_add_ps(_mm_mul_ps(s0, w01), d00);
                                d01 = _mm_add_ps(_mm_mul_ps(s0, w11), d01);
                            }
                            if (M > 1)
                            {
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 1)), m);
                                d10 = _mm_add_ps(_mm_mul_ps(s0, w00), d10);
                                d11 = _mm_add_ps(_mm_mul_ps(s0, w10), d11);
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 0)), m);
                                d10 = _mm_add_ps(_mm_mul_ps(s0, w01), d10);
                                d11 = _mm_add_ps(_mm_mul_ps(s0, w11), d11);
                            }
                            if (M > 2)
                            {
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 1)), m);
                                d20 = _mm_add_ps(_mm_mul_ps(s0, w00), d20);
                                d21 = _mm_add_ps(_mm_mul_ps(s0, w10), d21);
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 0)), m);
                                d20 = _mm_add_ps(_mm_mul_ps(s0, w01), d20);
                                d21 = _mm_add_ps(_mm_mul_ps(s0, w11), d21);
                            }
                            if (M > 3)
                            {
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 1)), m);
                                d30 = _mm_add_ps(_mm_mul_ps(s0, w00), d30);
                                d31 = _mm_add_ps(_mm_mul_ps(s0, w10), d31);
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 0)), m);
                                d30 = _mm_add_ps(_mm_mul_ps(s0, w01), d30);
                                d31 = _mm_add_ps(_mm_mul_ps(s0, w11), d31);
                            }
                            if (M > 4)
                            {
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 1)), m);
                                d40 = _mm_add_ps(_mm_mul_ps(s0, w00), d40);
                                d41 = _mm_add_ps(_mm_mul_ps(s0, w10), d41);
                                s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 0)), m);
                                d40 = _mm_add_ps(_mm_mul_ps(s0, w01), d40);
                                d41 = _mm_add_ps(_mm_mul_ps(s0, w11), d41);
                            }
#else
                            Load1(weight + 0 * DF, w00, w01);
                            Load1(weight + 1 * DF, w10, w11);
                            if (M > 0) s0 = SetBf16(src0[offs + 0]), s1 = SetBf16(src0[offs + 1]), Madd1(d00, s0, w00, s1, w01), Madd1(d01, s0, w10, s1, w11);
                            if (M > 1) s0 = SetBf16(src1[offs + 0]), s1 = SetBf16(src1[offs + 1]), Madd1(d10, s0, w00, s1, w01), Madd1(d11, s0, w10, s1, w11);
                            if (M > 2) s0 = SetBf16(src2[offs + 0]), s1 = SetBf16(src2[offs + 1]), Madd1(d20, s0, w00, s1, w01), Madd1(d21, s0, w10, s1, w11);
                            if (M > 3) s0 = SetBf16(src3[offs + 0]), s1 = SetBf16(src3[offs + 1]), Madd1(d30, s0, w00, s1, w01), Madd1(d31, s0, w10, s1, w11);
                            if (M > 4) s0 = SetBf16(src4[offs + 0]), s1 = SetBf16(src4[offs + 1]), Madd1(d40, s0, w00, s1, w01), Madd1(d41, s0, w10, s1, w11);
#endif
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
#if BF16_CONV_VER == 2
                    if (M > 5) Save2<term, type>(dst, d50, d51, bias, params), dst += dD;
#endif
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, d00, d01, bias, params, dstC), dst += dD;
                    if (M > 1) Save2<term, type>(dst, d10, d11, bias, params, dstC), dst += dD;
                    if (M > 2) Save2<term, type>(dst, d20, d21, bias, params, dstC), dst += dD;
                    if (M > 3) Save2<term, type>(dst, d30, d31, bias, params, dstC), dst += dD;
                    if (M > 4) Save2<term, type>(dst, d40, d41, bias, params, dstC), dst += dD;
#if BF16_CONV_VER == 2
                    if (M > 5) Save2<term, type>(dst, d50, d51, bias, params, dstC), dst += dD;
#endif
                }
            }
            else
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm_setzero_ps();
                    if (M > 1) d10 = _mm_setzero_ps();
                    if (M > 2) d20 = _mm_setzero_ps();
                    if (M > 3) d30 = _mm_setzero_ps();
                    if (M > 4) d40 = _mm_setzero_ps();
#if BF16_CONV_VER == 2
                    if (M > 5) d50 = _mm_setzero_ps();
#endif
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_ps(dst + 0 * dD + 0);
                    if (M > 1) d10 = _mm_loadu_ps(dst + 1 * dD + 0);
                    if (M > 2) d20 = _mm_loadu_ps(dst + 2 * dD + 0);
                    if (M > 3) d30 = _mm_loadu_ps(dst + 3 * dD + 0);
                    if (M > 4) d40 = _mm_loadu_ps(dst + 4 * dD + 0);
#if BF16_CONV_VER == 2
                    if (M > 5) d50 = _mm_loadu_ps(dst + 5 * dD + 0);
#endif
                }
                for (size_t ky = 0; ky < kY; ++ky)
                {
                    for (size_t kx = 0; kx < kX; ++kx)
                    {
                        for (size_t offs = ky * dY + kx * dX, end = offs + srcC; offs < end; offs += 2)
                        {
#if BF16_CONV_VER == 2
                            w0 = _mm_and_ps(_mm_castsi128_ps(_mm_loadu_si128((__m128i*)(weight - 1) + 0)), m);
                            if (M > 0) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m), d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
                            if (M > 1) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 1)), m), d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
                            if (M > 2) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 1)), m), d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
                            if (M > 3) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 1)), m), d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30);
                            if (M > 4) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 1)), m), d40 = _mm_add_ps(_mm_mul_ps(s0, w0), d40);
                            if (M > 5) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src5 + offs - 1)), m), d50 = _mm_add_ps(_mm_mul_ps(s0, w0), d50);
                            w0 = _mm_and_ps(_mm_castsi128_ps(_mm_loadu_si128((__m128i*)(weight - 0) + 0)), m);
                            if (M > 0) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m), d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
                            if (M > 1) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 0)), m), d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
                            if (M > 2) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 0)), m), d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
                            if (M > 3) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 0)), m), d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30);
                            if (M > 4) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 0)), m), d40 = _mm_add_ps(_mm_mul_ps(s0, w0), d40);
                            if (M > 5) s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src5 + offs - 0)), m), d50 = _mm_add_ps(_mm_mul_ps(s0, w0), d50);
#elif BF16_CONV_VER == 1
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
#else
                            Load1(weight + 0 * DF, w00, w01);
                            if (M > 0) s0 = SetBf16(src0[offs + 0]), Madd1(d00, s0, w00, s1, w01);
                            if (M > 1) s0 = SetBf16(src1[offs + 0]), Madd1(d10, s0, w00, s1, w01);
                            if (M > 2) s0 = SetBf16(src2[offs + 0]), Madd1(d20, s0, w00, s1, w01);
                            if (M > 3) s0 = SetBf16(src3[offs + 0]), Madd1(d30, s0, w00, s1, w01);
                            if (M > 4) s0 = SetBf16(src4[offs + 0]), Madd1(d40, s0, w00, s1, w01);
#endif
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
#if BF16_CONV_VER == 2
                    if (M > 5) Save1<term, type>(dst, d50, bias, params), dst += dD;
#endif
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, d00, bias, params, dstC), dst += dD;
                    if (M > 1) Save1<term, type>(dst, d10, bias, params, dstC), dst += dD;
                    if (M > 2) Save1<term, type>(dst, d20, bias, params, dstC), dst += dD;
                    if (M > 3) Save1<term, type>(dst, d30, bias, params, dstC), dst += dD;
                    if (M > 4) Save1<term, type>(dst, d40, bias, params, dstC), dst += dD;
#if BF16_CONV_VER == 2
                    if (M > 5) Save1<term, type>(dst, d50, bias, params, dstC), dst += dD;
#endif
                }
            }
        }

        typedef void(*ConvolutionBf16NhwcDirect_2xM_Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a, 
            size_t srcC, size_t dstC, int zero, const uint16_t* weight, const __m128* bias, const __m128* params, float* dst);

        template<TermType term, SimdConvolutionActivationType type> ConvolutionBf16NhwcDirect_2xM_Ptr GetConvolutionBf16NhwcDirect_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return ConvolutionBf16NhwcDirect_2xM<term, type, 1>;
            case 2: return ConvolutionBf16NhwcDirect_2xM<term, type, 2>;
            case 3: return ConvolutionBf16NhwcDirect_2xM<term, type, 3>;
            case 4: return ConvolutionBf16NhwcDirect_2xM<term, type, 4>;
            case 5: return ConvolutionBf16NhwcDirect_2xM<term, type, 5>;
            case 6: return ConvolutionBf16NhwcDirect_2xM<term, type, 6>;
            }
            assert(0);
            return NULL;
        }

        template<TermType term, SimdConvolutionActivationType type> void ConvolutionBf16NhwcDirect_2(const uint16_t* src, const ConvParam32f& p, 
            const AlgParam& a, size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t n = BF16_MICRO_C;
            size_t dstWn = AlignLoAny(p.dstW, n);
            size_t m = p.dstW - dstWn;
            ConvolutionBf16NhwcDirect_2xM_Ptr convolution_2xN = GetConvolutionBf16NhwcDirect_2xM<term, type>(n);
            ConvolutionBf16NhwcDirect_2xM_Ptr convolution_2xM = GetConvolutionBf16NhwcDirect_2xM<term, type>(m);

            __m128 _params[2], _bias[2];
            _params[0] = _mm_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm_loadu_ps(bias + dc + 0);
                _bias[1] = _mm_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + dc + 0);
                    _params[1] = _mm_loadu_ps(params + dc + F);
                }
                for (size_t dy = 0; dy < dstH; dy++)
                {
                    float* d = dst + dy * p.dstW * p.dstC;
                    const uint16_t* s = src + dy * p.strideY * a.srcW * srcC;
                    size_t dx = 0;
                    for (; dx < dstWn; dx += n, d += n * p.dstC, s += n * p.strideX * srcC)
                        convolution_2xN(s, p, a, srcC, dC, zero, weight, _bias, _params, d);
                    for (; dx < p.dstW; dx += m, d += m * p.dstC, s += m * p.strideX * srcC)
                        convolution_2xM(s, p, a, srcC, dC, zero, weight, _bias, _params, d);
                }
                weight += p.kernelY * p.kernelX * srcC * DF;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam32f& p, const AlgParam& a, Convert& convert, Convolution* convolutions)
        {
            //if (p.Is1x1())
            //    convolution = ConvolutionNhwcDirect1x1_2<type>;
            //else
            {
                convert = ConvolutionBf16NhwcDirectConvert;
                convolutions[TermLast] = ConvolutionBf16NhwcDirect_2<TermLast, type>;
                convolutions[TermInterim] = ConvolutionBf16NhwcDirect_2<TermInterim, type>;
            }
        }

        SynetConvolution32fBf16Nhwc::SynetConvolution32fBf16Nhwc(const ConvParam32f & p)
            : Base::SynetConvolution32fBf16Nhwc(p)
        {
            SetAlgParam(F * 2, BF16_MICRO_C, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            const AlgParam& a = _alg;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, a, _convert, _convolutions); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, a, _convert, _convolutions); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, a, _convert, _convolutions); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, a, _convert, _convolutions); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, a, _convert, _convolutions); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, a, _convert, _convolutions); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, a, _convert, _convolutions); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, a, _convert, _convolutions); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, a, _convert, _convolutions); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, a, _convert, _convolutions); break;
            default: assert(0);
            }
        }
    }
#endif
}
