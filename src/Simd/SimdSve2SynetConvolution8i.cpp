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

#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        using AlgParam = SynetConvolution8iNhwcDirect::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDirect::ConvolutionPtr;

        SIMD_INLINE svuint8_t Set4(uint32_t value)
        {
            return svreinterpret_u8_u32(svdup_n_u32(value));
        }

        SIMD_INLINE svuint8_t Set4(const uint8_t* src)
        {
            uint32_t value = 0;
            memcpy(&value, src, sizeof(value));
            return Set4(value);
        }

        SIMD_INLINE svuint8_t Set4(const uint8_t* src, size_t tail, uint8_t zero)
        {
            uint8_t tmp[4] = { zero, zero, zero, zero };
            for (size_t i = 0; i < tail; ++i)
                tmp[i] = src[i];
            return Set4(tmp);
        }

        template<bool overflow> SIMD_INLINE void Madd4(svint32_t& sum, const svuint8_t& src, const int8_t* weight);

        template<> SIMD_INLINE void Madd4<false>(svint32_t& sum, const svuint8_t& src, const int8_t* weight)
        {
            const svbool_t body8 = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            svint8_t _weight = svld1_s8(body8, weight);
            svint8_t _src = svreinterpret_s8_u8(svsub_n_u8_x(body8, src, 128));
            svint32_t corr = svmul_n_s32_x(body32, svdot_s32(svdup_n_s32(0), svdup_n_s8(1), _weight), 128);
            sum = svadd_s32_x(body32, svdot_s32(sum, _src, _weight), corr);
        }

        template<> SIMD_INLINE void Madd4<true>(svint32_t& sum, const svuint8_t& src, const int8_t* weight)
        {
            const svbool_t body8 = svptrue_b8();
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            svint8_t _weight = svld1_s8(body8, weight);
            svint16_t sLo = svreinterpret_s16_u16(svmovlb_u16(src));
            svint16_t sHi = svreinterpret_s16_u16(svmovlt_u16(src));
            svint16_t wLo = svmovlb_s16(_weight);
            svint16_t wHi = svmovlt_s16(_weight);
            svint16_t lo = svmul_s16_x(body16, sLo, wLo);
            svint16_t hi = svmul_s16_x(body16, sHi, wHi);
            svint16_t pairs = svqadd_s16(lo, hi);
            svint16_t zero = svdup_n_s16(0);
            svint32_t sum0 = svaddlb_s32(pairs, zero);
            svint32_t sum1 = svaddlt_s32(pairs, zero);
            sum = svadd_s32_x(body32, sum, svadd_s32_x(body32, sum0, sum1));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE float Activate(float value, const float* params, size_t offset)
        {
            return Base::Activate<type>(value, params, offset);
        }

        template<Term8iType term> struct Term8i
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, const svint32_t& sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t upper, size_t tail);
        };

        template<> struct Term8i<Term8iLast8u>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, const svint32_t& sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t upper, size_t tail)
            {
                int32_t sums[SIMD_SVE2_VECTOR_SIZE_MAX / sizeof(int32_t)];
                int32_t _upper = upper & 0xFF;
                svst1_s32(svwhilelt_b32((size_t)0, tail), sums, sum);
                for (size_t i = 0; i < tail; ++i)
                {
                    float value = Activate<type>(float(sums[i]) * norm[i] + bias[i], params, i);
                    dst[i] = (uint8_t)Simd::RestrictRange(Simd::Round(value * scale[i] + shift[i]), 0, _upper);
                }
            }
        };

        template<> struct Term8i<Term8iLast32f>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, const svint32_t& sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t upper, size_t tail)
            {
                int32_t sums[SIMD_SVE2_VECTOR_SIZE_MAX / sizeof(int32_t)];
                float* dst32f = (float*)dst;
                svst1_s32(svwhilelt_b32((size_t)0, tail), sums, sum);
                for (size_t i = 0; i < tail; ++i)
                    dst32f[i] = Activate<type>(float(sums[i]) * norm[i] + bias[i], params, i);
            }
        };

        template<> struct Term8i<Term8iInterim>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, const svint32_t& sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t upper, size_t tail)
            {
                svst1_s32(svwhilelt_b32((size_t)0, tail), buf, sum);
            }
        };

        SIMD_INLINE svint32_t LoadSum(int first, const int32_t* buf, size_t tail)
        {
            return first ? svdup_n_s32(0) : svld1_s32(svwhilelt_b32((size_t)0, tail), buf);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, const svint32_t& sum,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t upper, size_t tail)
        {
            Term8i<term>::template Save<type>(dst, buf, sum, norm, bias, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, const svint32_t& sum0,
            const svint32_t& sum1, const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t upper, size_t size, size_t F, size_t dstC)
        {
            Save1<term, type>(dst, buf, sum0, norm, bias, params, scale, shift, upper, F);
            Save1<term, type>(dst + F * size, buf + F, sum1, norm + F, bias + F,
                type == ::SimdConvolutionActivationPrelu ? params + F : params, scale + F, shift + F, upper, dstC - F);
        }

        SIMD_INLINE svuint8_t Set4(const uint8_t* src, size_t tail)
        {
            return tail == 4 ? Set4(src) : Set4(src, tail, 0);
        }

#define SIMD_SVE2_DECL_D0() svint32_t d00, d10, d20, d30, d40, d50, d60, d70, d80, d90, dA0, dB0
#define SIMD_SVE2_DECL_D1() svint32_t d01, d11, d21, d31, d41, d51, d61, d71, d81, d91, dA1, dB1
#define SIMD_SVE2_INIT_D0(I) if (M > 0x##I) d##I##0 = LoadSum(first, buf + 0x##I * dB, Simd::Min(F, dstC))
#define SIMD_SVE2_INIT_D1(I) if (M > 0x##I) d##I##1 = LoadSum(first, buf + 0x##I * dB + F, dstC - F)
#define SIMD_SVE2_MADD_SRC_1(I) if (M > 0x##I) s0 = Set4(src + 0x##I * step + offs, tail), Madd4<overflow>(d##I##0, s0, weight0)
#define SIMD_SVE2_MADD_SRC_2(I) if (M > 0x##I) s0 = Set4(src + 0x##I * step + offs, tail), Madd4<overflow>(d##I##0, s0, weight0), Madd4<overflow>(d##I##1, s0, weight1)
#define SIMD_SVE2_MADD_ZERO_1(I) if (M > 0x##I) Madd4<overflow>(d##I##0, zero, weight0)
#define SIMD_SVE2_MADD_ZERO_2(I) if (M > 0x##I) Madd4<overflow>(d##I##0, zero, weight0), Madd4<overflow>(d##I##1, zero, weight1)
#define SIMD_SVE2_SAVE_1(I) if (M > 0x##I) Save1<term, type>(dst + 0x##I * dD, buf + 0x##I * dB, d##I##0, norm, bias, params, scale, shift, a.upper, dstC)
#define SIMD_SVE2_SAVE_2(I) if (M > 0x##I) Save2<term, type>(dst + 0x##I * dD, buf + 0x##I * dB, d##I##0, d##I##1, norm, bias, params, scale, shift, a.upper, a.size, F, dstC)
#define SIMD_SVE2_APPLY_12(MACRO) MACRO(0); MACRO(1); MACRO(2); MACRO(3); MACRO(4); MACRO(5); MACRO(6); MACRO(7); MACRO(8); MACRO(9); MACRO(A); MACRO(B)

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            const size_t F = a.F, A = F * 4, step = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const size_t srcCA = AlignLo(srcC, 4);
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            svuint8_t s0;
            SIMD_SVE2_DECL_D0();
            SIMD_SVE2_DECL_D1();
            SIMD_SVE2_APPLY_12(SIMD_SVE2_INIT_D0);
            if (dstC > F)
            {
                SIMD_SVE2_APPLY_12(SIMD_SVE2_INIT_D1);
                size_t offs = 0, tail = 4;
                const uint8_t* src = src0;
                for (; offs < srcCA; offs += 4, weight0 += A, weight1 += A)
                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_SRC_2);
                if (offs < srcC)
                {
                    tail = srcC - offs;
                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_SRC_2);
                }
                SIMD_SVE2_APPLY_12(SIMD_SVE2_SAVE_2);
            }
            else
            {
                size_t offs = 0, tail = 4;
                const uint8_t* src = src0;
                for (; offs < srcCA; offs += 4, weight0 += A)
                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_SRC_1);
                if (offs < srcC)
                {
                    tail = srcC - offs;
                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_SRC_1);
                }
                SIMD_SVE2_APPLY_12(SIMD_SVE2_SAVE_1);
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect_2xM(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t* weight0,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            const size_t F = a.F, A = F * 4, dY = p.srcW * p.srcC, dX = p.srcC, step = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const size_t srcCF = DivHi(srcC, 4), srcCA = AlignLo(srcC, 4), dW = (DivHi(p.srcC, 4) - srcCF) * A;
            const size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            const size_t sy = dy * p.strideY - p.padY, sx = dx * p.strideX - p.padX;
            svuint8_t s0, zero = Set4((uint32_t)a.zero);
            SIMD_SVE2_DECL_D0();
            SIMD_SVE2_DECL_D1();
            SIMD_SVE2_APPLY_12(SIMD_SVE2_INIT_D0);
            if (dstC > F)
            {
                SIMD_SVE2_APPLY_12(SIMD_SVE2_INIT_D1);
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            if (sx + kx < p.srcW && sx + kx + (M - 1) * p.strideX < p.srcW)
                            {
                                const uint8_t* src = src0 + (sy + ky) * dY + (sx + kx) * dX;
                                size_t offs = 0, tail = 4;
                                for (; offs < srcCA; offs += 4, weight0 += A, weight1 += A)
                                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_SRC_2);
                                if (offs < srcC)
                                {
                                    tail = srcC - offs;
                                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_SRC_2);
                                    weight0 += A, weight1 += A;
                                }
                            }
                            else if (a.zero)
                            {
                                for (size_t offs = 0; offs < srcC; offs += 4, weight0 += A, weight1 += A)
                                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_ZERO_2);
                            }
                            else
                                weight0 += srcCF * A, weight1 += srcCF * A;
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else if (a.zero)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            for (size_t offs = 0; offs < srcC; offs += 4, weight0 += A, weight1 += A)
                                SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_ZERO_2);
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                    {
                        weight0 += (srcCF * A + dW) * p.kernelX;
                        weight1 += (srcCF * A + dW) * p.kernelX;
                    }
                }
                SIMD_SVE2_APPLY_12(SIMD_SVE2_SAVE_2);
            }
            else
            {
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    if (sy + ky < p.srcH)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            if (sx + kx < p.srcW && sx + kx + (M - 1) * p.strideX < p.srcW)
                            {
                                const uint8_t* src = src0 + (sy + ky) * dY + (sx + kx) * dX;
                                size_t offs = 0, tail = 4;
                                for (; offs < srcCA; offs += 4, weight0 += A)
                                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_SRC_1);
                                if (offs < srcC)
                                {
                                    tail = srcC - offs;
                                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_SRC_1);
                                    weight0 += A;
                                }
                            }
                            else if (a.zero)
                            {
                                for (size_t offs = 0; offs < srcC; offs += 4, weight0 += A)
                                    SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_ZERO_1);
                            }
                            else
                                weight0 += srcCF * A;
                            weight0 += dW;
                        }
                    }
                    else if (a.zero)
                    {
                        for (size_t kx = 0; kx < kX; kx += p.dilationX)
                        {
                            for (size_t offs = 0; offs < srcC; offs += 4, weight0 += A)
                                SIMD_SVE2_APPLY_12(SIMD_SVE2_MADD_ZERO_1);
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += (srcCF * A + dW) * p.kernelX;
                }
                SIMD_SVE2_APPLY_12(SIMD_SVE2_SAVE_1);
            }
        }

#undef SIMD_SVE2_DECL_D0
#undef SIMD_SVE2_DECL_D1
#undef SIMD_SVE2_INIT_D0
#undef SIMD_SVE2_INIT_D1
#undef SIMD_SVE2_MADD_SRC_1
#undef SIMD_SVE2_MADD_SRC_2
#undef SIMD_SVE2_MADD_ZERO_1
#undef SIMD_SVE2_MADD_ZERO_2
#undef SIMD_SVE2_SAVE_1
#undef SIMD_SVE2_SAVE_2
#undef SIMD_SVE2_APPLY_12

        typedef void(*ConvolutionNhwcDirect1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC,
            const int8_t* weight0, const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first);

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect1x1_2xM_Ptr GetConvolutionNhwcDirect1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0x1>;
            case 0x2: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0x2>;
            case 0x3: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0x3>;
            case 0x4: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0x4>;
            case 0x5: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0x5>;
            case 0x6: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0x6>;
            case 0x7: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0x7>;
            case 0x8: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0x8>;
            case 0x9: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0x9>;
            case 0xA: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0xA>;
            case 0xB: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0xB>;
            case 0xC: return ConvolutionNhwcDirect1x1_2xM<overflow, term, type, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const uint8_t* src,
            const ConvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            const size_t F = a.F, DF = 2 * F, n = 12, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xN = GetConvolutionNhwcDirect1x1_2xM<overflow, term, type>(n);
            ConvolutionNhwcDirect1x1_2xM_Ptr convolutionNhwcDirect1x1_2xM = GetConvolutionNhwcDirect1x1_2xM<overflow, term, type>(m);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                const float* _params = type == ::SimdConvolutionActivationPrelu ? params + dc : params;
                const uint8_t* s = src + yBeg * p.srcW * p.srcC;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += p.srcC * n, b += p.dstC * n, d += p.dstC * a.size * n)
                    convolutionNhwcDirect1x1_2xN(s, p, a, srcC, dC, weight, norm + dc, bias + dc, _params, scale + dc, shift + dc, b, d, first);
                for (; i < n1; i += m, s += p.srcC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                    convolutionNhwcDirect1x1_2xM(s, p, a, srcC, dC, weight, norm + dc, bias + dc, _params, scale + dc, shift + dc, b, d, first);
                weight += DivHi(p.srcC, 4) * DF * 4;
            }
        }

        typedef void(*ConvolutionNhwcDirect_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t dy, size_t dx, size_t srcC, size_t dstC,
            const int8_t* weight0, const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first);

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> ConvolutionNhwcDirect_2xM_Ptr GetConvolutionNhwcDirect_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0x1>;
            case 0x2: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0x2>;
            case 0x3: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0x3>;
            case 0x4: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0x4>;
            case 0x5: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0x5>;
            case 0x6: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0x6>;
            case 0x7: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0x7>;
            case 0x8: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0x8>;
            case 0x9: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0x9>;
            case 0xA: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0xA>;
            case 0xB: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0xB>;
            case 0xC: return ConvolutionNhwcDirect_2xM<overflow, term, type, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const uint8_t* src,
            const ConvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            const size_t F = a.F, DF = 2 * F, n = 12, noseW = p.NoseW(), bodyW = p.BodyW(), bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2x1 = GetConvolutionNhwcDirect_2xM<overflow, term, type>(1);
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2xN = GetConvolutionNhwcDirect_2xM<overflow, term, type>(n);
            ConvolutionNhwcDirect_2xM_Ptr convolutionNhwcDirect_2xM = GetConvolutionNhwcDirect_2xM<overflow, term, type>(m);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                const float* _params = type == ::SimdConvolutionActivationPrelu ? params + dc : params;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, norm + dc, bias + dc, _params, scale + dc, shift + dc, b, d, first);
                    for (; dx < bodyWn; dx += n, b += p.dstC * n, d += p.dstC * a.size * n)
                        convolutionNhwcDirect_2xN(src, p, a, dy, dx, srcC, dC, weight, norm + dc, bias + dc, _params, scale + dc, shift + dc, b, d, first);
                    for (; dx < bodyW; dx += m, b += p.dstC * m, d += p.dstC * a.size * m)
                        convolutionNhwcDirect_2xM(src, p, a, dy, dx, srcC, dC, weight, norm + dc, bias + dc, _params, scale + dc, shift + dc, b, d, first);
                    for (; dx < p.dstW; dx++, b += p.dstC, d += p.dstC * a.size)
                        convolutionNhwcDirect_2x1(src, p, a, dy, dx, srcC, dC, weight, norm + dc, bias + dc, _params, scale + dc, shift + dc, b, d, first);
                }
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * DF * 4;
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect(const uint8_t* src,
            const ConvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            const size_t F = a.F;
            const size_t dY = p.srcW * p.srcC, dX = p.srcC, dD = p.dstC * a.size, dB = p.dstC;
            const size_t dW = p.kernelY * p.kernelX * DivHi(p.srcC, 4) * F * 4;
            const size_t kY = p.kernelY * p.dilationY, kX = p.kernelX * p.dilationX;
            for (size_t dc = 0; dc < dstC; dc += F)
            {
                size_t dC = Simd::Min(F, dstC - dc);
                const int8_t* weight0 = weight + dc / F * dW;
                const float* norm0 = norm + dc;
                const float* bias0 = bias + dc;
                const float* params0 = type == ::SimdConvolutionActivationPrelu ? params + dc : params;
                const float* scale0 = scale + dc;
                const float* shift0 = shift + dc;
                svbool_t tail = svwhilelt_b32((size_t)0, dC);
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    size_t sy = dy * p.strideY - p.padY;
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                    {
                        size_t sx = dx * p.strideX - p.padX;
                        int32_t* b = buf + dy * p.dstW * dB + dx * dB + dc;
                        uint8_t* d = dst + (dy * p.dstW * p.dstC + dx * p.dstC + dc) * a.size;
                        svint32_t sum = first ? svdup_n_s32(0) : svld1_s32(tail, b);
                        const int8_t* w = weight0;
                        for (size_t ky = 0; ky < kY; ky += p.dilationY)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                if (sy + ky < p.srcH && sx + kx < p.srcW)
                                {
                                    const uint8_t* ps = src + (sy + ky) * dY + (sx + kx) * dX;
                                    size_t offs = 0, aligned = AlignLo(srcC, 4);
                                    for (; offs < aligned; offs += 4, w += F * 4)
                                        Madd4<overflow>(sum, Set4(ps + offs), w);
                                    if (offs < srcC)
                                        Madd4<overflow>(sum, Set4(ps + offs, srcC - offs, 0), w), w += F * 4;
                                }
                                else if (a.zero)
                                {
                                    svuint8_t zero = Set4((uint32_t)a.zero);
                                    for (size_t offs = 0; offs < srcC; offs += 4, w += F * 4)
                                        Madd4<overflow>(sum, zero, w);
                                }
                                else
                                    w += DivHi(srcC, 4) * F * 4;
                                w += (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * F * 4;
                            }
                        }
                        Term8i<term>::template Save<type>(d, b, sum, norm0, bias0, params0, scale0, shift0, a.upper, dC);
                    }
                }
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType activation> void SetDirect1x1(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            d[term] = ConvolutionNhwcDirect1x1_2<overflow, term, activation>;
        }

        template<Term8iType term, SimdConvolutionActivationType activation> void SetDirect1x1(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            if (Base::Overflow(p.compatibility))
                SetDirect1x1<true, term, activation>(p, a, d);
            else
                SetDirect1x1<false, term, activation>(p, a, d);
        }

        template<SimdConvolutionActivationType activation> void SetDirect1x1(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            SetDirect1x1<Term8iLast8u, activation>(p, a, d);
            SetDirect1x1<Term8iLast32f, activation>(p, a, d);
            SetDirect1x1<Term8iInterim, SimdConvolutionActivationIdentity>(p, a, d);
        }

        void SetDirect1x1(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetDirect1x1<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationRelu: SetDirect1x1<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationLeakyRelu: SetDirect1x1<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationRestrictRange: SetDirect1x1<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationPrelu: SetDirect1x1<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationElu: SetDirect1x1<SimdConvolutionActivationElu>(p, a, d); break;
            case SimdConvolutionActivationHswish: SetDirect1x1<SimdConvolutionActivationHswish>(p, a, d); break;
            case SimdConvolutionActivationMish: SetDirect1x1<SimdConvolutionActivationMish>(p, a, d); break;
            case SimdConvolutionActivationHardSigmoid: SetDirect1x1<SimdConvolutionActivationHardSigmoid>(p, a, d); break;
            case SimdConvolutionActivationSwish: SetDirect1x1<SimdConvolutionActivationSwish>(p, a, d); break;
            case SimdConvolutionActivationGelu: SetDirect1x1<SimdConvolutionActivationGelu>(p, a, d); break;
            default: assert(0);
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType activation> void SetDirectAny(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            d[term] = ConvolutionNhwcDirect_2<overflow, term, activation>;
        }

        template<Term8iType term, SimdConvolutionActivationType activation> void SetDirectAny(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            if (Base::Overflow(p.compatibility))
                SetDirectAny<true, term, activation>(p, a, d);
            else
                SetDirectAny<false, term, activation>(p, a, d);
        }

        template<SimdConvolutionActivationType activation> void SetDirectAny(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            SetDirectAny<Term8iLast8u, activation>(p, a, d);
            SetDirectAny<Term8iLast32f, activation>(p, a, d);
            SetDirectAny<Term8iInterim, SimdConvolutionActivationIdentity>(p, a, d);
        }

        void SetDirectAny(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetDirectAny<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationRelu: SetDirectAny<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationLeakyRelu: SetDirectAny<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationRestrictRange: SetDirectAny<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationPrelu: SetDirectAny<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationElu: SetDirectAny<SimdConvolutionActivationElu>(p, a, d); break;
            case SimdConvolutionActivationHswish: SetDirectAny<SimdConvolutionActivationHswish>(p, a, d); break;
            case SimdConvolutionActivationMish: SetDirectAny<SimdConvolutionActivationMish>(p, a, d); break;
            case SimdConvolutionActivationHardSigmoid: SetDirectAny<SimdConvolutionActivationHardSigmoid>(p, a, d); break;
            case SimdConvolutionActivationSwish: SetDirectAny<SimdConvolutionActivationSwish>(p, a, d); break;
            case SimdConvolutionActivationGelu: SetDirectAny<SimdConvolutionActivationGelu>(p, a, d); break;
            default: assert(0);
            }
        }

        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam& p)
            : Base::SynetConvolution8iNhwcDirect(p)
        {
            size_t F = svcntw();
            SetAlgParam(F, 2 * F, 12, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (p.Is1x1())
                SetDirect1x1(p, _alg, _convolutions);
            else
                SetDirectAny(p, _alg, _convolutions);
            _convertSrc = Sve2::SynetConvert32fTo8u;
        }

        bool SynetConvolution8iNhwcDirect::Preferable(const ConvParam& p)
        {
            if (p.trans != SimdTrue || p.group != 1)
                return false;
            return true;
        }

        //---------------------------------------------------------------------

        void* SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam param(batch, conv, compatibility);
            if (!param.Valid(SimdTensorData32f, SimdTensorData8u))
                return NULL;
            else if (SynetConvolution8iNhwcDirect::Preferable(param))
                return new SynetConvolution8iNhwcDirect(param);
            else
                return new Base::SynetConvolution8iGemmNN(param);
        }
    }
#endif
}
