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

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type, int M> void ConvolutionNhwcDirect1x1_2xM(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight0,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            const size_t F = a.F, A = F * 4, step = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const size_t srcCA = AlignLo(srcC, 4);
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            svuint8_t s0;
            svint32_t d00, d10, d20, d30, d40, d50, d60, d70, d80, d90, dA0, dB0;
            svint32_t d01, d11, d21, d31, d41, d51, d61, d71, d81, d91, dA1, dB1;
            if (M > 0x0) d00 = LoadSum(first, buf + 0x0 * dB, Simd::Min(F, dstC));
            if (M > 0x1) d10 = LoadSum(first, buf + 0x1 * dB, Simd::Min(F, dstC));
            if (M > 0x2) d20 = LoadSum(first, buf + 0x2 * dB, Simd::Min(F, dstC));
            if (M > 0x3) d30 = LoadSum(first, buf + 0x3 * dB, Simd::Min(F, dstC));
            if (M > 0x4) d40 = LoadSum(first, buf + 0x4 * dB, Simd::Min(F, dstC));
            if (M > 0x5) d50 = LoadSum(first, buf + 0x5 * dB, Simd::Min(F, dstC));
            if (M > 0x6) d60 = LoadSum(first, buf + 0x6 * dB, Simd::Min(F, dstC));
            if (M > 0x7) d70 = LoadSum(first, buf + 0x7 * dB, Simd::Min(F, dstC));
            if (M > 0x8) d80 = LoadSum(first, buf + 0x8 * dB, Simd::Min(F, dstC));
            if (M > 0x9) d90 = LoadSum(first, buf + 0x9 * dB, Simd::Min(F, dstC));
            if (M > 0xA) dA0 = LoadSum(first, buf + 0xA * dB, Simd::Min(F, dstC));
            if (M > 0xB) dB0 = LoadSum(first, buf + 0xB * dB, Simd::Min(F, dstC));
            if (dstC > F)
            {
                if (M > 0x0) d01 = LoadSum(first, buf + 0x0 * dB + F, dstC - F);
                if (M > 0x1) d11 = LoadSum(first, buf + 0x1 * dB + F, dstC - F);
                if (M > 0x2) d21 = LoadSum(first, buf + 0x2 * dB + F, dstC - F);
                if (M > 0x3) d31 = LoadSum(first, buf + 0x3 * dB + F, dstC - F);
                if (M > 0x4) d41 = LoadSum(first, buf + 0x4 * dB + F, dstC - F);
                if (M > 0x5) d51 = LoadSum(first, buf + 0x5 * dB + F, dstC - F);
                if (M > 0x6) d61 = LoadSum(first, buf + 0x6 * dB + F, dstC - F);
                if (M > 0x7) d71 = LoadSum(first, buf + 0x7 * dB + F, dstC - F);
                if (M > 0x8) d81 = LoadSum(first, buf + 0x8 * dB + F, dstC - F);
                if (M > 0x9) d91 = LoadSum(first, buf + 0x9 * dB + F, dstC - F);
                if (M > 0xA) dA1 = LoadSum(first, buf + 0xA * dB + F, dstC - F);
                if (M > 0xB) dB1 = LoadSum(first, buf + 0xB * dB + F, dstC - F);
                size_t offs = 0, tail = 4;
                const uint8_t* src = src0;
                for (; offs < srcCA; offs += 4, weight0 += A, weight1 += A)
                {
                    if (M > 0x0) s0 = Set4(src + 0x0 * step + offs, tail), Madd4<overflow>(d00, s0, weight0), Madd4<overflow>(d01, s0, weight1);
                    if (M > 0x1) s0 = Set4(src + 0x1 * step + offs, tail), Madd4<overflow>(d10, s0, weight0), Madd4<overflow>(d11, s0, weight1);
                    if (M > 0x2) s0 = Set4(src + 0x2 * step + offs, tail), Madd4<overflow>(d20, s0, weight0), Madd4<overflow>(d21, s0, weight1);
                    if (M > 0x3) s0 = Set4(src + 0x3 * step + offs, tail), Madd4<overflow>(d30, s0, weight0), Madd4<overflow>(d31, s0, weight1);
                    if (M > 0x4) s0 = Set4(src + 0x4 * step + offs, tail), Madd4<overflow>(d40, s0, weight0), Madd4<overflow>(d41, s0, weight1);
                    if (M > 0x5) s0 = Set4(src + 0x5 * step + offs, tail), Madd4<overflow>(d50, s0, weight0), Madd4<overflow>(d51, s0, weight1);
                    if (M > 0x6) s0 = Set4(src + 0x6 * step + offs, tail), Madd4<overflow>(d60, s0, weight0), Madd4<overflow>(d61, s0, weight1);
                    if (M > 0x7) s0 = Set4(src + 0x7 * step + offs, tail), Madd4<overflow>(d70, s0, weight0), Madd4<overflow>(d71, s0, weight1);
                    if (M > 0x8) s0 = Set4(src + 0x8 * step + offs, tail), Madd4<overflow>(d80, s0, weight0), Madd4<overflow>(d81, s0, weight1);
                    if (M > 0x9) s0 = Set4(src + 0x9 * step + offs, tail), Madd4<overflow>(d90, s0, weight0), Madd4<overflow>(d91, s0, weight1);
                    if (M > 0xA) s0 = Set4(src + 0xA * step + offs, tail), Madd4<overflow>(dA0, s0, weight0), Madd4<overflow>(dA1, s0, weight1);
                    if (M > 0xB) s0 = Set4(src + 0xB * step + offs, tail), Madd4<overflow>(dB0, s0, weight0), Madd4<overflow>(dB1, s0, weight1);
                }
                if (offs < srcC)
                {
                    tail = srcC - offs;
                    if (M > 0x0) s0 = Set4(src + 0x0 * step + offs, tail), Madd4<overflow>(d00, s0, weight0), Madd4<overflow>(d01, s0, weight1);
                    if (M > 0x1) s0 = Set4(src + 0x1 * step + offs, tail), Madd4<overflow>(d10, s0, weight0), Madd4<overflow>(d11, s0, weight1);
                    if (M > 0x2) s0 = Set4(src + 0x2 * step + offs, tail), Madd4<overflow>(d20, s0, weight0), Madd4<overflow>(d21, s0, weight1);
                    if (M > 0x3) s0 = Set4(src + 0x3 * step + offs, tail), Madd4<overflow>(d30, s0, weight0), Madd4<overflow>(d31, s0, weight1);
                    if (M > 0x4) s0 = Set4(src + 0x4 * step + offs, tail), Madd4<overflow>(d40, s0, weight0), Madd4<overflow>(d41, s0, weight1);
                    if (M > 0x5) s0 = Set4(src + 0x5 * step + offs, tail), Madd4<overflow>(d50, s0, weight0), Madd4<overflow>(d51, s0, weight1);
                    if (M > 0x6) s0 = Set4(src + 0x6 * step + offs, tail), Madd4<overflow>(d60, s0, weight0), Madd4<overflow>(d61, s0, weight1);
                    if (M > 0x7) s0 = Set4(src + 0x7 * step + offs, tail), Madd4<overflow>(d70, s0, weight0), Madd4<overflow>(d71, s0, weight1);
                    if (M > 0x8) s0 = Set4(src + 0x8 * step + offs, tail), Madd4<overflow>(d80, s0, weight0), Madd4<overflow>(d81, s0, weight1);
                    if (M > 0x9) s0 = Set4(src + 0x9 * step + offs, tail), Madd4<overflow>(d90, s0, weight0), Madd4<overflow>(d91, s0, weight1);
                    if (M > 0xA) s0 = Set4(src + 0xA * step + offs, tail), Madd4<overflow>(dA0, s0, weight0), Madd4<overflow>(dA1, s0, weight1);
                    if (M > 0xB) s0 = Set4(src + 0xB * step + offs, tail), Madd4<overflow>(dB0, s0, weight0), Madd4<overflow>(dB1, s0, weight1);
                }
                if (M > 0x0) Save2<term, type>(dst + 0x0 * dD, buf + 0x0 * dB, d00, d01, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x1) Save2<term, type>(dst + 0x1 * dD, buf + 0x1 * dB, d10, d11, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x2) Save2<term, type>(dst + 0x2 * dD, buf + 0x2 * dB, d20, d21, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x3) Save2<term, type>(dst + 0x3 * dD, buf + 0x3 * dB, d30, d31, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x4) Save2<term, type>(dst + 0x4 * dD, buf + 0x4 * dB, d40, d41, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x5) Save2<term, type>(dst + 0x5 * dD, buf + 0x5 * dB, d50, d51, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x6) Save2<term, type>(dst + 0x6 * dD, buf + 0x6 * dB, d60, d61, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x7) Save2<term, type>(dst + 0x7 * dD, buf + 0x7 * dB, d70, d71, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x8) Save2<term, type>(dst + 0x8 * dD, buf + 0x8 * dB, d80, d81, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x9) Save2<term, type>(dst + 0x9 * dD, buf + 0x9 * dB, d90, d91, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0xA) Save2<term, type>(dst + 0xA * dD, buf + 0xA * dB, dA0, dA1, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0xB) Save2<term, type>(dst + 0xB * dD, buf + 0xB * dB, dB0, dB1, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
            }
            else
            {
                size_t offs = 0, tail = 4;
                const uint8_t* src = src0;
                for (; offs < srcCA; offs += 4, weight0 += A)
                {
                    if (M > 0x0) s0 = Set4(src + 0x0 * step + offs, tail), Madd4<overflow>(d00, s0, weight0);
                    if (M > 0x1) s0 = Set4(src + 0x1 * step + offs, tail), Madd4<overflow>(d10, s0, weight0);
                    if (M > 0x2) s0 = Set4(src + 0x2 * step + offs, tail), Madd4<overflow>(d20, s0, weight0);
                    if (M > 0x3) s0 = Set4(src + 0x3 * step + offs, tail), Madd4<overflow>(d30, s0, weight0);
                    if (M > 0x4) s0 = Set4(src + 0x4 * step + offs, tail), Madd4<overflow>(d40, s0, weight0);
                    if (M > 0x5) s0 = Set4(src + 0x5 * step + offs, tail), Madd4<overflow>(d50, s0, weight0);
                    if (M > 0x6) s0 = Set4(src + 0x6 * step + offs, tail), Madd4<overflow>(d60, s0, weight0);
                    if (M > 0x7) s0 = Set4(src + 0x7 * step + offs, tail), Madd4<overflow>(d70, s0, weight0);
                    if (M > 0x8) s0 = Set4(src + 0x8 * step + offs, tail), Madd4<overflow>(d80, s0, weight0);
                    if (M > 0x9) s0 = Set4(src + 0x9 * step + offs, tail), Madd4<overflow>(d90, s0, weight0);
                    if (M > 0xA) s0 = Set4(src + 0xA * step + offs, tail), Madd4<overflow>(dA0, s0, weight0);
                    if (M > 0xB) s0 = Set4(src + 0xB * step + offs, tail), Madd4<overflow>(dB0, s0, weight0);
                }
                if (offs < srcC)
                {
                    tail = srcC - offs;
                    if (M > 0x0) s0 = Set4(src + 0x0 * step + offs, tail), Madd4<overflow>(d00, s0, weight0);
                    if (M > 0x1) s0 = Set4(src + 0x1 * step + offs, tail), Madd4<overflow>(d10, s0, weight0);
                    if (M > 0x2) s0 = Set4(src + 0x2 * step + offs, tail), Madd4<overflow>(d20, s0, weight0);
                    if (M > 0x3) s0 = Set4(src + 0x3 * step + offs, tail), Madd4<overflow>(d30, s0, weight0);
                    if (M > 0x4) s0 = Set4(src + 0x4 * step + offs, tail), Madd4<overflow>(d40, s0, weight0);
                    if (M > 0x5) s0 = Set4(src + 0x5 * step + offs, tail), Madd4<overflow>(d50, s0, weight0);
                    if (M > 0x6) s0 = Set4(src + 0x6 * step + offs, tail), Madd4<overflow>(d60, s0, weight0);
                    if (M > 0x7) s0 = Set4(src + 0x7 * step + offs, tail), Madd4<overflow>(d70, s0, weight0);
                    if (M > 0x8) s0 = Set4(src + 0x8 * step + offs, tail), Madd4<overflow>(d80, s0, weight0);
                    if (M > 0x9) s0 = Set4(src + 0x9 * step + offs, tail), Madd4<overflow>(d90, s0, weight0);
                    if (M > 0xA) s0 = Set4(src + 0xA * step + offs, tail), Madd4<overflow>(dA0, s0, weight0);
                    if (M > 0xB) s0 = Set4(src + 0xB * step + offs, tail), Madd4<overflow>(dB0, s0, weight0);
                }
                if (M > 0x0) Save1<term, type>(dst + 0x0 * dD, buf + 0x0 * dB, d00, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x1) Save1<term, type>(dst + 0x1 * dD, buf + 0x1 * dB, d10, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x2) Save1<term, type>(dst + 0x2 * dD, buf + 0x2 * dB, d20, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x3) Save1<term, type>(dst + 0x3 * dD, buf + 0x3 * dB, d30, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x4) Save1<term, type>(dst + 0x4 * dD, buf + 0x4 * dB, d40, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x5) Save1<term, type>(dst + 0x5 * dD, buf + 0x5 * dB, d50, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x6) Save1<term, type>(dst + 0x6 * dD, buf + 0x6 * dB, d60, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x7) Save1<term, type>(dst + 0x7 * dD, buf + 0x7 * dB, d70, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x8) Save1<term, type>(dst + 0x8 * dD, buf + 0x8 * dB, d80, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x9) Save1<term, type>(dst + 0x9 * dD, buf + 0x9 * dB, d90, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0xA) Save1<term, type>(dst + 0xA * dD, buf + 0xA * dB, dA0, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0xB) Save1<term, type>(dst + 0xB * dD, buf + 0xB * dB, dB0, norm, bias, params, scale, shift, a.upper, dstC);
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
            svint32_t d00, d10, d20, d30, d40, d50, d60, d70, d80, d90, dA0, dB0;
            svint32_t d01, d11, d21, d31, d41, d51, d61, d71, d81, d91, dA1, dB1;
            if (M > 0x0) d00 = LoadSum(first, buf + 0x0 * dB, Simd::Min(F, dstC));
            if (M > 0x1) d10 = LoadSum(first, buf + 0x1 * dB, Simd::Min(F, dstC));
            if (M > 0x2) d20 = LoadSum(first, buf + 0x2 * dB, Simd::Min(F, dstC));
            if (M > 0x3) d30 = LoadSum(first, buf + 0x3 * dB, Simd::Min(F, dstC));
            if (M > 0x4) d40 = LoadSum(first, buf + 0x4 * dB, Simd::Min(F, dstC));
            if (M > 0x5) d50 = LoadSum(first, buf + 0x5 * dB, Simd::Min(F, dstC));
            if (M > 0x6) d60 = LoadSum(first, buf + 0x6 * dB, Simd::Min(F, dstC));
            if (M > 0x7) d70 = LoadSum(first, buf + 0x7 * dB, Simd::Min(F, dstC));
            if (M > 0x8) d80 = LoadSum(first, buf + 0x8 * dB, Simd::Min(F, dstC));
            if (M > 0x9) d90 = LoadSum(first, buf + 0x9 * dB, Simd::Min(F, dstC));
            if (M > 0xA) dA0 = LoadSum(first, buf + 0xA * dB, Simd::Min(F, dstC));
            if (M > 0xB) dB0 = LoadSum(first, buf + 0xB * dB, Simd::Min(F, dstC));
            if (dstC > F)
            {
                if (M > 0x0) d01 = LoadSum(first, buf + 0x0 * dB + F, dstC - F);
                if (M > 0x1) d11 = LoadSum(first, buf + 0x1 * dB + F, dstC - F);
                if (M > 0x2) d21 = LoadSum(first, buf + 0x2 * dB + F, dstC - F);
                if (M > 0x3) d31 = LoadSum(first, buf + 0x3 * dB + F, dstC - F);
                if (M > 0x4) d41 = LoadSum(first, buf + 0x4 * dB + F, dstC - F);
                if (M > 0x5) d51 = LoadSum(first, buf + 0x5 * dB + F, dstC - F);
                if (M > 0x6) d61 = LoadSum(first, buf + 0x6 * dB + F, dstC - F);
                if (M > 0x7) d71 = LoadSum(first, buf + 0x7 * dB + F, dstC - F);
                if (M > 0x8) d81 = LoadSum(first, buf + 0x8 * dB + F, dstC - F);
                if (M > 0x9) d91 = LoadSum(first, buf + 0x9 * dB + F, dstC - F);
                if (M > 0xA) dA1 = LoadSum(first, buf + 0xA * dB + F, dstC - F);
                if (M > 0xB) dB1 = LoadSum(first, buf + 0xB * dB + F, dstC - F);
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
                                {
                                    if (M > 0x0) s0 = Set4(src + 0x0 * step + offs, tail), Madd4<overflow>(d00, s0, weight0), Madd4<overflow>(d01, s0, weight1);
                                    if (M > 0x1) s0 = Set4(src + 0x1 * step + offs, tail), Madd4<overflow>(d10, s0, weight0), Madd4<overflow>(d11, s0, weight1);
                                    if (M > 0x2) s0 = Set4(src + 0x2 * step + offs, tail), Madd4<overflow>(d20, s0, weight0), Madd4<overflow>(d21, s0, weight1);
                                    if (M > 0x3) s0 = Set4(src + 0x3 * step + offs, tail), Madd4<overflow>(d30, s0, weight0), Madd4<overflow>(d31, s0, weight1);
                                    if (M > 0x4) s0 = Set4(src + 0x4 * step + offs, tail), Madd4<overflow>(d40, s0, weight0), Madd4<overflow>(d41, s0, weight1);
                                    if (M > 0x5) s0 = Set4(src + 0x5 * step + offs, tail), Madd4<overflow>(d50, s0, weight0), Madd4<overflow>(d51, s0, weight1);
                                    if (M > 0x6) s0 = Set4(src + 0x6 * step + offs, tail), Madd4<overflow>(d60, s0, weight0), Madd4<overflow>(d61, s0, weight1);
                                    if (M > 0x7) s0 = Set4(src + 0x7 * step + offs, tail), Madd4<overflow>(d70, s0, weight0), Madd4<overflow>(d71, s0, weight1);
                                    if (M > 0x8) s0 = Set4(src + 0x8 * step + offs, tail), Madd4<overflow>(d80, s0, weight0), Madd4<overflow>(d81, s0, weight1);
                                    if (M > 0x9) s0 = Set4(src + 0x9 * step + offs, tail), Madd4<overflow>(d90, s0, weight0), Madd4<overflow>(d91, s0, weight1);
                                    if (M > 0xA) s0 = Set4(src + 0xA * step + offs, tail), Madd4<overflow>(dA0, s0, weight0), Madd4<overflow>(dA1, s0, weight1);
                                    if (M > 0xB) s0 = Set4(src + 0xB * step + offs, tail), Madd4<overflow>(dB0, s0, weight0), Madd4<overflow>(dB1, s0, weight1);
                                }
                                if (offs < srcC)
                                {
                                    tail = srcC - offs;
                                    if (M > 0x0) s0 = Set4(src + 0x0 * step + offs, tail), Madd4<overflow>(d00, s0, weight0), Madd4<overflow>(d01, s0, weight1);
                                    if (M > 0x1) s0 = Set4(src + 0x1 * step + offs, tail), Madd4<overflow>(d10, s0, weight0), Madd4<overflow>(d11, s0, weight1);
                                    if (M > 0x2) s0 = Set4(src + 0x2 * step + offs, tail), Madd4<overflow>(d20, s0, weight0), Madd4<overflow>(d21, s0, weight1);
                                    if (M > 0x3) s0 = Set4(src + 0x3 * step + offs, tail), Madd4<overflow>(d30, s0, weight0), Madd4<overflow>(d31, s0, weight1);
                                    if (M > 0x4) s0 = Set4(src + 0x4 * step + offs, tail), Madd4<overflow>(d40, s0, weight0), Madd4<overflow>(d41, s0, weight1);
                                    if (M > 0x5) s0 = Set4(src + 0x5 * step + offs, tail), Madd4<overflow>(d50, s0, weight0), Madd4<overflow>(d51, s0, weight1);
                                    if (M > 0x6) s0 = Set4(src + 0x6 * step + offs, tail), Madd4<overflow>(d60, s0, weight0), Madd4<overflow>(d61, s0, weight1);
                                    if (M > 0x7) s0 = Set4(src + 0x7 * step + offs, tail), Madd4<overflow>(d70, s0, weight0), Madd4<overflow>(d71, s0, weight1);
                                    if (M > 0x8) s0 = Set4(src + 0x8 * step + offs, tail), Madd4<overflow>(d80, s0, weight0), Madd4<overflow>(d81, s0, weight1);
                                    if (M > 0x9) s0 = Set4(src + 0x9 * step + offs, tail), Madd4<overflow>(d90, s0, weight0), Madd4<overflow>(d91, s0, weight1);
                                    if (M > 0xA) s0 = Set4(src + 0xA * step + offs, tail), Madd4<overflow>(dA0, s0, weight0), Madd4<overflow>(dA1, s0, weight1);
                                    if (M > 0xB) s0 = Set4(src + 0xB * step + offs, tail), Madd4<overflow>(dB0, s0, weight0), Madd4<overflow>(dB1, s0, weight1);
                                    weight0 += A, weight1 += A;
                                }
                            }
                            else if (a.zero)
                            {
                                for (size_t offs = 0; offs < srcC; offs += 4, weight0 += A, weight1 += A)
                                {
                                    if (M > 0x0) Madd4<overflow>(d00, zero, weight0), Madd4<overflow>(d01, zero, weight1);
                                    if (M > 0x1) Madd4<overflow>(d10, zero, weight0), Madd4<overflow>(d11, zero, weight1);
                                    if (M > 0x2) Madd4<overflow>(d20, zero, weight0), Madd4<overflow>(d21, zero, weight1);
                                    if (M > 0x3) Madd4<overflow>(d30, zero, weight0), Madd4<overflow>(d31, zero, weight1);
                                    if (M > 0x4) Madd4<overflow>(d40, zero, weight0), Madd4<overflow>(d41, zero, weight1);
                                    if (M > 0x5) Madd4<overflow>(d50, zero, weight0), Madd4<overflow>(d51, zero, weight1);
                                    if (M > 0x6) Madd4<overflow>(d60, zero, weight0), Madd4<overflow>(d61, zero, weight1);
                                    if (M > 0x7) Madd4<overflow>(d70, zero, weight0), Madd4<overflow>(d71, zero, weight1);
                                    if (M > 0x8) Madd4<overflow>(d80, zero, weight0), Madd4<overflow>(d81, zero, weight1);
                                    if (M > 0x9) Madd4<overflow>(d90, zero, weight0), Madd4<overflow>(d91, zero, weight1);
                                    if (M > 0xA) Madd4<overflow>(dA0, zero, weight0), Madd4<overflow>(dA1, zero, weight1);
                                    if (M > 0xB) Madd4<overflow>(dB0, zero, weight0), Madd4<overflow>(dB1, zero, weight1);
                                }
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
                            {
                                if (M > 0x0) Madd4<overflow>(d00, zero, weight0), Madd4<overflow>(d01, zero, weight1);
                                if (M > 0x1) Madd4<overflow>(d10, zero, weight0), Madd4<overflow>(d11, zero, weight1);
                                if (M > 0x2) Madd4<overflow>(d20, zero, weight0), Madd4<overflow>(d21, zero, weight1);
                                if (M > 0x3) Madd4<overflow>(d30, zero, weight0), Madd4<overflow>(d31, zero, weight1);
                                if (M > 0x4) Madd4<overflow>(d40, zero, weight0), Madd4<overflow>(d41, zero, weight1);
                                if (M > 0x5) Madd4<overflow>(d50, zero, weight0), Madd4<overflow>(d51, zero, weight1);
                                if (M > 0x6) Madd4<overflow>(d60, zero, weight0), Madd4<overflow>(d61, zero, weight1);
                                if (M > 0x7) Madd4<overflow>(d70, zero, weight0), Madd4<overflow>(d71, zero, weight1);
                                if (M > 0x8) Madd4<overflow>(d80, zero, weight0), Madd4<overflow>(d81, zero, weight1);
                                if (M > 0x9) Madd4<overflow>(d90, zero, weight0), Madd4<overflow>(d91, zero, weight1);
                                if (M > 0xA) Madd4<overflow>(dA0, zero, weight0), Madd4<overflow>(dA1, zero, weight1);
                                if (M > 0xB) Madd4<overflow>(dB0, zero, weight0), Madd4<overflow>(dB1, zero, weight1);
                            }
                            weight0 += dW, weight1 += dW;
                        }
                    }
                    else
                    {
                        weight0 += (srcCF * A + dW) * p.kernelX;
                        weight1 += (srcCF * A + dW) * p.kernelX;
                    }
                }
                if (M > 0x0) Save2<term, type>(dst + 0x0 * dD, buf + 0x0 * dB, d00, d01, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x1) Save2<term, type>(dst + 0x1 * dD, buf + 0x1 * dB, d10, d11, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x2) Save2<term, type>(dst + 0x2 * dD, buf + 0x2 * dB, d20, d21, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x3) Save2<term, type>(dst + 0x3 * dD, buf + 0x3 * dB, d30, d31, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x4) Save2<term, type>(dst + 0x4 * dD, buf + 0x4 * dB, d40, d41, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x5) Save2<term, type>(dst + 0x5 * dD, buf + 0x5 * dB, d50, d51, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x6) Save2<term, type>(dst + 0x6 * dD, buf + 0x6 * dB, d60, d61, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x7) Save2<term, type>(dst + 0x7 * dD, buf + 0x7 * dB, d70, d71, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x8) Save2<term, type>(dst + 0x8 * dD, buf + 0x8 * dB, d80, d81, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0x9) Save2<term, type>(dst + 0x9 * dD, buf + 0x9 * dB, d90, d91, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0xA) Save2<term, type>(dst + 0xA * dD, buf + 0xA * dB, dA0, dA1, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
                if (M > 0xB) Save2<term, type>(dst + 0xB * dD, buf + 0xB * dB, dB0, dB1, norm, bias, params, scale, shift, a.upper, a.size, F, dstC);
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
                                {
                                    if (M > 0x0) s0 = Set4(src + 0x0 * step + offs, tail), Madd4<overflow>(d00, s0, weight0);
                                    if (M > 0x1) s0 = Set4(src + 0x1 * step + offs, tail), Madd4<overflow>(d10, s0, weight0);
                                    if (M > 0x2) s0 = Set4(src + 0x2 * step + offs, tail), Madd4<overflow>(d20, s0, weight0);
                                    if (M > 0x3) s0 = Set4(src + 0x3 * step + offs, tail), Madd4<overflow>(d30, s0, weight0);
                                    if (M > 0x4) s0 = Set4(src + 0x4 * step + offs, tail), Madd4<overflow>(d40, s0, weight0);
                                    if (M > 0x5) s0 = Set4(src + 0x5 * step + offs, tail), Madd4<overflow>(d50, s0, weight0);
                                    if (M > 0x6) s0 = Set4(src + 0x6 * step + offs, tail), Madd4<overflow>(d60, s0, weight0);
                                    if (M > 0x7) s0 = Set4(src + 0x7 * step + offs, tail), Madd4<overflow>(d70, s0, weight0);
                                    if (M > 0x8) s0 = Set4(src + 0x8 * step + offs, tail), Madd4<overflow>(d80, s0, weight0);
                                    if (M > 0x9) s0 = Set4(src + 0x9 * step + offs, tail), Madd4<overflow>(d90, s0, weight0);
                                    if (M > 0xA) s0 = Set4(src + 0xA * step + offs, tail), Madd4<overflow>(dA0, s0, weight0);
                                    if (M > 0xB) s0 = Set4(src + 0xB * step + offs, tail), Madd4<overflow>(dB0, s0, weight0);
                                }
                                if (offs < srcC)
                                {
                                    tail = srcC - offs;
                                    if (M > 0x0) s0 = Set4(src + 0x0 * step + offs, tail), Madd4<overflow>(d00, s0, weight0);
                                    if (M > 0x1) s0 = Set4(src + 0x1 * step + offs, tail), Madd4<overflow>(d10, s0, weight0);
                                    if (M > 0x2) s0 = Set4(src + 0x2 * step + offs, tail), Madd4<overflow>(d20, s0, weight0);
                                    if (M > 0x3) s0 = Set4(src + 0x3 * step + offs, tail), Madd4<overflow>(d30, s0, weight0);
                                    if (M > 0x4) s0 = Set4(src + 0x4 * step + offs, tail), Madd4<overflow>(d40, s0, weight0);
                                    if (M > 0x5) s0 = Set4(src + 0x5 * step + offs, tail), Madd4<overflow>(d50, s0, weight0);
                                    if (M > 0x6) s0 = Set4(src + 0x6 * step + offs, tail), Madd4<overflow>(d60, s0, weight0);
                                    if (M > 0x7) s0 = Set4(src + 0x7 * step + offs, tail), Madd4<overflow>(d70, s0, weight0);
                                    if (M > 0x8) s0 = Set4(src + 0x8 * step + offs, tail), Madd4<overflow>(d80, s0, weight0);
                                    if (M > 0x9) s0 = Set4(src + 0x9 * step + offs, tail), Madd4<overflow>(d90, s0, weight0);
                                    if (M > 0xA) s0 = Set4(src + 0xA * step + offs, tail), Madd4<overflow>(dA0, s0, weight0);
                                    if (M > 0xB) s0 = Set4(src + 0xB * step + offs, tail), Madd4<overflow>(dB0, s0, weight0);
                                    weight0 += A;
                                }
                            }
                            else if (a.zero)
                            {
                                for (size_t offs = 0; offs < srcC; offs += 4, weight0 += A)
                                {
                                    if (M > 0x0) Madd4<overflow>(d00, zero, weight0);
                                    if (M > 0x1) Madd4<overflow>(d10, zero, weight0);
                                    if (M > 0x2) Madd4<overflow>(d20, zero, weight0);
                                    if (M > 0x3) Madd4<overflow>(d30, zero, weight0);
                                    if (M > 0x4) Madd4<overflow>(d40, zero, weight0);
                                    if (M > 0x5) Madd4<overflow>(d50, zero, weight0);
                                    if (M > 0x6) Madd4<overflow>(d60, zero, weight0);
                                    if (M > 0x7) Madd4<overflow>(d70, zero, weight0);
                                    if (M > 0x8) Madd4<overflow>(d80, zero, weight0);
                                    if (M > 0x9) Madd4<overflow>(d90, zero, weight0);
                                    if (M > 0xA) Madd4<overflow>(dA0, zero, weight0);
                                    if (M > 0xB) Madd4<overflow>(dB0, zero, weight0);
                                }
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
                            {
                                if (M > 0x0) Madd4<overflow>(d00, zero, weight0);
                                if (M > 0x1) Madd4<overflow>(d10, zero, weight0);
                                if (M > 0x2) Madd4<overflow>(d20, zero, weight0);
                                if (M > 0x3) Madd4<overflow>(d30, zero, weight0);
                                if (M > 0x4) Madd4<overflow>(d40, zero, weight0);
                                if (M > 0x5) Madd4<overflow>(d50, zero, weight0);
                                if (M > 0x6) Madd4<overflow>(d60, zero, weight0);
                                if (M > 0x7) Madd4<overflow>(d70, zero, weight0);
                                if (M > 0x8) Madd4<overflow>(d80, zero, weight0);
                                if (M > 0x9) Madd4<overflow>(d90, zero, weight0);
                                if (M > 0xA) Madd4<overflow>(dA0, zero, weight0);
                                if (M > 0xB) Madd4<overflow>(dB0, zero, weight0);
                            }
                            weight0 += dW;
                        }
                    }
                    else
                        weight0 += (srcCF * A + dW) * p.kernelX;
                }
                if (M > 0x0) Save1<term, type>(dst + 0x0 * dD, buf + 0x0 * dB, d00, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x1) Save1<term, type>(dst + 0x1 * dD, buf + 0x1 * dB, d10, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x2) Save1<term, type>(dst + 0x2 * dD, buf + 0x2 * dB, d20, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x3) Save1<term, type>(dst + 0x3 * dD, buf + 0x3 * dB, d30, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x4) Save1<term, type>(dst + 0x4 * dD, buf + 0x4 * dB, d40, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x5) Save1<term, type>(dst + 0x5 * dD, buf + 0x5 * dB, d50, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x6) Save1<term, type>(dst + 0x6 * dD, buf + 0x6 * dB, d60, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x7) Save1<term, type>(dst + 0x7 * dD, buf + 0x7 * dB, d70, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x8) Save1<term, type>(dst + 0x8 * dD, buf + 0x8 * dB, d80, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0x9) Save1<term, type>(dst + 0x9 * dD, buf + 0x9 * dB, d90, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0xA) Save1<term, type>(dst + 0xA * dD, buf + 0xA * dB, dA0, norm, bias, params, scale, shift, a.upper, dstC);
                if (M > 0xB) Save1<term, type>(dst + 0xB * dD, buf + 0xB * dB, dB0, norm, bias, params, scale, shift, a.upper, dstC);
            }
        }

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
            const size_t F = a.F, DF = 2 * F, n = 1, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
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
            const size_t F = a.F, DF = 2 * F, n = 1, noseW = p.NoseW(), bodyW = p.BodyW(), bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
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
