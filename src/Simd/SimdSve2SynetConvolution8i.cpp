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

        template<bool overflow> SIMD_INLINE void Madd4(svint32_t& sum, const svuint8_t& src, const int8_t* weight)
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
            svint16_t pairs;
            if (overflow)
                pairs = svqadd_s16(lo, hi);
            else
                pairs = svadd_s16_x(body16, lo, hi);
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
                svst1_s32(svwhilelt_b32((size_t)0, tail), sums, sum);
                for (size_t i = 0; i < tail; ++i)
                {
                    float value = Activate<type>(float(sums[i]) * norm[i] + bias[i], params, i);
                    dst[i] = (uint8_t)Simd::RestrictRange(Simd::Round(value * scale[i] + shift[i]), 0, upper);
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

        template<bool overflow, Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            d[term] = ConvolutionNhwcDirect<overflow, term, activation>;
        }

        template<Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            if (Base::Overflow(p.compatibility))
                Set<true, term, activation>(p, a, d);
            else
                Set<false, term, activation>(p, a, d);
        }

        template<SimdConvolutionActivationType activation> void Set(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
        {
            Set<Term8iLast8u, activation>(p, a, d);
            Set<Term8iLast32f, activation>(p, a, d);
            Set<Term8iInterim, SimdConvolutionActivationIdentity>(p, a, d);
        }

        static void Set(const ConvParam& p, const AlgParam& a, ConvolutionPtr* d)
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
            case SimdConvolutionActivationGelu: Set<SimdConvolutionActivationGelu>(p, a, d); break;
            default: assert(0);
            }
        }

        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam& p)
            : Base::SynetConvolution8iNhwcDirect(p)
        {
            size_t F = svcntw();
            SetAlgParam(F, F, 1, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            Set(p, _alg, _convolutions);
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
