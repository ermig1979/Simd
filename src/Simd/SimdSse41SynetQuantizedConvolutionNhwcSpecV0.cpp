/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetQuantizedDepthwise.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        typedef Base::SynetQuantizedConvolutionNhwcSpecV0::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcSpecV0::ConvolutionPtr Convolution;

        SIMD_INLINE void Copy(const uint8_t* src, size_t size, uint8_t* dst)
        {
            size_t i = 0;
            for (; i < size; ++i)
                dst[i] = src[i];
            for (; i < A; ++i)
                dst[i] = 0;
        }

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcSpecV0Reorder(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint8_t* dst)
        {
            assert(a.microC == A);
            __m128i _zero = _mm_set1_epi8(zero);
            size_t srcCA = Simd::AlignLo(p.srcC, A), tailC = p.srcC - srcCA;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        _mm_storeu_si128((__m128i*)(dst + c * cD + s * sD), _zero);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            _mm_storeu_si128((__m128i*)(dst + c * cD + s * sD), _zero);
                    dst += a.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCA; sc += A)
                        Sse41::Copy(src + sc, dst + sc * cD);
                    if (tailC)
                        Sse41::Copy(src + sc, tailC, dst + sc * cD);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        _mm_storeu_si128((__m128i*)(dst + c * cD + s * sD), _zero);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        _mm_storeu_si128((__m128i*)(dst + c * cD + s * sD), _zero);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<int M> void QuantizedConvolutionNhwcSpecV0_2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, size_t dstC, int update, const int8_t* weight0, int32_t* dst)
        {
            __m128i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dD = a.macroD, dX = a.microC;
            const int8_t* weight1 = weight0 + a.K * F;
            const uint8_t* src1 = src0 + 1 * dX;
            const uint8_t* src2 = src0 + 2 * dX;
            const uint8_t* src3 = src0 + 3 * dX;
            const uint8_t* src4 = src0 + 4 * dX;
            if (dstC > F)
            {
                if (update)
                {
                    if (M > 0x0) d00 = _mm_loadu_si128((__m128i*)(dst + 0 * dD + 0)), d01 = _mm_loadu_si128((__m128i*)(dst + 0 * dD + F));
                    if (M > 0x1) d10 = _mm_loadu_si128((__m128i*)(dst + 1 * dD + 0)), d11 = _mm_loadu_si128((__m128i*)(dst + 1 * dD + F));
                    if (M > 0x2) d20 = _mm_loadu_si128((__m128i*)(dst + 2 * dD + 0)), d21 = _mm_loadu_si128((__m128i*)(dst + 2 * dD + F));
                    if (M > 0x3) d30 = _mm_loadu_si128((__m128i*)(dst + 3 * dD + 0)), d31 = _mm_loadu_si128((__m128i*)(dst + 3 * dD + F));
                    if (M > 0x4) d40 = _mm_loadu_si128((__m128i*)(dst + 4 * dD + 0)), d41 = _mm_loadu_si128((__m128i*)(dst + 4 * dD + F));
                }
                else
                {
                    if (M > 0x0) d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                    if (M > 0x1) d10 = _mm_setzero_si128(), d11 = _mm_setzero_si128();
                    if (M > 0x2) d20 = _mm_setzero_si128(), d21 = _mm_setzero_si128();
                    if (M > 0x3) d30 = _mm_setzero_si128(), d31 = _mm_setzero_si128();
                    if (M > 0x4) d40 = _mm_setzero_si128(), d41 = _mm_setzero_si128();
                }
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs = offset[k], end = offs + dX; offs < end; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight0);
                        w1 = _mm_loadu_si128((__m128i*)weight1);
                        if (M > 0x0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                        if (M > 0x1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                        if (M > 0x2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                        if (M > 0x3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                        if (M > 0x4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                        weight0 += A;
                        weight1 += A;
                    }
                }
                if (M > 0x0) _mm_storeu_si128((__m128i*)(dst + 0 * dD + 0), d00), _mm_storeu_si128((__m128i*)(dst + 0 * dD + F), d01);
                if (M > 0x1) _mm_storeu_si128((__m128i*)(dst + 1 * dD + 0), d10), _mm_storeu_si128((__m128i*)(dst + 1 * dD + F), d11);
                if (M > 0x2) _mm_storeu_si128((__m128i*)(dst + 2 * dD + 0), d20), _mm_storeu_si128((__m128i*)(dst + 2 * dD + F), d21);
                if (M > 0x3) _mm_storeu_si128((__m128i*)(dst + 3 * dD + 0), d30), _mm_storeu_si128((__m128i*)(dst + 3 * dD + F), d31);
                if (M > 0x4) _mm_storeu_si128((__m128i*)(dst + 4 * dD + 0), d40), _mm_storeu_si128((__m128i*)(dst + 4 * dD + F), d41);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) d00 = _mm_loadu_si128((__m128i*)(dst + 0 * dD + 0));
                    if (M > 0x1) d10 = _mm_loadu_si128((__m128i*)(dst + 1 * dD + 0));
                    if (M > 0x2) d20 = _mm_loadu_si128((__m128i*)(dst + 2 * dD + 0));
                    if (M > 0x3) d30 = _mm_loadu_si128((__m128i*)(dst + 3 * dD + 0));
                    if (M > 0x4) d40 = _mm_loadu_si128((__m128i*)(dst + 4 * dD + 0));
                }
                else
                {
                    if (M > 0x0) d00 = _mm_setzero_si128();
                    if (M > 0x1) d10 = _mm_setzero_si128();
                    if (M > 0x2) d20 = _mm_setzero_si128();
                    if (M > 0x3) d30 = _mm_setzero_si128();
                    if (M > 0x4) d40 = _mm_setzero_si128();
                }
                for (size_t k = 0; k < nK; k += 1)
                {
                    for (size_t offs = offset[k], end = offs + dX; offs < end; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight0);
                        if (M > 0x0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                        if (M > 0x1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                        if (M > 0x2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                        if (M > 0x3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                        if (M > 0x4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                        weight0 += A;
                    }
                }
                if (M > 0x0) _mm_storeu_si128((__m128i*)(dst + 0 * dD + 0), d00);
                if (M > 0x1) _mm_storeu_si128((__m128i*)(dst + 1 * dD + 0), d10);
                if (M > 0x2) _mm_storeu_si128((__m128i*)(dst + 2 * dD + 0), d20);
                if (M > 0x3) _mm_storeu_si128((__m128i*)(dst + 3 * dD + 0), d30);
                if (M > 0x4) _mm_storeu_si128((__m128i*)(dst + 4 * dD + 0), d40);
            }
        }

        typedef void(*QuantizedConvolutionNhwcSpecV0_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, size_t dstC, int update, const int8_t* weight0, int32_t* dst);

        static QuantizedConvolutionNhwcSpecV0_2xM_Ptr GetQuantizedConvolutionNhwcSpecV0_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return QuantizedConvolutionNhwcSpecV0_2xM<0x1>;
            case 0x2: return QuantizedConvolutionNhwcSpecV0_2xM<0x2>;
            case 0x3: return QuantizedConvolutionNhwcSpecV0_2xM<0x3>;
            case 0x4: return QuantizedConvolutionNhwcSpecV0_2xM<0x4>;
            case 0x5: return QuantizedConvolutionNhwcSpecV0_2xM<0x5>;
            }
            assert(0);
            return NULL;
        }

        static void QuantizedConvolutionNhwcSpecV0_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t nK, int update, const int8_t* weight, int32_t* dst)
        {
            size_t n1 = dstH * a.srcW - a.gapH, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.K * DF;
            size_t dD = a.macroD, dS = a.microC;
            QuantizedConvolutionNhwcSpecV0_2xM_Ptr convolution_2xN = GetQuantizedConvolutionNhwcSpecV0_2xM(n);
            QuantizedConvolutionNhwcSpecV0_2xM_Ptr convolution_2xM = GetQuantizedConvolutionNhwcSpecV0_2xM(m);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                for (; i < nn; i += n)
                    convolution_2xN(src + i * dS, p, a, offs, nK, dC, update, weight, dst + i * dD);
                for (; i < n1; i += m)
                    convolution_2xM(src + i * dS, p, a, offs, nK, dC, update, weight, dst + i * dD);
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> void SynetQuantizedConvolutionNhwcSpecV0Postprocess(const int32_t* src, const ConvParam& p, const AlgParam& a, 
            size_t dstC, size_t dyBeg, size_t dyEnd, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst)
        {
            size_t dstCF = AlignLo(dstC, F), tailD = dstC - dstCF;
            size_t rowGap = a.gapH * a.macroD;
            src += dyBeg * a.srcW * a.macroD;
            dst += dyBeg * p.dstW * p.dstC * a.elem;
            __m128 _sNorm, _iScale, _params[2], _dNorm;
            __m128i _src, _dZero = _mm_set1_epi32(dZero), _sBias, _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm_set1_epi32(-iZero);
                _iHi = _mm_set1_epi32(255 - iZero);
                _iScale = _mm_set1_ps(iScale);
                _dNorm = _mm_set1_ps(dNorm);
                _params[0] = _mm_set1_ps(params[0]);
                _params[1] = _mm_set1_ps(params[1]);
            }
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < dstCF; dc += F)
                    {
                        _src = _mm_loadu_si128((__m128i*)(src + dc));
                        _sBias = _mm_loadu_si128((__m128i*)(sBias + dc));
                        _sNorm = _mm_loadu_ps(sNorm + dc);
                        if (type == SimdConvolutionActivationPrelu)
                            _params[0] = _mm_loadu_ps(params + dc);
                        Save1<Term8iLast8u, type>(dst + dc, _src, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                    }
                    if (tailD)
                    {
                        _src = _mm_loadu_si128((__m128i*)(src + dc));
                        _sBias = _mm_loadu_si128((__m128i*)(sBias + dc));
                        _sNorm = _mm_loadu_ps(sNorm + dc);
                        if (type == SimdConvolutionActivationPrelu)
                            _params[0] = _mm_loadu_ps(params + dc);
                        Save1<Term8iLast8u, type>(dst + dc, _src, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tailD);
                    }
                    src += a.macroD;
                    dst += p.dstC * a.elem;
                }
                src += rowGap;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcSpecV0::SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcSpecV0(p)
        {
            SetAlgParam(F, F * 2, 5, F * 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            AlgParam& a = _alg;
            if (_src8u)
            {
                _preprocess = QuantizedConvolutionNhwcSpecV0Reorder;
            }
            else
                assert(0);
            _convolution = QuantizedConvolutionNhwcSpecV0_2;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: _postprocess = SynetQuantizedConvolutionNhwcSpecV0Postprocess<SimdConvolutionActivationGelu>; break;
            default:
                _postprocess = NULL;
                assert(0);
            }
        }
    }
#endif
}
