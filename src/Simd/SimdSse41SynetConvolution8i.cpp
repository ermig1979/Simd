/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdSse2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        using AlgParam = SynetConvolution8iNhwcDirect::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDirect::ConvolutionPtr;
        using Term8iType = Base::SynetConvolution8iNhwcDirect::Term8iType;

        SIMD_INLINE __m128i Set4(const uint8_t* src)
        {
            return _mm_set1_epi32(*(int32_t*)src);
        }

        template<bool overflow> void Madd4(__m128i& i32, __m128i u8, __m128i i8);

        template<> SIMD_INLINE void Madd4<true>(__m128i& i32, __m128i u8, __m128i i8)
        {
            i32 = _mm_add_epi32(i32, _mm_madd_epi16(_mm_maddubs_epi16(u8, i8), Sse2::K16_0001));
        }

        template<> SIMD_INLINE void Madd4<false>(__m128i& i32, __m128i u8, __m128i i8)
        {
            __m128i lo = _mm_madd_epi16(UnpackU8<0>(u8), UnpackI8<0>(i8));
            __m128i hi = _mm_madd_epi16(UnpackU8<1>(u8), UnpackI8<1>(i8));
            i32 = _mm_add_epi32(i32, _mm_hadd_epi32(lo, hi));
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2x1(const uint8_t * src0, 
            const ConvParam8i& p, const AlgParam & a, size_t dy, size_t dx, size_t srcC, size_t dstC, const int8_t * weight0, 
            const __m128i * bias, const __m128i * params, const __m128 * scale, const __m128* shift, int32_t * buf, uint8_t* dst)
        {
            __m128i d00, d01, s0, w0, w1;
            size_t dW = (DivHi(p.srcC, 4) - DivHi(srcC, 4)) * A, dY = p.srcW * p.srcC, dX = p.srcC;
            const int8_t* weight1 = weight0 + p.kernelY * p.kernelX * DivHi(p.srcC, 4) * A;
            __m128i norm = _mm_set1_epi32(a.norm);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm_setzero_si128();
                d01 = _mm_setzero_si128();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm_loadu_si128((__m128i*)weight0);
                                w1 = _mm_loadu_si128((__m128i*)weight1);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d01, s0, w1);
                                weight0 += A;
                                weight1 += A;
                            }
                        }
                        else
                        {
                            s0 = _mm_set1_epi32(a.zero);
                            for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                            {
                                w0 = _mm_loadu_si128((__m128i*)weight0);
                                w1 = _mm_loadu_si128((__m128i*)weight1);
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d01, s0, w1);
                                weight0 += A;
                                weight1 += A;
                            }
                        }
                        weight0 += dW;
                        weight1 += dW;
                    }
                }
                if (dstC == DF)
                {
                    Term<term>::template Save<type, 0>(dst, buf, d00, norm, bias, params, scale, shift);
                    Term<term>::template Save<type, 1>(dst, buf, d01, norm, bias, params, scale, shift);
                }
                else
                {
                    Term<term>::template Save<type, 0>(dst, buf, d00, norm, bias, params, scale, shift);
                    Term<term>::template Save<type, 1>(dst, buf, d01, norm, bias, params, scale, shift, dstC - F);
                }
            }
            else
            {
                d00 = _mm_setzero_si128();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sy + ky) * dY + (sx + kx) * dX, end = offs + srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm_loadu_si128((__m128i*)weight0);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                weight0 += A;
                            }
                        }
                        else
                        {
                            s0 = _mm_set1_epi32(a.zero);
                            for (size_t offs = 0, end = srcC; offs < end; offs += 4)
                            {
                                w0 = _mm_loadu_si128((__m128i*)weight0);
                                Madd4<overflow>(d00, s0, w0);
                                weight0 += A;
                            }
                        }
                        weight0 += dW;
                    }
                }
                if (dstC == F)
                {
                    Term<term>::template Save<type, 0>(dst, buf, d00, norm, bias, params, scale, shift);
                }
                else
                {
                    Term<term>::template Save<type, 0>(dst, buf, d00, norm, bias, params, scale, shift, dstC);
                }
            }
        }

        template<bool overflow, Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect_2(const uint8_t* src, 
            const ConvParam8i & p, const AlgParam & a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight,
            const int32_t* bias, const int32_t * params, const float * scale, const float* shift, int32_t* buf, uint8_t* dst)
        {
            size_t noseH = p.NoseH(), noseW = p.NoseW(), bodyH = p.BodyH(), bodyW = p.BodyW();
            //size_t bodyW6 = AlignLoAny(bodyW - noseW, 6 * p.strideX) + noseW;
            size_t tailH = p.dstH, tailW = p.dstW;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;
            __m128i _params[2], _bias[2];
            _params[0] = _mm_setzero_si128();
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm_set1_epi32(a.high);
            __m128 _scale[2], _shift[2];

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm_loadu_si128((__m128i*)(bias + dc + 0));
                _bias[1] = _mm_loadu_si128((__m128i*)(bias + dc + F));
                _scale[0] = _mm_loadu_ps(scale + dc + 0);
                _scale[1] = _mm_loadu_ps(scale + dc + F);
                _shift[0] = _mm_loadu_ps(shift + dc + 0);
                _shift[1] = _mm_loadu_ps(shift + dc + F);

                uint8_t * d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t * b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t dy = yBeg;
                for (; dy < noseH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                }
                for (; dy < bodyH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                }
                for (; dy < tailH && dy < yEnd; dy++)
                {
                    size_t dx = 0;
                    for (; dx < noseW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < bodyW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                    for (; dx < tailW; dx++, b += p.dstC, d += p.dstC * a.size)
                        ConvolutionNhwcDirect_2x1<overflow, term, type>(src, p, a, dy, dx, srcC, dC, weight, _bias, _params, _scale, _shift, b, d);
                }
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * DA;
            }
        }

        template <bool overflow, Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam & a, ConvolutionPtr * d)
        {
            //if (p.Is1x1())
            //{
            //    switch (microD)
            //    {
            //    case 2 * F: convolution = ConvolutionNhwcDirect1x1_2<type>; break;
            //    default:
            //        assert(0);
            //    }
            //}
            //else
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
            if (p.compatibility & SimdSynetCompatibilityOverflow16i)
                Set<true, term, activation>(p, a, d);
            else
                Set<false, term, activation>(p, a, d);
        }        
        
        template<SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iFirst, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iIterim, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iLast8u, activation>(p, a, d);
            Set<Base::SynetConvolution8iNhwcDirect::Term8iLast32f, activation>(p, a, d);
        }

        static void Set(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr * d)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationIdentity>(p, a, d); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRelu>(p, a, d); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            default: assert(0);
            }
        }

        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam8i& p)
            : Base::SynetConvolution8iNhwcDirect(p)
        {
            SetAlgParam(F, 2 * F, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            Set(p, _alg, _convolutions);
            _convertSrc = Sse2::SynetConvert32fTo8u;
        }

        bool SynetConvolution8iNhwcDirect::Preferable(const ConvParam8i& p)
        {
            if (p.trans != SimdTrue || p.group != 1)
                return false;
            return true;
        }

        //---------------------------------------------------------------------

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
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
