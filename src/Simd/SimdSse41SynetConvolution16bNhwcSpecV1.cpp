/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse41
    {
        typedef Base::SynetConvolution16bNhwcSpecV1::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcSpecV1::PostprocessPtr PostprocessPtr;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcSpecV1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t padH = a.padH * p.srcC, padHDF = AlignLo(padH, DF);
            size_t sizeH = p.srcW * p.srcC, sizeHDF = AlignLo(sizeH, DF);
            if (dyBeg == 0)
            {
                size_t padV = a.padV * a.srcW * p.srcC;
                memset(dst, 0, padV * sizeof(uint16_t));
                dst += padV;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * p.srcC;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (padH)
                {
                    size_t i = 0;
                    for (; i < padHDF; i += DF)
                        Sse41::SetZero(dst + i);
                    for (; i < padH; i += 1)
                        dst[i] = 0;
                    dst += padH;
                }
                size_t x = 0;
                for (; x < sizeH; x += DF)
                    Sse41::Float32ToBFloat16(src + x, dst + x); 
                for (; x < sizeH; ++x)
                    dst[x] = Base::Float32ToBFloat16(src[x]);
                src += sizeH;
                dst += sizeH;
            }
            if (end)
            {
                size_t padE = a.padE * p.srcC;
                memset(dst, 0, padE * sizeof(uint16_t));
                dst += padE;
            }
        }

        static void Reorder16bNhwcSpecV1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t padH = a.padH * p.srcC, padHDF = AlignLo(padH, DF);
            size_t sizeH = p.srcW * p.srcC, sizeHDF = AlignLo(sizeH, DF);
            if (dyBeg == 0)
            {
                size_t padV = a.padV * a.srcW * p.srcC;
                memset(dst, 0, padV * sizeof(uint16_t));
                dst += padV;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * p.srcC;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (padH)
                {
                    size_t i = 0;
                    for (; i < padHDF; i += DF)
                        Sse41::SetZero(dst + i);
                    for (; i < padH; i += 1)
                        dst[i] = 0;
                    dst += padH;
                }
                memcpy(dst, src, sizeH * sizeof(uint16_t));
                src += sizeH;
                dst += sizeH;
            }
            if (end)
            {
                size_t padE = a.padE * p.srcC;
                memset(dst, 0, padE * sizeof(uint16_t));
                dst += padE;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<int M> void Convolution16bNhwcSpecV1_2xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t K, size_t dstC, int zero, const uint16_t* weight0, float* dst)
        {
            __m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11, m = _mm_castsi128_ps(Bf16::MASK);
            size_t dD = a.macroD, dS = p.srcC;
            const uint16_t* weight1 = weight0 + a.K * F;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            if (dstC > F)
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
                    if (M > 1) d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
                    if (M > 2) d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
                    if (M > 3) d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
                    if (M > 4) d40 = _mm_setzero_ps(), d41 = _mm_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_ps(dst + 0 * dD + 0), d01 = _mm_loadu_ps(dst + 0 * dD + F);
                    if (M > 1) d10 = _mm_loadu_ps(dst + 1 * dD + 0), d11 = _mm_loadu_ps(dst + 1 * dD + F);
                    if (M > 2) d20 = _mm_loadu_ps(dst + 2 * dD + 0), d21 = _mm_loadu_ps(dst + 2 * dD + F);
                    if (M > 3) d30 = _mm_loadu_ps(dst + 3 * dD + 0), d31 = _mm_loadu_ps(dst + 3 * dD + F);
                    if (M > 4) d40 = _mm_loadu_ps(dst + 4 * dD + 0), d41 = _mm_loadu_ps(dst + 4 * dD + F);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    size_t offs = *offset++;
                    w01 = _mm_loadu_ps((float*)weight0);
                    w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                    w01 = _mm_and_ps(w01, m);
                    w11 = _mm_loadu_ps((float*)weight1);
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
                    weight0 += DF;
                    weight1 += DF;
                }
                if (M > 0) _mm_storeu_ps(dst + 0 * dD + 0, d00), _mm_storeu_ps(dst + 0 * dD + F, d01);
                if (M > 1) _mm_storeu_ps(dst + 1 * dD + 0, d10), _mm_storeu_ps(dst + 1 * dD + F, d11);
                if (M > 2) _mm_storeu_ps(dst + 2 * dD + 0, d20), _mm_storeu_ps(dst + 2 * dD + F, d21);
                if (M > 3) _mm_storeu_ps(dst + 3 * dD + 0, d30), _mm_storeu_ps(dst + 3 * dD + F, d31);
                if (M > 4) _mm_storeu_ps(dst + 4 * dD + 0, d40), _mm_storeu_ps(dst + 4 * dD + F, d41);
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
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_ps(dst + 0 * dD + 0);
                    if (M > 1) d10 = _mm_loadu_ps(dst + 1 * dD + 0);
                    if (M > 2) d20 = _mm_loadu_ps(dst + 2 * dD + 0);
                    if (M > 3) d30 = _mm_loadu_ps(dst + 3 * dD + 0);
                    if (M > 4) d40 = _mm_loadu_ps(dst + 4 * dD + 0);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    size_t offs = *offset++;
                    w01 = _mm_loadu_ps((float*)weight0);
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
                    weight0 += DF;
                }
                if (M > 0) _mm_storeu_ps(dst + 0 * dD + 0, d00);
                if (M > 1) _mm_storeu_ps(dst + 1 * dD + 0, d10);
                if (M > 2) _mm_storeu_ps(dst + 2 * dD + 0, d20);
                if (M > 3) _mm_storeu_ps(dst + 3 * dD + 0, d30);
                if (M > 4) _mm_storeu_ps(dst + 4 * dD + 0, d40);
            }
        }

        typedef void(*Convolution16bNhwcSpecV1_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t K, size_t dstC, int zero, const uint16_t* weight0, float* dst);

        static Convolution16bNhwcSpecV1_2xM_Ptr GetConvolution16bNhwcSpecV1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Convolution16bNhwcSpecV1_2xM<1>;
            case 2: return Convolution16bNhwcSpecV1_2xM<2>;
            case 3: return Convolution16bNhwcSpecV1_2xM<3>;
            case 4: return Convolution16bNhwcSpecV1_2xM<4>;
            case 5: return Convolution16bNhwcSpecV1_2xM<5>;
            }
            assert(0);
            return NULL;
        }

        static void Convolution16bNhwcSpecV1_2(const uint16_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t K, int zero, const uint16_t* weight, float* dst)
        {
            size_t n1 = dstH * a.srcW + 1 - p.kernelX, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.K * DF;
            size_t dD = a.macroD, dS = p.srcC;
            Convolution16bNhwcSpecV1_2xM_Ptr convolution_2xN = GetConvolution16bNhwcSpecV1_2xM(n);
            Convolution16bNhwcSpecV1_2xM_Ptr convolution_2xM = GetConvolution16bNhwcSpecV1_2xM(m);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                for (; i < nn; i += n)
                    convolution_2xN(src + i * dS, p, a, offs, K, dC, zero, weight, dst + i * dD);
                for (; i < n1; i += m)
                    convolution_2xM(src + i * dS, p, a, offs, K, dC, zero, weight, dst + i * dD);
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        //-----------------------------------------------------------------------------------------

        SynetConvolution16bNhwcSpecV1::SynetConvolution16bNhwcSpecV1(const ConvParam & p)
            : Base::SynetConvolution16bNhwcSpecV1(p)
        {
            SetAlgParam(F, F * 2, 5, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
                _preprocess = Reorder16bNhwcSpecV1;
            else
                _preprocess = Convert16bNhwcSpecV1;
            _convolution = Convolution16bNhwcSpecV1_2;
            //switch (p.activation)
            //{
            //case SimdConvolutionActivationIdentity: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationRelu: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationLeakyRelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationRestrictRange: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationPrelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationElu: SetPostprocess<SimdConvolutionActivationElu>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationHswish: SetPostprocess<SimdConvolutionActivationHswish>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationMish: SetPostprocess<SimdConvolutionActivationMish>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationHardSigmoid: SetPostprocess<SimdConvolutionActivationHardSigmoid>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationSwish: SetPostprocess<SimdConvolutionActivationSwish>(p, _alg, _postprocess); break;
            //case SimdConvolutionActivationGelu: SetPostprocess<SimdConvolutionActivationGelu>(p, _alg, _postprocess); break;
            //default: assert(0);
            //}
        }
    }
#endif
}
