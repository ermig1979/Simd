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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)  
    namespace AmxBf16
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using InputPtr = Base::SynetMergedConvolution16b::InputConvolutionPtr;

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int flush, int M> static SIMD_INLINE void ApplyMx1(const float* src, float* dst0, float* dst1, float* dst2, float* dst3, const __m512* bias, const __m512* params)
        {
            if (M > 0)
            {
                _mm512_storeu_ps(dst0, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 0 * 256), bias[0]), params, 0));
                if (flush) _mm_prefetch((const char*)dst0, _MM_HINT_NTA);
            }
            if (M > 1)
            {
                _mm512_storeu_ps(dst1, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 1 * 256), bias[1]), params, 1));
                if (flush) _mm_prefetch((const char*)dst1, _MM_HINT_NTA);
            }
            if (M > 2)
            {
                _mm512_storeu_ps(dst1, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 2 * 256), bias[2]), params, 2));
                if (flush) _mm_prefetch((const char*)dst2, _MM_HINT_NTA);
            }
            if (M > 3)
            {
                _mm512_storeu_ps(dst1, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 3 * 256), bias[3]), params, 3));
                if (flush) _mm_prefetch((const char*)dst3, _MM_HINT_NTA);
            }
        }

        template<SimdConvolutionActivationType type, int flush, int M, int N> static SIMD_INLINE void ApplyMxN(const float* src, float* dst0, float* dst1, float* dst2, float* dst3, const __m512* bias, const __m512* params)
        {
            if (N > 0) ApplyMx1<type, flush, M>(src + 0 * F, dst0 + 0 * F, dst1 + 0 * F, dst2 + 0 * F, dst3 + 0 * F, bias, params);
            if (N > 1) ApplyMx1<type, flush, M>(src + 1 * F, dst0 + 1 * F, dst1 + 1 * F, dst2 + 1 * F, dst3 + 1 * F, bias, params);
            if (N > 2) ApplyMx1<type, flush, M>(src + 2 * F, dst0 + 2 * F, dst1 + 2 * F, dst2 + 2 * F, dst3 + 2 * F, bias, params);
            if (N > 3) ApplyMx1<type, flush, M>(src + 3 * F, dst0 + 3 * F, dst1 + 3 * F, dst2 + 3 * F, dst3 + 3 * F, bias, params);
            if (N > 4) ApplyMx1<type, flush, M>(src + 4 * F, dst0 + 4 * F, dst1 + 4 * F, dst2 + 4 * F, dst3 + 4 * F, bias, params);
            if (N > 5) ApplyMx1<type, flush, M>(src + 5 * F, dst0 + 5 * F, dst1 + 5 * F, dst2 + 5 * F, dst3 + 5 * F, bias, params);
            if (N > 6) ApplyMx1<type, flush, M>(src + 6 * F, dst0 + 6 * F, dst1 + 6 * F, dst2 + 6 * F, dst3 + 6 * F, bias, params);
            if (N > 7) ApplyMx1<type, flush, M>(src + 7 * F, dst0 + 7 * F, dst1 + 7 * F, dst2 + 7 * F, dst3 + 7 * F, bias, params);
        }

        //-----------------------------------------------------------------------------------------

        //-----------------------------------------------------------------------------------------

        //-----------------------------------------------------------------------------------------

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int apply> void InputConvolution1x1_4V2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* buf, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcC = AlignHi(p.srcC, a.miK), y0 = a.bufH[0] ? yBeg : 0;
            __m512 _bias[4], _params[4];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 16;
            size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLo(i1, n), i = i1 - in;
            size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLo(e1, n), e = e1 - en;

            //if (yInt == yBeg)
            //{
            //    if (en)
            //    {
            //        SetTileConfFull();
            //        for (size_t dc = 0; dc < maC; dc += DF)
            //        {
            //            size_t dC = Simd::Min(DF, maC - dc);
            //            _bias[0] = _mm512_loadu_ps(bias + dc + 0);
            //            _bias[1] = _mm512_loadu_ps(bias + dc + F);
            //            if (type == ::SimdConvolutionActivationPrelu)
            //            {
            //                _params[0] = _mm512_loadu_ps(params + dc + 0);
            //                _params[1] = _mm512_loadu_ps(params + dc + F);
            //            }
            //            if (dC > F)
            //            {
            //                const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
            //                float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
            //                InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, e1, weight, _bias, _params, buf, dst0, dst1);
            //            }
            //            else
            //            {
            //                const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
            //                float* dst0 = dst + (yInt & dstM) * p.dstW * F;
            //                InputConvolution1x1_Nx32V1<type, 1, 1, apply>(src0, p, a, e1, weight, _bias, _params, buf, dst0, NULL);
            //            }
            //            dst += a.bufH[1] * p.dstW * DF;
            //            weight += srcC * DF;
            //        }
            //    }
            //    else if (e1)
            //    {
            //        InputConvolution1x1V1Ptr conv_Ex2 = e > 16 ? InputConvolution1x1_2x2V1<type, 0> : InputConvolution1x1_1x2V1<type, 0>;
            //        InputConvolution1x1V1Ptr conv_Ex1 = e > 16 ? InputConvolution1x1_2x1V1<type, 0> : InputConvolution1x1_1x1V1<type, 0>;
            //        if (e > 16)
            //            SetTileConf2x2(e, 32);
            //        else
            //            SetTileConf1x2(e, 32);
            //        for (size_t dc = 0; dc < maC; dc += DF)
            //        {
            //            size_t dC = Simd::Min(DF, maC - dc);
            //            _bias[0] = _mm512_loadu_ps(bias + dc + 0);
            //            _bias[1] = _mm512_loadu_ps(bias + dc + F);
            //            if (type == ::SimdConvolutionActivationPrelu)
            //            {
            //                _params[0] = _mm512_loadu_ps(params + dc + 0);
            //                _params[1] = _mm512_loadu_ps(params + dc + F);
            //            }
            //            if (dC > F)
            //            {
            //                const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
            //                float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
            //                conv_Ex2(src0, p, a, e, weight, _bias, _params, buf, dst0, dst1);
            //            }
            //            else
            //            {
            //                const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
            //                float* dst0 = dst + (yInt & dstM) * p.dstW * F;
            //                conv_Ex1(src0, p, a, e, weight, _bias, _params, buf, dst0, NULL);
            //            }
            //            dst += a.bufH[1] * p.dstW * DF;
            //            weight += srcC * DF;
            //        }
            //    }
            //}
            //else
            //{
            //    InputConvolution1x1V1Ptr conv_Ix2 = i > 16 ? InputConvolution1x1_2x2V1<type, 1> : InputConvolution1x1_1x2V1<type, 1>;
            //    InputConvolution1x1V1Ptr conv_Ix1 = i > 16 ? InputConvolution1x1_2x1V1<type, 1> : InputConvolution1x1_1x1V1<type, 1>;
            //    InputConvolution1x1V1Ptr conv_Ex2 = e > 16 ? InputConvolution1x1_2x2V1<type, 1> : InputConvolution1x1_1x2V1<type, 1>;
            //    InputConvolution1x1V1Ptr conv_Ex1 = e > 16 ? InputConvolution1x1_2x1V1<type, 1> : InputConvolution1x1_1x1V1<type, 1>;
            //    for (size_t dc = 0; dc < maC; dc += DF)
            //    {
            //        size_t dC = Simd::Min(DF, maC - dc);
            //        _bias[0] = _mm512_loadu_ps(bias + dc + 0);
            //        _bias[1] = _mm512_loadu_ps(bias + dc + F);
            //        if (type == ::SimdConvolutionActivationPrelu)
            //        {
            //            _params[0] = _mm512_loadu_ps(params + dc + 0);
            //            _params[1] = _mm512_loadu_ps(params + dc + F);
            //        }
            //        if (dC > F)
            //        {
            //            if (yInt > yBeg)
            //            {
            //                SetTileConfFull();
            //                const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
            //                float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
            //                if(in)
            //                    InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, i1, weight, _bias, _params, buf, dst0, dst1);
            //                else if (i)
            //                    conv_Ix2(src0, p, a, i, weight, _bias, _params, buf, dst0, dst1);
            //            }
            //            if (yEnd > yInt)
            //            {
            //                SetTileConfFull();
            //                const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
            //                float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
            //                if (in)
            //                    InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, e1, weight, _bias, _params, buf, dst0, dst1);
            //                else if (e)
            //                    conv_Ex2(src0, p, a, e, weight, _bias, _params, buf, dst0, dst1);
            //            }
            //        }
            //        else
            //        {
            //            if (yInt > yBeg)
            //            {
            //                SetTileConfFull();
            //                const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
            //                float* dst0 = dst + (yBeg & dstM) * p.dstW * F;
            //                if(in)
            //                    InputConvolution1x1_Nx32V1<type, 1, 1, apply>(src0, p, a, i1, weight, _bias, _params, buf, dst0, NULL);
            //                else if (i)
            //                    conv_Ix1(src0, p, a, i, weight, _bias, _params, buf, dst0, NULL);
            //            }
            //            if (yEnd > yInt)
            //            {
            //                SetTileConfFull();
            //                const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
            //                float* dst0 = dst + (yInt & dstM) * p.dstW * F;
            //                if(en)
            //                    InputConvolution1x1_Nx32V1<type, 1, 1, apply>(src0, p, a, e1, weight, _bias, _params, buf, dst0, NULL);
            //                else if (e)
            //                    conv_Ex1(src0, p, a, e, weight, _bias, _params, buf, dst0, NULL);
            //            }
            //        }
            //        dst += a.bufH[1] * p.dstW * DF;
            //        weight += srcC * DF;
            //    }
            //}
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInputV2(const ConvParam& p, InputPtr& input)
        {
            if (Is1x1(p))
            {
                if (p.srcC >= 96)
                    input = InputConvolution1x1_4V2<type, 1>;
                else
                    input = InputConvolution1x1_4V2<type, 2>;
            }
            else
                assert(0);
        }

        void SetInputV2(const ConvParam& p, InputPtr& input)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetInputV2<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationRelu: SetInputV2<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationLeakyRelu: SetInputV2<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationRestrictRange: SetInputV2<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationPrelu: SetInputV2<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationElu: SetInputV2<SimdConvolutionActivationElu>(p, input); break;
            case SimdConvolutionActivationHswish: SetInputV2<SimdConvolutionActivationHswish>(p, input); break;
            case SimdConvolutionActivationMish: SetInputV2<SimdConvolutionActivationMish>(p, input); break;
            case SimdConvolutionActivationHardSigmoid: SetInputV2<SimdConvolutionActivationHardSigmoid>(p, input); break;
            case SimdConvolutionActivationSwish: SetInputV2<SimdConvolutionActivationSwish>(p, input); break;
            case SimdConvolutionActivationGelu: SetInputV2<SimdConvolutionActivationGelu>(p, input); break;
            }
        }
    }
#endif
}
