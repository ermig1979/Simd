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

        SIMD_INLINE void Float32ToBFloat16Tail(const float* src, size_t size, uint16_t* dst)
        {
            size_t i = 0;
            for (; i < size; ++i)
                dst[i] = Base::Float32ToBFloat16(src[i]);
            for (; i < DF; ++i)
                dst[i] = 0;
        }

        SIMD_INLINE void Copy(const uint16_t* src, size_t size, uint16_t* dst)
        {
            size_t i = 0;
            for (; i < size; ++i)
                dst[i] = src[i];
            for (; i < DF; ++i)
                dst[i] = 0;
        }

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcSpecV1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            if (dyBeg == 0)
            {
                size_t size = a.padV * a.srcW * p.srcC, sizeDF = AlignLo(size, DF), i = 0;
                for (; i < sizeDF; i += DF)
                    Sse41::SetZero(dst + i);
                for (; i < size; i += 1)
                    dst[i] = 0;
                //memset(dst, 0, size * sizeof(uint16_t));
                dst += size;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * p.srcC;
            }
            //for (size_t sy = syBeg; sy < syEnd; ++sy)
            //{
            //    if (p.padX)
            //    {
            //        for (size_t s = 0; s < p.padX; ++s)
            //            for (size_t c = 0; c < a.srcC; c += a.microC)
            //                Sse41::SetZero(dst + c * cD + s * sD);
            //        dst += p.padX * sD;
            //    }
            //    for (size_t sx = 0; sx < p.srcW; ++sx)
            //    {
            //        size_t sc = 0;
            //        for (; sc < srcCDF; sc += DF)
            //            Sse41::Float32ToBFloat16(src + sc, dst + sc * cD);
            //        if (tailC)
            //            Sse41::Float32ToBFloat16Tail(src + sc, tailC, dst + sc * cD);
            //        src += p.srcC;
            //        dst += sD;
            //    }
            //    if (p.padW)
            //    {
            //        for (size_t s = 0; s < p.padW; ++s)
            //            for (size_t c = 0; c < a.srcC; c += a.microC)
            //                Sse41::SetZero(dst + c * cD + s * sD);
            //        dst += p.padW * sD;
            //    }
            //}
            //if (dyEnd == p.dstH)
            //{
            //    for (size_t s = 0, n = p.padH * a.srcW; s < n; ++s)
            //        for (size_t c = 0; c < a.srcC; c += a.microC)
            //            Sse41::SetZero(dst + c * cD + s * sD);
            //    dst += p.padH * a.srcW * sD;
            //}
        }

        static void Reorder16bNhwcSpecV1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            //assert(a.microC == DF);
            //const uint16_t* src = (uint16_t*)src8;
            //size_t srcCDF = Simd::AlignLo(p.srcC, DF), tailC = p.srcC - srcCDF;
            //size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            //size_t cD = a.batch * a.srcH * a.srcW, sD = a.microC;
            //if (dyBeg == 0)
            //{
            //    for (size_t s = 0, n = p.padY * a.srcW; s < n; ++s)
            //        for (size_t c = 0; c < a.srcC; c += a.microC)
            //            Sse41::SetZero(dst + c * cD + s * sD);
            //    dst += p.padY * a.srcW * sD;
            //    syBeg = 0;
            //}
            //else
            //{
            //    syBeg = dyBeg + syPad;
            //    src += syBeg * p.srcW * p.srcC;
            //    dst += (dyBeg + p.kernelY - 1) * a.srcW * sD;
            //}
            //for (size_t sy = syBeg; sy < syEnd; ++sy)
            //{
            //    if (p.padX)
            //    {
            //        for (size_t s = 0; s < p.padX; ++s)
            //            for (size_t c = 0; c < a.srcC; c += a.microC)
            //                Sse41::SetZero(dst + c * cD + s * sD);
            //        dst += p.padX * sD;
            //    }
            //    for (size_t sx = 0; sx < p.srcW; ++sx)
            //    {
            //        size_t sc = 0;
            //        for (; sc < srcCDF; sc += DF)
            //            Sse41::Copy(src + sc, dst + sc * cD);
            //        if (tailC)
            //            Sse41::Copy(src + sc, tailC, dst + sc * cD);
            //        src += p.srcC;
            //        dst += sD;
            //    }
            //    if (p.padW)
            //    {
            //        for (size_t s = 0; s < p.padW; ++s)
            //            for (size_t c = 0; c < a.srcC; c += a.microC)
            //                Sse41::SetZero(dst + c * cD + s * sD);
            //        dst += p.padW * sD;
            //    }
            //}
            //if (dyEnd == p.dstH)
            //{
            //    for (size_t s = 0, n = p.padH * a.srcW; s < n; ++s)
            //        for (size_t c = 0; c < a.srcC; c += a.microC)
            //            Sse41::SetZero(dst + c * cD + s * sD);
            //    dst += p.padH * a.srcW * sD;
            //}
        }

        //-----------------------------------------------------------------------------------------


        //-----------------------------------------------------------------------------------------

        //-----------------------------------------------------------------------------------------

        SynetConvolution16bNhwcSpecV1::SynetConvolution16bNhwcSpecV1(const ConvParam & p)
            : Base::SynetConvolution16bNhwcSpecV1(p)
        {
            SetAlgParam(F, F * 2, 5, F * 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
                _preprocess = Reorder16bNhwcSpecV1;
            else
                _preprocess = Convert16bNhwcSpecV1;
            //_convolution = Convolution16bNhwcSpecV0_2;
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
