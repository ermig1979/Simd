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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if (defined(SIMD_AMX_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)   
    namespace Amx
    {
        SynetMergedConvolution8iCdc::SynetMergedConvolution8iCdc(const MergConvParam8i& p)
#if defined(SIMD_AMX_EMULATE)
            : Avx512bw::SynetMergedConvolution8iCdc(p)
#else
            : Avx512vnni::SynetMergedConvolution8iCdc(p)
#endif
        {
            SetSize(Avx512bw::F);
            _cvt32fTo8u = _s8u ? NULL : Avx512bw::Convert32fTo8u;
            if(_param.conv[0].Is1x1())
                Avx512bw::SetInput(_param.conv[0], _input);
            else
            {
#if defined(SIMD_AMX_EMULATE)
                Avx512bw::SetInput(_param.conv[0], _input);
#else
                Avx512vnni::SetInput(_param.conv[0], _input);
#endif
            }
            Avx512bw::SetDepthwise(_param.conv[1], _depthwise);
            Avx512bw::SetOutput(_param.conv[2], _output);
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iCd::SynetMergedConvolution8iCd(const MergConvParam8i& p)
#if defined(SIMD_AMX_EMULATE)
            : Avx512bw::SynetMergedConvolution8iCd(p)
#else
            : Avx512vnni::SynetMergedConvolution8iCd(p)
#endif
        {
            SetSize(Avx512bw::F);
            _cvt32fTo8u = _s8u ? NULL : Avx512bw::Convert32fTo8u;
            if (_param.conv[0].Is1x1())
                Avx512bw::SetInput(_param.conv[0], _input);
            else
            {
#if defined(SIMD_AMX_EMULATE)
                Avx512bw::SetInput(_param.conv[0], _input);
#else
                Avx512vnni::SetInput(_param.conv[0], _input);
#endif
            }
            Avx512bw::SetDepthwise(_param.conv[1], _depthwise);
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iDc::SynetMergedConvolution8iDc(const MergConvParam8i& p)
#if defined(SIMD_AMX_EMULATE)
            : Avx512bw::SynetMergedConvolution8iDc(p)
#else
            : Avx512vnni::SynetMergedConvolution8iDc(p)
#endif
        {
            SetSize(Avx512bw::F);
            _cvt8uTo32f = _s8u ? (Convert8uTo32fPtr)Avx512bw::Convert8uTo32f : NULL;
            Avx512bw::SetDepthwise(_param.conv[0], _depthwise);
            Avx512bw::SetOutput(_param.conv[1], _output);
        }

        //---------------------------------------------------------------------

        void* SynetMergedConvolution8iInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdSynetCompatibilityType compatibility)
        {
            MergConvParam8i param(batch, convs, count, compatibility);
            if (!param.Valid())
                return NULL;
            if (SynetMergedConvolution8iCdc::Preferable(param))
                return new Amx::SynetMergedConvolution8iCdc(param);
            else if (SynetMergedConvolution8iCd::Preferable(param))
                return new Amx::SynetMergedConvolution8iCd(param);
            else if (SynetMergedConvolution8iDc::Preferable(param))
                return new Amx::SynetMergedConvolution8iDc(param);
            else
                return new Base::SynetMergedConvolution8i(param);
        }
    }
#endif
}
