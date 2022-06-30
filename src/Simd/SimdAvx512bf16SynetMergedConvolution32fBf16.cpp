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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdAvx512bf16.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Avx512bf16
	{
        void ConvertFp32ToBf16(const float* src, const ConvParam32f& p, size_t yBeg, size_t yEnd, uint16_t* dst, size_t bufH)
        {
            size_t size = p.srcW * p.srcC, mask = bufH - 1;
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, bufH));
            if (yInt > yBeg)
                Float32ToBFloat16(src + yBeg * size, (yInt - yBeg) * size, dst + (yBeg & mask) * size);
            if (yEnd > yInt)
                Float32ToBFloat16(src + yInt * size, (yEnd - yInt) * size, dst + (yInt & mask) * size);
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32fBf16Cdc::SynetMergedConvolution32fBf16Cdc(const MergConvParam32f& p)
            : Avx512bw::SynetMergedConvolution32fBf16Cdc(p)
        {
            if (p.conv[2].dstC > HF)
            {
                SetSize(Avx512bw::F);
                _convert = ConvertFp32ToBf16;
                SetInput(_param.conv[0], _input);
                SetDepthwise(_param.conv[1], _depthwise);
                SetOutput(_param.conv[2], _output);
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32fBf16Cd::SynetMergedConvolution32fBf16Cd(const MergConvParam32f& p)
            : Avx512bw::SynetMergedConvolution32fBf16Cd(p)
        {
            if (p.conv[1].dstC > HF)
            {
                SetSize(Avx512bw::F);
                _convert = ConvertFp32ToBf16;
                SetInput(_param.conv[0], _input);
                SetDepthwise(_param.conv[1], _depthwise);
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution32fBf16Dc::SynetMergedConvolution32fBf16Dc(const MergConvParam32f& p)
            : Avx512bw::SynetMergedConvolution32fBf16Dc(p)
        {
            if (p.conv[0].dstC > HF && p.conv[1].dstC > HF)
            {
                SetSize(Avx512bw::F);
                SetDepthwise(_param.conv[0], _depthwise);
                SetOutput(_param.conv[1], _output);
            }
        }

        //---------------------------------------------------------------------

        void* SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add, SimdSynetCompatibilityType compatibility)
        {
            MergConvParam32f param(batch, convs, count, add, compatibility);
            if (!param.Valid())
                return NULL;
            if (Base::Bf16Soft(compatibility) || Base::Bf16Hard(compatibility))
            {
                if (Base::SynetMergedConvolution32fBf16Cdc::Preferable(param))
                    return new Avx512bf16::SynetMergedConvolution32fBf16Cdc(param);
                else if (Base::SynetMergedConvolution32fBf16Cd::Preferable(param))
                    return new Avx512bf16::SynetMergedConvolution32fBf16Cd(param);
                else if (Base::SynetMergedConvolution32fBf16Dc::Preferable(param))
                    return new Avx512bf16::SynetMergedConvolution32fBf16Dc(param);
                else
                    return new Base::SynetMergedConvolution32fBf16(param);
            }
            else 
                return Avx512bw::SynetMergedConvolution32fInit(batch, convs, count, add, compatibility);
        }
	}
#endif
}
