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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Sse41
	{
		SynetMergedConvolution32fCdc::SynetMergedConvolution32fCdc(const MergConvParam& p)
			: Base::SynetMergedConvolution32fCdc(p)
		{
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), F);
			SetInput(p.conv[0], _convolution + 0);
			SetDepthwise(p.conv[1], false, _convolution + 1);
			SetOutput(p.conv[2], _convolution + 2);
		}

		//-------------------------------------------------------------------------------------------------

		SynetMergedConvolution32fCd::SynetMergedConvolution32fCd(const MergConvParam& p)
			: Base::SynetMergedConvolution32fCd(p)
		{
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), F);
			SetInput(_param.conv[0], _convolution + 0);
			SetDepthwise(_param.conv[1], true, _convolution + 1);
		}

		//-------------------------------------------------------------------------------------------------

		SynetMergedConvolution32fDc::SynetMergedConvolution32fDc(const MergConvParam& p)
			: Base::SynetMergedConvolution32fDc(p)
		{
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), F);
			SetDepthwise(p.conv[0], false, _convolution + 0);
			SetOutput(p.conv[1], _convolution + 1);
		}

		//-------------------------------------------------------------------------------------------------

		void* SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add)
		{
			MergConvParam param(batch, convs, count, add, SimdSynetCompatibilityDefault);
			if (!param.Valid(SimdTensorData32f))
				return NULL;
			if (SynetMergedConvolution32fCdc::Preferable(param))
				return new Sse41::SynetMergedConvolution32fCdc(param);
			else if (SynetMergedConvolution32fCd::Preferable(param))
				return new Sse41::SynetMergedConvolution32fCd(param);
			else if (SynetMergedConvolution32fDc::Preferable(param))
				return new Sse41::SynetMergedConvolution32fDc(param);
			else
				return new Base::SynetMergedConvolution32f(param);
		}
	}
#endif
}
