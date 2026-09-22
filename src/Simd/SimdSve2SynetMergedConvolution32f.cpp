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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
	namespace Sve2
	{
		SynetMergedConvolution32fCdc::SynetMergedConvolution32fCdc(const MergConvParam& p)
			: Base::SynetMergedConvolution32fCdc(p)
		{
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), svcntw());
			SetInput(p.conv[0], _convolution + 0);
			SetDepthwise(p.conv[1], false, _convolution + 1);
			SetOutput(p.conv[2], _convolution + 2);
		}

		void SynetMergedConvolution32fCdc::ReorderFirstWeight(const float* src, float* dst) const
		{
			const SimdConvolutionParameters& p = _param.conv[0];
			const size_t F = svcntw(), DF = F * 2;
			const svbool_t ptrue = svptrue_b32();
			size_t K = p.kernelY * p.kernelX * p.srcC, N = p.dstC, NDF = AlignLo(N, DF);
			for (size_t j = 0; j < NDF; j += DF)
			{
				const float* ps = src;
				for (size_t k = 0; k < K; ++k, dst += DF, ps += N)
				{
					svst1_f32(ptrue, dst + 0, svld1_f32(ptrue, ps + 0));
					svst1_f32(ptrue, dst + F, svld1_f32(ptrue, ps + F));
				}
				src += DF;
			}
			if (NDF < N)
			{
				size_t T = N - NDF;
				const svbool_t mask0 = svwhilelt_b32((uint64_t)0, (uint64_t)T);
				const svbool_t mask1 = svwhilelt_b32((uint64_t)F, (uint64_t)T);
				const float* ps = src;
				for (size_t k = 0; k < K; ++k, dst += DF, ps += N)
				{
					svst1_f32(ptrue, dst + 0, svld1_f32(mask0, ps + 0));
					svst1_f32(ptrue, dst + F, svld1_f32(mask1, ps + F));
				}
			}
		}

		void SynetMergedConvolution32fCdc::ReorderSecondWeight(const float* src, float* dst) const
		{
			const SimdConvolutionParameters& p = _param.conv[1];
			const size_t F = svcntw();
			const svbool_t ptrue = svptrue_b32();
			size_t K = p.kernelY * p.kernelX, N = p.dstC, NF = AlignLo(N, F);
			for (size_t j = 0; j < NF; j += F)
			{
				const float* ps = src;
				for (size_t k = 0; k < K; ++k, dst += F, ps += N)
					svst1_f32(ptrue, dst, svld1_f32(ptrue, ps));
				src += F;
			}
			if (NF < N)
			{
				const svbool_t mask = svwhilelt_b32((uint64_t)0, (uint64_t)(N - NF));
				const float* ps = src;
				for (size_t k = 0; k < K; ++k, dst += F, ps += N)
					svst1_f32(ptrue, dst, svld1_f32(mask, ps));
			}
		}

		void SynetMergedConvolution32fCdc::ReorderThirdWeight(const float* src, float* dst) const
		{
			const SimdConvolutionParameters& p = _param.conv[2];
			const size_t F = svcntw(), DF = F * 2;
			const svbool_t ptrue = svptrue_b32();
			size_t srcC = p.srcC, N = p.dstC, NDF = AlignLo(N, DF);
			for (size_t m = 0; m < srcC; m += _maC)
			{
				size_t K = Simd::Min(srcC, m + _maC) - m;
				const float* src0 = src;
				for (size_t j = 0; j < NDF; j += DF)
				{
					const float* ps = src0;
					for (size_t k = 0; k < K; ++k, dst += DF, ps += N)
					{
						svst1_f32(ptrue, dst + 0, svld1_f32(ptrue, ps + 0));
						svst1_f32(ptrue, dst + F, svld1_f32(ptrue, ps + F));
					}
					src0 += DF;
				}
				if (NDF < N)
				{
					size_t T = N - NDF;
					const svbool_t mask0 = svwhilelt_b32((uint64_t)0, (uint64_t)T);
					const svbool_t mask1 = svwhilelt_b32((uint64_t)F, (uint64_t)T);
					const float* ps = src0;
					for (size_t k = 0; k < K; ++k, dst += DF, ps += N)
					{
						svst1_f32(ptrue, dst + 0, svld1_f32(mask0, ps + 0));
						svst1_f32(ptrue, dst + F, svld1_f32(mask1, ps + F));
					}
				}
				src += N * K;
			}
		}

		//-------------------------------------------------------------------------------------------------

		SynetMergedConvolution32fCd::SynetMergedConvolution32fCd(const MergConvParam& p)
			: Base::SynetMergedConvolution32fCd(p)
		{
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), svcntw());
			SetInput(_param.conv[0], _convolution + 0);
			SetDepthwise(_param.conv[1], true, _convolution + 1);
		}

		//-------------------------------------------------------------------------------------------------

		SynetMergedConvolution32fDc::SynetMergedConvolution32fDc(const MergConvParam& p)
			: Base::SynetMergedConvolution32fDc(p)
		{
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), svcntw());
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
				return new Sve2::SynetMergedConvolution32fCdc(param);
			else if (SynetMergedConvolution32fCd::Preferable(param))
				return new Sve2::SynetMergedConvolution32fCd(param);
			else if (SynetMergedConvolution32fDc::Preferable(param))
				return new Sve2::SynetMergedConvolution32fDc(param);
			else
				return new Base::SynetMergedConvolution32f(param);
		}
	}
#endif
}
