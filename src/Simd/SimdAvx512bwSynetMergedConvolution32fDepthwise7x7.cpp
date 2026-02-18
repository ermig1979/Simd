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
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Avx512bw
    {
		template<SimdConvolutionActivationType type> void DepthwiseConvolution_k7p3d1s1w4(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			assert(IsKernel(p, 7) && IsPad(p, 3) && IsStride(p, 1) && IsDilation(p, 1) && Aligned(p.srcW, 4));

			size_t dstC = srcC, dstW = p.dstW, srcH = p.srcH, end = dstW - 4;
			size_t sM = (bufH[0] - 1), sD = bufH[0] ? bufH[0] * p.srcW * F : F, sX = bufH[0] ? F : p.srcC, sY = sX * p.srcW;
			size_t dM = (bufH[1] - 1), dX = (bufH[1] ? F : p.dstC), dY = p.dstW * dX, dD = bufH[1] ? bufH[1] * dY : F;
			size_t wD = 49 * F;

			__m512 s0, s1, w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3, _params[2];
			_params[0] = _mm512_set1_ps(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				_params[1] = _mm512_set1_ps(params[1]);
			for (size_t c = 0; c < dstC; c += F)
			{
				__m512 _bias = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
				if (type == ::SimdConvolutionActivationPrelu)
					_params[0] = _mm512_loadu_ps(params + c);
				__mmask16 tail = TailMask16(dstC - c);
				for (size_t dy = yBeg; dy < yEnd; ++dy)
				{
					for (size_t dx = 0; dx < dstW; dx += 4)
					{
						d0 = _bias, d1 = _bias, d2 = _bias, d3 = _bias;
						for (size_t ky = 0; ky < 7; ++ky)
						{
							size_t sy = dy + ky - 3;
							const float* ps = src + (sy & sM) * sY + (dx - 3) * sX;
							const float* pw = weight + ky * 7 * F;
							if (sy < srcH)
							{
								w0 = _mm512_maskz_loadu_ps(tail, pw + 0 * F);
								w1 = _mm512_maskz_loadu_ps(tail, pw + 1 * F);
								w2 = _mm512_maskz_loadu_ps(tail, pw + 2 * F);
								if (dx)
								{
									s0 = _mm512_maskz_loadu_ps(tail, ps + 0 * sX);
									d0 = _mm512_fmadd_ps(s0, w0, d0);

									s1 = _mm512_maskz_loadu_ps(tail, ps + 1 * sX);
									d0 = _mm512_fmadd_ps(s1, w1, d0);
									d1 = _mm512_fmadd_ps(s1, w0, d1);

									s0 = _mm512_maskz_loadu_ps(tail, ps + 2 * sX);
									d0 = _mm512_fmadd_ps(s0, w2, d0);
									d1 = _mm512_fmadd_ps(s0, w1, d1);
									d2 = _mm512_fmadd_ps(s0, w0, d2);
								}
								s1 = _mm512_maskz_loadu_ps(tail, ps + 3 * sX);
								w3 = _mm512_maskz_loadu_ps(tail, pw + 3 * F);
								d0 = _mm512_fmadd_ps(s1, w3, d0);
								d1 = _mm512_fmadd_ps(s1, w2, d1);
								d2 = _mm512_fmadd_ps(s1, w1, d2);
								d3 = _mm512_fmadd_ps(s1, w0, d3);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 4 * sX);
								w4 = _mm512_maskz_loadu_ps(tail, pw + 4 * F);
								d0 = _mm512_fmadd_ps(s0, w4, d0);
								d1 = _mm512_fmadd_ps(s0, w3, d1);
								d2 = _mm512_fmadd_ps(s0, w2, d2);
								d3 = _mm512_fmadd_ps(s0, w1, d3);

								s1 = _mm512_maskz_loadu_ps(tail, ps + 5 * sX);
								w5 = _mm512_maskz_loadu_ps(tail, pw + 5 * F);
								d0 = _mm512_fmadd_ps(s1, w5, d0);
								d1 = _mm512_fmadd_ps(s1, w4, d1);
								d2 = _mm512_fmadd_ps(s1, w3, d2);
								d3 = _mm512_fmadd_ps(s1, w2, d3);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 6 * sX);
								w6 = _mm512_maskz_loadu_ps(tail, pw + 6 * F);
								d0 = _mm512_fmadd_ps(s0, w6, d0);
								d1 = _mm512_fmadd_ps(s0, w5, d1);
								d2 = _mm512_fmadd_ps(s0, w4, d2);
								d3 = _mm512_fmadd_ps(s0, w3, d3);
								if (dx < end)
								{
									s1 = _mm512_maskz_loadu_ps(tail, ps + 7 * sX);
									d1 = _mm512_fmadd_ps(s1, w6, d1);
									d2 = _mm512_fmadd_ps(s1, w5, d2);
									d3 = _mm512_fmadd_ps(s1, w4, d3);

									s0 = _mm512_maskz_loadu_ps(tail, ps + 8 * sX);
									d2 = _mm512_fmadd_ps(s0, w6, d2);
									d3 = _mm512_fmadd_ps(s0, w5, d3);

									s1 = _mm512_maskz_loadu_ps(tail, ps + 9 * sX);
									d3 = _mm512_fmadd_ps(s1, w6, d3);
								}
							}
						}
						float* pd = dst + (dy & dM) * dY + dx * dX;
						_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(d0, _params, 0));
						_mm512_mask_storeu_ps(pd + 1 * dX, tail, Activate<type>(d1, _params, 0));
						_mm512_mask_storeu_ps(pd + 2 * dX, tail, Activate<type>(d2, _params, 0));
						_mm512_mask_storeu_ps(pd + 3 * dX, tail, Activate<type>(d3, _params, 0));
					}
				}
				src += sD;
				dst += dD;
				weight += wD;
			}
		}

		//-------------------------------------------------------------------------------------------------------

		template<SimdConvolutionActivationType type> void DepthwiseConvolution_k7p3d1s1w6(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			assert(IsKernel(p, 7) && IsPad(p, 3) && IsStride(p, 1) && IsDilation(p, 1) && AlignedAny(p.srcW, 6));

			size_t dstC = srcC, dstW = p.dstW, srcH = p.srcH, end = dstW - 6;
			size_t sM = (bufH[0] - 1), sD = bufH[0] ? bufH[0] * p.srcW * F : F, sX = bufH[0] ? F : p.srcC, sY = sX * p.srcW;
			size_t dM = (bufH[1] - 1), dX = (bufH[1] ? F : p.dstC), dY = p.dstW * dX, dD = bufH[1] ? bufH[1] * dY : F;
			size_t wD = 49 * F;

			__m512 s0, s1, w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3, d4, d5, _params[2];
			_params[0] = _mm512_set1_ps(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				_params[1] = _mm512_set1_ps(params[1]);
			for (size_t c = 0; c < dstC; c += F)
			{
				__m512 _bias = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
				if (type == ::SimdConvolutionActivationPrelu)
					_params[0] = _mm512_loadu_ps(params + c);
				__mmask16 tail = TailMask16(dstC - c);
				for (size_t dy = yBeg; dy < yEnd; ++dy)
				{
					for (size_t dx = 0; dx < dstW; dx += 6)
					{
						d0 = _bias, d1 = _bias, d2 = _bias, d3 = _bias, d4 = _bias, d5 = _bias;
						for (size_t ky = 0; ky < 7; ++ky)
						{
							size_t sy = dy + ky - 3;
							const float* ps = src + (sy & sM) * sY + (dx - 3) * sX;
							const float* pw = weight + ky * 7 * F;
							if (sy < srcH)
							{
								w0 = _mm512_maskz_loadu_ps(tail, pw + 0 * F);
								w1 = _mm512_maskz_loadu_ps(tail, pw + 1 * F);
								w2 = _mm512_maskz_loadu_ps(tail, pw + 2 * F);
								if (dx)
								{
									s0 = _mm512_maskz_loadu_ps(tail, ps + 0 * sX);
									d0 = _mm512_fmadd_ps(s0, w0, d0);

									s1 = _mm512_maskz_loadu_ps(tail, ps + 1 * sX);
									d0 = _mm512_fmadd_ps(s1, w1, d0);
									d1 = _mm512_fmadd_ps(s1, w0, d1);

									s0 = _mm512_maskz_loadu_ps(tail, ps + 2 * sX);
									d0 = _mm512_fmadd_ps(s0, w2, d0);
									d1 = _mm512_fmadd_ps(s0, w1, d1);
									d2 = _mm512_fmadd_ps(s0, w0, d2);
								}
								s1 = _mm512_maskz_loadu_ps(tail, ps + 3 * sX);
								w3 = _mm512_maskz_loadu_ps(tail, pw + 3 * F);
								d0 = _mm512_fmadd_ps(s1, w3, d0);
								d1 = _mm512_fmadd_ps(s1, w2, d1);
								d2 = _mm512_fmadd_ps(s1, w1, d2);
								d3 = _mm512_fmadd_ps(s1, w0, d3);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 4 * sX);
								w4 = _mm512_maskz_loadu_ps(tail, pw + 4 * F);
								d0 = _mm512_fmadd_ps(s0, w4, d0);
								d1 = _mm512_fmadd_ps(s0, w3, d1);
								d2 = _mm512_fmadd_ps(s0, w2, d2);
								d3 = _mm512_fmadd_ps(s0, w1, d3);
								d4 = _mm512_fmadd_ps(s0, w0, d4);

								s1 = _mm512_maskz_loadu_ps(tail, ps + 5 * sX);
								w5 = _mm512_maskz_loadu_ps(tail, pw + 5 * F);
								d0 = _mm512_fmadd_ps(s1, w5, d0);
								d1 = _mm512_fmadd_ps(s1, w4, d1);
								d2 = _mm512_fmadd_ps(s1, w3, d2);
								d3 = _mm512_fmadd_ps(s1, w2, d3);
								d4 = _mm512_fmadd_ps(s1, w1, d4);
								d5 = _mm512_fmadd_ps(s1, w0, d5);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 6 * sX);
								w6 = _mm512_maskz_loadu_ps(tail, pw + 6 * F);
								d0 = _mm512_fmadd_ps(s0, w6, d0);
								d1 = _mm512_fmadd_ps(s0, w5, d1);
								d2 = _mm512_fmadd_ps(s0, w4, d2);
								d3 = _mm512_fmadd_ps(s0, w3, d3);
								d4 = _mm512_fmadd_ps(s0, w2, d4);
								d5 = _mm512_fmadd_ps(s0, w1, d5);

								s1 = _mm512_maskz_loadu_ps(tail, ps + 7 * sX);
								d1 = _mm512_fmadd_ps(s1, w6, d1);
								d2 = _mm512_fmadd_ps(s1, w5, d2);
								d3 = _mm512_fmadd_ps(s1, w4, d3);
								d4 = _mm512_fmadd_ps(s1, w3, d4);
								d5 = _mm512_fmadd_ps(s1, w2, d5);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 8 * sX);
								d2 = _mm512_fmadd_ps(s0, w6, d2);
								d3 = _mm512_fmadd_ps(s0, w5, d3);
								d4 = _mm512_fmadd_ps(s0, w4, d4);
								d5 = _mm512_fmadd_ps(s0, w3, d5);

								if (dx < end)
								{
									s1 = _mm512_maskz_loadu_ps(tail, ps + 9 * sX);
									d3 = _mm512_fmadd_ps(s1, w6, d3);
									d4 = _mm512_fmadd_ps(s1, w5, d4);
									d5 = _mm512_fmadd_ps(s1, w4, d5);

									s0 = _mm512_maskz_loadu_ps(tail, ps + 10 * sX);
									d4 = _mm512_fmadd_ps(s0, w6, d4);
									d5 = _mm512_fmadd_ps(s0, w5, d5);

									s1 = _mm512_maskz_loadu_ps(tail, ps + 11 * sX);
									d5 = _mm512_fmadd_ps(s1, w6, d5);
								}
							}
						}
						float* pd = dst + (dy & dM) * dY + dx * dX;
						_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(d0, _params, 0));
						_mm512_mask_storeu_ps(pd + 1 * dX, tail, Activate<type>(d1, _params, 0));
						_mm512_mask_storeu_ps(pd + 2 * dX, tail, Activate<type>(d2, _params, 0));
						_mm512_mask_storeu_ps(pd + 3 * dX, tail, Activate<type>(d3, _params, 0));
						_mm512_mask_storeu_ps(pd + 4 * dX, tail, Activate<type>(d4, _params, 0));
						_mm512_mask_storeu_ps(pd + 5 * dX, tail, Activate<type>(d5, _params, 0));
					}
				}
				src += sD;
				dst += dD;
				weight += wD;
			}
		}

		//-------------------------------------------------------------------------------------------------------

		template<SimdConvolutionActivationType type> void DepthwiseConvolution_k7p3d1s1w8(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			assert(IsKernel(p, 7) && IsPad(p, 3) && IsStride(p, 1) && IsDilation(p, 1) && Aligned(p.srcW, 8));

			size_t dstC = srcC, dstW = p.dstW, srcH = p.srcH, end = dstW - 8;
			size_t sM = (bufH[0] - 1), sD = bufH[0] ? bufH[0] * p.srcW * F : F, sX = bufH[0] ? F : p.srcC, sY = sX * p.srcW;
			size_t dM = (bufH[1] - 1), dX = (bufH[1] ? F : p.dstC), dY = p.dstW * dX, dD = bufH[1] ? bufH[1] * dY : F;
			size_t wD = 49 * F;

			__m512 s0, s1, w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3, d4, d5, d6, d7, _params[2];
			_params[0] = _mm512_set1_ps(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				_params[1] = _mm512_set1_ps(params[1]);
			for (size_t c = 0; c < dstC; c += F)
			{
				__m512 _bias = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
				if (type == ::SimdConvolutionActivationPrelu)
					_params[0] = _mm512_loadu_ps(params + c);
				__mmask16 tail = TailMask16(dstC - c);
				for (size_t dy = yBeg; dy < yEnd; ++dy)
				{
					for (size_t dx = 0; dx < dstW; dx += 8)
					{
						d0 = _bias, d1 = _bias, d2 = _bias, d3 = _bias, d4 = _bias, d5 = _bias, d6 = _bias, d7 = _bias;
						for (size_t ky = 0; ky < 7; ++ky)
						{
							size_t sy = dy + ky - 3;
							const float* ps = src + (sy & sM) * sY + (dx - 3) * sX;
							const float* pw = weight + ky * 7 * F;
							if (sy < srcH)
							{
								w0 = _mm512_maskz_loadu_ps(tail, pw + 0 * F);
								w1 = _mm512_maskz_loadu_ps(tail, pw + 1 * F);
								w2 = _mm512_maskz_loadu_ps(tail, pw + 2 * F);
								if (dx)
								{
									s0 = _mm512_maskz_loadu_ps(tail, ps + 0 * sX);
									d0 = _mm512_fmadd_ps(s0, w0, d0);

									s1 = _mm512_maskz_loadu_ps(tail, ps + 1 * sX);
									d0 = _mm512_fmadd_ps(s1, w1, d0);
									d1 = _mm512_fmadd_ps(s1, w0, d1);

									s0 = _mm512_maskz_loadu_ps(tail, ps + 2 * sX);
									d0 = _mm512_fmadd_ps(s0, w2, d0);
									d1 = _mm512_fmadd_ps(s0, w1, d1);
									d2 = _mm512_fmadd_ps(s0, w0, d2);
								}
								s1 = _mm512_maskz_loadu_ps(tail, ps + 3 * sX);
								w3 = _mm512_maskz_loadu_ps(tail, pw + 3 * F);
								d0 = _mm512_fmadd_ps(s1, w3, d0);
								d1 = _mm512_fmadd_ps(s1, w2, d1);
								d2 = _mm512_fmadd_ps(s1, w1, d2);
								d3 = _mm512_fmadd_ps(s1, w0, d3);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 4 * sX);
								w4 = _mm512_maskz_loadu_ps(tail, pw + 4 * F);
								d0 = _mm512_fmadd_ps(s0, w4, d0);
								d1 = _mm512_fmadd_ps(s0, w3, d1);
								d2 = _mm512_fmadd_ps(s0, w2, d2);
								d3 = _mm512_fmadd_ps(s0, w1, d3);
								d4 = _mm512_fmadd_ps(s0, w0, d4);

								s1 = _mm512_maskz_loadu_ps(tail, ps + 5 * sX);
								w5 = _mm512_maskz_loadu_ps(tail, pw + 5 * F);
								d0 = _mm512_fmadd_ps(s1, w5, d0);
								d1 = _mm512_fmadd_ps(s1, w4, d1);
								d2 = _mm512_fmadd_ps(s1, w3, d2);
								d3 = _mm512_fmadd_ps(s1, w2, d3);
								d4 = _mm512_fmadd_ps(s1, w1, d4);
								d5 = _mm512_fmadd_ps(s1, w0, d5);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 6 * sX);
								w6 = _mm512_maskz_loadu_ps(tail, pw + 6 * F);
								d0 = _mm512_fmadd_ps(s0, w6, d0);
								d1 = _mm512_fmadd_ps(s0, w5, d1);
								d2 = _mm512_fmadd_ps(s0, w4, d2);
								d3 = _mm512_fmadd_ps(s0, w3, d3);
								d4 = _mm512_fmadd_ps(s0, w2, d4);
								d5 = _mm512_fmadd_ps(s0, w1, d5);
								d6 = _mm512_fmadd_ps(s0, w0, d6);

								s1 = _mm512_maskz_loadu_ps(tail, ps + 7 * sX);
								d1 = _mm512_fmadd_ps(s1, w6, d1);
								d2 = _mm512_fmadd_ps(s1, w5, d2);
								d3 = _mm512_fmadd_ps(s1, w4, d3);
								d4 = _mm512_fmadd_ps(s1, w3, d4);
								d5 = _mm512_fmadd_ps(s1, w2, d5);
								d6 = _mm512_fmadd_ps(s1, w1, d6);
								d7 = _mm512_fmadd_ps(s1, w0, d7);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 8 * sX);
								d2 = _mm512_fmadd_ps(s0, w6, d2);
								d3 = _mm512_fmadd_ps(s0, w5, d3);
								d4 = _mm512_fmadd_ps(s0, w4, d4);
								d5 = _mm512_fmadd_ps(s0, w3, d5);
								d6 = _mm512_fmadd_ps(s0, w2, d6);
								d7 = _mm512_fmadd_ps(s0, w1, d7);

								s1 = _mm512_maskz_loadu_ps(tail, ps + 9 * sX);
								d3 = _mm512_fmadd_ps(s1, w6, d3);
								d4 = _mm512_fmadd_ps(s1, w5, d4);
								d5 = _mm512_fmadd_ps(s1, w4, d5);
								d6 = _mm512_fmadd_ps(s1, w3, d6);
								d7 = _mm512_fmadd_ps(s1, w2, d7);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 10 * sX);
								d4 = _mm512_fmadd_ps(s0, w6, d4);
								d5 = _mm512_fmadd_ps(s0, w5, d5);
								d6 = _mm512_fmadd_ps(s0, w4, d6);
								d7 = _mm512_fmadd_ps(s0, w3, d7);

								if (dx < end)
								{
									s1 = _mm512_maskz_loadu_ps(tail, ps + 11 * sX);
									d5 = _mm512_fmadd_ps(s1, w6, d5);
									d6 = _mm512_fmadd_ps(s1, w5, d6);
									d7 = _mm512_fmadd_ps(s1, w4, d7);

									s0 = _mm512_maskz_loadu_ps(tail, ps + 12 * sX);
									d6 = _mm512_fmadd_ps(s0, w6, d6);
									d7 = _mm512_fmadd_ps(s0, w5, d7);

									s1 = _mm512_maskz_loadu_ps(tail, ps + 13 * sX);
									d7 = _mm512_fmadd_ps(s1, w6, d7);
								}
							}
						}
						float* pd = dst + (dy & dM) * dY + dx * dX;
						_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(d0, _params, 0));
						_mm512_mask_storeu_ps(pd + 1 * dX, tail, Activate<type>(d1, _params, 0));
						_mm512_mask_storeu_ps(pd + 2 * dX, tail, Activate<type>(d2, _params, 0));
						_mm512_mask_storeu_ps(pd + 3 * dX, tail, Activate<type>(d3, _params, 0));
						_mm512_mask_storeu_ps(pd + 4 * dX, tail, Activate<type>(d4, _params, 0));
						_mm512_mask_storeu_ps(pd + 5 * dX, tail, Activate<type>(d5, _params, 0));
						_mm512_mask_storeu_ps(pd + 6 * dX, tail, Activate<type>(d6, _params, 0));
						_mm512_mask_storeu_ps(pd + 7 * dX, tail, Activate<type>(d7, _params, 0));
					}
				}
				src += sD;
				dst += dD;
				weight += wD;
			}
		}

		//-------------------------------------------------------------------------------------------------------

		template <SimdConvolutionActivationType type> bool SetDepthwise7x7(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			if (p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 8))
				convolution[0] = DepthwiseConvolution_k7p3d1s1w8<type>;
			else if (p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && AlignedAny(p.srcW, 6))
				convolution[0] = DepthwiseConvolution_k7p3d1s1w6<type>;
			else if (p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 4))
				convolution[0] = DepthwiseConvolution_k7p3d1s1w4<type>;
			else
				return false;
			return true;
		}

		bool SetDepthwise7x7(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			switch (p.activation)
			{
			case SimdConvolutionActivationIdentity: return SetDepthwise7x7<SimdConvolutionActivationRestrictRange>(p, convolution);
			case SimdConvolutionActivationRelu: return SetDepthwise7x7<SimdConvolutionActivationRestrictRange>(p, convolution);
			case SimdConvolutionActivationLeakyRelu: return SetDepthwise7x7<SimdConvolutionActivationPrelu>(p, convolution);
			case SimdConvolutionActivationRestrictRange: return SetDepthwise7x7<SimdConvolutionActivationRestrictRange>(p, convolution);
			case SimdConvolutionActivationPrelu: return SetDepthwise7x7<SimdConvolutionActivationPrelu>(p, convolution);
			case SimdConvolutionActivationElu: return SetDepthwise7x7<SimdConvolutionActivationElu>(p, convolution);
			case SimdConvolutionActivationHswish: return SetDepthwise7x7<SimdConvolutionActivationHswish>(p, convolution);
			case SimdConvolutionActivationMish: return SetDepthwise7x7<SimdConvolutionActivationMish>(p, convolution);
			case SimdConvolutionActivationHardSigmoid: return SetDepthwise7x7<SimdConvolutionActivationHardSigmoid>(p, convolution);
			case SimdConvolutionActivationSwish: return SetDepthwise7x7<SimdConvolutionActivationSwish>(p, convolution);
			case SimdConvolutionActivationGelu: return SetDepthwise7x7<SimdConvolutionActivationGelu>(p, convolution);
			default: 
				return false;
			}
		}
	}
#endif
}
