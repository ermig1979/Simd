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
		template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = srcC;
			size_t sM = (bufH[0] - 1), sD = bufH[0] ? bufH[0] * p.srcW * F : F, sX = bufH[0] ? F : p.srcC, sY = sX * p.srcW;
			size_t dM = (bufH[1] - 1), dX = (bufH[1] ? F : p.dstC), dY = p.dstW * dX, dy0 = bufH[1] ? yBeg : 0, dD = bufH[1] ? bufH[1] * dY : F;
			size_t wD = p.kernelY * p.kernelX * F, ssX = strideX * sX;
			size_t noseY = NoseH(p), bodyY = BodyH(p), noseX = NoseW(p), bodyX = BodyW(p);
			size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
			size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
			size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;
			size_t dstCF = AlignLo(dstC, F);

			__m512 _params[2];
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
					float* pd = dst + (dy & dM) * dY;
					if (dy >= noseY && dy < bodyY)
					{
						size_t dx = 0;
						for (; dx < noseX; dx += 1, pd += dX)
						{
							__m512 sum0 = _bias;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * p.strideY + ky - padY;
								for (size_t kx = 0; kx < p.kernelX; ++kx)
								{
									size_t sx = dx * p.strideX + kx - padX;
									if (sx < p.srcW)
									{
										const float* pw = weight + (ky * p.kernelX + kx) * F;
										const float* ps = src + (sy & sM) * sY + sx * sX;
										sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum0);
									}
								}
							}
							_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(sum0, _params, 0));
						}
						for (; dx < bodyX8; dx += 8, pd += 8 * dX)
						{
							__m512 sum0 = _bias;
							__m512 sum1 = _bias;
							__m512 sum2 = _bias;
							__m512 sum3 = _bias;
							__m512 sum4 = _bias;
							__m512 sum5 = _bias;
							__m512 sum6 = _bias;
							__m512 sum7 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									__m512 w0 = _mm512_loadu_ps(pw);
									sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * ssX), w0, sum0);
									sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * ssX), w0, sum1);
									sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * ssX), w0, sum2);
									sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 3 * ssX), w0, sum3);
									sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 4 * ssX), w0, sum4);
									sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 5 * ssX), w0, sum5);
									sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 6 * ssX), w0, sum6);
									sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 7 * ssX), w0, sum7);
								}
							}
							_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(sum0, _params, 0));
							_mm512_mask_storeu_ps(pd + 1 * dX, tail, Activate<type>(sum1, _params, 0));
							_mm512_mask_storeu_ps(pd + 2 * dX, tail, Activate<type>(sum2, _params, 0));
							_mm512_mask_storeu_ps(pd + 3 * dX, tail, Activate<type>(sum3, _params, 0));
							_mm512_mask_storeu_ps(pd + 4 * dX, tail, Activate<type>(sum4, _params, 0));
							_mm512_mask_storeu_ps(pd + 5 * dX, tail, Activate<type>(sum5, _params, 0));
							_mm512_mask_storeu_ps(pd + 6 * dX, tail, Activate<type>(sum6, _params, 0));
							_mm512_mask_storeu_ps(pd + 7 * dX, tail, Activate<type>(sum7, _params, 0));
						}
						for (; dx < bodyX4; dx += 4, pd += 4 * dX)
						{
							__m512 sum0 = _bias;
							__m512 sum1 = _bias;
							__m512 sum2 = _bias;
							__m512 sum3 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									__m512 w0 = _mm512_loadu_ps(pw);
									sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * ssX), w0, sum0);
									sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * ssX), w0, sum1);
									sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * ssX), w0, sum2);
									sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 3 * ssX), w0, sum3);
								}
							}
							_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(sum0, _params, 0));
							_mm512_mask_storeu_ps(pd + 1 * dX, tail, Activate<type>(sum1, _params, 0));
							_mm512_mask_storeu_ps(pd + 2 * dX, tail, Activate<type>(sum2, _params, 0));
							_mm512_mask_storeu_ps(pd + 3 * dX, tail, Activate<type>(sum3, _params, 0));
						}
						for (; dx < bodyX2; dx += 2, pd += 2 * dX)
						{
							__m512 sum0 = _bias;
							__m512 sum1 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									__m512 w0 = _mm512_loadu_ps(pw);
									sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * ssX), w0, sum0);
									sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * ssX), w0, sum1);
								}
							}
							_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(sum0, _params, 0));
							_mm512_mask_storeu_ps(pd + 1 * dX, tail, Activate<type>(sum1, _params, 0));
						}
						for (; dx < bodyX; dx += 1, pd += dX)
						{
							__m512 sum0 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									__m512 w0 = _mm512_loadu_ps(pw);
									sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps), w0, sum0);
								}
							}
							_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(sum0, _params, 0));
						}
						for (; dx < p.dstW; dx += 1, pd += dX)
						{
							__m512 sum0 = _bias;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								for (size_t kx = 0; kx < p.kernelX; ++kx)
								{
									size_t sx = dx * strideX + kx - padX;
									if (sx < p.srcW)
									{
										const float* pw = weight + (ky * p.kernelX + kx) * F;
										const float* ps = src + (sy & sM) * sY + sx * sX;
										sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum0);
									}
								}
							}
							_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(sum0, _params, 0));
						}
					}
					else
					{
						for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
						{
							__m512 sum0 = _bias;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								if (sy < p.srcH)
								{
									for (size_t kx = 0; kx < p.kernelX; ++kx)
									{
										size_t sx = dx * strideX + kx - padX;
										if (sx < p.srcW)
										{
											const float* pw = weight + (ky * p.kernelX + kx) * F;
											const float* ps = src + (sy & sM) * sY + sx * sX;
											sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum0);
										}
									}
								}
							}
							_mm512_mask_storeu_ps(pd + 0 * dX, tail, Activate<type>(sum0, _params, 0));
						}
					}
				}
				src += sD;
				dst += dD;
				weight += wD;
			}
		}

		//-------------------------------------------------------------------------------------------------------

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x2(
			const float* src0, const float* src1, size_t sX, const __m512* weight, const __m512& bias, const __m512* params, float* dst, __mmask16 tail)
		{
			__m512 sum0 = bias, sum1 = _mm512_setzero_ps();
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum1);
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum1);
			_mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(sum0, sum1), params, 0));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
			const float* src0, const float* src1, size_t sX, const __m512* weight, const __m512& bias, const __m512* params, float* dst, __mmask16 tail)
		{
			__m512 sum0 = bias, sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum1);
			sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 2 * sX), weight[2], sum2);
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum1);
			sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 2 * sX), weight[5], sum2);
			_mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), params, 0));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
			const float* src0, const float* src1, const float* src2, size_t sX, const __m512* weight, const __m512& bias, const __m512* params, float* dst, __mmask16 tail)
		{
			__m512 sum0 = bias, sum1 = _mm512_setzero_ps();
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum1);
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum1);
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 0 * sX), weight[6], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 1 * sX), weight[7], sum1);
			_mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(sum0, sum1), params, 0));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
			const float* src0, const float* src1, const float* src2, size_t sX, const __m512* weight, const __m512& bias, const __m512* params, float* dst, __mmask16 tail)
		{
			__m512 sum0 = bias, sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * sX), weight[0], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * sX), weight[1], sum1);
			sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 2 * sX), weight[2], sum2);
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * sX), weight[3], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * sX), weight[4], sum1);
			sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 2 * sX), weight[5], sum2);
			sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 0 * sX), weight[6], sum0);
			sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 1 * sX), weight[7], sum1);
			sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 2 * sX), weight[8], sum2);
			_mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), params, 0));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x2(
			const float* src0, const float* src1, const float* src2, size_t sX, const __m512* weight, const __m512& bias, const __m512* params, float* dst, size_t dstC, __mmask16 tail)
		{
			__m512 sum0 = bias, sum1 = bias, s0;

			s0 = _mm512_loadu_ps(src0 + 0 * sX);
			sum0 = _mm512_fmadd_ps(s0, weight[0], sum0);
			s0 = _mm512_loadu_ps(src0 + 1 * sX);
			sum0 = _mm512_fmadd_ps(s0, weight[1], sum0);
			sum1 = _mm512_fmadd_ps(s0, weight[0], sum1);
			s0 = _mm512_loadu_ps(src0 + 2 * sX);
			sum0 = _mm512_fmadd_ps(s0, weight[2], sum0);
			sum1 = _mm512_fmadd_ps(s0, weight[1], sum1);
			s0 = _mm512_loadu_ps(src0 + 3 * sX);
			sum1 = _mm512_fmadd_ps(s0, weight[2], sum1);

			s0 = _mm512_loadu_ps(src1 + 0 * sX);
			sum0 = _mm512_fmadd_ps(s0, weight[3], sum0);
			s0 = _mm512_loadu_ps(src1 + 1 * sX);
			sum0 = _mm512_fmadd_ps(s0, weight[4], sum0);
			sum1 = _mm512_fmadd_ps(s0, weight[3], sum1);
			s0 = _mm512_loadu_ps(src1 + 2 * sX);
			sum0 = _mm512_fmadd_ps(s0, weight[5], sum0);
			sum1 = _mm512_fmadd_ps(s0, weight[4], sum1);
			s0 = _mm512_loadu_ps(src1 + 3 * sX);
			sum1 = _mm512_fmadd_ps(s0, weight[5], sum1);

			s0 = _mm512_loadu_ps(src2 + 0 * sX);
			sum0 = _mm512_fmadd_ps(s0, weight[6], sum0);
			s0 = _mm512_loadu_ps(src2 + 1 * sX);
			sum0 = _mm512_fmadd_ps(s0, weight[7], sum0);
			sum1 = _mm512_fmadd_ps(s0, weight[6], sum1);
			s0 = _mm512_loadu_ps(src2 + 2 * sX);
			sum0 = _mm512_fmadd_ps(s0, weight[8], sum0);
			sum1 = _mm512_fmadd_ps(s0, weight[7], sum1);
			s0 = _mm512_loadu_ps(src2 + 3 * sX);
			sum1 = _mm512_fmadd_ps(s0, weight[8], sum1);

			_mm512_mask_storeu_ps(dst + 0 * dstC, tail, Activate<type>(sum0, params, 0));
			_mm512_mask_storeu_ps(dst + 1 * dstC, tail, Activate<type>(sum1, params, 0));
		}

		template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = srcC;
			size_t sM = (bufH[0] - 1), sD = bufH[0] ? bufH[0] * p.srcW * F : F, sX = bufH[0] ? F : p.srcC, sY = sX * p.srcW;
			size_t dM = (bufH[1] - 1), dX = (bufH[1] ? F : p.dstC), dY = p.dstW * dX, dy0 = bufH[1] ? yBeg : 0, dD = bufH[1] ? bufH[1] * dY : F;
			size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX) * sX;
			size_t xMainEnd = p.dstW - p.padW, xMainEnd2 = AlignLo(xMainEnd - padX, 2) * (p.strideX == 1 ? 1 : 0) + padX;
			size_t yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

			__m512 _params[2];
			_params[0] = _mm512_set1_ps(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				_params[1] = _mm512_set1_ps(params[1]);
			for (size_t c = 0; c < srcC; c += F)
			{
				__m512 _weight[9];
				for (size_t i = 0; i < 9; ++i)
					_weight[i] = _mm512_loadu_ps(weight + i * F);
				__m512 _bias = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
				if (type == ::SimdConvolutionActivationPrelu)
					_params[0] = _mm512_loadu_ps(params + c);
				__mmask16 tail = TailMask16(srcC - c);

				size_t dy = yBeg;
				if (yBeg == 0 && padY)
				{
					size_t sy = 0, dx = 0;
					const float* src0 = src + ((sy + 0) & sM) * sY;
					const float* src1 = src + ((sy + 1) & sM) * sY;
					float* pDst = dst + (dy & dM) * dY;
					if (padX)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 4, _bias, _params, pDst, tail),
						pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
					for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
						ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, sX, _weight + 3, _bias, _params, pDst, tail);
					if (padW)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 3, _bias, _params, pDst, tail);
					dy++;
				}
				for (; dy < yMainEnd; ++dy)
				{
					size_t sy = dy * strideY - padY, dx = 0;
					const float* src0 = src + ((sy + 0) & sM) * sY;
					const float* src1 = src + ((sy + 1) & sM) * sY;
					const float* src2 = src + ((sy + 2) & sM) * sY;
					float* pDst = dst + (dy & dM) * dY;
					if (padX)
						ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, sX, _weight + 1, _bias, _params, pDst, tail),
						pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
					for (; dx < xMainEnd2; dx += 2, pDst += 2 * dX, src0 += 2 * ssX, src1 += 2 * ssX, src2 += 2 * ssX)
						ConvolutionDepthwise3x3Main1x2<type>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, dX, tail);
					for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
						ConvolutionDepthwise3x3Main1x1<type>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, tail);
					if (padW)
						ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, tail);
				}
				if (dy < yEnd)
				{
					size_t sy = dy * strideY - padY, dx = 0;
					const float* src0 = src + ((sy + 0) & sM) * sY;
					const float* src1 = src + ((sy + 1) & sM) * sY;
					float* pDst = dst + (dy & dM) * dY;
					if (padX)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 1, _bias, _params, pDst, tail),
						pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
					for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
						ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, sX, _weight + 0, _bias, _params, pDst, tail);
					if (padW)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 0, _bias, _params, pDst, tail);
				}
				src += sD;
				dst += dD;
				weight += wD;
			}
		}

		//-------------------------------------------------------------------------------------------------------

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

		template<SimdConvolutionActivationType type> void DepthwiseConvolution_k3p1d1s1w6(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			assert(IsKernel(p, 3) && IsPad(p, 1) && IsStride(p, 1) && IsDilation(p, 1) && AlignedAny(p.srcW, 6));

			size_t dstC = srcC, dstW = p.dstW, srcH = p.srcH, end = dstW - 6;
			size_t sM = (bufH[0] - 1), sD = bufH[0] ? bufH[0] * p.srcW * F : F, sX = bufH[0] ? F : p.srcC, sY = sX * p.srcW;
			size_t dM = (bufH[1] - 1), dX = (bufH[1] ? F : p.dstC), dY = p.dstW * dX, dD = bufH[1] ? bufH[1] * dY : F;
			size_t wD = 9 * F;

			__m512 s0, s1, d0, d1, d2, d3, d4, d5, _params[2], _weight[9], w0, w1, w2;
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
				for (size_t i = 0; i < 9; ++i)
					_weight[i] = _mm512_loadu_ps(weight + i * F);
				for (size_t dy = yBeg; dy < yEnd; ++dy)
				{
					for (size_t dx = 0; dx < dstW; dx += 6)
					{
						d0 = _bias, d1 = _bias, d2 = _bias, d3 = _bias, d4 = _bias, d5 = _bias;
						for (size_t ky = 0; ky < 3; ++ky)
						{
							size_t sy = dy + ky - 1;
							const float* ps = src + (sy & sM) * sY + (dx - 1) * sX;
							const __m512 * pw = _weight + ky * 3;
							if (sy < srcH)
							{
								w0 = pw[0];
								if (dx)
								{
									s0 = _mm512_maskz_loadu_ps(tail, ps + 0 * sX);
									d0 = _mm512_fmadd_ps(s0, w0, d0);
								}

								w1 = pw[1];
								s1 = _mm512_maskz_loadu_ps(tail, ps + 1 * sX);
								d0 = _mm512_fmadd_ps(s1, w1, d0);
								d1 = _mm512_fmadd_ps(s1, w0, d1);

								w2 = pw[2];
								s0 = _mm512_maskz_loadu_ps(tail, ps + 2 * sX);
								d0 = _mm512_fmadd_ps(s0, w2, d0);
								d1 = _mm512_fmadd_ps(s0, w1, d1);
								d2 = _mm512_fmadd_ps(s0, w0, d2);

								s1 = _mm512_maskz_loadu_ps(tail, ps + 3 * sX);
								d1 = _mm512_fmadd_ps(s1, w2, d1);
								d2 = _mm512_fmadd_ps(s1, w1, d2);
								d3 = _mm512_fmadd_ps(s1, w0, d3);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 4 * sX);
								d2 = _mm512_fmadd_ps(s0, w2, d2);
								d3 = _mm512_fmadd_ps(s0, w1, d3);
								d4 = _mm512_fmadd_ps(s0, w0, d4);

								s1 = _mm512_maskz_loadu_ps(tail, ps + 5 * sX);
								d3 = _mm512_fmadd_ps(s1, w2, d3);
								d4 = _mm512_fmadd_ps(s1, w1, d4);
								d5 = _mm512_fmadd_ps(s1, w0, d5);

								s0 = _mm512_maskz_loadu_ps(tail, ps + 6 * sX);
								d4 = _mm512_fmadd_ps(s0, w2, d4);
								d5 = _mm512_fmadd_ps(s0, w1, d5);

								if (dx < end)
								{
									s1 = _mm512_maskz_loadu_ps(tail, ps + 7 * sX);
									d5 = _mm512_fmadd_ps(s1, w2, d5);
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


		template <SimdConvolutionActivationType type> void SetDepthwise(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			if (p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 8))
				convolution[0] = DepthwiseConvolution_k7p3d1s1w8<type>;
			else if (p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && AlignedAny(p.srcW, 6))
				convolution[0] = DepthwiseConvolution_k7p3d1s1w6<type>;
			else if (p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 4))
				convolution[0] = DepthwiseConvolution_k7p3d1s1w4<type>;
			else if (p.IsKernel(3) && p.IsPad(1) && p.IsStride(1) && p.IsDilation(1) && AlignedAny(p.srcW, 6))
				convolution[0] = DepthwiseConvolution_k3p1d1s1w6<type>;
			else if (p.kernelY == 3)
				convolution[0] = DepthwiseConvolution3x3<type>;
			else
				convolution[0] = DepthwiseConvolution<type>;
		}

		void SetDepthwise(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			switch (p.activation)
			{
			case SimdConvolutionActivationIdentity: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, convolution); break;
			case SimdConvolutionActivationRelu: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, convolution); break;
			case SimdConvolutionActivationLeakyRelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, convolution); break;
			case SimdConvolutionActivationRestrictRange: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, convolution); break;
			case SimdConvolutionActivationPrelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, convolution); break;
			case SimdConvolutionActivationElu: SetDepthwise<SimdConvolutionActivationElu>(p, convolution); break;
			case SimdConvolutionActivationHswish: SetDepthwise<SimdConvolutionActivationHswish>(p, convolution); break;
			case SimdConvolutionActivationMish: SetDepthwise<SimdConvolutionActivationMish>(p, convolution); break;
			case SimdConvolutionActivationHardSigmoid: SetDepthwise<SimdConvolutionActivationHardSigmoid>(p, convolution); break;
			case SimdConvolutionActivationSwish: SetDepthwise<SimdConvolutionActivationSwish>(p, convolution); break;
			case SimdConvolutionActivationGelu: SetDepthwise<SimdConvolutionActivationGelu>(p, convolution); break;
			default: assert(0);
			}
		}
	}
#endif
}
