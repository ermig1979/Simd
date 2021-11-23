/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Avx2
	{
		namespace Dc
		{
			template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
			{
				size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
				size_t srcX = p.srcC, srcW = p.srcW * srcX, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F, strideXC = strideX * srcX;
				size_t dstM = (bufH[1] - 1), dstS = bufH[1] * dstW;
				size_t noseY = (p.padY + p.strideY - 1) / p.strideY;
				size_t bodyY = (p.srcH + p.padY + p.strideY - p.kernelY) / p.strideY;
				size_t noseX = (p.padX + p.strideX - 1) / p.strideX;
				size_t bodyX = (p.srcW + p.padX + p.strideX - p.kernelX) / p.strideX;
				size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
				size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
				size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;

				__m256 _params[2];
				_params[0] = _mm256_set1_ps(params[0]);
				if (type == SimdConvolutionActivationRestrictRange ||
					type == SimdConvolutionActivationHswish ||
					type == SimdConvolutionActivationHardSigmoid)
					_params[1] = _mm256_set1_ps(params[1]);
				for (size_t c = 0; c < srcC; c += F)
				{
					__m256 _bias = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
					if (type == ::SimdConvolutionActivationPrelu)
						_params[0] = _mm256_loadu_ps(params + c);

					for (size_t dy = yBeg; dy < yEnd; ++dy)
					{
						float* pd = dst + (dy & dstM) * dstW;
						if (dy >= noseY && dy < bodyY)
						{
							size_t dx = 0;
							for (; dx < noseX; ++dx, pd += F)
							{
								__m256 sum = _bias;
								for (size_t ky = 0; ky < p.kernelY; ++ky)
								{
									size_t sy = dy * p.strideY + ky - padY;
									for (size_t kx = 0; kx < p.kernelX; ++kx)
									{
										size_t sx = dx * p.strideX + kx - padX;
										if (sx < p.srcW)
										{
											const float* pw = weight + (ky * p.kernelX + kx) * F;
											const float* ps = src + sy * srcW + sx * srcX;
											sum = _mm256_fmadd_ps(_mm256_loadu_ps(ps), _mm256_loadu_ps(pw), sum);
										}
									}
								}
								_mm256_storeu_ps(pd, Activate<type>(sum, _params, 0));
							}
							for (; dx < bodyX8; dx += 8, pd += 8 * F)
							{
								__m256 sum0 = _bias;
								__m256 sum1 = _bias;
								__m256 sum2 = _bias;
								__m256 sum3 = _bias;
								__m256 sum4 = _bias;
								__m256 sum5 = _bias;
								__m256 sum6 = _bias;
								__m256 sum7 = _bias;
								const float* pw = weight;
								for (size_t ky = 0; ky < p.kernelY; ++ky)
								{
									size_t sy = dy * strideY + ky - padY;
									const float* ps = src + sy * srcW + (dx * strideX - padX) * srcX;
									for (size_t kx = 0; kx < p.kernelX; ++kx, ps += srcX, pw += F)
									{
										__m256 w0 = _mm256_loadu_ps(pw);
										sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 0 * strideXC), w0, sum0);
										sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 1 * strideXC), w0, sum1);
										sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 2 * strideXC), w0, sum2);
										sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 3 * strideXC), w0, sum3);
										sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 4 * strideXC), w0, sum4);
										sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 5 * strideXC), w0, sum5);
										sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 6 * strideXC), w0, sum6);
										sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 7 * strideXC), w0, sum7);
									}
								}
								_mm256_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
								_mm256_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
								_mm256_storeu_ps(pd + 2 * F, Activate<type>(sum2, _params, 0));
								_mm256_storeu_ps(pd + 3 * F, Activate<type>(sum3, _params, 0));
								_mm256_storeu_ps(pd + 4 * F, Activate<type>(sum4, _params, 0));
								_mm256_storeu_ps(pd + 5 * F, Activate<type>(sum5, _params, 0));
								_mm256_storeu_ps(pd + 6 * F, Activate<type>(sum6, _params, 0));
								_mm256_storeu_ps(pd + 7 * F, Activate<type>(sum7, _params, 0));
							}
							for (; dx < bodyX4; dx += 4, pd += 4 * F)
							{
								__m256 sum0 = _bias;
								__m256 sum1 = _bias;
								__m256 sum2 = _bias;
								__m256 sum3 = _bias;
								const float* pw = weight;
								for (size_t ky = 0; ky < p.kernelY; ++ky)
								{
									size_t sy = dy * strideY + ky - padY;
									const float* ps = src + sy * srcW + (dx * strideX - padX) * srcX;
									for (size_t kx = 0; kx < p.kernelX; ++kx, ps += srcX, pw += F)
									{
										__m256 w0 = _mm256_loadu_ps(pw);
										sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 0 * strideXC), w0, sum0);
										sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 1 * strideXC), w0, sum1);
										sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 2 * strideXC), w0, sum2);
										sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 3 * strideXC), w0, sum3);
									}
								}
								_mm256_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
								_mm256_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
								_mm256_storeu_ps(pd + 2 * F, Activate<type>(sum2, _params, 0));
								_mm256_storeu_ps(pd + 3 * F, Activate<type>(sum3, _params, 0));
							}
							for (; dx < bodyX2; dx += 2, pd += 2 * F)
							{
								__m256 sum0 = _bias;
								__m256 sum1 = _bias;
								const float* pw = weight;
								for (size_t ky = 0; ky < p.kernelY; ++ky)
								{
									size_t sy = dy * strideY + ky - padY;
									const float* ps = src + sy * srcW + (dx * strideX - padX) * srcX;
									for (size_t kx = 0; kx < p.kernelX; ++kx, ps += srcX, pw += F)
									{
										__m256 w0 = _mm256_loadu_ps(pw);
										sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 0 * strideXC), w0, sum0);
										sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(ps + 1 * strideXC), w0, sum1);
									}
								}
								_mm256_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
								_mm256_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
							}
							for (; dx < bodyX; ++dx, pd += F)
							{
								__m256 sum = _bias;
								const float* pw = weight;
								for (size_t ky = 0; ky < p.kernelY; ++ky)
								{
									size_t sy = dy * strideY + ky - padY;
									const float* ps = src + sy * srcW + (dx * strideX - padX) * srcX;
									for (size_t kx = 0; kx < p.kernelX; ++kx, ps += srcX, pw += F)
									{
										__m256 w0 = _mm256_loadu_ps(pw);
										sum = _mm256_fmadd_ps(_mm256_loadu_ps(ps), w0, sum);
									}
								}
								_mm256_storeu_ps(pd, Activate<type>(sum, _params, 0));
							}
							for (; dx < p.dstW; ++dx, pd += F)
							{
								__m256 sum = _bias;
								for (size_t ky = 0; ky < p.kernelY; ++ky)
								{
									size_t sy = dy * strideY + ky - padY;
									for (size_t kx = 0; kx < p.kernelX; ++kx)
									{
										size_t sx = dx * strideX + kx - padX;
										if (sx < p.srcW)
										{
											const float* pw = weight + (ky * p.kernelX + kx) * F;
											const float* ps = src + sy * srcW + sx * srcX;
											sum = _mm256_fmadd_ps(_mm256_loadu_ps(ps), _mm256_loadu_ps(pw), sum);
										}
									}
								}
								_mm256_storeu_ps(pd, Activate<type>(sum, _params, 0));
							}
						}
						else
						{
							for (size_t dx = 0; dx < p.dstW; ++dx, pd += F)
							{
								__m256 sum = _bias;
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
												const float* ps = src + sy * srcW + sx * srcX;
												sum = _mm256_fmadd_ps(_mm256_loadu_ps(ps), _mm256_loadu_ps(pw), sum);
											}
										}
									}
								}
								_mm256_storeu_ps(pd, Activate<type>(sum, _params, 0));
							}
						}
					}
					src += F;
					dst += dstS;
					weight += weightS;
				}
			}

			//---------------------------------------------------------------------

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x2(
				const float* src0, const float* src1, size_t srcC, const __m256* weight, const __m256& bias, const __m256* params, float* dst)
			{
				__m256 sum0 = bias, sum1 = _mm256_setzero_ps();
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 0 * srcC), weight[0], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 1 * srcC), weight[1], sum1);
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 0 * srcC), weight[3], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 1 * srcC), weight[4], sum1);
				_mm256_storeu_ps(dst, Activate<type>(_mm256_add_ps(sum0, sum1), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
				const float* src0, const float* src1, size_t srcC, const __m256* weight, const __m256& bias, const __m256* params, float* dst)
			{
				__m256 sum0 = bias, sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 0 * srcC), weight[0], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 1 * srcC), weight[1], sum1);
				sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 2 * srcC), weight[2], sum2);
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 0 * srcC), weight[3], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 1 * srcC), weight[4], sum1);
				sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 2 * srcC), weight[5], sum2);
				_mm256_storeu_ps(dst, Activate<type>(_mm256_add_ps(_mm256_add_ps(sum0, sum1), sum2), params, 0));
		}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
				const float* src0, const float* src1, const float* src2, size_t srcC, const __m256* weight, const __m256& bias, const __m256* params, float* dst)
			{
				__m256 sum0 = bias, sum1 = _mm256_setzero_ps();
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 0 * srcC), weight[0], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 1 * srcC), weight[1], sum1);
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 0 * srcC), weight[3], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 1 * srcC), weight[4], sum1);
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src2 + 0 * srcC), weight[6], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src2 + 1 * srcC), weight[7], sum1);
				_mm256_storeu_ps(dst, Activate<type>(_mm256_add_ps(sum0, sum1), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
				const float* src0, const float* src1, const float* src2, size_t srcC, const __m256* weight, const __m256& bias, const __m256* params, float* dst)
			{
				__m256 sum0 = bias, sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 0 * srcC), weight[0], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 1 * srcC), weight[1], sum1);
				sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(src0 + 2 * srcC), weight[2], sum2);
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 0 * srcC), weight[3], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 1 * srcC), weight[4], sum1);
				sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(src1 + 2 * srcC), weight[5], sum2);
				sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(src2 + 0 * srcC), weight[6], sum0);
				sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(src2 + 1 * srcC), weight[7], sum1);
				sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(src2 + 2 * srcC), weight[8], sum2);
				_mm256_storeu_ps(dst, Activate<type>(_mm256_add_ps(_mm256_add_ps(sum0, sum1), sum2), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x2(
				const float* src0, const float* src1, const float* src2, size_t srcC, const __m256* weight, const __m256& bias, const __m256* params, float* dst)
			{
				__m256 sum0 = bias, sum1 = bias, s0;

				s0 = _mm256_loadu_ps(src0 + 0 * srcC);
				sum0 = _mm256_fmadd_ps(s0, weight[0], sum0);
				s0 = _mm256_loadu_ps(src0 + 1 * srcC);
				sum0 = _mm256_fmadd_ps(s0, weight[1], sum0);
				sum1 = _mm256_fmadd_ps(s0, weight[0], sum1);
				s0 = _mm256_loadu_ps(src0 + 2 * srcC);
				sum0 = _mm256_fmadd_ps(s0, weight[2], sum0);
				sum1 = _mm256_fmadd_ps(s0, weight[1], sum1);
				s0 = _mm256_loadu_ps(src0 + 3 * srcC);
				sum1 = _mm256_fmadd_ps(s0, weight[2], sum1);

				s0 = _mm256_loadu_ps(src1 + 0 * srcC);
				sum0 = _mm256_fmadd_ps(s0, weight[3], sum0);
				s0 = _mm256_loadu_ps(src1 + 1 * srcC);
				sum0 = _mm256_fmadd_ps(s0, weight[4], sum0);
				sum1 = _mm256_fmadd_ps(s0, weight[3], sum1);
				s0 = _mm256_loadu_ps(src1 + 2 * srcC);
				sum0 = _mm256_fmadd_ps(s0, weight[5], sum0);
				sum1 = _mm256_fmadd_ps(s0, weight[4], sum1);
				s0 = _mm256_loadu_ps(src1 + 3 * srcC);
				sum1 = _mm256_fmadd_ps(s0, weight[5], sum1);

				s0 = _mm256_loadu_ps(src2 + 0 * srcC);
				sum0 = _mm256_fmadd_ps(s0, weight[6], sum0);
				s0 = _mm256_loadu_ps(src2 + 1 * srcC);
				sum0 = _mm256_fmadd_ps(s0, weight[7], sum0);
				sum1 = _mm256_fmadd_ps(s0, weight[6], sum1);
				s0 = _mm256_loadu_ps(src2 + 2 * srcC);
				sum0 = _mm256_fmadd_ps(s0, weight[8], sum0);
				sum1 = _mm256_fmadd_ps(s0, weight[7], sum1);
				s0 = _mm256_loadu_ps(src2 + 3 * srcC);
				sum1 = _mm256_fmadd_ps(s0, weight[8], sum1);

				_mm256_storeu_ps(dst + 0 * F, Activate<type>(sum0, params, 0));
				_mm256_storeu_ps(dst + 1 * F, Activate<type>(sum1, params, 0));
			}

			template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
			{
				size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
				size_t srcX = p.srcC, srcW = p.srcW * srcX, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F;
				size_t dstM = (bufH[1] - 1), srcS = bufH[0] * srcW, dstS = bufH[1] * dstW;
				size_t xStep = srcX * p.strideX, xStep0 = (p.strideX - p.padX) * srcX;
				size_t xMainEnd = p.dstW - p.padW, xMainEnd2 = AlignLo(xMainEnd - padX, 2) * (p.strideX == 1 ? 1 : 0) + padX;
				size_t yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

				__m256 _params[2];
				_params[0] = _mm256_set1_ps(params[0]);
				if (type == SimdConvolutionActivationRestrictRange ||
					type == SimdConvolutionActivationHswish ||
					type == SimdConvolutionActivationHardSigmoid)
					_params[1] = _mm256_set1_ps(params[1]);
				for (size_t c = 0; c < srcC; c += F)
				{
					__m256 _weight[9];
					for (size_t i = 0; i < 9; ++i)
						_weight[i] = _mm256_loadu_ps(weight + i * F);
					__m256 _bias = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
					if (type == ::SimdConvolutionActivationPrelu)
						_params[0] = _mm256_loadu_ps(params + c);

					size_t dy = yBeg;
					if (yBeg == 0 && padY)
					{
						size_t sy = 0, dx = 0;
						const float* src0 = src + (sy + 0) * srcW;
						const float* src1 = src + (sy + 1) * srcW;
						float* pDst = dst + (dy & dstM) * dstW;
						if (padX)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, srcX, _weight + 4, _bias, _params, pDst), pDst += F, dx++, src0 += xStep0, src1 += xStep0;
						for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep)
							ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, srcX, _weight + 3, _bias, _params, pDst);
						if (padW)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, srcX, _weight + 3, _bias, _params, pDst);
						dy++;
					}
					for (; dy < yMainEnd; ++dy)
					{
						size_t sy = dy * strideY - padY, dx = 0;
						const float* src0 = src + (sy + 0) * srcW;
						const float* src1 = src + (sy + 1) * srcW;
						const float* src2 = src + (sy + 2) * srcW;
						float* pDst = dst + (dy & dstM) * dstW;
						if (padX)
							ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, srcX, _weight + 1, _bias, _params, pDst), pDst += F, dx++, src0 += xStep0, src1 += xStep0, src2 += xStep0;
						for (; dx < xMainEnd2; dx += 2, pDst += F * 2, src0 += xStep * 2, src1 += xStep * 2, src2 += xStep * 2)
							ConvolutionDepthwise3x3Main1x2<type>(src0, src1, src2, srcX, _weight + 0, _bias, _params, pDst);
						for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep, src2 += xStep)
							ConvolutionDepthwise3x3Main1x1<type>(src0, src1, src2, srcX, _weight + 0, _bias, _params, pDst);
						if (padW)
							ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, srcX, _weight + 0, _bias, _params, pDst);
					}
					if (dy < yEnd)
					{
						size_t sy = dy * strideY - padY, dx = 0;
						const float* src0 = src + (sy + 0) * srcW;
						const float* src1 = src + (sy + 1) * srcW;
						float* pDst = dst + (dy & dstM) * dstW;
						if (padX)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, srcX, _weight + 1, _bias, _params, pDst), pDst += F, dx++, src0 += xStep0, src1 += xStep0;
						for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep)
							ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, srcX, _weight + 0, _bias, _params, pDst);
						if (padW)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, srcX, _weight + 0, _bias, _params, pDst);
					}
					src += F;
					dst += dstS;
					weight += weightS;
				}
			}

			//---------------------------------------------------------------------

			template <SimdConvolutionActivationType type> void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32fCdc::ConvolutionPtr* c)
			{
				switch (t)
				{
				case 1:
					if (p.conv[i].kernelY == 3)
						c[i + 0] = DepthwiseConvolution3x3<type>;
					else
						c[i + 0] = DepthwiseConvolution<type>;
					break;
				default:
					assert(0);
				}
			}
		}

		//---------------------------------------------------------------------

		SynetMergedConvolution32fDc::SynetMergedConvolution32fDc(const MergConvParam32f& p)
			: Avx::SynetMergedConvolution32fDc(p)
		{
			SynetMergedConvolution32fDc::Set(_param, 1, 0, _convolution);
			SynetMergedConvolution32fCdc::Set(_param, 2, 1, _convolution);
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Avx::F);
		}

		void SynetMergedConvolution32fDc::Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c)
		{
			switch (p.conv[i].activation)
			{
			case SimdConvolutionActivationIdentity: Dc::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
			case SimdConvolutionActivationRelu: Dc::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
			case SimdConvolutionActivationLeakyRelu: Dc::Set<SimdConvolutionActivationPrelu>(p, t, i, c); break;
			case SimdConvolutionActivationRestrictRange: Dc::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
			case SimdConvolutionActivationPrelu: Dc::Set<SimdConvolutionActivationPrelu>(p, t, i, c); break;
			case SimdConvolutionActivationElu: Dc::Set<SimdConvolutionActivationElu>(p, t, i, c); break;
			case SimdConvolutionActivationHswish: Dc::Set<SimdConvolutionActivationHswish>(p, t, i, c); break;
			case SimdConvolutionActivationMish: Dc::Set<SimdConvolutionActivationMish>(p, t, i, c); break;
			case SimdConvolutionActivationHardSigmoid: Dc::Set<SimdConvolutionActivationHardSigmoid>(p, t, i, c); break;
			case SimdConvolutionActivationSwish: Dc::Set<SimdConvolutionActivationSwish>(p, t, i, c); break;
			default: assert(0);
			}
		}
	}
#endif//SIMD_AVX2_ENABLE
}
