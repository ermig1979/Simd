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
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Neon
	{
		namespace Cd
		{
			SIMD_INLINE void Save(float* ptr, float32x4_t val, size_t tail)
			{
				float tmp[F];
				Store<false>(tmp, val);
				for (size_t i = 0; i < tail; ++i)
					ptr[i] = tmp[i];
			}

			template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
			{
				size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
				size_t srcW = p.srcW * F, weightS = p.kernelY * p.kernelX * F, strideXF = strideX * F;
				size_t srcM = (bufH[0] - 1), srcS = bufH[0] * srcW, dstS = p.dstW*p.dstC;
				size_t noseY = (p.padY + p.strideY - 1) / p.strideY;
				size_t bodyY = (p.srcH + p.padY + p.strideY - p.kernelY) / p.strideY;
				size_t noseX = (p.padX + p.strideX - 1) / p.strideX;
				size_t bodyX = (p.srcW + p.padX + p.strideX - p.kernelX) / p.strideX;
				size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
				size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
				size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;
				size_t srcCF = AlignLo(srcC, F);

				float32x4_t _params[2];
				_params[0] = vdupq_n_f32(params[0]);
				if (type == SimdConvolutionActivationRestrictRange ||
					type == SimdConvolutionActivationHswish ||
					type == SimdConvolutionActivationHardSigmoid)
					_params[1] = vdupq_n_f32(params[1]);
				for (size_t c = 0; c < srcC; c += F)
				{
					float32x4_t _bias = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
					if (type == ::SimdConvolutionActivationPrelu)
						_params[0] = Load<false>(params + c);

					if (c == srcCF)
					{
						size_t tail = srcC - srcCF;
						for (size_t dy = yBeg; dy < yEnd; ++dy)
						{
							float* pd = dst + dy * dstS;
							for (size_t dx = 0; dx < p.dstW; ++dx, pd += srcC)
							{
								float32x4_t sum = _bias;
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
												const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
												sum = vmlaq_f32(sum, Load<false>(ps), Load<false>(pw));
											}
										}
									}
								}
								Save(pd, Activate<type>(sum, _params, 0), tail);
							}
						}					
					}
					else
					{
						for (size_t dy = yBeg; dy < yEnd; ++dy)
						{
							float* pd = dst + dy * dstS;
							if (dy >= noseY && dy < bodyY)
							{
								size_t dx = 0;
								for (; dx < noseX; ++dx, pd += srcC)
								{
									float32x4_t sum = _bias;
									for (size_t ky = 0; ky < p.kernelY; ++ky)
									{
										size_t sy = dy * p.strideY + ky - padY;
										for (size_t kx = 0; kx < p.kernelX; ++kx)
										{
											size_t sx = dx * p.strideX + kx - padX;
											if (sx < p.srcW)
											{
												const float* pw = weight + (ky * p.kernelX + kx) * F;
												const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
												sum = vmlaq_f32(sum, Load<false>(ps), Load<false>(pw));
											}
										}
									}
									Store<false>(pd, Activate<type>(sum, _params, 0));
								}
								for (; dx < bodyX8; dx += 8, pd += 8 * srcC)
								{
									float32x4_t sum0 = _bias;
									float32x4_t sum1 = _bias;
									float32x4_t sum2 = _bias;
									float32x4_t sum3 = _bias;
									float32x4_t sum4 = _bias;
									float32x4_t sum5 = _bias;
									float32x4_t sum6 = _bias;
									float32x4_t sum7 = _bias;
									const float* pw = weight;
									for (size_t ky = 0; ky < p.kernelY; ++ky)
									{
										size_t sy = dy * strideY + ky - padY;
										const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
										for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
										{
											float32x4_t w0 = Load<false>(pw);
											sum0 = vmlaq_f32(sum0, Load<false>(ps + 0 * strideXF), w0);
											sum1 = vmlaq_f32(sum1, Load<false>(ps + 1 * strideXF), w0);
											sum2 = vmlaq_f32(sum2, Load<false>(ps + 2 * strideXF), w0);
											sum3 = vmlaq_f32(sum3, Load<false>(ps + 3 * strideXF), w0);
											sum4 = vmlaq_f32(sum4, Load<false>(ps + 4 * strideXF), w0);
											sum5 = vmlaq_f32(sum5, Load<false>(ps + 5 * strideXF), w0);
											sum6 = vmlaq_f32(sum6, Load<false>(ps + 6 * strideXF), w0);
											sum7 = vmlaq_f32(sum7, Load<false>(ps + 7 * strideXF), w0);
										}
									}
									Store<false>(pd + 0 * srcC, Activate<type>(sum0, _params, 0));
									Store<false>(pd + 1 * srcC, Activate<type>(sum1, _params, 0));
									Store<false>(pd + 2 * srcC, Activate<type>(sum2, _params, 0));
									Store<false>(pd + 3 * srcC, Activate<type>(sum3, _params, 0));
									Store<false>(pd + 4 * srcC, Activate<type>(sum4, _params, 0));
									Store<false>(pd + 5 * srcC, Activate<type>(sum5, _params, 0));
									Store<false>(pd + 6 * srcC, Activate<type>(sum6, _params, 0));
									Store<false>(pd + 7 * srcC, Activate<type>(sum7, _params, 0));
								}
								for (; dx < bodyX4; dx += 4, pd += 4 * srcC)
								{
									float32x4_t sum0 = _bias;
									float32x4_t sum1 = _bias;
									float32x4_t sum2 = _bias;
									float32x4_t sum3 = _bias;
									const float* pw = weight;
									for (size_t ky = 0; ky < p.kernelY; ++ky)
									{
										size_t sy = dy * strideY + ky - padY;
										const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
										for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
										{
											float32x4_t w0 = Load<false>(pw);
											sum0 = vmlaq_f32(sum0, Load<false>(ps + 0 * strideXF), w0);
											sum1 = vmlaq_f32(sum1, Load<false>(ps + 1 * strideXF), w0);
											sum2 = vmlaq_f32(sum2, Load<false>(ps + 2 * strideXF), w0);
											sum3 = vmlaq_f32(sum3, Load<false>(ps + 3 * strideXF), w0);
										}
									}
									Store<false>(pd + 0 * srcC, Activate<type>(sum0, _params, 0));
									Store<false>(pd + 1 * srcC, Activate<type>(sum1, _params, 0));
									Store<false>(pd + 2 * srcC, Activate<type>(sum2, _params, 0));
									Store<false>(pd + 3 * srcC, Activate<type>(sum3, _params, 0));
								}
								for (; dx < bodyX2; dx += 2, pd += 2 * srcC)
								{
									float32x4_t sum0 = _bias;
									float32x4_t sum1 = _bias;
									const float* pw = weight;
									for (size_t ky = 0; ky < p.kernelY; ++ky)
									{
										size_t sy = dy * strideY + ky - padY;
										const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
										for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
										{
											float32x4_t w0 = Load<false>(pw);
											sum0 = vmlaq_f32(sum0, Load<false>(ps + 0 * strideXF), w0);
											sum1 = vmlaq_f32(sum1, Load<false>(ps + 1 * strideXF), w0);
										}
									}
									Store<false>(pd + 0 * srcC, Activate<type>(sum0, _params, 0));
									Store<false>(pd + 1 * srcC, Activate<type>(sum1, _params, 0));
								}
								for (; dx < bodyX; ++dx, pd += srcC)
								{
									float32x4_t sum = _bias;
									const float* pw = weight;
									for (size_t ky = 0; ky < p.kernelY; ++ky)
									{
										size_t sy = dy * strideY + ky - padY;
										const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
										for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
										{
											float32x4_t w0 = Load<false>(pw);
											sum = vmlaq_f32(sum, Load<false>(ps + 0 * strideXF), w0);
										}
									}
									Store<false>(pd, Activate<type>(sum, _params, 0));
								}
								for (; dx < p.dstW; ++dx, pd += srcC)
								{
									float32x4_t sum = _bias;
									for (size_t ky = 0; ky < p.kernelY; ++ky)
									{
										size_t sy = dy * strideY + ky - padY;
										for (size_t kx = 0; kx < p.kernelX; ++kx)
										{
											size_t sx = dx * strideX + kx - padX;
											if (sx < p.srcW)
											{
												const float* pw = weight + (ky * p.kernelX + kx) * F;
												const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
												sum = vmlaq_f32(sum, Load<false>(ps), Load<false>(pw));
											}
										}
									}
									Store<false>(pd, Activate<type>(sum, _params, 0));
								}
							}
							else
							{
								for (size_t dx = 0; dx < p.dstW; ++dx, pd += srcC)
								{
									float32x4_t sum = _bias;
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
													const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
													sum = vmlaq_f32(sum, Load<false>(ps), Load<false>(pw));
												}
											}
										}
									}
									Store<false>(pd, Activate<type>(sum, _params, 0));
								}
							}
						}
					}

					src += srcS;
					dst += F;
					weight += weightS;
				}
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x2(
				const float* src0, const float* src1, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, float* dst)
			{
				float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f);
				sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * F), weight[0]);
				sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * F), weight[1]);
				sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * F), weight[3]);
				sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * F), weight[4]);
				Store<false>(dst, Activate<type>(vaddq_f32(sum0, sum1), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
				const float* src0, const float* src1, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, float* dst)
			{
				float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f), sum2 = vdupq_n_f32(0.0f);
				sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * F), weight[0]);
				sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * F), weight[1]);
				sum2 = vmlaq_f32(sum2, Load<false>(src0 + 2 * F), weight[2]);
				sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * F), weight[3]);
				sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * F), weight[4]);
				sum2 = vmlaq_f32(sum2, Load<false>(src1 + 2 * F), weight[5]);
				Store<false>(dst, Activate<type>(vaddq_f32(vaddq_f32(sum0, sum1), sum2), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
				const float* src0, const float* src1, const float* src2, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, float* dst)
			{
				float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f);
				sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * F), weight[0]);
				sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * F), weight[1]);
				sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * F), weight[3]);
				sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * F), weight[4]);
				sum0 = vmlaq_f32(sum0, Load<false>(src2 + 0 * F), weight[6]);
				sum1 = vmlaq_f32(sum1, Load<false>(src2 + 1 * F), weight[7]);
				Store<false>(dst, Activate<type>(vaddq_f32(sum0, sum1), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
				const float* src0, const float* src1, const float* src2, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, float* dst)
			{
				float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f), sum2 = vdupq_n_f32(0.0f);
				sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * F), weight[0]);
				sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * F), weight[1]);
				sum2 = vmlaq_f32(sum2, Load<false>(src0 + 2 * F), weight[2]);
				sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * F), weight[3]);
				sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * F), weight[4]);
				sum2 = vmlaq_f32(sum2, Load<false>(src1 + 2 * F), weight[5]);
				sum0 = vmlaq_f32(sum0, Load<false>(src2 + 0 * F), weight[6]);
				sum1 = vmlaq_f32(sum1, Load<false>(src2 + 1 * F), weight[7]);
				sum2 = vmlaq_f32(sum2, Load<false>(src2 + 2 * F), weight[8]);
				Store<false>(dst, Activate<type>(vaddq_f32(vaddq_f32(sum0, sum1), sum2), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x2(
				const float* src0, const float* src1, const float* src2, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, float* dst, size_t dstC)
			{
				float32x4_t sum0 = bias, sum1 = bias, s0;

				s0 = Load<false>(src0 + 0 * F);
				sum0 = vmlaq_f32(sum0, s0, weight[0]);
				s0 = Load<false>(src0 + 1 * F);
				sum0 = vmlaq_f32(sum0, s0, weight[1]);
				sum1 = vmlaq_f32(sum1, s0, weight[0]);
				s0 = Load<false>(src0 + 2 * F);
				sum0 = vmlaq_f32(sum0, s0, weight[2]);
				sum1 = vmlaq_f32(sum1, s0, weight[1]);
				s0 = Load<false>(src0 + 3 * F);
				sum1 = vmlaq_f32(sum1, s0, weight[2]);

				s0 = Load<false>(src1 + 0 * F);
				sum0 = vmlaq_f32(sum0, s0, weight[3]);
				s0 = Load<false>(src1 + 1 * F);
				sum0 = vmlaq_f32(sum0, s0, weight[4]);
				sum1 = vmlaq_f32(sum1, s0, weight[3]);
				s0 = Load<false>(src1 + 2 * F);
				sum0 = vmlaq_f32(sum0, s0, weight[5]);
				sum1 = vmlaq_f32(sum1, s0, weight[4]);
				s0 = Load<false>(src1 + 3 * F);
				sum1 = vmlaq_f32(sum1, s0, weight[5]);

				s0 = Load<false>(src2 + 0 * F);
				sum0 = vmlaq_f32(sum0, s0, weight[6]);
				s0 = Load<false>(src2 + 1 * F);
				sum0 = vmlaq_f32(sum0, s0, weight[7]);
				sum1 = vmlaq_f32(sum1, s0, weight[6]);
				s0 = Load<false>(src2 + 2 * F);
				sum0 = vmlaq_f32(sum0, s0, weight[8]);
				sum1 = vmlaq_f32(sum1, s0, weight[7]);
				s0 = Load<false>(src2 + 3 * F);
				sum1 = vmlaq_f32(sum1, s0, weight[8]);

				Store<false>(dst + 0 * dstC, Activate<type>(sum0, params, 0));
				Store<false>(dst + 1 * dstC, Activate<type>(sum1, params, 0));
			}

			template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
			{
				size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
				size_t srcW = p.srcW * F, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F;
				size_t srcM = (bufH[0] - 1), srcS = bufH[0] * srcW, dstS = p.dstW * p.dstC;
				size_t xStep = F * p.strideX, xStep0 = (p.strideX - p.padX) * F;
				size_t xMainEnd = p.dstW - p.padW, xMainEnd2 = AlignLo(xMainEnd - padX, 2)*(p.strideX == 1 ? 1 : 0) + padX;
				size_t yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

				float32x4_t _params[2];
				_params[0] = vdupq_n_f32(params[0]);
				if (type == SimdConvolutionActivationRestrictRange ||
					type == SimdConvolutionActivationHswish ||
					type == SimdConvolutionActivationHardSigmoid)
					_params[1] = vdupq_n_f32(params[1]);
				for (size_t c = 0; c < srcC; c += F)
				{
					float32x4_t _weight[9];
					for (size_t i = 0; i < 9; ++i)
						_weight[i] = Load<false>(weight + i * F);
					float32x4_t _bias = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
					if (type == ::SimdConvolutionActivationPrelu)
						_params[0] = Load<false>(params + c);

					size_t dy = yBeg;
					if (yBeg == 0 && padY)
					{
						size_t sy = 0, dx = 0;
						const float* src0 = src + ((sy + 0) & srcM) * srcW;
						const float* src1 = src + ((sy + 1) & srcM) * srcW;
						float* pDst = dst + dy * dstS;
						if (padX)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 4, _bias, _params, pDst), pDst += p.dstC, dx++, src0 += xStep0, src1 += xStep0;
						for (; dx < xMainEnd; dx++, pDst += p.dstC, src0 += xStep, src1 += xStep)
							ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, _weight + 3, _bias, _params, pDst);
						if (padW)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 3, _bias, _params, pDst);
						dy++;
					}
					for (; dy < yMainEnd; ++dy)
					{
						size_t sy = dy * strideY - padY, dx = 0;
						const float* src0 = src + ((sy + 0) & srcM) * srcW;
						const float* src1 = src + ((sy + 1) & srcM) * srcW;
						const float* src2 = src + ((sy + 2) & srcM) * srcW;
						float* pDst = dst + dy * dstS;
						if (padX)
							ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 1, _bias, _params, pDst), pDst += p.dstC, dx++, src0 += xStep0, src1 += xStep0, src2 += xStep0;
						for (; dx < xMainEnd2; dx += 2, pDst += 2* p.dstC, src0 += 2*xStep, src1 += 2*xStep, src2 += 2*xStep)
							ConvolutionDepthwise3x3Main1x2<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst, srcC);
						for (; dx < xMainEnd; dx++, pDst += p.dstC, src0 += xStep, src1 += xStep, src2 += xStep)
							ConvolutionDepthwise3x3Main1x1<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst);
						if (padW)
							ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst);
					}
					if (dy < yEnd)
					{
						size_t sy = dy * strideY - padY, dx = 0;
						const float* src0 = src + ((sy + 0) & srcM) * srcW;
						const float* src1 = src + ((sy + 1) & srcM) * srcW;
						float* pDst = dst + dy * dstS;
						if (padX)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 1, _bias, _params, pDst), pDst += p.dstC, dx++, src0 += xStep0, src1 += xStep0;
						for (; dx < xMainEnd; dx++, pDst += p.dstC, src0 += xStep, src1 += xStep)
							ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, _weight + 0, _bias, _params, pDst);
						if (padW)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 0, _bias, _params, pDst);
					}
					src += srcS;
					dst += F;
					weight += weightS;
				}
			}

			//---------------------------------------------------------------------

			template <SimdConvolutionActivationType type> void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32fCd::ConvolutionPtr * c)
			{
				switch (t)
				{
				case 1:
					if (p.conv[i].kernelY == 3 && Aligned(p.conv[i].dstC, F))
						c[i] = DepthwiseConvolution3x3<type>;
					else
						c[i] = DepthwiseConvolution<type>;
					break;
				default:
					assert(0);
				}
			}
		}

		//---------------------------------------------------------------------

		SynetMergedConvolution32fCd::SynetMergedConvolution32fCd(const MergConvParam32f& p)
			: Base::SynetMergedConvolution32fCd(p)
		{
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), F);
			SynetMergedConvolution32fCdc::Set(_param, 0, 0, _convolution);
			SynetMergedConvolution32fCd::Set(_param, 1, 1, _convolution);
		}

		void SynetMergedConvolution32fCd::Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c)
		{
			switch (p.conv[i].activation)
			{
			case SimdConvolutionActivationIdentity: Cd::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
			case SimdConvolutionActivationRelu: Cd::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
			case SimdConvolutionActivationLeakyRelu: Cd::Set<SimdConvolutionActivationPrelu>(p, t, i, c); break;
			case SimdConvolutionActivationRestrictRange: Cd::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
			case SimdConvolutionActivationPrelu: Cd::Set<SimdConvolutionActivationPrelu>(p, t, i, c); break;
			case SimdConvolutionActivationElu: Cd::Set<SimdConvolutionActivationElu>(p, t, i, c); break;
			case SimdConvolutionActivationHswish: Cd::Set<SimdConvolutionActivationHswish>(p, t, i, c); break;
			case SimdConvolutionActivationMish: Cd::Set<SimdConvolutionActivationMish>(p, t, i, c); break;
			case SimdConvolutionActivationHardSigmoid: Cd::Set<SimdConvolutionActivationHardSigmoid>(p, t, i, c); break;
			case SimdConvolutionActivationSwish: Cd::Set<SimdConvolutionActivationSwish>(p, t, i, c); break;
			default: assert(0);
			}
		}
	}
#endif//SIMD_NEON_ENABLE
}
