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
		template<SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate(svfloat32_t value, const float* params, const svbool_t& mask);

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationIdentity>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			return value;
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationRelu>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			return svmax_n_f32_x(mask, value, 0.0f);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationLeakyRelu>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			return svmla_n_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), svmin_n_f32_x(mask, value, 0.0f), params[0]);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationRestrictRange>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			return svmin_n_f32_x(mask, svmax_n_f32_x(mask, value, params[0]), params[1]);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationPrelu>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), svld1_f32(mask, params), svmin_n_f32_x(mask, value, 0.0f));
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationElu>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			svfloat32_t neg = svmul_n_f32_x(mask, svsub_n_f32_x(mask, Exponent(mask, value), 1.0f), params[0]);
			return svsel_f32(svcmplt_n_f32(mask, value, 0.0f), neg, value);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHswish>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			svfloat32_t shift = svdup_n_f32(params[0]);
			svfloat32_t scale = svdup_n_f32(params[1]);
			svfloat32_t upper = svmin_f32_x(mask, value, shift);
			svfloat32_t positive = svmax_n_f32_x(mask, svadd_f32_x(mask, upper, shift), 0.0f);
			return svmul_f32_x(mask, svmul_f32_x(mask, positive, scale), value);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationMish>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			svfloat32_t exp = svmin_f32_x(mask, Exponent(mask, value), svdup_n_f32(params[0]));
			return svmul_f32_x(mask, value, Tanh(mask, Logarithm(mask, svadd_n_f32_x(mask, exp, 1.0f))));
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHardSigmoid>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			return svmax_n_f32_x(mask, svmin_n_f32_x(mask, svmla_n_f32_x(mask, svdup_n_f32(params[1]), value, params[0]), 1.0f), 0.0f);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationSwish>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			svfloat32_t exp = Exponent(mask, svneg_f32_x(mask, svmul_n_f32_x(mask, value, params[0])));
			return svdiv_f32_x(mask, value, svadd_n_f32_x(mask, exp, 1.0f));
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationGelu>(svfloat32_t value, const float* params, const svbool_t& mask)
		{
			svfloat32_t t = svmul_n_f32_x(mask, value, 0.70710678118654752440f);
			return svmul_f32_x(mask, svmul_n_f32_x(mask, t, 0.70710678118654752440f), svadd_n_f32_x(mask, Erf(mask, t), 1.0f));
		}

		//-------------------------------------------------------------------------------------------------------

		template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			const size_t F = svcntw();
			size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = srcC;
			size_t sM = (bufH[0] - 1), sD = bufH[0] ? bufH[0] * p.srcW * F : F, sX = bufH[0] ? F : p.srcC, sY = sX * p.srcW;
			size_t dM = (bufH[1] - 1), dX = (bufH[1] ? F : p.dstC), dY = p.dstW * dX, dy0 = bufH[1] ? yBeg : 0, dD = bufH[1] ? bufH[1] * dY : F;
			size_t wD = p.kernelY * p.kernelX * F, ssX = strideX * sX;
			size_t noseY = NoseH(p), bodyY = BodyH(p), noseX = NoseW(p), bodyX = BodyW(p);
			size_t bodyS = bodyX > noseX ? bodyX - noseX : 0;
			size_t bodyX2 = AlignLo(bodyS, 2) + noseX;
			size_t bodyX4 = AlignLo(bodyS, 4) + noseX;
			size_t bodyX8 = AlignLo(bodyS, 8) + noseX;
			size_t dstCF = AlignLo(dstC, F);

			for (size_t c = 0; c < dstC; c += F)
			{
				svbool_t mask = svwhilelt_b32((uint64_t)0, (uint64_t)Simd::Min(F, dstC - c));
				svfloat32_t _bias = bias ? svld1_f32(mask, bias + c) : svdup_n_f32(0.0f);
				const float* pParams = (type == ::SimdConvolutionActivationPrelu) ? params + c : params;
				if (c == dstCF)
				{
					for (size_t dy = yBeg; dy < yEnd; ++dy)
					{
						float* pd = dst + (dy & dM) * dY;
						for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
						{
							svfloat32_t sum0 = _bias;
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
											sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, ps), svld1_f32(mask, pw));
										}
									}
								}
							}
							svst1_f32(mask, pd, Activate<type>(sum0, pParams, mask));
						}
					}
					return;
				}
				const svbool_t body = svptrue_b32();
				for (size_t dy = yBeg; dy < yEnd; ++dy)
				{
					float* pd = dst + (dy & dM) * dY;
					if (dy >= noseY && dy < bodyY)
					{
						size_t dx = 0;
						for (; dx < noseX; dx += 1, pd += dX)
						{
							svfloat32_t sum0 = _bias;
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
										sum0 = svmla_f32_x(body, sum0, svld1_f32(body, ps), svld1_f32(body, pw));
									}
								}
							}
							svst1_f32(body, pd + 0 * dX, Activate<type>(sum0, pParams, body));
						}
						for (; dx < bodyX8; dx += 8, pd += 8 * dX)
						{
							svfloat32_t sum0 = _bias;
							svfloat32_t sum1 = _bias;
							svfloat32_t sum2 = _bias;
							svfloat32_t sum3 = _bias;
							svfloat32_t sum4 = _bias;
							svfloat32_t sum5 = _bias;
							svfloat32_t sum6 = _bias;
							svfloat32_t sum7 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									svfloat32_t w0 = svld1_f32(body, pw);
									sum0 = svmla_f32_x(body, sum0, svld1_f32(body, ps + 0 * ssX), w0);
									sum1 = svmla_f32_x(body, sum1, svld1_f32(body, ps + 1 * ssX), w0);
									sum2 = svmla_f32_x(body, sum2, svld1_f32(body, ps + 2 * ssX), w0);
									sum3 = svmla_f32_x(body, sum3, svld1_f32(body, ps + 3 * ssX), w0);
									sum4 = svmla_f32_x(body, sum4, svld1_f32(body, ps + 4 * ssX), w0);
									sum5 = svmla_f32_x(body, sum5, svld1_f32(body, ps + 5 * ssX), w0);
									sum6 = svmla_f32_x(body, sum6, svld1_f32(body, ps + 6 * ssX), w0);
									sum7 = svmla_f32_x(body, sum7, svld1_f32(body, ps + 7 * ssX), w0);
								}
							}
							svst1_f32(body, pd + 0 * dX, Activate<type>(sum0, pParams, body));
							svst1_f32(body, pd + 1 * dX, Activate<type>(sum1, pParams, body));
							svst1_f32(body, pd + 2 * dX, Activate<type>(sum2, pParams, body));
							svst1_f32(body, pd + 3 * dX, Activate<type>(sum3, pParams, body));
							svst1_f32(body, pd + 4 * dX, Activate<type>(sum4, pParams, body));
							svst1_f32(body, pd + 5 * dX, Activate<type>(sum5, pParams, body));
							svst1_f32(body, pd + 6 * dX, Activate<type>(sum6, pParams, body));
							svst1_f32(body, pd + 7 * dX, Activate<type>(sum7, pParams, body));
						}
						for (; dx < bodyX4; dx += 4, pd += 4 * dX)
						{
							svfloat32_t sum0 = _bias;
							svfloat32_t sum1 = _bias;
							svfloat32_t sum2 = _bias;
							svfloat32_t sum3 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									svfloat32_t w0 = svld1_f32(body, pw);
									sum0 = svmla_f32_x(body, sum0, svld1_f32(body, ps + 0 * ssX), w0);
									sum1 = svmla_f32_x(body, sum1, svld1_f32(body, ps + 1 * ssX), w0);
									sum2 = svmla_f32_x(body, sum2, svld1_f32(body, ps + 2 * ssX), w0);
									sum3 = svmla_f32_x(body, sum3, svld1_f32(body, ps + 3 * ssX), w0);
								}
							}
							svst1_f32(body, pd + 0 * dX, Activate<type>(sum0, pParams, body));
							svst1_f32(body, pd + 1 * dX, Activate<type>(sum1, pParams, body));
							svst1_f32(body, pd + 2 * dX, Activate<type>(sum2, pParams, body));
							svst1_f32(body, pd + 3 * dX, Activate<type>(sum3, pParams, body));
						}
						for (; dx < bodyX2; dx += 2, pd += 2 * dX)
						{
							svfloat32_t sum0 = _bias;
							svfloat32_t sum1 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									svfloat32_t w0 = svld1_f32(body, pw);
									sum0 = svmla_f32_x(body, sum0, svld1_f32(body, ps + 0 * ssX), w0);
									sum1 = svmla_f32_x(body, sum1, svld1_f32(body, ps + 1 * ssX), w0);
								}
							}
							svst1_f32(body, pd + 0 * dX, Activate<type>(sum0, pParams, body));
							svst1_f32(body, pd + 1 * dX, Activate<type>(sum1, pParams, body));
						}
						for (; dx < bodyX; dx += 1, pd += dX)
						{
							svfloat32_t sum0 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									svfloat32_t w0 = svld1_f32(body, pw);
									sum0 = svmla_f32_x(body, sum0, svld1_f32(body, ps), w0);
								}
							}
							svst1_f32(body, pd + 0 * dX, Activate<type>(sum0, pParams, body));
						}
						for (; dx < p.dstW; dx += 1, pd += dX)
						{
							svfloat32_t sum0 = _bias;
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
										sum0 = svmla_f32_x(body, sum0, svld1_f32(body, ps), svld1_f32(body, pw));
									}
								}
							}
							svst1_f32(body, pd + 0 * dX, Activate<type>(sum0, pParams, body));
						}
					}
					else
					{
						for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
						{
							svfloat32_t sum0 = _bias;
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
											sum0 = svmla_f32_x(body, sum0, svld1_f32(body, ps), svld1_f32(body, pw));
										}
									}
								}
							}
							svst1_f32(body, pd + 0 * dX, Activate<type>(sum0, pParams, body));
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
			const float* src0, const float* src1, size_t sX, const float* weight, size_t F, svfloat32_t bias, const float* params, const svbool_t& mask, float* dst)
		{
			svfloat32_t sum0 = bias, sum1 = svdup_n_f32(0.0f);
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src0 + 0 * sX), svld1_f32(mask, weight + 0 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src0 + 1 * sX), svld1_f32(mask, weight + 1 * F));
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src1 + 0 * sX), svld1_f32(mask, weight + 3 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src1 + 1 * sX), svld1_f32(mask, weight + 4 * F));
			svst1_f32(mask, dst, Activate<type>(svadd_f32_x(mask, sum0, sum1), params, mask));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
			const float* src0, const float* src1, size_t sX, const float* weight, size_t F, svfloat32_t bias, const float* params, const svbool_t& mask, float* dst)
		{
			svfloat32_t sum0 = bias, sum1 = svdup_n_f32(0.0f), sum2 = svdup_n_f32(0.0f);
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src0 + 0 * sX), svld1_f32(mask, weight + 0 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src0 + 1 * sX), svld1_f32(mask, weight + 1 * F));
			sum2 = svmla_f32_x(mask, sum2, svld1_f32(mask, src0 + 2 * sX), svld1_f32(mask, weight + 2 * F));
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src1 + 0 * sX), svld1_f32(mask, weight + 3 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src1 + 1 * sX), svld1_f32(mask, weight + 4 * F));
			sum2 = svmla_f32_x(mask, sum2, svld1_f32(mask, src1 + 2 * sX), svld1_f32(mask, weight + 5 * F));
			svst1_f32(mask, dst, Activate<type>(svadd_f32_x(mask, svadd_f32_x(mask, sum0, sum1), sum2), params, mask));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
			const float* src0, const float* src1, const float* src2, size_t sX, const float* weight, size_t F, svfloat32_t bias, const float* params, const svbool_t& mask, float* dst)
		{
			svfloat32_t sum0 = bias, sum1 = svdup_n_f32(0.0f);
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src0 + 0 * sX), svld1_f32(mask, weight + 0 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src0 + 1 * sX), svld1_f32(mask, weight + 1 * F));
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src1 + 0 * sX), svld1_f32(mask, weight + 3 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src1 + 1 * sX), svld1_f32(mask, weight + 4 * F));
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src2 + 0 * sX), svld1_f32(mask, weight + 6 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src2 + 1 * sX), svld1_f32(mask, weight + 7 * F));
			svst1_f32(mask, dst, Activate<type>(svadd_f32_x(mask, sum0, sum1), params, mask));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
			const float* src0, const float* src1, const float* src2, size_t sX, const float* weight, size_t F, svfloat32_t bias, const float* params, const svbool_t& mask, float* dst)
		{
			svfloat32_t sum0 = bias, sum1 = svdup_n_f32(0.0f), sum2 = svdup_n_f32(0.0f);
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src0 + 0 * sX), svld1_f32(mask, weight + 0 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src0 + 1 * sX), svld1_f32(mask, weight + 1 * F));
			sum2 = svmla_f32_x(mask, sum2, svld1_f32(mask, src0 + 2 * sX), svld1_f32(mask, weight + 2 * F));
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src1 + 0 * sX), svld1_f32(mask, weight + 3 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src1 + 1 * sX), svld1_f32(mask, weight + 4 * F));
			sum2 = svmla_f32_x(mask, sum2, svld1_f32(mask, src1 + 2 * sX), svld1_f32(mask, weight + 5 * F));
			sum0 = svmla_f32_x(mask, sum0, svld1_f32(mask, src2 + 0 * sX), svld1_f32(mask, weight + 6 * F));
			sum1 = svmla_f32_x(mask, sum1, svld1_f32(mask, src2 + 1 * sX), svld1_f32(mask, weight + 7 * F));
			sum2 = svmla_f32_x(mask, sum2, svld1_f32(mask, src2 + 2 * sX), svld1_f32(mask, weight + 8 * F));
			svst1_f32(mask, dst, Activate<type>(svadd_f32_x(mask, svadd_f32_x(mask, sum0, sum1), sum2), params, mask));
		}

		template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			const size_t F = svcntw();
			const svbool_t mask = svptrue_b32();
			size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = srcC;
			size_t sM = (bufH[0] - 1), sD = bufH[0] ? bufH[0] * p.srcW * F : F, sX = bufH[0] ? F : p.srcC, sY = sX * p.srcW;
			size_t dM = (bufH[1] - 1), dX = (bufH[1] ? F : p.dstC), dY = p.dstW * dX, dy0 = bufH[1] ? yBeg : 0, dD = bufH[1] ? bufH[1] * dY : F;
			size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX) * sX;
			size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

			for (size_t c = 0; c < srcC; c += F)
			{
				const float* _weight = weight;
				svfloat32_t _bias = bias ? svld1_f32(mask, bias + c) : svdup_n_f32(0.0f);
				const float* pParams = (type == ::SimdConvolutionActivationPrelu) ? params + c : params;

				size_t dy = yBeg;
				if (yBeg == 0 && padY)
				{
					size_t sy = 0, dx = 0;
					const float* src0 = src + ((sy + 0) & sM) * sY;
					const float* src1 = src + ((sy + 1) & sM) * sY;
					float* pDst = dst + (dy & dM) * dY;
					if (padX)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 4 * F, F, _bias, pParams, mask, pDst),
						pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
					for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
						ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, sX, _weight + 3 * F, F, _bias, pParams, mask, pDst);
					if (padW)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 3 * F, F, _bias, pParams, mask, pDst);
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
						ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, sX, _weight + 1 * F, F, _bias, pParams, mask, pDst),
						pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
					for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
						ConvolutionDepthwise3x3Main1x1<type>(src0, src1, src2, sX, _weight + 0 * F, F, _bias, pParams, mask, pDst);
					if (padW)
						ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, sX, _weight + 0 * F, F, _bias, pParams, mask, pDst);
				}
				if (dy < yEnd)
				{
					size_t sy = dy * strideY - padY, dx = 0;
					const float* src0 = src + ((sy + 0) & sM) * sY;
					const float* src1 = src + ((sy + 1) & sM) * sY;
					float* pDst = dst + (dy & dM) * dY;
					if (padX)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 1 * F, F, _bias, pParams, mask, pDst),
						pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
					for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
						ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, sX, _weight + 0 * F, F, _bias, pParams, mask, pDst);
					if (padW)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 0 * F, F, _bias, pParams, mask, pDst);
				}
				src += sD;
				dst += dD;
				weight += wD;
			}
		}

		//-------------------------------------------------------------------------------------------------------

		template <SimdConvolutionActivationType type> void SetDepthwise(const ConvParam& p, bool last, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			const size_t F = svcntw();
			if (p.IsKernel(3) && (!last || Aligned(p.dstC, F)) && p.srcH > 1 && p.srcW > 1)
				convolution[0] = DepthwiseConvolution3x3<type>;
			else
				convolution[0] = DepthwiseConvolution<type>;
		}

		void SetDepthwise(const ConvParam& p, bool last, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			switch (p.activation)
			{
			case SimdConvolutionActivationIdentity: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, last, convolution); break;
			case SimdConvolutionActivationRelu: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, last, convolution); break;
			case SimdConvolutionActivationLeakyRelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, last, convolution); break;
			case SimdConvolutionActivationRestrictRange: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, last, convolution); break;
			case SimdConvolutionActivationPrelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, last, convolution); break;
			case SimdConvolutionActivationElu: SetDepthwise<SimdConvolutionActivationElu>(p, last, convolution); break;
			case SimdConvolutionActivationHswish: SetDepthwise<SimdConvolutionActivationHswish>(p, last, convolution); break;
			case SimdConvolutionActivationMish: SetDepthwise<SimdConvolutionActivationMish>(p, last, convolution); break;
			case SimdConvolutionActivationHardSigmoid: SetDepthwise<SimdConvolutionActivationHardSigmoid>(p, last, convolution); break;
			case SimdConvolutionActivationSwish: SetDepthwise<SimdConvolutionActivationSwish>(p, last, convolution); break;
			case SimdConvolutionActivationGelu: SetDepthwise<SimdConvolutionActivationGelu>(p, last, convolution); break;
			default: assert(0);
			}
		}
	}
#endif
}
