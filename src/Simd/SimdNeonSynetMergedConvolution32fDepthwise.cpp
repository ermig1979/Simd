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
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Neon
	{
		SIMD_INLINE void Save(float* ptr, float32x4_t val, size_t tail)
		{
			float tmp[F];
			Store<false>(tmp, val);
			for (size_t i = 0; i < tail; ++i)
				ptr[i] = tmp[i];
		}

		//-------------------------------------------------------------------------------------------------------

		template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
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

			float32x4_t _params[2];
			_params[0] = vdupq_n_f32(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				_params[1] = vdupq_n_f32(params[1]);
			for (size_t c = 0; c < dstC; c += F)
			{
				float32x4_t _bias = bias ? Load<false>(bias + c) : vdupq_n_f32(0.0f);
				if (type == ::SimdConvolutionActivationPrelu)
					_params[0] = Load<false>(params + c);
				if (c == dstCF)
				{
					size_t tail = dstC - dstCF;
					for (size_t dy = yBeg; dy < yEnd; ++dy)
					{
						float* pd = dst + (dy & dM) * dY;
						for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
						{
							float32x4_t sum0 = _bias;
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
											sum0 = vmlaq_f32(sum0, Load<false>(ps), Load<false>(pw));
										}
									}
								}
							}
							Save(pd, Activate<type>(sum0, _params, 0), tail);
						}
					}
					return;
				}
				for (size_t dy = yBeg; dy < yEnd; ++dy)
				{
					float* pd = dst + (dy & dM) * dY;
					if (dy >= noseY && dy < bodyY)
					{
						size_t dx = 0;
						for (; dx < noseX; dx += 1, pd += dX)
						{
							float32x4_t sum0 = _bias;
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
										sum0 = vmlaq_f32(sum0, Load<false>(ps), Load<false>(pw));
									}
								}
							}
							Store<false>(pd + 0 * dX, Activate<type>(sum0, _params, 0));
						}
						for (; dx < bodyX8; dx += 8, pd += 8 * dX)
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
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									float32x4_t w0 = Load<false>(pw);
									sum0 = vmlaq_f32(sum0, Load<false>(ps + 0 * ssX), w0);
									sum1 = vmlaq_f32(sum1, Load<false>(ps + 1 * ssX), w0);
									sum2 = vmlaq_f32(sum2, Load<false>(ps + 2 * ssX), w0);
									sum3 = vmlaq_f32(sum3, Load<false>(ps + 3 * ssX), w0);
									sum4 = vmlaq_f32(sum4, Load<false>(ps + 4 * ssX), w0);
									sum5 = vmlaq_f32(sum5, Load<false>(ps + 5 * ssX), w0);
									sum6 = vmlaq_f32(sum6, Load<false>(ps + 6 * ssX), w0);
									sum7 = vmlaq_f32(sum7, Load<false>(ps + 7 * ssX), w0);
								}
							}
							Store<false>(pd + 0 * dX, Activate<type>(sum0, _params, 0));
							Store<false>(pd + 1 * dX, Activate<type>(sum1, _params, 0));
							Store<false>(pd + 2 * dX, Activate<type>(sum2, _params, 0));
							Store<false>(pd + 3 * dX, Activate<type>(sum3, _params, 0));
							Store<false>(pd + 4 * dX, Activate<type>(sum4, _params, 0));
							Store<false>(pd + 5 * dX, Activate<type>(sum5, _params, 0));
							Store<false>(pd + 6 * dX, Activate<type>(sum6, _params, 0));
							Store<false>(pd + 7 * dX, Activate<type>(sum7, _params, 0));
						}
						for (; dx < bodyX4; dx += 4, pd += 4 * dX)
						{
							float32x4_t sum0 = _bias;
							float32x4_t sum1 = _bias;
							float32x4_t sum2 = _bias;
							float32x4_t sum3 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									float32x4_t w0 = Load<false>(pw);
									sum0 = vmlaq_f32(sum0, Load<false>(ps + 0 * ssX), w0);
									sum1 = vmlaq_f32(sum1, Load<false>(ps + 1 * ssX), w0);
									sum2 = vmlaq_f32(sum2, Load<false>(ps + 2 * ssX), w0);
									sum3 = vmlaq_f32(sum3, Load<false>(ps + 3 * ssX), w0);
								}
							}
							Store<false>(pd + 0 * dX, Activate<type>(sum0, _params, 0));
							Store<false>(pd + 1 * dX, Activate<type>(sum1, _params, 0));
							Store<false>(pd + 2 * dX, Activate<type>(sum2, _params, 0));
							Store<false>(pd + 3 * dX, Activate<type>(sum3, _params, 0));
						}
						for (; dx < bodyX2; dx += 2, pd += 2 * dX)
						{
							float32x4_t sum0 = _bias;
							float32x4_t sum1 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									float32x4_t w0 = Load<false>(pw);
									sum0 = vmlaq_f32(sum0, Load<false>(ps + 0 * ssX), w0);
									sum1 = vmlaq_f32(sum1, Load<false>(ps + 1 * ssX), w0);
								}
							}
							Store<false>(pd + 0 * dX, Activate<type>(sum0, _params, 0));
							Store<false>(pd + 1 * dX, Activate<type>(sum1, _params, 0));
						}
						for (; dx < bodyX; dx += 1, pd += dX)
						{
							float32x4_t sum0 = _bias;
							const float* pw = weight;
							for (size_t ky = 0; ky < p.kernelY; ++ky)
							{
								size_t sy = dy * strideY + ky - padY;
								const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
								for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
								{
									float32x4_t w0 = Load<false>(pw);
									sum0 = vmlaq_f32(sum0, Load<false>(ps), w0);
								}
							}
							Store<false>(pd + 0 * dX, Activate<type>(sum0, _params, 0));
						}
						for (; dx < p.dstW; dx += 1, pd += dX)
						{
							float32x4_t sum0 = _bias;
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
										sum0 = vmlaq_f32(sum0, Load<false>(ps), Load<false>(pw));
									}
								}
							}
							Store<false>(pd + 0 * dX, Activate<type>(sum0, _params, 0));
						}
					}
					else
					{
						for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
						{
							float32x4_t sum0 = _bias;
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
											sum0 = vmlaq_f32(sum0, Load<false>(ps), Load<false>(pw));
										}
									}
								}
							}
							Store<false>(pd + 0 * dX, Activate<type>(sum0, _params, 0));
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
			const float* src0, const float* src1, size_t sX, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, float* dst)
		{
			float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f);
			sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * sX), weight[0]);
			sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * sX), weight[1]);
			sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * sX), weight[3]);
			sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * sX), weight[4]);
			Store<false>(dst, Activate<type>(vaddq_f32(sum0, sum1), params, 0));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
			const float* src0, const float* src1, size_t sX, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, float* dst)
		{
			float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f), sum2 = vdupq_n_f32(0.0f);
			sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * sX), weight[0]);
			sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * sX), weight[1]);
			sum2 = vmlaq_f32(sum2, Load<false>(src0 + 2 * sX), weight[2]);
			sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * sX), weight[3]);
			sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * sX), weight[4]);
			sum2 = vmlaq_f32(sum2, Load<false>(src1 + 2 * sX), weight[5]);
			Store<false>(dst, Activate<type>(vaddq_f32(vaddq_f32(sum0, sum1), sum2), params, 0));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
			const float* src0, const float* src1, const float* src2, size_t sX, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, float* dst)
		{
			float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f);
			sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * sX), weight[0]);
			sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * sX), weight[1]);
			sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * sX), weight[3]);
			sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * sX), weight[4]);
			sum0 = vmlaq_f32(sum0, Load<false>(src2 + 0 * sX), weight[6]);
			sum1 = vmlaq_f32(sum1, Load<false>(src2 + 1 * sX), weight[7]);
			Store<false>(dst, Activate<type>(vaddq_f32(sum0, sum1), params, 0));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
			const float* src0, const float* src1, const float* src2, size_t sX, const float32x4_t* weight, const float32x4_t& bias, const float32x4_t* params, float* dst)
		{
			float32x4_t sum0 = bias, sum1 = vdupq_n_f32(0.0f), sum2 = vdupq_n_f32(0.0f);
			sum0 = vmlaq_f32(sum0, Load<false>(src0 + 0 * sX), weight[0]);
			sum1 = vmlaq_f32(sum1, Load<false>(src0 + 1 * sX), weight[1]);
			sum2 = vmlaq_f32(sum2, Load<false>(src0 + 2 * sX), weight[2]);
			sum0 = vmlaq_f32(sum0, Load<false>(src1 + 0 * sX), weight[3]);
			sum1 = vmlaq_f32(sum1, Load<false>(src1 + 1 * sX), weight[4]);
			sum2 = vmlaq_f32(sum2, Load<false>(src1 + 2 * sX), weight[5]);
			sum0 = vmlaq_f32(sum0, Load<false>(src2 + 0 * sX), weight[6]);
			sum1 = vmlaq_f32(sum1, Load<false>(src2 + 1 * sX), weight[7]);
			sum2 = vmlaq_f32(sum2, Load<false>(src2 + 2 * sX), weight[8]);
			Store<false>(dst, Activate<type>(vaddq_f32(vaddq_f32(sum0, sum1), sum2), params, 0));
		}

		template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = srcC;
			size_t sM = (bufH[0] - 1), sD = bufH[0] ? bufH[0] * p.srcW * F : F, sX = bufH[0] ? F : p.srcC, sY = sX * p.srcW;
			size_t dM = (bufH[1] - 1), dX = (bufH[1] ? F : p.dstC), dY = p.dstW * dX, dy0 = bufH[1] ? yBeg : 0, dD = bufH[1] ? bufH[1] * dY : F;
			size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX) * sX;
			size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

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
					const float* src0 = src + ((sy + 0) & sM) * sY;
					const float* src1 = src + ((sy + 1) & sM) * sY;
					float* pDst = dst + (dy & dM) * dY;
					if (padX)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 4, _bias, _params, pDst), 
						pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
					for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
						ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, sX, _weight + 3, _bias, _params, pDst);
					if (padW)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 3, _bias, _params, pDst);
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
						ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, sX, _weight + 1, _bias, _params, pDst), 
						pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
					for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
						ConvolutionDepthwise3x3Main1x1<type>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst);
					if (padW)
						ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst);
				}
				if (dy < yEnd)
				{
					size_t sy = dy * strideY - padY, dx = 0;
					const float* src0 = src + ((sy + 0) & sM) * sY;
					const float* src1 = src + ((sy + 1) & sM) * sY;
					float* pDst = dst + (dy & dM) * dY;
					if (padX)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 1, _bias, _params, pDst), 
						pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
					for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
						ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, sX, _weight + 0, _bias, _params, pDst);
					if (padW)
						ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, sX, _weight + 0, _bias, _params, pDst);
				}
				src += sD;
				dst += dD;
				weight += wD;
			}
		}

		//-------------------------------------------------------------------------------------------------------

		template <SimdConvolutionActivationType type> void SetDepthwise(const ConvParam& p, bool last, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
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
