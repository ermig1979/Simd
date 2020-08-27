/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#if defined(SIMD_SSE2_ENABLE)
	namespace Sse2
	{
		namespace Dc
		{
			template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst)
			{
				size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
				size_t srcW = p.srcW * F, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F, strideXF = strideX * F;
				size_t srcM = (bufH[0] - 1), dstM = (bufH[1] - 1), srcS = bufH[0] * srcW, dstS = bufH[1] * dstW;
				size_t noseY = (p.padY + p.strideY - 1) / p.strideY;
				size_t bodyY = (p.srcH + p.padY + p.strideY - p.kernelY) / p.strideY;
				size_t noseX = (p.padX + p.strideX - 1) / p.strideX;
				size_t bodyX = (p.srcW + p.padX + p.strideX - p.kernelX) / p.strideX;
				size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
				size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
				size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;

				__m128 _params[2];
				_params[0] = _mm_set1_ps(params[0]);
				if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
					_params[1] = _mm_set1_ps(params[1]);
				for (size_t c = 0; c < srcC; c += F)
				{
					__m128 _bias = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
					if (type == ::SimdConvolutionActivationPrelu)
						_params[0] = _mm_loadu_ps(params + c);

					for (size_t dy = yBeg; dy < yEnd; ++dy)
					{
						float* pd = dst + (dy & dstM) * dstW;
						//if (dy >= noseY && dy < bodyY)
						//{
						//	size_t dx = 0;
						//	for (; dx < noseX; ++dx, pd += F)
						//	{
						//		__m128 sum = _bias;
						//		for (size_t ky = 0; ky < p.kernelY; ++ky)
						//		{
						//			size_t sy = dy * p.strideY + ky - padY;
						//			for (size_t kx = 0; kx < p.kernelX; ++kx)
						//			{
						//				size_t sx = dx * p.strideX + kx - padX;
						//				if (sx < p.srcW)
						//				{
						//					const float* pw = weight + (ky * p.kernelX + kx) * F;
						//					const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
						//					sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
						//				}
						//			}
						//		}
						//		_mm_storeu_ps(pd, Activate<type>(sum, _params, 0));
						//	}
						//	for (; dx < bodyX8; dx += 8, pd += 8 * F)
						//	{
						//		__m128 sum0 = _bias;
						//		__m128 sum1 = _bias;
						//		__m128 sum2 = _bias;
						//		__m128 sum3 = _bias;
						//		__m128 sum4 = _bias;
						//		__m128 sum5 = _bias;
						//		__m128 sum6 = _bias;
						//		__m128 sum7 = _bias;
						//		const float* pw = weight;
						//		for (size_t ky = 0; ky < p.kernelY; ++ky)
						//		{
						//			size_t sy = dy * strideY + ky - padY;
						//			const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
						//			for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
						//			{
						//				__m128 w0 = _mm_loadu_ps(pw);
						//				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * strideXF), w0), sum0);
						//				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * strideXF), w0), sum1);
						//				sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * strideXF), w0), sum2);
						//				sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * strideXF), w0), sum3);
						//				sum4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 4 * strideXF), w0), sum4);
						//				sum5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 5 * strideXF), w0), sum5);
						//				sum6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 6 * strideXF), w0), sum6);
						//				sum7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 7 * strideXF), w0), sum7);
						//			}
						//		}
						//		_mm_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
						//		_mm_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
						//		_mm_storeu_ps(pd + 2 * F, Activate<type>(sum2, _params, 0));
						//		_mm_storeu_ps(pd + 3 * F, Activate<type>(sum3, _params, 0));
						//		_mm_storeu_ps(pd + 4 * F, Activate<type>(sum4, _params, 0));
						//		_mm_storeu_ps(pd + 5 * F, Activate<type>(sum5, _params, 0));
						//		_mm_storeu_ps(pd + 6 * F, Activate<type>(sum6, _params, 0));
						//		_mm_storeu_ps(pd + 7 * F, Activate<type>(sum7, _params, 0));
						//	}
						//	for (; dx < bodyX4; dx += 4, pd += 4 * F)
						//	{
						//		__m128 sum0 = _bias;
						//		__m128 sum1 = _bias;
						//		__m128 sum2 = _bias;
						//		__m128 sum3 = _bias;
						//		const float* pw = weight;
						//		for (size_t ky = 0; ky < p.kernelY; ++ky)
						//		{
						//			size_t sy = dy * strideY + ky - padY;
						//			const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
						//			for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
						//			{
						//				__m128 w0 = _mm_loadu_ps(pw);
						//				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * strideXF), w0), sum0);
						//				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * strideXF), w0), sum1);
						//				sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * strideXF), w0), sum2);
						//				sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * strideXF), w0), sum3);
						//			}
						//		}
						//		_mm_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
						//		_mm_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
						//		_mm_storeu_ps(pd + 2 * F, Activate<type>(sum2, _params, 0));
						//		_mm_storeu_ps(pd + 3 * F, Activate<type>(sum3, _params, 0));
						//	}
						//	for (; dx < bodyX2; dx += 2, pd += 2 * F)
						//	{
						//		__m128 sum0 = _bias;
						//		__m128 sum1 = _bias;
						//		const float* pw = weight;
						//		for (size_t ky = 0; ky < p.kernelY; ++ky)
						//		{
						//			size_t sy = dy * strideY + ky - padY;
						//			const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
						//			for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
						//			{
						//				__m128 w0 = _mm_loadu_ps(pw);
						//				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * strideXF), w0), sum0);
						//				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * strideXF), w0), sum1);
						//			}
						//		}
						//		_mm_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
						//		_mm_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
						//	}
						//	for (; dx < bodyX; ++dx, pd += F)
						//	{
						//		__m128 sum = _bias;
						//		const float* pw = weight;
						//		for (size_t ky = 0; ky < p.kernelY; ++ky)
						//		{
						//			size_t sy = dy * strideY + ky - padY;
						//			const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
						//			for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
						//			{
						//				__m128 w0 = _mm_loadu_ps(pw);
						//				sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), w0), sum);
						//			}
						//		}
						//		_mm_storeu_ps(pd, Activate<type>(sum, _params, 0));
						//	}
						//	for (; dx < p.dstW; ++dx, pd += F)
						//	{
						//		__m128 sum = _bias;
						//		for (size_t ky = 0; ky < p.kernelY; ++ky)
						//		{
						//			size_t sy = dy * strideY + ky - padY;
						//			for (size_t kx = 0; kx < p.kernelX; ++kx)
						//			{
						//				size_t sx = dx * strideX + kx - padX;
						//				if (sx < p.srcW)
						//				{
						//					const float* pw = weight + (ky * p.kernelX + kx) * F;
						//					const float* ps = src + ((sy & srcM) * p.srcW + sx) * F;
						//					sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
						//				}
						//			}
						//		}
						//		_mm_storeu_ps(pd, Activate<type>(sum, _params, 0));
						//	}
						//}
						//else
						{
							for (size_t dx = 0; dx < p.dstW; ++dx, pd += F)
							{
								__m128 sum = _bias;
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
												sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
											}
										}
									}
								}
								_mm_storeu_ps(pd, Activate<type>(sum, _params, 0));
							}
						}
					}
					src += srcS;
					dst += dstS;
					weight += weightS;
				}
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x2(
				const float* src0, const float* src1, const __m128* weight, const __m128& bias, const __m128* params, float* dst)
			{
				__m128 sum0 = bias, sum1 = _mm_setzero_ps();
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
				_mm_storeu_ps(dst, Activate<type>(_mm_add_ps(sum0, sum1), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
				const float* src0, const float* src1, const __m128* weight, const __m128& bias, const __m128* params, float* dst)
			{
				__m128 sum0 = bias, sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps();
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
				sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * F), weight[2]), sum2);
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
				sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * F), weight[5]), sum2);
				_mm_storeu_ps(dst, Activate<type>(_mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
				const float* src0, const float* src1, const float* src2, const __m128* weight, const __m128& bias, const __m128* params, float* dst)
			{
				__m128 sum0 = bias, sum1 = _mm_setzero_ps();
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * F), weight[6]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * F), weight[7]), sum1);
				_mm_storeu_ps(dst, Activate<type>(_mm_add_ps(sum0, sum1), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
				const float* src0, const float* src1, const float* src2, const __m128* weight, const __m128& bias, const __m128* params, float* dst)
			{
				__m128 sum0 = bias, sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps();
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
				sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * F), weight[2]), sum2);
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
				sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * F), weight[5]), sum2);
				sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * F), weight[6]), sum0);
				sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * F), weight[7]), sum1);
				sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 2 * F), weight[8]), sum2);
				_mm_storeu_ps(dst, Activate<type>(_mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params, 0));
			}

			template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst)
			{
				size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
				size_t srcW = p.srcW * F, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F;
				size_t srcM = (bufH[0] - 1), dstM = (bufH[1] - 1), srcS = bufH[0] * srcW, dstS = bufH[1] * dstW;
				size_t xStep = F * p.strideX, xStep0 = (p.strideX - p.padX) * F;
				size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

				__m128 _params[2];
				_params[0] = _mm_set1_ps(params[0]);
				if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
					_params[1] = _mm_set1_ps(params[1]);
				for (size_t c = 0; c < srcC; c += F)
				{
					__m128 _weight[9];
					for (size_t i = 0; i < 9; ++i)
						_weight[i] = _mm_loadu_ps(weight + i * F);
					__m128 _bias = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
					if (type == ::SimdConvolutionActivationPrelu)
						_params[0] = _mm_loadu_ps(params + c);

					size_t dy = yBeg;
					if (yBeg == 0 && padY)
					{
						size_t sy = 0, dx = 0;
						const float* src0 = src + ((sy + 0) & srcM) * srcW;
						const float* src1 = src + ((sy + 1) & srcM) * srcW;
						float* pDst = dst + (dy & dstM) * dstW;
						if (padX)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 4, _bias, _params, pDst), pDst += F, dx++, src0 += xStep0, src1 += xStep0;
						for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep)
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
						float* pDst = dst + (dy & dstM) * dstW;
						if (padX)
							ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 1, _bias, _params, pDst), pDst += F, dx++, src0 += xStep0, src1 += xStep0, src2 += xStep0;
						for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep, src2 += xStep)
							ConvolutionDepthwise3x3Main1x1<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst);
						if (padW)
							ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst);
					}
					if (dy < yEnd)
					{
						size_t sy = dy * strideY - padY, dx = 0;
						const float* src0 = src + ((sy + 0) & srcM) * srcW;
						const float* src1 = src + ((sy + 1) & srcM) * srcW;
						float* pDst = dst + (dy & dstM) * dstW;
						if (padX)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 1, _bias, _params, pDst), pDst += F, dx++, src0 += xStep0, src1 += xStep0;
						for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep)
							ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, _weight + 0, _bias, _params, pDst);
						if (padW)
							ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 0, _bias, _params, pDst);
					}
					src += srcS;
					dst += dstS;
					weight += weightS;
				}
			}

			//---------------------------------------------------------------------

			template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x6(const float* src, size_t srcC, size_t srcS,
				const float* weight, const __m128* bias, const __m128* params, float* dst, size_t dstC, size_t tail)
			{
				__m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
				if (tail > F)
				{
					d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
					d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
					d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
					d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
					d40 = _mm_setzero_ps(), d41 = _mm_setzero_ps();
					d50 = _mm_setzero_ps(), d51 = _mm_setzero_ps();
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm_loadu_ps(weight + 0);
							w1 = _mm_loadu_ps(weight + F);
							s0 = _mm_set1_ps(src[i + 0 * F]);
							d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
							d01 = _mm_add_ps(_mm_mul_ps(s0, w1), d01);
							s0 = _mm_set1_ps(src[i + 1 * F]);
							d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
							d11 = _mm_add_ps(_mm_mul_ps(s0, w1), d11);
							s0 = _mm_set1_ps(src[i + 2 * F]);
							d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
							d21 = _mm_add_ps(_mm_mul_ps(s0, w1), d21);
							s0 = _mm_set1_ps(src[i + 3 * F]);
							d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30);
							d31 = _mm_add_ps(_mm_mul_ps(s0, w1), d31);
							s0 = _mm_set1_ps(src[i + 4 * F]);
							d40 = _mm_add_ps(_mm_mul_ps(s0, w0), d40);
							d41 = _mm_add_ps(_mm_mul_ps(s0, w1), d41);
							s0 = _mm_set1_ps(src[i + 5 * F]);
							d50 = _mm_add_ps(_mm_mul_ps(s0, w0), d50);
							d51 = _mm_add_ps(_mm_mul_ps(s0, w1), d51);
						}
						src += srcS;
					}
					if (tail == DF)
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d01, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d11, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d21, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d31, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d41, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d51, bias, params);
					}
					else
					{
						tail -= F;
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d31, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d41, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d51, bias, params, tail);
					}
				}
				else
				{
					d00 = _mm_setzero_ps();
					d10 = _mm_setzero_ps();
					d20 = _mm_setzero_ps();
					d30 = _mm_setzero_ps();
					d40 = _mm_setzero_ps();
					d50 = _mm_setzero_ps();
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm_loadu_ps(weight + 0);
							s0 = _mm_set1_ps(src[i + 0 * F]);
							d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
							s0 = _mm_set1_ps(src[i + 1 * F]);
							d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
							s0 = _mm_set1_ps(src[i + 2 * F]);
							d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
							s0 = _mm_set1_ps(src[i + 3 * F]);
							d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30);
							s0 = _mm_set1_ps(src[i + 4 * F]);
							d40 = _mm_add_ps(_mm_mul_ps(s0, w0), d40);
							s0 = _mm_set1_ps(src[i + 5 * F]);
							d50 = _mm_add_ps(_mm_mul_ps(s0, w0), d50);
						}
						src += srcS;
					}
					if (tail == F)
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
					}
					else
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, tail);
					}
				}
			}

			template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x4(const float* src, size_t srcC, size_t srcS,
				const float* weight, const __m128* bias, const __m128* params, float* dst, size_t dstC, size_t tail)
			{
				__m128 d00, d01, d10, d11, d20, d21, d30, d31, s0, w0, w1;
				if (tail > F)
				{
					d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
					d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
					d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
					d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm_loadu_ps(weight + 0);
							w1 = _mm_loadu_ps(weight + F);
							s0 = _mm_set1_ps(src[i + 0 * F]);
							d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
							d01 = _mm_add_ps(_mm_mul_ps(s0, w1), d01);
							s0 = _mm_set1_ps(src[i + 1 * F]);
							d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
							d11 = _mm_add_ps(_mm_mul_ps(s0, w1), d11);
							s0 = _mm_set1_ps(src[i + 2 * F]);
							d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
							d21 = _mm_add_ps(_mm_mul_ps(s0, w1), d21);
							s0 = _mm_set1_ps(src[i + 3 * F]);
							d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30);
							d31 = _mm_add_ps(_mm_mul_ps(s0, w1), d31);
						}
						src += srcS;
					}
					if (tail == DF)
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d01, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d11, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d21, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d31, bias, params);
					}
					else
					{
						tail -= F;
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d31, bias, params, tail);
					}
				}
				else
				{
					d00 = _mm_setzero_ps();
					d10 = _mm_setzero_ps();
					d20 = _mm_setzero_ps();
					d30 = _mm_setzero_ps();
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm_loadu_ps(weight + 0);
							s0 = _mm_set1_ps(src[i + 0 * F]);
							d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
							s0 = _mm_set1_ps(src[i + 1 * F]);
							d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
							s0 = _mm_set1_ps(src[i + 2 * F]);
							d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
							s0 = _mm_set1_ps(src[i + 3 * F]);
							d30 = _mm_add_ps(_mm_mul_ps(s0, w0), d30);
						}
						src += srcS;
					}
					if (tail == F)
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
					}
					else
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, tail);
					}
				}
			}

			template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x3(const float* src, size_t srcC, size_t srcS,
				const float* weight, const __m128* bias, const __m128* params, float* dst, size_t dstC, size_t tail)
			{
				__m128 d00, d01, d10, d11, d20, d21, s0, w0, w1;
				if (tail > F)
				{
					d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
					d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
					d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm_loadu_ps(weight + 0);
							w1 = _mm_loadu_ps(weight + F);
							s0 = _mm_set1_ps(src[i + 0 * F]);
							d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
							d01 = _mm_add_ps(_mm_mul_ps(s0, w1), d01);
							s0 = _mm_set1_ps(src[i + 1 * F]);
							d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
							d11 = _mm_add_ps(_mm_mul_ps(s0, w1), d11);
							s0 = _mm_set1_ps(src[i + 2 * F]);
							d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
							d21 = _mm_add_ps(_mm_mul_ps(s0, w1), d21);
						}
						src += srcS;
					}
					if (tail == DF)
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d01, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d11, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d21, bias, params);
					}
					else
					{
						tail -= F;
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tail);
					}
				}
				else
				{
					d00 = _mm_setzero_ps();
					d10 = _mm_setzero_ps();
					d20 = _mm_setzero_ps();
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm_loadu_ps(weight + 0);
							s0 = _mm_set1_ps(src[i + 0 * F]);
							d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
							s0 = _mm_set1_ps(src[i + 1 * F]);
							d10 = _mm_add_ps(_mm_mul_ps(s0, w0), d10);
							s0 = _mm_set1_ps(src[i + 2 * F]);
							d20 = _mm_add_ps(_mm_mul_ps(s0, w0), d20);
						}
						src += srcS;
					}
					if (tail == F)
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
					}
					else
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tail);
						dst += dstC;
						Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tail);
					}
				}
			}

			template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x1(const float* src, size_t srcC, size_t srcS,
				const float* weight, const __m128* bias, const __m128* params, float* dst, size_t dstC, size_t tail)
			{
				__m128 d00, d01, s0, w0, w1;
				if (tail > F)
				{
					d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm_loadu_ps(weight + 0);
							w1 = _mm_loadu_ps(weight + F);
							s0 = _mm_set1_ps(src[i + 0 * F]);
							d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
							d01 = _mm_add_ps(_mm_mul_ps(s0, w1), d01);
						}
						src += srcS;
					}
					if (tail == DF)
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d01, bias, params);
					}
					else
					{
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
						Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tail - F);
					}
				}
				else
				{
					d00 = _mm_setzero_ps();
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm_loadu_ps(weight + 0);
							s0 = _mm_set1_ps(src[i + 0 * F]);
							d00 = _mm_add_ps(_mm_mul_ps(s0, w0), d00);
						}
						src += srcS;
					}
					if (tail == F)
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
					else
						Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tail);
				}
			}

			template<TermType term, SimdConvolutionActivationType type> void OutputConvolution(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst)
			{
				assert(p.group == 1 && p.kernelY == 1 && p.strideY == 1);
				size_t srcH = p.srcH, srcW = p.srcW, dstW = p.dstW, dstC = p.dstC;
				size_t srcM = (bufH[1] - 1), srcS = bufH[1] * srcW * F;
				size_t dstW3 = AlignLoAny(dstW, 3), dstW6 = AlignLoAny(dstW, 6);
				__m128 _params[2], _bias[2];
				_params[0] = _mm_set1_ps(params[0]);
				if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
					_params[1] = _mm_set1_ps(params[1]);

				dst += yBeg * p.dstW * p.dstC;
				size_t dc = 0;
				for (; dc < dstC; dc += DF)
				{
					size_t tail = Simd::Min(DF, dstC - dc);
					_bias[0] = _mm_loadu_ps(bias + dc + 0);
					_bias[1] = _mm_loadu_ps(bias + dc + F);
					if (type == ::SimdConvolutionActivationPrelu)
					{
						_params[0] = _mm_loadu_ps(params + dc + 0);
						_params[1] = _mm_loadu_ps(params + dc + F);
					}
					float* pDst = dst + dc;
					for (size_t y = yBeg; y < yEnd; ++y)
					{
						const float* pSrc = src + (y & srcM) * srcW * F;
						size_t x = 0;
						//for (; x < dstW6; x += 6, pDst += 6 * dstC, pSrc += 6 * F)
						//	OutputConvolution_2x6<term, type>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail);
						//if (dstW - dstW6 == 4)
						//	OutputConvolution_2x4<term, type>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail), pDst += 4 * dstC;
						//else
						{
							//for (; x < dstW3; x += 3, pDst += 3 * dstC, pSrc += 3 * F)
							//	OutputConvolution_2x3<term, type>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail);
							for (; x < dstW; ++x, pDst += dstC, pSrc += F)
								OutputConvolution_2x1<term, type>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail);
						}
					}
					weight += srcC * DF;
				}
			}

			//---------------------------------------------------------------------

			template <SimdConvolutionActivationType type> void Set(const MergConvParam32f& p, size_t index, SynetMergedConvolution32fCdc::ConvolutionPtr * convolution)
			{
				switch (index)
				{
				case 0:
					//if (p.conv[1].kernelY == 3)
					//	convolution[index + 0] = DepthwiseConvolution3x3<type>;
					//else
						convolution[index + 0] = DepthwiseConvolution<type>;
					break;
				case 1:
					convolution[index + 0] = OutputConvolution<TermSingle, type>;
					convolution[index + 1] = OutputConvolution<TermFirst, SimdConvolutionActivationIdentity>;
					convolution[index + 2] = OutputConvolution<TermIterim, SimdConvolutionActivationIdentity>;
					convolution[index + 3] = OutputConvolution<TermLast, type>;
					break;
				default:
					assert(0);
				}
			}
		}

		//---------------------------------------------------------------------

		SynetMergedConvolution32fDc::SynetMergedConvolution32fDc(const MergConvParam32f& p)
			: Base::SynetMergedConvolution32fDc(p)
		{
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3(), Sse::F);
			for (size_t i = 0; i < _param.count; ++i)
				Set(p, i, _convolution);
		}

		void SynetMergedConvolution32fDc::Set(const MergConvParam32f& p, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c)
		{
			switch (p.conv[i].activation)
			{
			case SimdConvolutionActivationIdentity: Dc::Set<SimdConvolutionActivationRestrictRange>(p, i, c); break;
			case SimdConvolutionActivationRelu: Dc::Set<SimdConvolutionActivationRestrictRange>(p, i, c); break;
			case SimdConvolutionActivationLeakyRelu: Dc::Set<SimdConvolutionActivationPrelu>(p, i, c); break;
			case SimdConvolutionActivationRestrictRange: Dc::Set<SimdConvolutionActivationRestrictRange>(p, i, c); break;
			case SimdConvolutionActivationPrelu: Dc::Set<SimdConvolutionActivationPrelu>(p, i, c); break;
			case SimdConvolutionActivationElu: Dc::Set<SimdConvolutionActivationElu>(p, i, c); break;
			case SimdConvolutionActivationHswish: Dc::Set<SimdConvolutionActivationHswish>(p, i, c); break;
			default: assert(0);
			}
		}
	}
#endif//SIMD_SSE2_ENABLE
}
