/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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

		template <SimdConvolutionActivationType type> void SetDepthwise(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			convolution[0] = DepthwiseConvolution<type>;
		}

		void SetDepthwise(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			if (SetDepthwise7x7(p, convolution))
				return;
			if (SetDepthwise3x3(p, convolution))
				return;
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
