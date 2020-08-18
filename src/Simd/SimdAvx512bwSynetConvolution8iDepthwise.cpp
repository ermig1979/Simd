/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE)
    namespace Avx512bw
    {
		using AlgParam = SynetConvolution8iNhwcDepthwise::AlgParam;
		using ConvolutionPtr = SynetConvolution8iNhwcDepthwise::ConvolutionPtr;
		using Term8iType = Base::SynetConvolution8iNhwcDirect::Term8iType;

		SIMD_INLINE __m512i LoadAs32i(const uint8_t* src, __mmask16 tail = -1)
		{
			return _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, src));
		}

		SIMD_INLINE __m512i LoadAs32i(const int8_t* src, __mmask16 tail = -1)
		{
			return _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(tail, src));
		}

		template <Term8iType term, SimdConvolutionActivationType activation, bool nofma> void ConvolutionNhwcDepthwiseDefault(
			const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight, const float* norm,
			const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
		{
			__m512i zero = _mm512_set1_epi32(a.zero);
			__m128i upper = _mm_set1_epi32(a.upper);
			__m512i d00, d01, d02, d03, w0, w1, w2, w3, s0;
			size_t size = p.group;
			size_t sizeF = AlignLo(size, F);
			size_t sizeF2 = AlignLo(size, F * 2);
			size_t sizeF4 = AlignLo(size, F * 4);
			__mmask16 tail = TailMask16(size - sizeF);
			for (size_t dy = 0; dy < p.dstH; ++dy)
			{
				for (size_t dx = 0; dx < p.dstW; ++dx)
				{
					size_t i = 0;
					for (; i < sizeF4; i += F * 4)
					{
						d00 = _mm512_setzero_si512();
						d01 = _mm512_setzero_si512();
						d02 = _mm512_setzero_si512();
						d03 = _mm512_setzero_si512();
						for (size_t ky = 0; ky < p.kernelY; ++ky)
						{
							size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
							for (size_t kx = 0; kx < p.kernelX; ++kx)
							{
								size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
								size_t ow = (ky * p.kernelX + kx) * size + i;
								w0 = LoadAs32i(weight + ow + 0 * F);
								w1 = LoadAs32i(weight + ow + 1 * F);
								w2 = LoadAs32i(weight + ow + 2 * F);
								w3 = LoadAs32i(weight + ow + 3 * F);
								if (sy < p.srcH && sx < p.srcW)
								{
									size_t os = (sy * p.srcW + sx) * size + i;
									Madd4<true>(d00, LoadAs32i(src + os + 0 * F), w0);
									Madd4<true>(d01, LoadAs32i(src + os + 1 * F), w1);
									Madd4<true>(d02, LoadAs32i(src + os + 2 * F), w2);
									Madd4<true>(d03, LoadAs32i(src + os + 3 * F), w3);
								}
								else
								{
									Madd4<true>(d00, zero, w0);
									Madd4<true>(d01, zero, w1);
									Madd4<true>(d02, zero, w2);
									Madd4<true>(d03, zero, w3);
								}
							}
						}
						Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, i + F * 0);
						Save<term, activation, nofma>(dst, d01, norm, bias, params, scale, shift, upper, i + F * 1);
						Save<term, activation, nofma>(dst, d02, norm, bias, params, scale, shift, upper, i + F * 2);
						Save<term, activation, nofma>(dst, d03, norm, bias, params, scale, shift, upper, i + F * 3);
					}
					for (; i < sizeF2; i += F * 2)
					{
						d00 = _mm512_setzero_si512();
						d01 = _mm512_setzero_si512();
						for (size_t ky = 0; ky < p.kernelY; ++ky)
						{
							size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
							for (size_t kx = 0; kx < p.kernelX; ++kx)
							{
								size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
								size_t ow = (ky * p.kernelX + kx) * size + i;
								w0 = LoadAs32i(weight + ow + 0 * F);
								w1 = LoadAs32i(weight + ow + 1 * F);
								if (sy < p.srcH && sx < p.srcW)
								{
									size_t os = (sy * p.srcW + sx) * size + i;
									Madd4<true>(d00, LoadAs32i(src + os + 0 * F), w0);
									Madd4<true>(d01, LoadAs32i(src + os + 1 * F), w1);
								}
								else
								{
									Madd4<true>(d00, zero, w0);
									Madd4<true>(d01, zero, w1);
								}
							}
						}
						Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, i + F * 0);
						Save<term, activation, nofma>(dst, d01, norm, bias, params, scale, shift, upper, i + F * 1);
					}
					for (; i < sizeF; i += F)
					{
						d00 = _mm512_setzero_si512();
						for (size_t ky = 0; ky < p.kernelY; ++ky)
						{
							size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
							for (size_t kx = 0; kx < p.kernelX; ++kx)
							{
								size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
								w0 = LoadAs32i(weight + (ky * p.kernelX + kx) * size + i);
								if (sy < p.srcH && sx < p.srcW)
									s0 = LoadAs32i(src + (sy * p.srcW + sx) * size + i);
								else
									s0 = zero;
								Madd4<true>(d00, s0, w0);
							}
						}
						Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, i);
					}
					for (; i < size; i += F)
					{
						d00 = _mm512_setzero_si512();
						for (size_t ky = 0; ky < p.kernelY; ++ky)
						{
							size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
							for (size_t kx = 0; kx < p.kernelX; ++kx)
							{
								size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
								w0 = LoadAs32i(weight + (ky * p.kernelX + kx) * size + i, tail);
								if (sy < p.srcH && sx < p.srcW)
									s0 = LoadAs32i(src + (sy * p.srcW + sx) * size + i, tail);
								else
									s0 = zero;
								Madd4<true>(d00, s0, w0);
							}
						}
						Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, i, tail);
					}
					dst += p.dstC * a.size;
				}
			}
		}


        //---------------------------------------------------------------------

		template <Term8iType term, SimdConvolutionActivationType activation, bool nofma> void Set(const ConvParam8i& p, ConvolutionPtr& d)
		{
			//if (p.IsKernel(3) && p.IsDilation(1))
			//	d = ConvolutionNhwcDepthwise3x3<term, activation, nofma>;
			//else
				d = ConvolutionNhwcDepthwiseDefault<term, activation, nofma>;
		}

		template<Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, ConvolutionPtr& d)
		{
			if (Base::FmaAvoid(p.compatibility))
				Set<term, activation, true>(p, d);
			else
				Set<term, activation, false>(p, d);
		}

		template<SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, ConvolutionPtr& d)
		{
			if (p.dstT == SimdTensorData8u)
				Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u, activation>(p, d);
			else
				Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f, activation>(p, d);
		}

		static void Set(const ConvParam8i& p, ConvolutionPtr& d)
		{
			switch (p.activation)
			{
			case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, d); break;
			case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, d); break;
			case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, d); break;
			case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, d); break;
			case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, d); break;
			case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, d); break;
			case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, d); break;
			default: assert(0);
			}
		}

        SynetConvolution8iNhwcDepthwise::SynetConvolution8iNhwcDepthwise(const ConvParam8i& p)
            : Avx2::SynetConvolution8iNhwcDepthwise(p)
        {
            Set(p, _convolution);
            _convertSrc = Avx512bw::SynetConvert32fTo8u;
        }
    }
#endif
}
