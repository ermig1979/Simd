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
#include "Simd/SimdUpdate.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Avx2
	{
		template<TermType term, SimdConvolutionActivationType type, int M> SIMD_NOINLINE void OutputConvolution_2xM(const float* src, size_t srcC, size_t srcS,
			const float* weight, const __m256* bias, const __m256* params, float* dst, size_t dstC, size_t tail, int first)
		{
			__m256 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
			if (tail > F)
			{
				if (first)
				{
					if (M > 0) d00 = _mm256_setzero_ps(), d01 = _mm256_setzero_ps();
					if (M > 1) d10 = _mm256_setzero_ps(), d11 = _mm256_setzero_ps();
					if (M > 2) d20 = _mm256_setzero_ps(), d21 = _mm256_setzero_ps();
					if (M > 3) d30 = _mm256_setzero_ps(), d31 = _mm256_setzero_ps();
					if (M > 4) d40 = _mm256_setzero_ps(), d41 = _mm256_setzero_ps();
					if (M > 5) d50 = _mm256_setzero_ps(), d51 = _mm256_setzero_ps();
				}
				else
				{
					if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dstC + 0), d01 = _mm256_loadu_ps(dst + 0 * dstC + F);
					if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dstC + 0), d11 = _mm256_loadu_ps(dst + 1 * dstC + F);
					if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dstC + 0), d21 = _mm256_loadu_ps(dst + 2 * dstC + F);
					if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dstC + 0), d31 = _mm256_loadu_ps(dst + 3 * dstC + F);
					if (M > 4) d40 = _mm256_loadu_ps(dst + 4 * dstC + 0), d41 = _mm256_loadu_ps(dst + 4 * dstC + F);
					if (M > 5) d50 = _mm256_loadu_ps(dst + 5 * dstC + 0), d51 = _mm256_loadu_ps(dst + 5 * dstC + F);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = _mm256_loadu_ps(weight + 0);
						w1 = _mm256_loadu_ps(weight + F);
						if (M > 0) s0 = _mm256_set1_ps(src[i + 0 * F]), d00 = _mm256_fmadd_ps(s0, w0, d00), d01 = _mm256_fmadd_ps(s0, w1, d01);
						if (M > 1) s0 = _mm256_set1_ps(src[i + 1 * F]), d10 = _mm256_fmadd_ps(s0, w0, d10), d11 = _mm256_fmadd_ps(s0, w1, d11);
						if (M > 2) s0 = _mm256_set1_ps(src[i + 2 * F]), d20 = _mm256_fmadd_ps(s0, w0, d20), d21 = _mm256_fmadd_ps(s0, w1, d21);
						if (M > 3) s0 = _mm256_set1_ps(src[i + 3 * F]), d30 = _mm256_fmadd_ps(s0, w0, d30), d31 = _mm256_fmadd_ps(s0, w1, d31);
						if (M > 4) s0 = _mm256_set1_ps(src[i + 4 * F]), d40 = _mm256_fmadd_ps(s0, w0, d40), d41 = _mm256_fmadd_ps(s0, w1, d41);
						if (M > 5) s0 = _mm256_set1_ps(src[i + 5 * F]), d50 = _mm256_fmadd_ps(s0, w0, d50), d51 = _mm256_fmadd_ps(s0, w1, d51);
					}
					src += srcS;
				}
				if (tail == DF)
				{
					if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params), Term<term>::template Save<type, 1>(dst + F, d01, bias, params), dst += dstC;
					if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params), Term<term>::template Save<type, 1>(dst + F, d11, bias, params), dst += dstC;
					if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params), Term<term>::template Save<type, 1>(dst + F, d21, bias, params), dst += dstC;
					if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params), Term<term>::template Save<type, 1>(dst + F, d31, bias, params), dst += dstC;
					if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params), Term<term>::template Save<type, 1>(dst + F, d41, bias, params), dst += dstC;
					if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params), Term<term>::template Save<type, 1>(dst + F, d51, bias, params), dst += dstC;
				}
				else
				{
					tail -= F;
					if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params), Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tail), dst += dstC;
					if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params), Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tail), dst += dstC;
					if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params), Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tail), dst += dstC;
					if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params), Term<term>::template Save<type, 1>(dst + F, d31, bias, params, tail), dst += dstC;
					if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params), Term<term>::template Save<type, 1>(dst + F, d41, bias, params, tail), dst += dstC;
					if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params), Term<term>::template Save<type, 1>(dst + F, d51, bias, params, tail), dst += dstC;
				}
			}
			else
			{
				if (first)
				{
					if (M > 0) d00 = _mm256_setzero_ps();
					if (M > 1) d10 = _mm256_setzero_ps();
					if (M > 2) d20 = _mm256_setzero_ps();
					if (M > 3) d30 = _mm256_setzero_ps();
					if (M > 4) d40 = _mm256_setzero_ps();
					if (M > 5) d50 = _mm256_setzero_ps();
				}
				else
				{
					if (M > 0) d00 = _mm256_loadu_ps(dst + 0 * dstC + 0);
					if (M > 1) d10 = _mm256_loadu_ps(dst + 1 * dstC + 0);
					if (M > 2) d20 = _mm256_loadu_ps(dst + 2 * dstC + 0);
					if (M > 3) d30 = _mm256_loadu_ps(dst + 3 * dstC + 0);
					if (M > 4) d40 = _mm256_loadu_ps(dst + 4 * dstC + 0);
					if (M > 5) d50 = _mm256_loadu_ps(dst + 5 * dstC + 0);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = _mm256_loadu_ps(weight + 0);
						if (M > 0) s0 = _mm256_set1_ps(src[i + 0 * F]), d00 = _mm256_fmadd_ps(s0, w0, d00);
						if (M > 1) s0 = _mm256_set1_ps(src[i + 1 * F]), d10 = _mm256_fmadd_ps(s0, w0, d10);
						if (M > 2) s0 = _mm256_set1_ps(src[i + 2 * F]), d20 = _mm256_fmadd_ps(s0, w0, d20);
						if (M > 3) s0 = _mm256_set1_ps(src[i + 3 * F]), d30 = _mm256_fmadd_ps(s0, w0, d30);
						if (M > 4) s0 = _mm256_set1_ps(src[i + 4 * F]), d40 = _mm256_fmadd_ps(s0, w0, d40);
						if (M > 5) s0 = _mm256_set1_ps(src[i + 5 * F]), d50 = _mm256_fmadd_ps(s0, w0, d50);
					}
					src += srcS;
				}
				if (tail == F)
				{
					if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params), dst += dstC;
					if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params), dst += dstC;
					if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params), dst += dstC;
					if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params), dst += dstC;
					if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params), dst += dstC;
					if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params), dst += dstC;
				}
				else
				{
					if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tail), dst += dstC;
					if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tail), dst += dstC;
					if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tail), dst += dstC;
					if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, tail), dst += dstC;
					if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, tail), dst += dstC;
					if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, tail), dst += dstC;
				}
			}
		}

		typedef void(*OutputConvolution_2xM_Ptr)(const float* src, size_t srcC, size_t srcS, const float* weight, const __m256* bias, const __m256* params, float* dst, size_t dstC, size_t tail, int first);

		template<TermType term, SimdConvolutionActivationType type> OutputConvolution_2xM_Ptr GetOutputConvolution_2xM(size_t M)
		{
			switch (M)
			{
			case 0: return NULL;
			case 1: return OutputConvolution_2xM<term, type, 1>;
			case 2: return OutputConvolution_2xM<term, type, 2>;
			case 3: return OutputConvolution_2xM<term, type, 3>;
			case 4: return OutputConvolution_2xM<term, type, 4>;
			case 5: return OutputConvolution_2xM<term, type, 5>;
			case 6: return OutputConvolution_2xM<term, type, 6>;
			}
			assert(0);
			return NULL;
		}

		template<TermType term, SimdConvolutionActivationType type> SIMD_NOINLINE void OutputConvolution(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			assert(p.group == 1 && p.kernelY == 1 && p.strideY == 1);
			size_t srcH = p.srcH, srcW = p.srcW, dstW = p.dstW, dstC = p.dstC;
			size_t srcM = (bufH[1] - 1), srcS = bufH[1] * srcW * F;
			size_t yInt = Simd::Max(yBeg, yEnd & (~srcM)), nBeg = yBeg * srcW, nInt = yInt * srcW, nEnd = yEnd * srcW;
			size_t nInt6 = AlignLoAny(nInt - nBeg, 6) + nBeg, nEnd6 = AlignLoAny(nEnd - nInt, 6) + nInt, nIntTail = nInt - nInt6, nEndTail = nEnd - nEnd6;
			OutputConvolution_2xM_Ptr bodyInt = GetOutputConvolution_2xM<term, type>(6);
			OutputConvolution_2xM_Ptr tailInt = GetOutputConvolution_2xM<term, type>(nIntTail);
			OutputConvolution_2xM_Ptr bodyEnd = GetOutputConvolution_2xM<term, type>(6);
			OutputConvolution_2xM_Ptr tailEnd = GetOutputConvolution_2xM<term, type>(nEndTail);

			__m256 _params[2], _bias[2];
			_params[0] = _mm256_set1_ps(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				_params[1] = _mm256_set1_ps(params[1]);

			dst += yBeg * p.dstW * p.dstC;
			size_t dc = 0;
			for (; dc < dstC; dc += DF)
			{
				size_t tail = Simd::Min(DF, dstC - dc);
				_bias[0] = _mm256_loadu_ps(bias + dc + 0);
				_bias[1] = _mm256_loadu_ps(bias + dc + F);
				if (type == ::SimdConvolutionActivationPrelu)
				{
					_params[0] = _mm256_loadu_ps(params + dc + 0);
					_params[1] = _mm256_loadu_ps(params + dc + F);
				}
				float* pDst = dst + dc;
				const float* src0 = src + (yBeg & srcM) * srcW * F;
				const float* src1 = src + (yInt & srcM) * srcW * F;
				size_t dn = nBeg;
				for (; dn < nInt6; dn += 6, pDst += 6 * dstC, src0 += 6 * F)
					bodyInt(src0, srcC, srcS, weight, _bias, _params, pDst, dstC, tail, first);
				if (nIntTail)
					tailInt(src0, srcC, srcS, weight, _bias, _params, pDst, dstC, tail, first), dn += nIntTail, pDst += nIntTail * dstC, src0 += nIntTail * F;
				for (; dn < nEnd6; dn += 6, pDst += 6 * dstC, src1 += 6 * F)
					bodyEnd(src1, srcC, srcS, weight, _bias, _params, pDst, dstC, tail, first);
				if (nEndTail)
					tailEnd(src1, srcC, srcS, weight, _bias, _params, pDst, dstC, tail, first), dn += nEndTail, pDst += nEndTail * dstC, src1 += nEndTail * F;
				weight += srcC * DF;
			}
		}

		//-------------------------------------------------------------------------------------------------------

		template <SimdConvolutionActivationType type> void SetOutput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			convolution[0] = OutputConvolution<TermLast, type>;
			convolution[1] = OutputConvolution<TermInterim, SimdConvolutionActivationIdentity>;
		}

		void SetOutput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			switch (p.activation)
			{
			case SimdConvolutionActivationIdentity: SetOutput<SimdConvolutionActivationRestrictRange>(p, convolution); break;
			case SimdConvolutionActivationRelu: SetOutput<SimdConvolutionActivationRestrictRange>(p, convolution); break;
			case SimdConvolutionActivationLeakyRelu: SetOutput<SimdConvolutionActivationPrelu>(p, convolution); break;
			case SimdConvolutionActivationRestrictRange: SetOutput<SimdConvolutionActivationRestrictRange>(p, convolution); break;
			case SimdConvolutionActivationPrelu: SetOutput<SimdConvolutionActivationPrelu>(p, convolution); break;
			case SimdConvolutionActivationElu: SetOutput<SimdConvolutionActivationElu>(p, convolution); break;
			case SimdConvolutionActivationHswish: SetOutput<SimdConvolutionActivationHswish>(p, convolution); break;
			case SimdConvolutionActivationMish: SetOutput<SimdConvolutionActivationMish>(p, convolution); break;
			case SimdConvolutionActivationHardSigmoid: SetOutput<SimdConvolutionActivationHardSigmoid>(p, convolution); break;
			case SimdConvolutionActivationSwish: SetOutput<SimdConvolutionActivationSwish>(p, convolution); break;
			case SimdConvolutionActivationGelu: SetOutput<SimdConvolutionActivationGelu>(p, convolution); break;
			default: assert(0);
			}
		}
	}
#endif
}
