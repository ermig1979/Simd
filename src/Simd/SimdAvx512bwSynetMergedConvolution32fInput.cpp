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
		template<SimdConvolutionActivationType type, int M> SIMD_INLINE void InputConvolution1x1_2xM(const float* src0, size_t srcC,
			const float* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
		{
			__m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, s0, s1, w0, w1;
			if (M > 0x0) d00 = bias[0], d01 = bias[1];
			if (M > 0x1) d10 = bias[0], d11 = bias[1];
			if (M > 0x2) d20 = bias[0], d21 = bias[1];
			if (M > 0x3) d30 = bias[0], d31 = bias[1];
			if (M > 0x4) d40 = bias[0], d41 = bias[1];
			if (M > 0x5) d50 = bias[0], d51 = bias[1];
			if (M > 0x6) d60 = bias[0], d61 = bias[1];
			if (M > 0x7) d70 = bias[0], d71 = bias[1];
			if (M > 0x8) d80 = bias[0], d81 = bias[1];
			if (M > 0x9) d90 = bias[0], d91 = bias[1];
			if (M > 0xa) da0 = bias[0], da1 = bias[1];
			if (M > 0xb) db0 = bias[0], db1 = bias[1];			
			const float* src1 = src0 + 1 * srcC;
			const float* src2 = src0 + 2 * srcC;
			const float* src3 = src0 + 3 * srcC;
			const float* src4 = src0 + 4 * srcC;
			const float* src5 = src0 + 5 * srcC;
			for (size_t offs0 = 0, offs6 = offs0 + 6 * srcC, end = srcC; offs0 < end; offs0 += 1, offs6 += 1)
			{
				w0 = _mm512_loadu_ps(weight + 0);
				w1 = _mm512_loadu_ps(weight + F);
				if (M > 0x0) s0 = _mm512_set1_ps(src0[offs0]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
				if (M > 0x1) s1 = _mm512_set1_ps(src1[offs0]), d10 = _mm512_fmadd_ps(s1, w0, d10), d11 = _mm512_fmadd_ps(s1, w1, d11);
				if (M > 0x2) s0 = _mm512_set1_ps(src2[offs0]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
				if (M > 0x3) s1 = _mm512_set1_ps(src3[offs0]), d30 = _mm512_fmadd_ps(s1, w0, d30), d31 = _mm512_fmadd_ps(s1, w1, d31);
				if (M > 0x4) s0 = _mm512_set1_ps(src4[offs0]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
				if (M > 0x5) s1 = _mm512_set1_ps(src5[offs0]), d50 = _mm512_fmadd_ps(s1, w0, d50), d51 = _mm512_fmadd_ps(s1, w1, d51);
				if (M > 0x6) s0 = _mm512_set1_ps(src0[offs6]), d60 = _mm512_fmadd_ps(s0, w0, d60), d61 = _mm512_fmadd_ps(s0, w1, d61);
				if (M > 0x7) s1 = _mm512_set1_ps(src1[offs6]), d70 = _mm512_fmadd_ps(s1, w0, d70), d71 = _mm512_fmadd_ps(s1, w1, d71);
				if (M > 0x8) s0 = _mm512_set1_ps(src2[offs6]), d80 = _mm512_fmadd_ps(s0, w0, d80), d81 = _mm512_fmadd_ps(s0, w1, d81);
				if (M > 0x9) s1 = _mm512_set1_ps(src3[offs6]), d90 = _mm512_fmadd_ps(s1, w0, d90), d91 = _mm512_fmadd_ps(s1, w1, d91);
				if (M > 0xa) s0 = _mm512_set1_ps(src4[offs6]), da0 = _mm512_fmadd_ps(s0, w0, da0), da1 = _mm512_fmadd_ps(s0, w1, da1);
				if (M > 0xb) s1 = _mm512_set1_ps(src5[offs6]), db0 = _mm512_fmadd_ps(s1, w0, db0), db1 = _mm512_fmadd_ps(s1, w1, db1);
				weight += DF;
			}
			if (M > 0x0) _mm512_storeu_ps(dst0 + 0x0 * F, Activate<type>(d00, params, 0)), _mm512_storeu_ps(dst1 + 0x0 * F, Activate<type>(d01, params, 1));
			if (M > 0x1) _mm512_storeu_ps(dst0 + 0x1 * F, Activate<type>(d10, params, 0)), _mm512_storeu_ps(dst1 + 0x1 * F, Activate<type>(d11, params, 1));
			if (M > 0x2) _mm512_storeu_ps(dst0 + 0x2 * F, Activate<type>(d20, params, 0)), _mm512_storeu_ps(dst1 + 0x2 * F, Activate<type>(d21, params, 1));
			if (M > 0x3) _mm512_storeu_ps(dst0 + 0x3 * F, Activate<type>(d30, params, 0)), _mm512_storeu_ps(dst1 + 0x3 * F, Activate<type>(d31, params, 1));
			if (M > 0x4) _mm512_storeu_ps(dst0 + 0x4 * F, Activate<type>(d40, params, 0)), _mm512_storeu_ps(dst1 + 0x4 * F, Activate<type>(d41, params, 1));
			if (M > 0x5) _mm512_storeu_ps(dst0 + 0x5 * F, Activate<type>(d50, params, 0)), _mm512_storeu_ps(dst1 + 0x5 * F, Activate<type>(d51, params, 1));
			if (M > 0x6) _mm512_storeu_ps(dst0 + 0x6 * F, Activate<type>(d60, params, 0)), _mm512_storeu_ps(dst1 + 0x6 * F, Activate<type>(d61, params, 1));
			if (M > 0x7) _mm512_storeu_ps(dst0 + 0x7 * F, Activate<type>(d70, params, 0)), _mm512_storeu_ps(dst1 + 0x7 * F, Activate<type>(d71, params, 1));
			if (M > 0x8) _mm512_storeu_ps(dst0 + 0x8 * F, Activate<type>(d80, params, 0)), _mm512_storeu_ps(dst1 + 0x8 * F, Activate<type>(d81, params, 1));
			if (M > 0x9) _mm512_storeu_ps(dst0 + 0x9 * F, Activate<type>(d90, params, 0)), _mm512_storeu_ps(dst1 + 0x9 * F, Activate<type>(d91, params, 1));
			if (M > 0xa) _mm512_storeu_ps(dst0 + 0xa * F, Activate<type>(da0, params, 0)), _mm512_storeu_ps(dst1 + 0xa * F, Activate<type>(da1, params, 1));
			if (M > 0xb) _mm512_storeu_ps(dst0 + 0xb * F, Activate<type>(db0, params, 0)), _mm512_storeu_ps(dst1 + 0xb * F, Activate<type>(db1, params, 1));
		}

		typedef void(*InputConvolution1x1_2xM_Ptr)(const float* src0, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1);

		template<SimdConvolutionActivationType type> InputConvolution1x1_2xM_Ptr GetInputConvolution1x1_2xM(size_t M)
		{
			switch (M)
			{
			case 0x1: return InputConvolution1x1_2xM<type, 0x1>;
			case 0x2: return InputConvolution1x1_2xM<type, 0x2>;
			case 0x3: return InputConvolution1x1_2xM<type, 0x3>;
			case 0x4: return InputConvolution1x1_2xM<type, 0x4>;
			case 0x5: return InputConvolution1x1_2xM<type, 0x5>;
			case 0x6: return InputConvolution1x1_2xM<type, 0x6>;
			case 0x7: return InputConvolution1x1_2xM<type, 0x7>;
			case 0x8: return InputConvolution1x1_2xM<type, 0x8>;
			case 0x9: return InputConvolution1x1_2xM<type, 0x9>;
			case 0xa: return InputConvolution1x1_2xM<type, 0xa>;
			case 0xb: return InputConvolution1x1_2xM<type, 0xb>;
			case 0xc: return InputConvolution1x1_2xM<type, 0xc>;
			}
			return NULL;
		}

		template<SimdConvolutionActivationType type, int M> SIMD_INLINE void InputConvolution1x1_1xM(const float* src0, size_t srcC,
			const float* weight, const __m512* bias, const __m512* params, float* dst0)
		{
			__m512 d00, d10, d20, d30, d40, d50, d60, d70, d80, d90, da0, db0, s0, s1, w0;
			if (M > 0x0) d00 = bias[0];
			if (M > 0x1) d10 = bias[0];
			if (M > 0x2) d20 = bias[0];
			if (M > 0x3) d30 = bias[0];
			if (M > 0x4) d40 = bias[0];
			if (M > 0x5) d50 = bias[0];
			if (M > 0x6) d60 = bias[0];
			if (M > 0x7) d70 = bias[0];
			if (M > 0x8) d80 = bias[0];
			if (M > 0x9) d90 = bias[0];
			if (M > 0xa) da0 = bias[0];
			if (M > 0xb) db0 = bias[0];
			const float* src1 = src0 + 1 * srcC;
			const float* src2 = src0 + 2 * srcC;
			const float* src3 = src0 + 3 * srcC;
			const float* src4 = src0 + 4 * srcC;
			const float* src5 = src0 + 5 * srcC;
			for (size_t offs0 = 0, offs6 = offs0 + 6 * srcC, end = srcC; offs0 < end; offs0 += 1, offs6 += 1)
			{
				w0 = _mm512_loadu_ps(weight + 0);
				if (M > 0x0) s0 = _mm512_set1_ps(src0[offs0]), d00 = _mm512_fmadd_ps(s0, w0, d00);
				if (M > 0x1) s1 = _mm512_set1_ps(src1[offs0]), d10 = _mm512_fmadd_ps(s1, w0, d10);
				if (M > 0x2) s0 = _mm512_set1_ps(src2[offs0]), d20 = _mm512_fmadd_ps(s0, w0, d20);
				if (M > 0x3) s1 = _mm512_set1_ps(src3[offs0]), d30 = _mm512_fmadd_ps(s1, w0, d30);
				if (M > 0x4) s0 = _mm512_set1_ps(src4[offs0]), d40 = _mm512_fmadd_ps(s0, w0, d40);
				if (M > 0x5) s1 = _mm512_set1_ps(src5[offs0]), d50 = _mm512_fmadd_ps(s1, w0, d50);
				if (M > 0x6) s0 = _mm512_set1_ps(src0[offs6]), d60 = _mm512_fmadd_ps(s0, w0, d60);
				if (M > 0x7) s1 = _mm512_set1_ps(src1[offs6]), d70 = _mm512_fmadd_ps(s1, w0, d70);
				if (M > 0x8) s0 = _mm512_set1_ps(src2[offs6]), d80 = _mm512_fmadd_ps(s0, w0, d80);
				if (M > 0x9) s1 = _mm512_set1_ps(src3[offs6]), d90 = _mm512_fmadd_ps(s1, w0, d90);
				if (M > 0xa) s0 = _mm512_set1_ps(src4[offs6]), da0 = _mm512_fmadd_ps(s0, w0, da0);
				if (M > 0xb) s1 = _mm512_set1_ps(src5[offs6]), db0 = _mm512_fmadd_ps(s1, w0, db0);
				weight += DF;
			}
			if (M > 0x0) _mm512_storeu_ps(dst0 + 0x0 * F, Activate<type>(d00, params, 0));
			if (M > 0x1) _mm512_storeu_ps(dst0 + 0x1 * F, Activate<type>(d10, params, 0));
			if (M > 0x2) _mm512_storeu_ps(dst0 + 0x2 * F, Activate<type>(d20, params, 0));
			if (M > 0x3) _mm512_storeu_ps(dst0 + 0x3 * F, Activate<type>(d30, params, 0));
			if (M > 0x4) _mm512_storeu_ps(dst0 + 0x4 * F, Activate<type>(d40, params, 0));
			if (M > 0x5) _mm512_storeu_ps(dst0 + 0x5 * F, Activate<type>(d50, params, 0));
			if (M > 0x6) _mm512_storeu_ps(dst0 + 0x6 * F, Activate<type>(d60, params, 0));
			if (M > 0x7) _mm512_storeu_ps(dst0 + 0x7 * F, Activate<type>(d70, params, 0));
			if (M > 0x8) _mm512_storeu_ps(dst0 + 0x8 * F, Activate<type>(d80, params, 0));
			if (M > 0x9) _mm512_storeu_ps(dst0 + 0x9 * F, Activate<type>(d90, params, 0));
			if (M > 0xa) _mm512_storeu_ps(dst0 + 0xa * F, Activate<type>(da0, params, 0));
			if (M > 0xb) _mm512_storeu_ps(dst0 + 0xb * F, Activate<type>(db0, params, 0));
		}

		typedef void(*InputConvolution1x1_1xM_Ptr)(const float* src0, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst0);

		template<SimdConvolutionActivationType type> InputConvolution1x1_1xM_Ptr GetInputConvolution1x1_1xM(size_t M)
		{
			switch (M)
			{
			case 0x1: return InputConvolution1x1_1xM<type, 0x1>;
			case 0x2: return InputConvolution1x1_1xM<type, 0x2>;
			case 0x3: return InputConvolution1x1_1xM<type, 0x3>;
			case 0x4: return InputConvolution1x1_1xM<type, 0x4>;
			case 0x5: return InputConvolution1x1_1xM<type, 0x5>;
			case 0x6: return InputConvolution1x1_1xM<type, 0x6>;
			case 0x7: return InputConvolution1x1_1xM<type, 0x7>;
			case 0x8: return InputConvolution1x1_1xM<type, 0x8>;
			case 0x9: return InputConvolution1x1_1xM<type, 0x9>;
			case 0xa: return InputConvolution1x1_1xM<type, 0xa>;
			case 0xb: return InputConvolution1x1_1xM<type, 0xb>;
			case 0xc: return InputConvolution1x1_1xM<type, 0xc>;
			}
			return NULL;
		}

		template<SimdConvolutionActivationType type> void InputConvolution1x1(const float* src, const SimdConvolutionParameters& p,
			size_t dstC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
			size_t dstM = (bufH[0] - 1), dstS = bufH[0] * dstW * F;
			size_t dstCDF = AlignLo(dstC, DF);
			__m512 _params[2], _bias[2];
			_params[0] = _mm512_set1_ps(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				_params[1] = _mm512_set1_ps(params[1]);
			size_t yInt = Simd::Max(yBeg, yEnd & (~dstM)), nBeg = yBeg * dstW, nInt = yInt * dstW, nEnd = yEnd * dstW;
			size_t nInt12 = AlignLoAny(nInt - nBeg, 12) + nBeg, nEnd12 = AlignLoAny(nEnd - nInt, 12) + nInt, nIntTail = nInt - nInt12, nEndTail = nEnd - nEnd12;
			InputConvolution1x1_2xM_Ptr tailInt_2 = GetInputConvolution1x1_2xM<type>(nIntTail);
			InputConvolution1x1_2xM_Ptr tailEnd_2 = GetInputConvolution1x1_2xM<type>(nEndTail);

			size_t dc = 0;
			for (; dc < dstC; dc += DF)
			{
				_bias[0] = bias ? _mm512_loadu_ps(bias + dc + 0) : _mm512_setzero_ps();
				_bias[1] = bias ? _mm512_loadu_ps(bias + dc + F) : _mm512_setzero_ps();
				if (type == ::SimdConvolutionActivationPrelu)
				{
					_params[0] = _mm512_loadu_ps(params + dc + 0);
					_params[1] = _mm512_loadu_ps(params + dc + F);
				}
				const float* pS = src + yBeg * srcW * srcC;
				const float* pW = weight + dc * srcC;
				float* pD = dst + (dc / F) * dstS;
				float* dst0 = pD + (yBeg & dstM) * dstW * F;
				float* dst1 = pD + (yInt & dstM) * dstW * F;
				size_t dn = nBeg;
				if (dstC - dc > F)
				{
					for (; dn < nInt12; dn += 12, pS += 12 * srcC, dst0 += 12 * F)
						InputConvolution1x1_2xM<type, 12>(pS, srcC, pW, _bias, _params, dst0, dst0 + dstS);
					if (nIntTail)
						tailInt_2(pS, srcC, pW, _bias, _params, dst0, dst0 + dstS), pS += nIntTail * srcC, dn += nIntTail;
					for (; dn < nEnd12; dn += 12, pS += 12 * srcC, dst1 += 12 * F)
						InputConvolution1x1_2xM<type, 12>(pS, srcC, pW, _bias, _params, dst1, dst1 + dstS);
					if (nEndTail)
						tailEnd_2(pS, srcC, pW, _bias, _params, dst1, dst1 + dstS), pS += nEndTail * srcC, dn += nEndTail;
				}
				else
				{
					InputConvolution1x1_1xM_Ptr tailInt_1 = GetInputConvolution1x1_1xM<type>(nIntTail);
					InputConvolution1x1_1xM_Ptr tailEnd_1 = GetInputConvolution1x1_1xM<type>(nEndTail);
					for (; dn < nInt12; dn += 12, pS += 12 * srcC, dst0 += 12 * F)
						InputConvolution1x1_1xM<type, 12>(pS, srcC, pW, _bias, _params, dst0);
					if (nIntTail)
						tailInt_1(pS, srcC, pW, _bias, _params, dst0), pS += nIntTail * srcC, dn += nIntTail;
					for (; dn < nEnd12; dn += 12, pS += 12 * srcC, dst1 += 12 * F)
						InputConvolution1x1_1xM<type, 12>(pS, srcC, pW, _bias, _params, dst1);
					if (nEndTail)
						tailEnd_1(pS, srcC, pW, _bias, _params, dst1), pS += nEndTail * srcC, dn += nEndTail;
				}
			}
		}

		//-------------------------------------------------------------------------------------------------

		template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution_2x1(const float* src0, const SimdConvolutionParameters& p,
			size_t kH, size_t kW, const float* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
		{
			__m512 d00, d01, s0, w0, w1;
			d00 = bias[0];
			d01 = bias[1];
			size_t size = kW * p.srcC, tail = DF * (p.kernelX - kW) * p.srcC, stride = p.srcW * p.srcC;
			for (size_t ky = 0; ky < kH; ++ky)
			{
				for (size_t i = 0; i < size; ++i)
				{
					w0 = _mm512_loadu_ps(weight + 0);
					w1 = _mm512_loadu_ps(weight + F);
					s0 = _mm512_set1_ps(src0[i]);
					d00 = _mm512_fmadd_ps(s0, w0, d00);
					d01 = _mm512_fmadd_ps(s0, w1, d01);
					weight += DF;
				}
				weight += tail;
				src0 += stride;
			}
			_mm512_storeu_ps(dst0, Activate<type>(d00, params, 0));
			_mm512_storeu_ps(dst1, Activate<type>(d01, params, 1));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution_1x1(const float* src0, const SimdConvolutionParameters& p,
			size_t kH, size_t kW, const float* weight, const __m512* bias, const __m512* params, float* dst0)
		{
			__m512 d00, s0, w0;
			d00 = bias[0];
			size_t size = kW * p.srcC, tail = DF * (p.kernelX - kW) * p.srcC, stride = p.srcW * p.srcC;
			for (size_t ky = 0; ky < kH; ++ky)
			{
				for (size_t i = 0; i < size; ++i)
				{
					w0 = _mm512_loadu_ps(weight + 0);
					s0 = _mm512_set1_ps(src0[i]);
					d00 = _mm512_fmadd_ps(s0, w0, d00);
					weight += DF;
				}
				weight += tail;
				src0 += stride;
			}
			_mm512_storeu_ps(dst0, Activate<type>(d00, params, 0));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution_2x6(const float* src0, const SimdConvolutionParameters& p,
			size_t kH, size_t kW, const float* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
		{
			__m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
			d00 = bias[0], d01 = bias[1];
			d10 = bias[0], d11 = bias[1];
			d20 = bias[0], d21 = bias[1];
			d30 = bias[0], d31 = bias[1];
			d40 = bias[0], d41 = bias[1];
			d50 = bias[0], d51 = bias[1];
			size_t size = kW * p.srcC, tail = DF * (p.kernelX - kW) * p.srcC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
			const float* src1 = src0 + 1 * step;
			const float* src2 = src0 + 2 * step;
			const float* src3 = src0 + 3 * step;
			const float* src4 = src0 + 4 * step;
			const float* src5 = src0 + 5 * step;
			for (size_t ky = 0; ky < kH; ++ky)
			{
				size_t offset = ky * stride;
				for (size_t end = offset + size; offset < end; ++offset)
				{
					w0 = _mm512_loadu_ps(weight + 0);
					w1 = _mm512_loadu_ps(weight + F);
					s0 = _mm512_set1_ps(src0[offset]);
					d00 = _mm512_fmadd_ps(s0, w0, d00);
					d01 = _mm512_fmadd_ps(s0, w1, d01);
					s0 = _mm512_set1_ps(src1[offset]);
					d10 = _mm512_fmadd_ps(s0, w0, d10);
					d11 = _mm512_fmadd_ps(s0, w1, d11);
					s0 = _mm512_set1_ps(src2[offset]);
					d20 = _mm512_fmadd_ps(s0, w0, d20);
					d21 = _mm512_fmadd_ps(s0, w1, d21);
					s0 = _mm512_set1_ps(src3[offset]);
					d30 = _mm512_fmadd_ps(s0, w0, d30);
					d31 = _mm512_fmadd_ps(s0, w1, d31);
					s0 = _mm512_set1_ps(src4[offset]);
					d40 = _mm512_fmadd_ps(s0, w0, d40);
					d41 = _mm512_fmadd_ps(s0, w1, d41);
					s0 = _mm512_set1_ps(src5[offset]);
					d50 = _mm512_fmadd_ps(s0, w0, d50);
					d51 = _mm512_fmadd_ps(s0, w1, d51);
					weight += DF;
				}
				weight += tail;
			}
			_mm512_storeu_ps(dst0 + 0 * F, Activate<type>(d00, params, 0));
			_mm512_storeu_ps(dst1 + 0 * F, Activate<type>(d01, params, 1));
			_mm512_storeu_ps(dst0 + 1 * F, Activate<type>(d10, params, 0));
			_mm512_storeu_ps(dst1 + 1 * F, Activate<type>(d11, params, 1));
			_mm512_storeu_ps(dst0 + 2 * F, Activate<type>(d20, params, 0));
			_mm512_storeu_ps(dst1 + 2 * F, Activate<type>(d21, params, 1));
			_mm512_storeu_ps(dst0 + 3 * F, Activate<type>(d30, params, 0));
			_mm512_storeu_ps(dst1 + 3 * F, Activate<type>(d31, params, 1));
			_mm512_storeu_ps(dst0 + 4 * F, Activate<type>(d40, params, 0));
			_mm512_storeu_ps(dst1 + 4 * F, Activate<type>(d41, params, 1));
			_mm512_storeu_ps(dst0 + 5 * F, Activate<type>(d50, params, 0));
			_mm512_storeu_ps(dst1 + 5 * F, Activate<type>(d51, params, 1));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution_1x6(const float* src0, const SimdConvolutionParameters& p,
			size_t kH, size_t kW, const float* weight, const __m512* bias, const __m512* params, float* dst0)
		{
			__m512 d00, d10, d20, d30, d40, d50, s0, w0;
			d00 = bias[0];
			d10 = bias[0];
			d20 = bias[0];
			d30 = bias[0];
			d40 = bias[0];
			d50 = bias[0];
			size_t size = kW * p.srcC, tail = DF * (p.kernelX - kW) * p.srcC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
			const float* src1 = src0 + 1 * step;
			const float* src2 = src0 + 2 * step;
			const float* src3 = src0 + 3 * step;
			const float* src4 = src0 + 4 * step;
			const float* src5 = src0 + 5 * step;
			for (size_t ky = 0; ky < kH; ++ky)
			{
				size_t offset = ky * stride;
				for (size_t end = offset + size; offset < end; ++offset)
				{
					w0 = _mm512_loadu_ps(weight + 0);
					s0 = _mm512_set1_ps(src0[offset]);
					d00 = _mm512_fmadd_ps(s0, w0, d00);
					s0 = _mm512_set1_ps(src1[offset]);
					d10 = _mm512_fmadd_ps(s0, w0, d10);
					s0 = _mm512_set1_ps(src2[offset]);
					d20 = _mm512_fmadd_ps(s0, w0, d20);
					s0 = _mm512_set1_ps(src3[offset]);
					d30 = _mm512_fmadd_ps(s0, w0, d30);
					s0 = _mm512_set1_ps(src4[offset]);
					d40 = _mm512_fmadd_ps(s0, w0, d40);
					s0 = _mm512_set1_ps(src5[offset]);
					d50 = _mm512_fmadd_ps(s0, w0, d50);
					weight += DF;
				}
				weight += tail;
			}
			_mm512_storeu_ps(dst0 + 0 * F, Activate<type>(d00, params, 0));
			_mm512_storeu_ps(dst0 + 1 * F, Activate<type>(d10, params, 0));
			_mm512_storeu_ps(dst0 + 2 * F, Activate<type>(d20, params, 0));
			_mm512_storeu_ps(dst0 + 3 * F, Activate<type>(d30, params, 0));
			_mm512_storeu_ps(dst0 + 4 * F, Activate<type>(d40, params, 0));
			_mm512_storeu_ps(dst0 + 5 * F, Activate<type>(d50, params, 0));
		}

		template<SimdConvolutionActivationType type> void InputConvolution(const float* src, const SimdConvolutionParameters& p,
			size_t dstC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
			size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX;
			size_t dstM = (bufH[0] - 1), dstS = bufH[0] * dstW * F;
			size_t dstCDF = AlignLo(dstC, DF);
			if (dstC - F > dstCDF)
				dstCDF += DF;

			size_t noseH = p.padY, noseW = p.padX;
			size_t bodyH = p.srcH - p.kernelY + 1 + noseH, bodyW = p.srcW - p.kernelX + 1 + noseW;
			size_t bodyW6 = AlignLoAny(bodyW - noseW, 6 * p.strideX) + noseW;
			size_t tailH = bodyH + p.padH, tailW = bodyW + p.padW;
			size_t wS = p.srcC * p.dstC;
			size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;

			__m512 _params[2], _bias[2];
			_params[0] = _mm512_set1_ps(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				_params[1] = _mm512_set1_ps(params[1]);

			size_t dc = 0;
			for (; dc < dstCDF; dc += DF)
			{
				_bias[0] = bias ? _mm512_loadu_ps(bias + dc + 0) : _mm512_setzero_ps();
				_bias[1] = bias ? _mm512_loadu_ps(bias + dc + F) : _mm512_setzero_ps();
				if (type == ::SimdConvolutionActivationPrelu)
				{
					_params[0] = _mm512_loadu_ps(params + dc + 0);
					_params[1] = _mm512_loadu_ps(params + dc + F);
				}
				size_t dy = yBeg, sy = dy * strideY;
				for (; sy < noseH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS, * dst1 = dst0 + dstS;
					size_t sx = 0;
					const float* s = src;
					const float* w = weight + (noseH - sy) * kernelX * DF * srcC;
					for (; sx < noseW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s, p, kY + sy, kX + sx, w + (noseW - sx) * srcC * DF, _bias, _params, dst0, dst1);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F, dst1 += 6 * F)
						InputConvolution_2x6<type>(s + (sx - noseW) * srcC, p, kY + sy, kernelX, w, _bias, _params, dst0, dst1);
					for (; sx < bodyW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kY + sy, kernelX, w, _bias, _params, dst0, dst1);
					for (; sx < tailW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kY + sy, kW - sx, w, _bias, _params, dst0, dst1);
				}
				for (; sy < bodyH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS, * dst1 = dst0 + dstS;
					size_t sx = 0;
					const float* s = src + (sy - noseH) * srcW * srcC;
					const float* w = weight;
					for (; sx < noseW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s, p, kernelY, kX + sx, w + (noseW - sx) * srcC * DF, _bias, _params, dst0, dst1);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F, dst1 += 6 * F)
						InputConvolution_2x6<type>(s + (sx - noseW) * srcC, p, kernelY, kernelX, w, _bias, _params, dst0, dst1);
					for (; sx < bodyW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kernelY, kernelX, w, _bias, _params, dst0, dst1);
					for (; sx < tailW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kernelY, kW - sx, w, _bias, _params, dst0, dst1);
				}
				for (; sy < tailH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS, * dst1 = dst0 + dstS;
					size_t sx = 0;
					const float* s = src + (sy - noseH) * srcW * srcC;
					const float* w = weight;
					for (; sx < noseW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s, p, kH - sy, kX + sx, w + (noseW - sx) * srcC * DF, _bias, _params, dst0, dst1);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F, dst1 += 6 * F)
						InputConvolution_2x6<type>(s + (sx - noseW) * srcC, p, kH - sy, kernelX, w, _bias, _params, dst0, dst1);
					for (; sx < bodyW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kH - sy, kernelX, w, _bias, _params, dst0, dst1);
					for (; sx < tailW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kH - sy, kW - sx, w, _bias, _params, dst0, dst1);
				}
				weight += kernelY * kernelX * srcC * DF;
			}
			if (dc < dstC)
			{
				_bias[0] = bias ? _mm512_loadu_ps(bias + dc) : _mm512_setzero_ps();
				if (type == ::SimdConvolutionActivationPrelu)
					_params[0] = _mm512_loadu_ps(params + dc);
				size_t dy = yBeg, sy = dy * strideY;
				for (; sy < noseH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS;
					size_t sx = 0;
					const float* s = src;
					const float* w = weight + (noseH - sy) * kernelX * DF * srcC;
					for (; sx < noseW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s, p, kY + sy, kX + sx, w + (noseW - sx) * srcC * DF, _bias, _params, dst0);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F)
						InputConvolution_1x6<type>(s + (sx - noseW) * srcC, p, kY + sy, kernelX, w, _bias, _params, dst0);
					for (; sx < bodyW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kY + sy, kernelX, w, _bias, _params, dst0);
					for (; sx < tailW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kY + sy, kW - sx, w, _bias, _params, dst0);
				}
				for (; sy < bodyH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS;
					size_t sx = 0;
					const float* s = src + (sy - noseH) * srcW * srcC;
					const float* w = weight;
					for (; sx < noseW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s, p, kernelY, kX + sx, w + (noseW - sx) * srcC * DF, _bias, _params, dst0);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F)
						InputConvolution_1x6<type>(s + (sx - noseW) * srcC, p, kernelY, kernelX, w, _bias, _params, dst0);
					for (; sx < bodyW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kernelY, kernelX, w, _bias, _params, dst0);
					for (; sx < tailW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kernelY, kW - sx, w, _bias, _params, dst0);
				}
				for (; sy < tailH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS;
					size_t sx = 0;
					const float* s = src + (sy - noseH) * srcW * srcC;
					const float* w = weight;
					for (; sx < noseW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s, p, kH - sy, kX + sx, w + (noseW - sx) * srcC * DF, _bias, _params, dst0);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F)
						InputConvolution_1x6<type>(s + (sx - noseW) * srcC, p, kH - sy, kernelX, w, _bias, _params, dst0);
					for (; sx < bodyW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kH - sy, kernelX, w, _bias, _params, dst0);
					for (; sx < tailW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kH - sy, kW - sx, w, _bias, _params, dst0);
				}
			}
		}

		//-------------------------------------------------------------------------------------------------

		template <SimdConvolutionActivationType type> void SetInput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			if (p.kernelY == 1 && p.strideY == 1)
				convolution[0] = InputConvolution1x1<type>;
			else
				convolution[0] = InputConvolution<type>;
		}

		void SetInput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			switch (p.activation)
			{
			case SimdConvolutionActivationIdentity: SetInput<SimdConvolutionActivationRestrictRange>(p, convolution); break;
			case SimdConvolutionActivationRelu: SetInput<SimdConvolutionActivationRestrictRange>(p, convolution); break;
			case SimdConvolutionActivationLeakyRelu: SetInput<SimdConvolutionActivationPrelu>(p, convolution); break;
			case SimdConvolutionActivationRestrictRange: SetInput<SimdConvolutionActivationRestrictRange>(p, convolution); break;
			case SimdConvolutionActivationPrelu: SetInput<SimdConvolutionActivationPrelu>(p, convolution); break;
			case SimdConvolutionActivationElu: SetInput<SimdConvolutionActivationElu>(p, convolution); break;
			case SimdConvolutionActivationHswish: SetInput<SimdConvolutionActivationHswish>(p, convolution); break;
			case SimdConvolutionActivationMish: SetInput<SimdConvolutionActivationMish>(p, convolution); break;
			case SimdConvolutionActivationHardSigmoid: SetInput<SimdConvolutionActivationHardSigmoid>(p, convolution); break;
			case SimdConvolutionActivationSwish: SetInput<SimdConvolutionActivationSwish>(p, convolution); break;
			case SimdConvolutionActivationGelu: SetInput<SimdConvolutionActivationGelu>(p, convolution); break;
			default: assert(0);
			}
		}
	}
#endif
}
