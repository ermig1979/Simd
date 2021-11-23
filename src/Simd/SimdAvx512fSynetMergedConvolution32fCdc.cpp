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
#if defined(SIMD_AVX512F_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Avx512f
	{
		namespace Cdc
		{
			template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_2x12(const float* src0, size_t srcC,
				const float* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
			{
				__m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, s0, w0, w1;
				d00 = bias[0], d01 = bias[1];
				d10 = bias[0], d11 = bias[1];
				d20 = bias[0], d21 = bias[1];
				d30 = bias[0], d31 = bias[1];
				d40 = bias[0], d41 = bias[1];
				d50 = bias[0], d51 = bias[1];
				d60 = bias[0], d61 = bias[1];
				d70 = bias[0], d71 = bias[1];
				d80 = bias[0], d81 = bias[1];
				d90 = bias[0], d91 = bias[1];
				da0 = bias[0], da1 = bias[1];
				db0 = bias[0], db1 = bias[1];
				const float* src1 = src0 + 1 * srcC;
				const float* src2 = src0 + 2 * srcC;
				const float* src3 = src0 + 3 * srcC;
				const float* src4 = src0 + 4 * srcC;
				const float* src5 = src0 + 5 * srcC;
				for (size_t sc0 = 0, sc6 = 6 * srcC; sc0 < srcC; ++sc0, ++sc6)
				{
					w0 = _mm512_loadu_ps(weight + 0);
					w1 = _mm512_loadu_ps(weight + F);
					s0 = _mm512_set1_ps(src0[sc0]);
					d00 = _mm512_fmadd_ps(s0, w0, d00);
					d01 = _mm512_fmadd_ps(s0, w1, d01);
					s0 = _mm512_set1_ps(src1[sc0]);
					d10 = _mm512_fmadd_ps(s0, w0, d10);
					d11 = _mm512_fmadd_ps(s0, w1, d11);
					s0 = _mm512_set1_ps(src2[sc0]);
					d20 = _mm512_fmadd_ps(s0, w0, d20);
					d21 = _mm512_fmadd_ps(s0, w1, d21);
					s0 = _mm512_set1_ps(src3[sc0]);
					d30 = _mm512_fmadd_ps(s0, w0, d30);
					d31 = _mm512_fmadd_ps(s0, w1, d31);
					s0 = _mm512_set1_ps(src4[sc0]);
					d40 = _mm512_fmadd_ps(s0, w0, d40);
					d41 = _mm512_fmadd_ps(s0, w1, d41);
					s0 = _mm512_set1_ps(src5[sc0]);
					d50 = _mm512_fmadd_ps(s0, w0, d50);
					d51 = _mm512_fmadd_ps(s0, w1, d51);
					s0 = _mm512_set1_ps(src0[sc6]);
					d60 = _mm512_fmadd_ps(s0, w0, d60);
					d61 = _mm512_fmadd_ps(s0, w1, d61);
					s0 = _mm512_set1_ps(src1[sc6]);
					d70 = _mm512_fmadd_ps(s0, w0, d70);
					d71 = _mm512_fmadd_ps(s0, w1, d71);
					s0 = _mm512_set1_ps(src2[sc6]);
					d80 = _mm512_fmadd_ps(s0, w0, d80);
					d81 = _mm512_fmadd_ps(s0, w1, d81);
					s0 = _mm512_set1_ps(src3[sc6]);
					d90 = _mm512_fmadd_ps(s0, w0, d90);
					d91 = _mm512_fmadd_ps(s0, w1, d91);
					s0 = _mm512_set1_ps(src4[sc6]);
					da0 = _mm512_fmadd_ps(s0, w0, da0);
					da1 = _mm512_fmadd_ps(s0, w1, da1);
					s0 = _mm512_set1_ps(src5[sc6]);
					db0 = _mm512_fmadd_ps(s0, w0, db0);
					db1 = _mm512_fmadd_ps(s0, w1, db1);
					weight += DF;
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
				_mm512_storeu_ps(dst0 + 6 * F, Activate<type>(d60, params, 0));
				_mm512_storeu_ps(dst1 + 6 * F, Activate<type>(d61, params, 1));
				_mm512_storeu_ps(dst0 + 7 * F, Activate<type>(d70, params, 0));
				_mm512_storeu_ps(dst1 + 7 * F, Activate<type>(d71, params, 1));
				_mm512_storeu_ps(dst0 + 8 * F, Activate<type>(d80, params, 0));
				_mm512_storeu_ps(dst1 + 8 * F, Activate<type>(d81, params, 1));
				_mm512_storeu_ps(dst0 + 9 * F, Activate<type>(d90, params, 0));
				_mm512_storeu_ps(dst1 + 9 * F, Activate<type>(d91, params, 1));
				_mm512_storeu_ps(dst0 + 10 * F, Activate<type>(da0, params, 0));
				_mm512_storeu_ps(dst1 + 10 * F, Activate<type>(da1, params, 1));
				_mm512_storeu_ps(dst0 + 11 * F, Activate<type>(db0, params, 0));
				_mm512_storeu_ps(dst1 + 11 * F, Activate<type>(db1, params, 1));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_2x6(const float* src0, size_t srcC,
				const float* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
			{
				__m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
				d00 = bias[0], d01 = bias[1];
				d10 = bias[0], d11 = bias[1];
				d20 = bias[0], d21 = bias[1];
				d30 = bias[0], d31 = bias[1];
				d40 = bias[0], d41 = bias[1];
				d50 = bias[0], d51 = bias[1];
				const float* src1 = src0 + 1 * srcC;
				const float* src2 = src0 + 2 * srcC;
				const float* src3 = src0 + 3 * srcC;
				const float* src4 = src0 + 4 * srcC;
				const float* src5 = src0 + 5 * srcC;
				for (size_t sc = 0; sc < srcC; ++sc)
				{
					w0 = _mm512_loadu_ps(weight + 0);
					w1 = _mm512_loadu_ps(weight + F);
					s0 = _mm512_set1_ps(src0[sc]);
					d00 = _mm512_fmadd_ps(s0, w0, d00);
					d01 = _mm512_fmadd_ps(s0, w1, d01);
					s0 = _mm512_set1_ps(src1[sc]);
					d10 = _mm512_fmadd_ps(s0, w0, d10);
					d11 = _mm512_fmadd_ps(s0, w1, d11);
					s0 = _mm512_set1_ps(src2[sc]);
					d20 = _mm512_fmadd_ps(s0, w0, d20);
					d21 = _mm512_fmadd_ps(s0, w1, d21);
					s0 = _mm512_set1_ps(src3[sc]);
					d30 = _mm512_fmadd_ps(s0, w0, d30);
					d31 = _mm512_fmadd_ps(s0, w1, d31);
					s0 = _mm512_set1_ps(src4[sc]);
					d40 = _mm512_fmadd_ps(s0, w0, d40);
					d41 = _mm512_fmadd_ps(s0, w1, d41);
					s0 = _mm512_set1_ps(src5[sc]);
					d50 = _mm512_fmadd_ps(s0, w0, d50);
					d51 = _mm512_fmadd_ps(s0, w1, d51);
					weight += DF;
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

			template<SimdConvolutionActivationType type, int M> SIMD_INLINE void InputConvolution1x1_2xM(const float* src0, size_t srcC,
				const float* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1)
			{
				__m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
				if (M > 0) d00 = bias[0], d01 = bias[1];
				if (M > 1) d10 = bias[0], d11 = bias[1];
				if (M > 2) d20 = bias[0], d21 = bias[1];
				if (M > 3) d30 = bias[0], d31 = bias[1];
				if (M > 4) d40 = bias[0], d41 = bias[1];
				if (M > 5) d50 = bias[0], d51 = bias[1];
				const float* src1 = src0 + 1 * srcC;
				const float* src2 = src0 + 2 * srcC;
				const float* src3 = src0 + 3 * srcC;
				const float* src4 = src0 + 4 * srcC;
				const float* src5 = src0 + 5 * srcC;
				for (size_t sc = 0; sc < srcC; ++sc)
				{
					w0 = _mm512_loadu_ps(weight + 0);
					w1 = _mm512_loadu_ps(weight + F);
					if (M > 0) s0 = _mm512_set1_ps(src0[sc]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
					if (M > 1) s0 = _mm512_set1_ps(src1[sc]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
					if (M > 2) s0 = _mm512_set1_ps(src2[sc]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
					if (M > 3) s0 = _mm512_set1_ps(src3[sc]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
					if (M > 4) s0 = _mm512_set1_ps(src4[sc]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
					if (M > 5) s0 = _mm512_set1_ps(src5[sc]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
					weight += DF;
				}
				if (M > 0) _mm512_storeu_ps(dst0 + 0 * F, Activate<type>(d00, params, 0)), _mm512_storeu_ps(dst1 + 0 * F, Activate<type>(d01, params, 1));
				if (M > 1) _mm512_storeu_ps(dst0 + 1 * F, Activate<type>(d10, params, 0)), _mm512_storeu_ps(dst1 + 1 * F, Activate<type>(d11, params, 1));
				if (M > 2) _mm512_storeu_ps(dst0 + 2 * F, Activate<type>(d20, params, 0)), _mm512_storeu_ps(dst1 + 2 * F, Activate<type>(d21, params, 1));
				if (M > 3) _mm512_storeu_ps(dst0 + 3 * F, Activate<type>(d30, params, 0)), _mm512_storeu_ps(dst1 + 3 * F, Activate<type>(d31, params, 1));
				if (M > 4) _mm512_storeu_ps(dst0 + 4 * F, Activate<type>(d40, params, 0)), _mm512_storeu_ps(dst1 + 4 * F, Activate<type>(d41, params, 1));
				if (M > 5) _mm512_storeu_ps(dst0 + 5 * F, Activate<type>(d50, params, 0)), _mm512_storeu_ps(dst1 + 5 * F, Activate<type>(d51, params, 1));
			}

			typedef void(*InputConvolution1x1_2xM_Ptr)(const float* src0, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst0, float* dst1);

			template<SimdConvolutionActivationType type> InputConvolution1x1_2xM_Ptr GetInputConvolution1x1_2xM(size_t M)
			{
				switch (M)
				{
				case 0: return InputConvolution1x1_2xM<type, 0>;
				case 1: return InputConvolution1x1_2xM<type, 1>;
				case 2: return InputConvolution1x1_2xM<type, 2>;
				case 3: return InputConvolution1x1_2xM<type, 3>;
				case 4: return InputConvolution1x1_2xM<type, 4>;
				case 5: return InputConvolution1x1_2xM<type, 5>;
				}
				assert(0);
				return NULL;
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_1x6(const float* src0, size_t srcC,
				const float* weight, const __m512* bias, const __m512* params, float* dst0)
			{
				__m512 d00, d10, d20, d30, d40, d50, s0, w0;
				d00 = bias[0];
				d10 = bias[0];
				d20 = bias[0];
				d30 = bias[0];
				d40 = bias[0];
				d50 = bias[0];
				const float* src1 = src0 + 1 * srcC;
				const float* src2 = src0 + 2 * srcC;
				const float* src3 = src0 + 3 * srcC;
				const float* src4 = src0 + 4 * srcC;
				const float* src5 = src0 + 5 * srcC;
				for (size_t sc = 0; sc < srcC; ++sc)
				{
					w0 = _mm512_loadu_ps(weight + 0);
					s0 = _mm512_set1_ps(src0[sc]);
					d00 = _mm512_fmadd_ps(s0, w0, d00);
					s0 = _mm512_set1_ps(src1[sc]);
					d10 = _mm512_fmadd_ps(s0, w0, d10);
					s0 = _mm512_set1_ps(src2[sc]);
					d20 = _mm512_fmadd_ps(s0, w0, d20);
					s0 = _mm512_set1_ps(src3[sc]);
					d30 = _mm512_fmadd_ps(s0, w0, d30);
					s0 = _mm512_set1_ps(src4[sc]);
					d40 = _mm512_fmadd_ps(s0, w0, d40);
					s0 = _mm512_set1_ps(src5[sc]);
					d50 = _mm512_fmadd_ps(s0, w0, d50);
					weight += DF;
				}
				_mm512_storeu_ps(dst0 + 0 * F, Activate<type>(d00, params, 0));
				_mm512_storeu_ps(dst0 + 1 * F, Activate<type>(d10, params, 0));
				_mm512_storeu_ps(dst0 + 2 * F, Activate<type>(d20, params, 0));
				_mm512_storeu_ps(dst0 + 3 * F, Activate<type>(d30, params, 0));
				_mm512_storeu_ps(dst0 + 4 * F, Activate<type>(d40, params, 0));
				_mm512_storeu_ps(dst0 + 5 * F, Activate<type>(d50, params, 0));
			}

			template<SimdConvolutionActivationType type, int M> SIMD_INLINE void InputConvolution1x1_1xM(const float* src0, size_t srcC,
				const float* weight, const __m512* bias, const __m512* params, float* dst0)
			{
				__m512 d00, d10, d20, d30, d40, d50, s0, w0;
				if (M > 0) d00 = bias[0];
				if (M > 1) d10 = bias[0];
				if (M > 2) d20 = bias[0];
				if (M > 3) d30 = bias[0];
				if (M > 4) d40 = bias[0];
				if (M > 5) d50 = bias[0];
				const float* src1 = src0 + 1 * srcC;
				const float* src2 = src0 + 2 * srcC;
				const float* src3 = src0 + 3 * srcC;
				const float* src4 = src0 + 4 * srcC;
				const float* src5 = src0 + 5 * srcC;
				for (size_t sc = 0; sc < srcC; ++sc)
				{
					w0 = _mm512_loadu_ps(weight + 0);
					if (M > 0) s0 = _mm512_set1_ps(src0[sc]), d00 = _mm512_fmadd_ps(s0, w0, d00);
					if (M > 1) s0 = _mm512_set1_ps(src1[sc]), d10 = _mm512_fmadd_ps(s0, w0, d10);
					if (M > 2) s0 = _mm512_set1_ps(src2[sc]), d20 = _mm512_fmadd_ps(s0, w0, d20);
					if (M > 3) s0 = _mm512_set1_ps(src3[sc]), d30 = _mm512_fmadd_ps(s0, w0, d30);
					if (M > 4) s0 = _mm512_set1_ps(src4[sc]), d40 = _mm512_fmadd_ps(s0, w0, d40);
					if (M > 5) s0 = _mm512_set1_ps(src5[sc]), d50 = _mm512_fmadd_ps(s0, w0, d50);
					weight += DF;
				}
				if (M > 0) _mm512_storeu_ps(dst0 + 0 * F, Activate<type>(d00, params, 0));
				if (M > 1) _mm512_storeu_ps(dst0 + 1 * F, Activate<type>(d10, params, 0));
				if (M > 2) _mm512_storeu_ps(dst0 + 2 * F, Activate<type>(d20, params, 0));
				if (M > 3) _mm512_storeu_ps(dst0 + 3 * F, Activate<type>(d30, params, 0));
				if (M > 4) _mm512_storeu_ps(dst0 + 4 * F, Activate<type>(d40, params, 0));
				if (M > 5) _mm512_storeu_ps(dst0 + 5 * F, Activate<type>(d50, params, 0));
			}

			typedef void(*InputConvolution1x1_1xM_Ptr)(const float* src0, size_t srcC, const float* weight, const __m512* bias, const __m512* params, float* dst0);

			template<SimdConvolutionActivationType type> InputConvolution1x1_1xM_Ptr GetInputConvolution1x1_1xM(size_t M)
			{
				switch (M)
				{
				case 0: return InputConvolution1x1_1xM<type, 0>;
				case 1: return InputConvolution1x1_1xM<type, 1>;
				case 2: return InputConvolution1x1_1xM<type, 2>;
				case 3: return InputConvolution1x1_1xM<type, 3>;
				case 4: return InputConvolution1x1_1xM<type, 4>;
				case 5: return InputConvolution1x1_1xM<type, 5>;
				}
				assert(0);
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
				size_t nInt6 = AlignLoAny(nInt - nBeg, 6) + nBeg, nEnd6 = AlignLoAny(nEnd - nInt, 6) + nInt, nIntTail = nInt - nInt6, nEndTail = nEnd - nEnd6;
				size_t nInt12 = AlignLoAny(nInt - nBeg, 12) + nBeg, nEnd12 = AlignLoAny(nEnd - nInt, 12) + nInt;
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
							InputConvolution1x1_2x12<type>(pS, srcC, pW, _bias, _params, dst0, dst0 + dstS);
						for (; dn < nInt6; dn += 6, pS += 6 * srcC, dst0 += 6 * F)
							InputConvolution1x1_2x6<type>(pS, srcC, pW, _bias, _params, dst0, dst0 + dstS);
						if (nIntTail)
							tailInt_2(pS, srcC, pW, _bias, _params, dst0, dst0 + dstS), pS += nIntTail * srcC, dn += nIntTail;
						for (; dn < nEnd12; dn += 12, pS += 12 * srcC, dst1 += 12 * F)
							InputConvolution1x1_2x12<type>(pS, srcC, pW, _bias, _params, dst1, dst1 + dstS);
						for (; dn < nEnd6; dn += 6, pS += 6 * srcC, dst1 += 6 * F)
							InputConvolution1x1_2x6<type>(pS, srcC, pW, _bias, _params, dst1, dst1 + dstS);
						if (nEndTail)
							tailEnd_2(pS, srcC, pW, _bias, _params, dst1, dst1 + dstS), pS += nEndTail * srcC, dn += nEndTail;
					}
					else
					{
						InputConvolution1x1_1xM_Ptr tailInt_1 = GetInputConvolution1x1_1xM<type>(nIntTail);
						InputConvolution1x1_1xM_Ptr tailEnd_1 = GetInputConvolution1x1_1xM<type>(nEndTail);
						for (; dn < nInt6; dn += 6, pS += 6 * srcC, dst0 += 6 * F)
							InputConvolution1x1_1x6<type>(pS, srcC, pW, _bias, _params, dst0);
						if (nIntTail)
							tailInt_1(pS, srcC, pW, _bias, _params, dst0), pS += nIntTail * srcC, dn += nIntTail;
						for (; dn < nEnd6; dn += 6, pS += 6 * srcC, dst1 += 6 * F)
							InputConvolution1x1_1x6<type>(pS, srcC, pW, _bias, _params, dst1);
						if (nEndTail)
							tailEnd_1(pS, srcC, pW, _bias, _params, dst1), pS += nEndTail * srcC, dn += nEndTail;
					}
				}
			}

			//---------------------------------------------------------------------

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

			//---------------------------------------------------------------------

			template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
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

				__m512 _params[2];
				_params[0] = _mm512_set1_ps(params[0]);
				if (type == SimdConvolutionActivationRestrictRange ||
					type == SimdConvolutionActivationHswish ||
					type == SimdConvolutionActivationHardSigmoid)
					_params[1] = _mm512_set1_ps(params[1]);
				for (size_t c = 0; c < srcC; c += F)
				{
					__m512 _bias = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
					if (type == ::SimdConvolutionActivationPrelu)
						_params[0] = _mm512_loadu_ps(params + c);

					for (size_t dy = yBeg; dy < yEnd; ++dy)
					{
						float* pd = dst + (dy & dstM) * dstW;
						if (dy >= noseY && dy < bodyY)
						{
							size_t dx = 0;
							for (; dx < noseX; ++dx, pd += F)
							{
								__m512 sum = _bias;
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
											sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
										}
									}
								}
								_mm512_storeu_ps(pd, Activate<type>(sum, _params, 0));
							}
							for (; dx < bodyX8; dx += 8, pd += 8 * F)
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
									const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
									for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
									{
										__m512 w0 = _mm512_loadu_ps(pw);
										sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * strideXF), w0, sum0);
										sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * strideXF), w0, sum1);
										sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * strideXF), w0, sum2);
										sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 3 * strideXF), w0, sum3);
										sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 4 * strideXF), w0, sum4);
										sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 5 * strideXF), w0, sum5);
										sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 6 * strideXF), w0, sum6);
										sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 7 * strideXF), w0, sum7);
									}
								}
								_mm512_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
								_mm512_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
								_mm512_storeu_ps(pd + 2 * F, Activate<type>(sum2, _params, 0));
								_mm512_storeu_ps(pd + 3 * F, Activate<type>(sum3, _params, 0));
								_mm512_storeu_ps(pd + 4 * F, Activate<type>(sum4, _params, 0));
								_mm512_storeu_ps(pd + 5 * F, Activate<type>(sum5, _params, 0));
								_mm512_storeu_ps(pd + 6 * F, Activate<type>(sum6, _params, 0));
								_mm512_storeu_ps(pd + 7 * F, Activate<type>(sum7, _params, 0));
							}
							for (; dx < bodyX4; dx += 4, pd += 4 * F)
							{
								__m512 sum0 = _bias;
								__m512 sum1 = _bias;
								__m512 sum2 = _bias;
								__m512 sum3 = _bias;
								const float* pw = weight;
								for (size_t ky = 0; ky < p.kernelY; ++ky)
								{
									size_t sy = dy * strideY + ky - padY;
									const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
									for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
									{
										__m512 w0 = _mm512_loadu_ps(pw);
										sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * strideXF), w0, sum0);
										sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * strideXF), w0, sum1);
										sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * strideXF), w0, sum2);
										sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 3 * strideXF), w0, sum3);
									}
								}
								_mm512_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
								_mm512_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
								_mm512_storeu_ps(pd + 2 * F, Activate<type>(sum2, _params, 0));
								_mm512_storeu_ps(pd + 3 * F, Activate<type>(sum3, _params, 0));
							}
							for (; dx < bodyX2; dx += 2, pd += 2 * F)
							{
								__m512 sum0 = _bias;
								__m512 sum1 = _bias;
								const float* pw = weight;
								for (size_t ky = 0; ky < p.kernelY; ++ky)
								{
									size_t sy = dy * strideY + ky - padY;
									const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
									for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
									{
										__m512 w0 = _mm512_loadu_ps(pw);
										sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * strideXF), w0, sum0);
										sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * strideXF), w0, sum1);
									}
								}
								_mm512_storeu_ps(pd + 0 * F, Activate<type>(sum0, _params, 0));
								_mm512_storeu_ps(pd + 1 * F, Activate<type>(sum1, _params, 0));
							}
							for (; dx < bodyX; ++dx, pd += F)
							{
								__m512 sum = _bias;
								const float* pw = weight;
								for (size_t ky = 0; ky < p.kernelY; ++ky)
								{
									size_t sy = dy * strideY + ky - padY;
									const float* ps = src + ((sy & srcM) * p.srcW + dx * strideX - padX) * F;
									for (size_t kx = 0; kx < p.kernelX; ++kx, ps += F, pw += F)
									{
										__m512 w0 = _mm512_loadu_ps(pw);
										sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), w0, sum);
									}
								}
								_mm512_storeu_ps(pd, Activate<type>(sum, _params, 0));
							}
							for (; dx < p.dstW; ++dx, pd += F)
							{
								__m512 sum = _bias;
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
											sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
										}
									}
								}
								_mm512_storeu_ps(pd, Activate<type>(sum, _params, 0));
							}
						}
						else
						{
							for (size_t dx = 0; dx < p.dstW; ++dx, pd += F)
							{
								__m512 sum = _bias;
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
												sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
											}
										}
									}
								}
								_mm512_storeu_ps(pd, Activate<type>(sum, _params, 0));
							}
						}
					}
					src += srcS;
					dst += dstS;
					weight += weightS;
				}
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x2(
				const float* src0, const float* src1, const __m512* weight, const __m512& bias, const __m512* params, float* dst)
			{
				__m512 sum0 = bias, sum1 = _mm512_setzero_ps();
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * F), weight[0], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * F), weight[1], sum1);
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * F), weight[3], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * F), weight[4], sum1);
				_mm512_storeu_ps(dst, Activate<type>(_mm512_add_ps(sum0, sum1), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
				const float* src0, const float* src1, const __m512* weight, const __m512& bias, const __m512* params, float* dst)
			{
				__m512 sum0 = bias, sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * F), weight[0], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * F), weight[1], sum1);
				sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 2 * F), weight[2], sum2);
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * F), weight[3], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * F), weight[4], sum1);
				sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 2 * F), weight[5], sum2);
				_mm512_storeu_ps(dst, Activate<type>(_mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
				const float* src0, const float* src1, const float* src2, const __m512* weight, const __m512& bias, const __m512* params, float* dst)
			{
				__m512 sum0 = bias, sum1 = _mm512_setzero_ps();
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * F), weight[0], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * F), weight[1], sum1);
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * F), weight[3], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * F), weight[4], sum1);
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 0 * F), weight[6], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 1 * F), weight[7], sum1);
				_mm512_storeu_ps(dst, Activate<type>(_mm512_add_ps(sum0, sum1), params, 0));
			}

			template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
				const float* src0, const float* src1, const float* src2, const __m512* weight, const __m512& bias, const __m512* params, float* dst)
			{
				__m512 sum0 = bias, sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 0 * F), weight[0], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 1 * F), weight[1], sum1);
				sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + 2 * F), weight[2], sum2);
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 0 * F), weight[3], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 1 * F), weight[4], sum1);
				sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + 2 * F), weight[5], sum2);
				sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 0 * F), weight[6], sum0);
				sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 1 * F), weight[7], sum1);
				sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src2 + 2 * F), weight[8], sum2);
				_mm512_storeu_ps(dst, Activate<type>(_mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), params, 0));
			}

			template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
			{
				size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
				size_t srcW = p.srcW * F, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F;
				size_t srcM = (bufH[0] - 1), dstM = (bufH[1] - 1), srcS = bufH[0] * srcW, dstS = bufH[1] * dstW;
				size_t xStep = F * p.strideX, xStep0 = (p.strideX - p.padX) * F;
				size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

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

			template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x12(const float* src, size_t srcC, size_t srcS,
				const float* weight, const __m512* bias, const __m512* params, float* dst, size_t dstC, const __mmask16 tails[2], int first)
			{
				__m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1, s0, w0, w1;
				if (tails[1])
				{
					if (first)
					{
						d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
						d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
						d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
						d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
						d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
						d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
						d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
						d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
						d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
						d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
						da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
						db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
					}
					else
					{
						d00 = _mm512_loadu_ps(dst + 0x0 * dstC + 0), d01 = _mm512_loadu_ps(dst + 0x0 * dstC + F);
						d10 = _mm512_loadu_ps(dst + 0x1 * dstC + 0), d11 = _mm512_loadu_ps(dst + 0x1 * dstC + F);
						d20 = _mm512_loadu_ps(dst + 0x2 * dstC + 0), d21 = _mm512_loadu_ps(dst + 0x2 * dstC + F);
						d30 = _mm512_loadu_ps(dst + 0x3 * dstC + 0), d31 = _mm512_loadu_ps(dst + 0x3 * dstC + F);
						d40 = _mm512_loadu_ps(dst + 0x4 * dstC + 0), d41 = _mm512_loadu_ps(dst + 0x4 * dstC + F);
						d50 = _mm512_loadu_ps(dst + 0x5 * dstC + 0), d51 = _mm512_loadu_ps(dst + 0x5 * dstC + F);
						d60 = _mm512_loadu_ps(dst + 0x6 * dstC + 0), d61 = _mm512_loadu_ps(dst + 0x6 * dstC + F);
						d70 = _mm512_loadu_ps(dst + 0x7 * dstC + 0), d71 = _mm512_loadu_ps(dst + 0x7 * dstC + F);
						d80 = _mm512_loadu_ps(dst + 0x8 * dstC + 0), d81 = _mm512_loadu_ps(dst + 0x8 * dstC + F);
						d90 = _mm512_loadu_ps(dst + 0x9 * dstC + 0), d91 = _mm512_loadu_ps(dst + 0x9 * dstC + F);
						da0 = _mm512_loadu_ps(dst + 0xa * dstC + 0), da1 = _mm512_loadu_ps(dst + 0xa * dstC + F);
						db0 = _mm512_loadu_ps(dst + 0xb * dstC + 0), db1 = _mm512_loadu_ps(dst + 0xb * dstC + F);
					}
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm512_loadu_ps(weight + 0);
							w1 = _mm512_loadu_ps(weight + F);
							s0 = _mm512_set1_ps(src[i + 0 * F]);
							d00 = _mm512_fmadd_ps(s0, w0, d00);
							d01 = _mm512_fmadd_ps(s0, w1, d01);
							s0 = _mm512_set1_ps(src[i + 1 * F]);
							d10 = _mm512_fmadd_ps(s0, w0, d10);
							d11 = _mm512_fmadd_ps(s0, w1, d11);
							s0 = _mm512_set1_ps(src[i + 2 * F]);
							d20 = _mm512_fmadd_ps(s0, w0, d20);
							d21 = _mm512_fmadd_ps(s0, w1, d21);
							s0 = _mm512_set1_ps(src[i + 3 * F]);
							d30 = _mm512_fmadd_ps(s0, w0, d30);
							d31 = _mm512_fmadd_ps(s0, w1, d31);
							s0 = _mm512_set1_ps(src[i + 4 * F]);
							d40 = _mm512_fmadd_ps(s0, w0, d40);
							d41 = _mm512_fmadd_ps(s0, w1, d41);
							s0 = _mm512_set1_ps(src[i + 5 * F]);
							d50 = _mm512_fmadd_ps(s0, w0, d50);
							d51 = _mm512_fmadd_ps(s0, w1, d51);
							s0 = _mm512_set1_ps(src[i + 6 * F]);
							d60 = _mm512_fmadd_ps(s0, w0, d60);
							d61 = _mm512_fmadd_ps(s0, w1, d61);
							s0 = _mm512_set1_ps(src[i + 7 * F]);
							d70 = _mm512_fmadd_ps(s0, w0, d70);
							d71 = _mm512_fmadd_ps(s0, w1, d71);
							s0 = _mm512_set1_ps(src[i + 8 * F]);
							d80 = _mm512_fmadd_ps(s0, w0, d80);
							d81 = _mm512_fmadd_ps(s0, w1, d81);
							s0 = _mm512_set1_ps(src[i + 9 * F]);
							d90 = _mm512_fmadd_ps(s0, w0, d90);
							d91 = _mm512_fmadd_ps(s0, w1, d91);
							s0 = _mm512_set1_ps(src[i + 10 * F]);
							da0 = _mm512_fmadd_ps(s0, w0, da0);
							da1 = _mm512_fmadd_ps(s0, w1, da1);
							s0 = _mm512_set1_ps(src[i + 11 * F]);
							db0 = _mm512_fmadd_ps(s0, w0, db0);
							db1 = _mm512_fmadd_ps(s0, w1, db1);
						}
						src += srcS;
					}
					Term<term>::template Save<type, 0>(dst + 0, d00, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d10, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d20, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d30, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d31, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d40, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d41, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d50, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d51, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d60, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d61, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d70, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d71, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d80, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d81, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d90, bias, params);
					Term<term>::template Save<type, 1>(dst + F, d91, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, da0, bias, params);
					Term<term>::template Save<type, 1>(dst + F, da1, bias, params, tails[1]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, db0, bias, params);
					Term<term>::template Save<type, 1>(dst + F, db1, bias, params, tails[1]);
				}
				else
				{
					if (first)
					{
						d00 = _mm512_setzero_ps();
						d10 = _mm512_setzero_ps();
						d20 = _mm512_setzero_ps();
						d30 = _mm512_setzero_ps();
						d40 = _mm512_setzero_ps();
						d50 = _mm512_setzero_ps();
						d60 = _mm512_setzero_ps();
						d70 = _mm512_setzero_ps();
						d80 = _mm512_setzero_ps();
						d90 = _mm512_setzero_ps();
						da0 = _mm512_setzero_ps();
						db0 = _mm512_setzero_ps();
					}
					else
					{
						d00 = _mm512_loadu_ps(dst + 0x0 * dstC + 0);
						d10 = _mm512_loadu_ps(dst + 0x1 * dstC + 0);
						d20 = _mm512_loadu_ps(dst + 0x2 * dstC + 0);
						d30 = _mm512_loadu_ps(dst + 0x3 * dstC + 0);
						d40 = _mm512_loadu_ps(dst + 0x4 * dstC + 0);
						d50 = _mm512_loadu_ps(dst + 0x5 * dstC + 0);
						d60 = _mm512_loadu_ps(dst + 0x6 * dstC + 0);
						d70 = _mm512_loadu_ps(dst + 0x7 * dstC + 0);
						d80 = _mm512_loadu_ps(dst + 0x8 * dstC + 0);
						d90 = _mm512_loadu_ps(dst + 0x9 * dstC + 0);
						da0 = _mm512_loadu_ps(dst + 0xa * dstC + 0);
						db0 = _mm512_loadu_ps(dst + 0xb * dstC + 0);
					}
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm512_loadu_ps(weight + 0);
							s0 = _mm512_set1_ps(src[i + 0 * F]);
							d00 = _mm512_fmadd_ps(s0, w0, d00);
							s0 = _mm512_set1_ps(src[i + 1 * F]);
							d10 = _mm512_fmadd_ps(s0, w0, d10);
							s0 = _mm512_set1_ps(src[i + 2 * F]);
							d20 = _mm512_fmadd_ps(s0, w0, d20);
							s0 = _mm512_set1_ps(src[i + 3 * F]);
							d30 = _mm512_fmadd_ps(s0, w0, d30);
							s0 = _mm512_set1_ps(src[i + 4 * F]);
							d40 = _mm512_fmadd_ps(s0, w0, d40);
							s0 = _mm512_set1_ps(src[i + 5 * F]);
							d50 = _mm512_fmadd_ps(s0, w0, d50);
							s0 = _mm512_set1_ps(src[i + 6 * F]);
							d60 = _mm512_fmadd_ps(s0, w0, d60);
							s0 = _mm512_set1_ps(src[i + 7 * F]);
							d70 = _mm512_fmadd_ps(s0, w0, d70);
							s0 = _mm512_set1_ps(src[i + 8 * F]);
							d80 = _mm512_fmadd_ps(s0, w0, d80);
							s0 = _mm512_set1_ps(src[i + 9 * F]);
							d90 = _mm512_fmadd_ps(s0, w0, d90);
							s0 = _mm512_set1_ps(src[i + 10 * F]);
							da0 = _mm512_fmadd_ps(s0, w0, da0);
							s0 = _mm512_set1_ps(src[i + 11 * F]);
							db0 = _mm512_fmadd_ps(s0, w0, db0);
						}
						src += srcS;
					}
					Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d60, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d70, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d80, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, d90, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, da0, bias, params, tails[0]);
					dst += dstC;
					Term<term>::template Save<type, 0>(dst + 0, db0, bias, params, tails[0]);
				}
			}

			template<TermType term, SimdConvolutionActivationType type, int M> void OutputConvolution_2xM(const float* src, size_t srcC, size_t srcS,
				const float* weight, const __m512* bias, const __m512* params, float* dst, size_t dstC, const __mmask16 tails[2], int first)
			{
				__m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
				if (tails[1])
				{
					if (first)
					{
						if (M > 0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
						if (M > 1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
						if (M > 2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
						if (M > 3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
						if (M > 4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
						if (M > 5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
					}
					else
					{
						if (M > 0) d00 = _mm512_loadu_ps(dst + 0x0 * dstC + 0), d01 = _mm512_loadu_ps(dst + 0x0 * dstC + F);
						if (M > 1) d10 = _mm512_loadu_ps(dst + 0x1 * dstC + 0), d11 = _mm512_loadu_ps(dst + 0x1 * dstC + F);
						if (M > 2) d20 = _mm512_loadu_ps(dst + 0x2 * dstC + 0), d21 = _mm512_loadu_ps(dst + 0x2 * dstC + F);
						if (M > 3) d30 = _mm512_loadu_ps(dst + 0x3 * dstC + 0), d31 = _mm512_loadu_ps(dst + 0x3 * dstC + F);
						if (M > 4) d40 = _mm512_loadu_ps(dst + 0x4 * dstC + 0), d41 = _mm512_loadu_ps(dst + 0x4 * dstC + F);
						if (M > 5) d50 = _mm512_loadu_ps(dst + 0x5 * dstC + 0), d51 = _mm512_loadu_ps(dst + 0x5 * dstC + F);
					}
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm512_loadu_ps(weight + 0);
							w1 = _mm512_loadu_ps(weight + F);
							if (M > 0) s0 = _mm512_set1_ps(src[i + 0 * F]), d00 = _mm512_fmadd_ps(s0, w0, d00), d01 = _mm512_fmadd_ps(s0, w1, d01);
							if (M > 1) s0 = _mm512_set1_ps(src[i + 1 * F]), d10 = _mm512_fmadd_ps(s0, w0, d10), d11 = _mm512_fmadd_ps(s0, w1, d11);
							if (M > 2) s0 = _mm512_set1_ps(src[i + 2 * F]), d20 = _mm512_fmadd_ps(s0, w0, d20), d21 = _mm512_fmadd_ps(s0, w1, d21);
							if (M > 3) s0 = _mm512_set1_ps(src[i + 3 * F]), d30 = _mm512_fmadd_ps(s0, w0, d30), d31 = _mm512_fmadd_ps(s0, w1, d31);
							if (M > 4) s0 = _mm512_set1_ps(src[i + 4 * F]), d40 = _mm512_fmadd_ps(s0, w0, d40), d41 = _mm512_fmadd_ps(s0, w1, d41);
							if (M > 5) s0 = _mm512_set1_ps(src[i + 5 * F]), d50 = _mm512_fmadd_ps(s0, w0, d50), d51 = _mm512_fmadd_ps(s0, w1, d51);
						}
						src += srcS;
					}
					if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params), Term<term>::template Save<type, 1>(dst + F, d01, bias, params, tails[1]), dst += dstC;
					if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params), Term<term>::template Save<type, 1>(dst + F, d11, bias, params, tails[1]), dst += dstC;
					if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params), Term<term>::template Save<type, 1>(dst + F, d21, bias, params, tails[1]), dst += dstC;
					if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params), Term<term>::template Save<type, 1>(dst + F, d31, bias, params, tails[1]), dst += dstC;
					if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params), Term<term>::template Save<type, 1>(dst + F, d41, bias, params, tails[1]), dst += dstC;
					if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params), Term<term>::template Save<type, 1>(dst + F, d51, bias, params, tails[1]), dst += dstC;
				}
				else
				{
					if (first)
					{
						if (M > 0) d00 = _mm512_setzero_ps();
						if (M > 1) d10 = _mm512_setzero_ps();
						if (M > 2) d20 = _mm512_setzero_ps();
						if (M > 3) d30 = _mm512_setzero_ps();
						if (M > 4) d40 = _mm512_setzero_ps();
						if (M > 5) d50 = _mm512_setzero_ps();
					}
					else
					{
						if (M > 0) d00 = _mm512_loadu_ps(dst + 0x0 * dstC + 0);
						if (M > 1) d10 = _mm512_loadu_ps(dst + 0x1 * dstC + 0);
						if (M > 2) d20 = _mm512_loadu_ps(dst + 0x2 * dstC + 0);
						if (M > 3) d30 = _mm512_loadu_ps(dst + 0x3 * dstC + 0);
						if (M > 4) d40 = _mm512_loadu_ps(dst + 0x4 * dstC + 0);
						if (M > 5) d50 = _mm512_loadu_ps(dst + 0x5 * dstC + 0);
					}
					for (size_t c = 0; c < srcC; c += F)
					{
						size_t n = Simd::Min(F, srcC - c);
						for (size_t i = 0; i < n; ++i, weight += DF)
						{
							w0 = _mm512_loadu_ps(weight + 0);
							if (M > 0) s0 = _mm512_set1_ps(src[i + 0 * F]), d00 = _mm512_fmadd_ps(s0, w0, d00);
							if (M > 1) s0 = _mm512_set1_ps(src[i + 1 * F]), d10 = _mm512_fmadd_ps(s0, w0, d10);
							if (M > 2) s0 = _mm512_set1_ps(src[i + 2 * F]), d20 = _mm512_fmadd_ps(s0, w0, d20);
							if (M > 3) s0 = _mm512_set1_ps(src[i + 3 * F]), d30 = _mm512_fmadd_ps(s0, w0, d30);
							if (M > 4) s0 = _mm512_set1_ps(src[i + 4 * F]), d40 = _mm512_fmadd_ps(s0, w0, d40);
							if (M > 5) s0 = _mm512_set1_ps(src[i + 5 * F]), d50 = _mm512_fmadd_ps(s0, w0, d50);
						}
						src += srcS;
					}
					if (M > 0) Term<term>::template Save<type, 0>(dst + 0, d00, bias, params, tails[0]), dst += dstC;
					if (M > 1) Term<term>::template Save<type, 0>(dst + 0, d10, bias, params, tails[0]), dst += dstC;
					if (M > 2) Term<term>::template Save<type, 0>(dst + 0, d20, bias, params, tails[0]), dst += dstC;
					if (M > 3) Term<term>::template Save<type, 0>(dst + 0, d30, bias, params, tails[0]), dst += dstC;
					if (M > 4) Term<term>::template Save<type, 0>(dst + 0, d40, bias, params, tails[0]), dst += dstC;
					if (M > 5) Term<term>::template Save<type, 0>(dst + 0, d50, bias, params, tails[0]), dst += dstC;
				}
			}

			typedef void(*OutputConvolution_2xM_Ptr)(const float* src, size_t srcC, size_t srcS, const float* weight, 
				const __m512* bias, const __m512* params, float* dst, size_t dstC, const __mmask16 tails[2], int first);

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

			template<TermType term, SimdConvolutionActivationType type> void OutputConvolution(const float* src, const SimdConvolutionParameters& p,
				size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
			{
				assert(p.group == 1 && p.kernelY == 1 && p.strideY == 1);
				size_t srcH = p.srcH, srcW = p.srcW, dstW = p.dstW, dstC = p.dstC;
				size_t srcM = (bufH[1] - 1), srcS = bufH[1] * srcW * F;
				size_t yInt = Simd::Max(yBeg, yEnd & (~srcM)), nBeg = yBeg * srcW, nInt = yInt * srcW, nEnd = yEnd * srcW;
				size_t nInt6 = AlignLoAny(nInt - nBeg, 6) + nBeg, nEnd6 = AlignLoAny(nEnd - nInt, 6) + nInt, nIntTail = nInt - nInt6, nEndTail = nEnd - nEnd6;
				size_t nInt12 = AlignLoAny(nInt - nBeg, 12) + nBeg, nEnd12 = AlignLoAny(nEnd - nInt, 12) + nInt;
				OutputConvolution_2xM_Ptr bodyInt = GetOutputConvolution_2xM<term, type>(6);
				OutputConvolution_2xM_Ptr tailInt = GetOutputConvolution_2xM<term, type>(nIntTail);
				OutputConvolution_2xM_Ptr bodyEnd = GetOutputConvolution_2xM<term, type>(6);
				OutputConvolution_2xM_Ptr tailEnd = GetOutputConvolution_2xM<term, type>(nEndTail);

				__m512 _params[2], _bias[2];
				_params[0] = _mm512_set1_ps(params[0]);
				if (type == SimdConvolutionActivationRestrictRange ||
					type == SimdConvolutionActivationHswish ||
					type == SimdConvolutionActivationHardSigmoid)
					_params[1] = _mm512_set1_ps(params[1]);

				dst += yBeg * p.dstW * p.dstC;
				size_t dc = 0;
				for (; dc < dstC; dc += DF)
				{
					size_t tail = Simd::Min(DF, dstC - dc);
					__mmask16 tails[2] = { TailMask16(tail), TailMask16(tail - F) };
					_bias[0] = _mm512_loadu_ps(bias + dc + 0);
					_bias[1] = _mm512_loadu_ps(bias + dc + F);
					if (type == ::SimdConvolutionActivationPrelu)
					{
						_params[0] = _mm512_loadu_ps(params + dc + 0);
						_params[1] = _mm512_loadu_ps(params + dc + F);
					}
					float* pDst = dst + dc;
					const float* src0 = src + (yBeg & srcM) * srcW * F;
					const float* src1 = src + (yInt & srcM) * srcW * F;
					size_t dn = nBeg;
					for (; dn < nInt12; dn += 12, pDst += 12 * dstC, src0 += 12 * F)
						OutputConvolution_2x12<term, type>(src0, srcC, srcS, weight, _bias, _params, pDst, dstC, tails, first);
					for (; dn < nInt6; dn += 6, pDst += 6 * dstC, src0 += 6 * F)
						bodyInt(src0, srcC, srcS, weight, _bias, _params, pDst, dstC, tails, first);
					if (nIntTail)
						tailInt(src0, srcC, srcS, weight, _bias, _params, pDst, dstC, tails, first), dn += nIntTail, pDst += nIntTail * dstC, src0 += nIntTail * F;
					for (; dn < nEnd12; dn += 12, pDst += 12 * dstC, src1 += 12 * F)
						OutputConvolution_2x12<term, type>(src1, srcC, srcS, weight, _bias, _params, pDst, dstC, tails, first);
					for (; dn < nEnd6; dn += 6, pDst += 6 * dstC, src1 += 6 * F)
						bodyEnd(src1, srcC, srcS, weight, _bias, _params, pDst, dstC, tails, first);
					if (nEndTail)
						tailEnd(src1, srcC, srcS, weight, _bias, _params, pDst, dstC, tails, first), dn += nEndTail, pDst += nEndTail * dstC, src1 += nEndTail * F;
					weight += srcC * DF;
				}
			}

			//---------------------------------------------------------------------

			template <SimdConvolutionActivationType type> void Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32fCdc::ConvolutionPtr* c)
			{
				switch (t)
				{
				case 0:
					if (p.conv[i].kernelY == 1 && p.conv[i].strideY == 1)
						c[i + 0] = InputConvolution1x1<type>;
					else
						c[i + 0] = InputConvolution<type>;
					break;
				case 1:
					if (p.conv[i].kernelY == 3)
						c[i + 0] = DepthwiseConvolution3x3<type>;
					else
						c[i + 0] = DepthwiseConvolution<type>;
					break;
				case 2:
					c[i + 0] = OutputConvolution<TermLast, type>;
					c[i + 1] = OutputConvolution<TermInterim, SimdConvolutionActivationIdentity>;
					break;
				default:
					assert(0);
				}
			}
		}

		//---------------------------------------------------------------------

		SynetMergedConvolution32fCdc::SynetMergedConvolution32fCdc(const MergConvParam32f& p)
			: Avx2::SynetMergedConvolution32fCdc(p)
		{
			for (size_t i = 0; i < _param.count; ++i)
				Set(p, i, i, _convolution);
			SetSize(Base::AlgCacheL1(), Base::AlgCacheL2() / 2, Base::AlgCacheL3(), Avx512f::F);
		}

		void SynetMergedConvolution32fCdc::Set(const MergConvParam32f& p, size_t t, size_t i, SynetMergedConvolution32f::ConvolutionPtr* c)
		{
			switch (p.conv[i].activation)
			{
			case SimdConvolutionActivationIdentity: Cdc::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
			case SimdConvolutionActivationRelu: Cdc::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
			case SimdConvolutionActivationLeakyRelu: Cdc::Set<SimdConvolutionActivationPrelu>(p, t, i, c); break;
			case SimdConvolutionActivationRestrictRange: Cdc::Set<SimdConvolutionActivationRestrictRange>(p, t, i, c); break;
			case SimdConvolutionActivationPrelu: Cdc::Set<SimdConvolutionActivationPrelu>(p, t, i, c); break;
			case SimdConvolutionActivationElu: Cdc::Set<SimdConvolutionActivationElu>(p, t, i, c); break;
			case SimdConvolutionActivationHswish: Cdc::Set<SimdConvolutionActivationHswish>(p, t, i, c); break;
			case SimdConvolutionActivationMish: Cdc::Set<SimdConvolutionActivationMish>(p, t, i, c); break;
			case SimdConvolutionActivationHardSigmoid: Cdc::Set<SimdConvolutionActivationHardSigmoid>(p, t, i, c); break;
			case SimdConvolutionActivationSwish: Cdc::Set<SimdConvolutionActivationSwish>(p, t, i, c); break;
			default: assert(0);
			}
		}

		//---------------------------------------------------------------------

		void* SynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add)
		{
			MergConvParam32f param(batch, convs, count, add);
			if (!param.Valid())
				return NULL;
			if (SynetMergedConvolution32fCdc::Preferable(param))
			{
				if (param.conv[1].dstC <= HF && param.conv[2].dstC <= HF)
					return new Avx2::SynetMergedConvolution32fCdc(param);
				else
					return new Avx512f::SynetMergedConvolution32fCdc(param);
			}
			else if (SynetMergedConvolution32fCd::Preferable(param))
			{
				if (param.conv[1].dstC <= HF)
					return new Avx2::SynetMergedConvolution32fCd(param);
				else
					return new Avx512f::SynetMergedConvolution32fCd(param);
			}
			else if (SynetMergedConvolution32fDc::Preferable(param))
			{
				if (param.conv[0].dstC <= HF || param.conv[1].dstC <= HF)
					return new Avx2::SynetMergedConvolution32fDc(param);
				else
					return new Avx512f::SynetMergedConvolution32fDc(param);
			}
			else
				return new Base::SynetMergedConvolution32f(param);
		}
	}
#endif//SIMD_AVX512f_ENABLE
}
