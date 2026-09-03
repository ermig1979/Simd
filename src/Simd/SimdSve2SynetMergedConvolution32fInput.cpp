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
		namespace
		{

		SIMD_INLINE svfloat32_t Erf(const svbool_t& mask, svfloat32_t x)
		{
			const svfloat32_t _1 = svdup_n_f32(1.0f);
			svfloat32_t a = svmin_f32_x(mask, svabs_f32_x(mask, x), svdup_n_f32(9.0f));
			svfloat32_t p = svdup_n_f32(0.0000430638f);
			p = svmla_f32_x(mask, svdup_n_f32(0.0002765672f), a, p);
			p = svmla_f32_x(mask, svdup_n_f32(0.0001520143f), a, p);
			p = svmla_f32_x(mask, svdup_n_f32(0.0092705272f), a, p);
			p = svmla_f32_x(mask, svdup_n_f32(0.0422820123f), a, p);
			p = svmla_f32_x(mask, svdup_n_f32(0.0705230784f), a, p);
			p = svmla_f32_x(mask, _1, a, p);
			p = svmul_f32_x(mask, p, p);
			p = svmul_f32_x(mask, p, p);
			p = svmul_f32_x(mask, p, p);
			p = svmul_f32_x(mask, p, p);
			svfloat32_t r = svsub_f32_x(mask, _1, svdiv_f32_x(mask, _1, p));
			return svsel_f32(svcmplt_n_f32(mask, x, 0.0f), svneg_f32_x(mask, r), r);
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask);

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationIdentity>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return value;
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationRelu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return svmax_n_f32_x(mask, value, 0.0f);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationLeakyRelu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), param0, svmin_n_f32_x(mask, value, 0.0f));
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationRestrictRange>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return svmin_f32_x(mask, svmax_f32_x(mask, param0, value), param1);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationPrelu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), param0, svmin_n_f32_x(mask, value, 0.0f));
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationElu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			svfloat32_t neg = svmul_f32_x(mask, param0, svsub_n_f32_x(mask, Exponent(mask, value), 1.0f));
			return svsel_f32(svcmplt_n_f32(mask, value, 0.0f), neg, value);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHswish>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			svfloat32_t upper = svmin_f32_x(mask, value, param0);
			svfloat32_t positive = svmax_n_f32_x(mask, svadd_f32_x(mask, upper, param0), 0.0f);
			return svmul_f32_x(mask, svmul_f32_x(mask, positive, param1), value);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationMish>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			svfloat32_t exp = svmin_f32_x(mask, Exponent(mask, value), param0);
			return svmul_f32_x(mask, value, Tanh(mask, Logarithm(mask, svadd_n_f32_x(mask, exp, 1.0f))));
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHardSigmoid>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return svmax_n_f32_x(mask, svmin_n_f32_x(mask, svmla_f32_x(mask, param1, value, param0), 1.0f), 0.0f);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationSwish>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return svdiv_f32_x(mask, value, svadd_n_f32_x(mask, Exponent(mask, svmul_f32_x(mask, svneg_f32_x(mask, value), param0)), 1.0f));
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationGelu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			svfloat32_t t = svmul_n_f32_x(mask, value, 0.70710678118654752440f);
			return svmul_f32_x(mask, svmul_n_f32_x(mask, t, 0.70710678118654752440f), svadd_n_f32_x(mask, Erf(mask, t), 1.0f));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate0(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return Activate<type>(value, param0, param1, mask);
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate1(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return Activate<type>(value, param0, param1, mask);
		}

		template<> SIMD_INLINE svfloat32_t Activate1<SimdConvolutionActivationPrelu>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return Activate<SimdConvolutionActivationPrelu>(value, param1, param1, mask);
		}

		//-------------------------------------------------------------------------------------------------

		template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution1x1_2x6(const float* src0, size_t srcC,
			const float* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst0, float* dst1)
		{
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
			d00 = bias0, d01 = bias1;
			d10 = bias0, d11 = bias1;
			d20 = bias0, d21 = bias1;
			d30 = bias0, d31 = bias1;
			d40 = bias0, d41 = bias1;
			d50 = bias0, d51 = bias1;
			const float* src1 = src0 + 1 * srcC;
			const float* src2 = src0 + 2 * srcC;
			const float* src3 = src0 + 3 * srcC;
			const float* src4 = src0 + 4 * srcC;
			const float* src5 = src0 + 5 * srcC;
			for (size_t sc = 0; sc < srcC; ++sc)
			{
				w0 = svld1_f32(mask, weight + 0);
				w1 = svld1_f32(mask, weight + F);
				s0 = svdup_n_f32(src0[sc]);
				d00 = svmla_f32_x(mask, d00, s0, w0);
				d01 = svmla_f32_x(mask, d01, s0, w1);
				s0 = svdup_n_f32(src1[sc]);
				d10 = svmla_f32_x(mask, d10, s0, w0);
				d11 = svmla_f32_x(mask, d11, s0, w1);
				s0 = svdup_n_f32(src2[sc]);
				d20 = svmla_f32_x(mask, d20, s0, w0);
				d21 = svmla_f32_x(mask, d21, s0, w1);
				s0 = svdup_n_f32(src3[sc]);
				d30 = svmla_f32_x(mask, d30, s0, w0);
				d31 = svmla_f32_x(mask, d31, s0, w1);
				s0 = svdup_n_f32(src4[sc]);
				d40 = svmla_f32_x(mask, d40, s0, w0);
				d41 = svmla_f32_x(mask, d41, s0, w1);
				s0 = svdup_n_f32(src5[sc]);
				d50 = svmla_f32_x(mask, d50, s0, w0);
				d51 = svmla_f32_x(mask, d51, s0, w1);
				weight += DF;
			}
			svst1_f32(mask, dst0 + 0 * F, Activate0<type>(d00, param0, param1, mask));
			svst1_f32(mask, dst1 + 0 * F, Activate1<type>(d01, param0, param1, mask));
			svst1_f32(mask, dst0 + 1 * F, Activate0<type>(d10, param0, param1, mask));
			svst1_f32(mask, dst1 + 1 * F, Activate1<type>(d11, param0, param1, mask));
			svst1_f32(mask, dst0 + 2 * F, Activate0<type>(d20, param0, param1, mask));
			svst1_f32(mask, dst1 + 2 * F, Activate1<type>(d21, param0, param1, mask));
			svst1_f32(mask, dst0 + 3 * F, Activate0<type>(d30, param0, param1, mask));
			svst1_f32(mask, dst1 + 3 * F, Activate1<type>(d31, param0, param1, mask));
			svst1_f32(mask, dst0 + 4 * F, Activate0<type>(d40, param0, param1, mask));
			svst1_f32(mask, dst1 + 4 * F, Activate1<type>(d41, param0, param1, mask));
			svst1_f32(mask, dst0 + 5 * F, Activate0<type>(d50, param0, param1, mask));
			svst1_f32(mask, dst1 + 5 * F, Activate1<type>(d51, param0, param1, mask));
		}

		template<SimdConvolutionActivationType type, int M> SIMD_INLINE void InputConvolution1x1_2xM(const float* src0, size_t srcC,
			const float* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst0, float* dst1)
		{
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
			if (M > 0) d00 = bias0, d01 = bias1;
			if (M > 1) d10 = bias0, d11 = bias1;
			if (M > 2) d20 = bias0, d21 = bias1;
			if (M > 3) d30 = bias0, d31 = bias1;
			if (M > 4) d40 = bias0, d41 = bias1;
			if (M > 5) d50 = bias0, d51 = bias1;
			const float* src1 = src0 + 1 * srcC;
			const float* src2 = src0 + 2 * srcC;
			const float* src3 = src0 + 3 * srcC;
			const float* src4 = src0 + 4 * srcC;
			const float* src5 = src0 + 5 * srcC;
			for (size_t sc = 0; sc < srcC; ++sc)
			{
				w0 = svld1_f32(mask, weight + 0);
				w1 = svld1_f32(mask, weight + F);
				if (M > 0) s0 = svdup_n_f32(src0[sc]), d00 = svmla_f32_x(mask, d00, s0, w0), d01 = svmla_f32_x(mask, d01, s0, w1);
				if (M > 1) s0 = svdup_n_f32(src1[sc]), d10 = svmla_f32_x(mask, d10, s0, w0), d11 = svmla_f32_x(mask, d11, s0, w1);
				if (M > 2) s0 = svdup_n_f32(src2[sc]), d20 = svmla_f32_x(mask, d20, s0, w0), d21 = svmla_f32_x(mask, d21, s0, w1);
				if (M > 3) s0 = svdup_n_f32(src3[sc]), d30 = svmla_f32_x(mask, d30, s0, w0), d31 = svmla_f32_x(mask, d31, s0, w1);
				if (M > 4) s0 = svdup_n_f32(src4[sc]), d40 = svmla_f32_x(mask, d40, s0, w0), d41 = svmla_f32_x(mask, d41, s0, w1);
				if (M > 5) s0 = svdup_n_f32(src5[sc]), d50 = svmla_f32_x(mask, d50, s0, w0), d51 = svmla_f32_x(mask, d51, s0, w1);
				weight += DF;
			}
			if (M > 0) svst1_f32(mask, dst0 + 0 * F, Activate0<type>(d00, param0, param1, mask)), svst1_f32(mask, dst1 + 0 * F, Activate1<type>(d01, param0, param1, mask));
			if (M > 1) svst1_f32(mask, dst0 + 1 * F, Activate0<type>(d10, param0, param1, mask)), svst1_f32(mask, dst1 + 1 * F, Activate1<type>(d11, param0, param1, mask));
			if (M > 2) svst1_f32(mask, dst0 + 2 * F, Activate0<type>(d20, param0, param1, mask)), svst1_f32(mask, dst1 + 2 * F, Activate1<type>(d21, param0, param1, mask));
			if (M > 3) svst1_f32(mask, dst0 + 3 * F, Activate0<type>(d30, param0, param1, mask)), svst1_f32(mask, dst1 + 3 * F, Activate1<type>(d31, param0, param1, mask));
			if (M > 4) svst1_f32(mask, dst0 + 4 * F, Activate0<type>(d40, param0, param1, mask)), svst1_f32(mask, dst1 + 4 * F, Activate1<type>(d41, param0, param1, mask));
			if (M > 5) svst1_f32(mask, dst0 + 5 * F, Activate0<type>(d50, param0, param1, mask)), svst1_f32(mask, dst1 + 5 * F, Activate1<type>(d51, param0, param1, mask));
		}

		typedef void(*InputConvolution1x1_2xM_Ptr)(const float* src0, size_t srcC, const float* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst0, float* dst1);

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
			const float* weight, svfloat32_t bias0, svfloat32_t param0, svfloat32_t param1, float* dst0)
		{
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			svfloat32_t d00, d10, d20, d30, d40, d50, s0, w0;
			d00 = bias0;
			d10 = bias0;
			d20 = bias0;
			d30 = bias0;
			d40 = bias0;
			d50 = bias0;
			const float* src1 = src0 + 1 * srcC;
			const float* src2 = src0 + 2 * srcC;
			const float* src3 = src0 + 3 * srcC;
			const float* src4 = src0 + 4 * srcC;
			const float* src5 = src0 + 5 * srcC;
			for (size_t sc = 0; sc < srcC; ++sc)
			{
				w0 = svld1_f32(mask, weight + 0);
				s0 = svdup_n_f32(src0[sc]);
				d00 = svmla_f32_x(mask, d00, s0, w0);
				s0 = svdup_n_f32(src1[sc]);
				d10 = svmla_f32_x(mask, d10, s0, w0);
				s0 = svdup_n_f32(src2[sc]);
				d20 = svmla_f32_x(mask, d20, s0, w0);
				s0 = svdup_n_f32(src3[sc]);
				d30 = svmla_f32_x(mask, d30, s0, w0);
				s0 = svdup_n_f32(src4[sc]);
				d40 = svmla_f32_x(mask, d40, s0, w0);
				s0 = svdup_n_f32(src5[sc]);
				d50 = svmla_f32_x(mask, d50, s0, w0);
				weight += DF;
			}
			svst1_f32(mask, dst0 + 0 * F, Activate0<type>(d00, param0, param1, mask));
			svst1_f32(mask, dst0 + 1 * F, Activate0<type>(d10, param0, param1, mask));
			svst1_f32(mask, dst0 + 2 * F, Activate0<type>(d20, param0, param1, mask));
			svst1_f32(mask, dst0 + 3 * F, Activate0<type>(d30, param0, param1, mask));
			svst1_f32(mask, dst0 + 4 * F, Activate0<type>(d40, param0, param1, mask));
			svst1_f32(mask, dst0 + 5 * F, Activate0<type>(d50, param0, param1, mask));
		}

		template<SimdConvolutionActivationType type, int M> SIMD_INLINE void InputConvolution1x1_1xM(const float* src0, size_t srcC,
			const float* weight, svfloat32_t bias0, svfloat32_t param0, svfloat32_t param1, float* dst0)
		{
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			svfloat32_t d00, d10, d20, d30, d40, d50, s0, w0;
			if (M > 0) d00 = bias0;
			if (M > 1) d10 = bias0;
			if (M > 2) d20 = bias0;
			if (M > 3) d30 = bias0;
			if (M > 4) d40 = bias0;
			if (M > 5) d50 = bias0;
			const float* src1 = src0 + 1 * srcC;
			const float* src2 = src0 + 2 * srcC;
			const float* src3 = src0 + 3 * srcC;
			const float* src4 = src0 + 4 * srcC;
			const float* src5 = src0 + 5 * srcC;
			for (size_t sc = 0; sc < srcC; ++sc)
			{
				w0 = svld1_f32(mask, weight + 0);
				if (M > 0) s0 = svdup_n_f32(src0[sc]), d00 = svmla_f32_x(mask, d00, s0, w0);
				if (M > 1) s0 = svdup_n_f32(src1[sc]), d10 = svmla_f32_x(mask, d10, s0, w0);
				if (M > 2) s0 = svdup_n_f32(src2[sc]), d20 = svmla_f32_x(mask, d20, s0, w0);
				if (M > 3) s0 = svdup_n_f32(src3[sc]), d30 = svmla_f32_x(mask, d30, s0, w0);
				if (M > 4) s0 = svdup_n_f32(src4[sc]), d40 = svmla_f32_x(mask, d40, s0, w0);
				if (M > 5) s0 = svdup_n_f32(src5[sc]), d50 = svmla_f32_x(mask, d50, s0, w0);
				weight += DF;
			}
			if (M > 0) svst1_f32(mask, dst0 + 0 * F, Activate0<type>(d00, param0, param1, mask));
			if (M > 1) svst1_f32(mask, dst0 + 1 * F, Activate0<type>(d10, param0, param1, mask));
			if (M > 2) svst1_f32(mask, dst0 + 2 * F, Activate0<type>(d20, param0, param1, mask));
			if (M > 3) svst1_f32(mask, dst0 + 3 * F, Activate0<type>(d30, param0, param1, mask));
			if (M > 4) svst1_f32(mask, dst0 + 4 * F, Activate0<type>(d40, param0, param1, mask));
			if (M > 5) svst1_f32(mask, dst0 + 5 * F, Activate0<type>(d50, param0, param1, mask));
		}

		typedef void(*InputConvolution1x1_1xM_Ptr)(const float* src0, size_t srcC, const float* weight, svfloat32_t bias0, svfloat32_t param0, svfloat32_t param1, float* dst0);

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
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
			size_t dstM = (bufH[0] - 1), dstS = bufH[0] * dstW * F;
			size_t dstCDF = AlignLo(dstC, DF);
			svfloat32_t param0 = svdup_n_f32(params[0]), param1 = svdup_n_f32(0.0f);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				param1 = svdup_n_f32(params[1]);
			size_t yInt = Simd::Max(yBeg, yEnd & (~dstM)), nBeg = yBeg * dstW, nInt = yInt * dstW, nEnd = yEnd * dstW;
			size_t nInt6 = AlignLoAny(nInt - nBeg, 6) + nBeg, nEnd6 = AlignLoAny(nEnd - nInt, 6) + nInt, nIntTail = nInt - nInt6, nEndTail = nEnd - nEnd6;
			InputConvolution1x1_2xM_Ptr tailInt_2 = GetInputConvolution1x1_2xM<type>(nIntTail);
			InputConvolution1x1_2xM_Ptr tailEnd_2 = GetInputConvolution1x1_2xM<type>(nEndTail);

			size_t dc = 0;
			for (; dc < dstC; dc += DF)
			{
				svfloat32_t bias0 = bias ? svld1_f32(mask, bias + dc + 0) : svdup_n_f32(0.0f);
				svfloat32_t bias1 = bias ? svld1_f32(mask, bias + dc + F) : svdup_n_f32(0.0f);
				if (type == ::SimdConvolutionActivationPrelu)
				{
					param0 = svld1_f32(mask, params + dc + 0);
					param1 = svld1_f32(mask, params + dc + F);
				}
				const float* pS = src + yBeg * srcW * srcC;
				const float* pW = weight + dc * srcC;
				float* pD = dst + (dc / F) * dstS;
				float* dst0 = pD + (yBeg & dstM) * dstW * F;
				float* dst1 = pD + (yInt & dstM) * dstW * F;
				size_t dn = nBeg;
				if (dstC - dc > F)
				{
					for (; dn < nInt6; dn += 6, pS += 6 * srcC, dst0 += 6 * F)
						InputConvolution1x1_2x6<type>(pS, srcC, pW, bias0, bias1, param0, param1, dst0, dst0 + dstS);
					if (nIntTail)
						tailInt_2(pS, srcC, pW, bias0, bias1, param0, param1, dst0, dst0 + dstS), pS += nIntTail * srcC, dn += nIntTail;
					for (; dn < nEnd6; dn += 6, pS += 6 * srcC, dst1 += 6 * F)
						InputConvolution1x1_2x6<type>(pS, srcC, pW, bias0, bias1, param0, param1, dst1, dst1 + dstS);
					if (nEndTail)
						tailEnd_2(pS, srcC, pW, bias0, bias1, param0, param1, dst1, dst1 + dstS), pS += nEndTail * srcC, dn += nEndTail;
				}
				else
				{
					InputConvolution1x1_1xM_Ptr tailInt_1 = GetInputConvolution1x1_1xM<type>(nIntTail);
					InputConvolution1x1_1xM_Ptr tailEnd_1 = GetInputConvolution1x1_1xM<type>(nEndTail);
					for (; dn < nInt6; dn += 6, pS += 6 * srcC, dst0 += 6 * F)
						InputConvolution1x1_1x6<type>(pS, srcC, pW, bias0, param0, param1, dst0);
					if (nIntTail)
						tailInt_1(pS, srcC, pW, bias0, param0, param1, dst0), pS += nIntTail * srcC, dn += nIntTail;
					for (; dn < nEnd6; dn += 6, pS += 6 * srcC, dst1 += 6 * F)
						InputConvolution1x1_1x6<type>(pS, srcC, pW, bias0, param0, param1, dst1);
					if (nEndTail)
						tailEnd_1(pS, srcC, pW, bias0, param0, param1, dst1), pS += nEndTail * srcC, dn += nEndTail;
				}
			}
		}

		//---------------------------------------------------------------------

		template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution_2x1(const float* src0, const SimdConvolutionParameters& p,
			size_t kH, size_t kW, const float* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst0, float* dst1)
		{
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			svfloat32_t d00, d01, s0, w0, w1;
			d00 = bias0;
			d01 = bias1;
			size_t size = kW * p.srcC, tail = DF * (p.kernelX - kW) * p.srcC, stride = p.srcW * p.srcC;
			for (size_t ky = 0; ky < kH; ++ky)
			{
				for (size_t i = 0; i < size; ++i)
				{
					w0 = svld1_f32(mask, weight + 0);
					w1 = svld1_f32(mask, weight + F);
					s0 = svdup_n_f32(src0[i]);
					d00 = svmla_f32_x(mask, d00, s0, w0);
					d01 = svmla_f32_x(mask, d01, s0, w1);
					weight += DF;
				}
				weight += tail;
				src0 += stride;
			}
			svst1_f32(mask, dst0, Activate0<type>(d00, param0, param1, mask));
			svst1_f32(mask, dst1, Activate1<type>(d01, param0, param1, mask));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution_1x1(const float* src0, const SimdConvolutionParameters& p,
			size_t kH, size_t kW, const float* weight, svfloat32_t bias0, svfloat32_t param0, svfloat32_t param1, float* dst0)
		{
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			svfloat32_t d00, s0, w0;
			d00 = bias0;
			size_t size = kW * p.srcC, tail = DF * (p.kernelX - kW) * p.srcC, stride = p.srcW * p.srcC;
			for (size_t ky = 0; ky < kH; ++ky)
			{
				for (size_t i = 0; i < size; ++i)
				{
					w0 = svld1_f32(mask, weight + 0);
					s0 = svdup_n_f32(src0[i]);
					d00 = svmla_f32_x(mask, d00, s0, w0);
					weight += DF;
				}
				weight += tail;
				src0 += stride;
			}
			svst1_f32(mask, dst0, Activate0<type>(d00, param0, param1, mask));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution_2x6(const float* src0, const SimdConvolutionParameters& p,
			size_t kH, size_t kW, const float* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst0, float* dst1)
		{
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
			d00 = bias0, d01 = bias1;
			d10 = bias0, d11 = bias1;
			d20 = bias0, d21 = bias1;
			d30 = bias0, d31 = bias1;
			d40 = bias0, d41 = bias1;
			d50 = bias0, d51 = bias1;
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
					w0 = svld1_f32(mask, weight + 0);
					w1 = svld1_f32(mask, weight + F);
					s0 = svdup_n_f32(src0[offset]);
					d00 = svmla_f32_x(mask, d00, s0, w0);
					d01 = svmla_f32_x(mask, d01, s0, w1);
					s0 = svdup_n_f32(src1[offset]);
					d10 = svmla_f32_x(mask, d10, s0, w0);
					d11 = svmla_f32_x(mask, d11, s0, w1);
					s0 = svdup_n_f32(src2[offset]);
					d20 = svmla_f32_x(mask, d20, s0, w0);
					d21 = svmla_f32_x(mask, d21, s0, w1);
					s0 = svdup_n_f32(src3[offset]);
					d30 = svmla_f32_x(mask, d30, s0, w0);
					d31 = svmla_f32_x(mask, d31, s0, w1);
					s0 = svdup_n_f32(src4[offset]);
					d40 = svmla_f32_x(mask, d40, s0, w0);
					d41 = svmla_f32_x(mask, d41, s0, w1);
					s0 = svdup_n_f32(src5[offset]);
					d50 = svmla_f32_x(mask, d50, s0, w0);
					d51 = svmla_f32_x(mask, d51, s0, w1);
					weight += DF;
				}
				weight += tail;
			}
			svst1_f32(mask, dst0 + 0 * F, Activate0<type>(d00, param0, param1, mask));
			svst1_f32(mask, dst1 + 0 * F, Activate1<type>(d01, param0, param1, mask));
			svst1_f32(mask, dst0 + 1 * F, Activate0<type>(d10, param0, param1, mask));
			svst1_f32(mask, dst1 + 1 * F, Activate1<type>(d11, param0, param1, mask));
			svst1_f32(mask, dst0 + 2 * F, Activate0<type>(d20, param0, param1, mask));
			svst1_f32(mask, dst1 + 2 * F, Activate1<type>(d21, param0, param1, mask));
			svst1_f32(mask, dst0 + 3 * F, Activate0<type>(d30, param0, param1, mask));
			svst1_f32(mask, dst1 + 3 * F, Activate1<type>(d31, param0, param1, mask));
			svst1_f32(mask, dst0 + 4 * F, Activate0<type>(d40, param0, param1, mask));
			svst1_f32(mask, dst1 + 4 * F, Activate1<type>(d41, param0, param1, mask));
			svst1_f32(mask, dst0 + 5 * F, Activate0<type>(d50, param0, param1, mask));
			svst1_f32(mask, dst1 + 5 * F, Activate1<type>(d51, param0, param1, mask));
		}

		template<SimdConvolutionActivationType type> SIMD_INLINE void InputConvolution_1x6(const float* src0, const SimdConvolutionParameters& p,
			size_t kH, size_t kW, const float* weight, svfloat32_t bias0, svfloat32_t param0, svfloat32_t param1, float* dst0)
		{
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			svfloat32_t d00, d10, d20, d30, d40, d50, s0, w0;
			d00 = bias0;
			d10 = bias0;
			d20 = bias0;
			d30 = bias0;
			d40 = bias0;
			d50 = bias0;
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
					w0 = svld1_f32(mask, weight + 0);
					s0 = svdup_n_f32(src0[offset]);
					d00 = svmla_f32_x(mask, d00, s0, w0);
					s0 = svdup_n_f32(src1[offset]);
					d10 = svmla_f32_x(mask, d10, s0, w0);
					s0 = svdup_n_f32(src2[offset]);
					d20 = svmla_f32_x(mask, d20, s0, w0);
					s0 = svdup_n_f32(src3[offset]);
					d30 = svmla_f32_x(mask, d30, s0, w0);
					s0 = svdup_n_f32(src4[offset]);
					d40 = svmla_f32_x(mask, d40, s0, w0);
					s0 = svdup_n_f32(src5[offset]);
					d50 = svmla_f32_x(mask, d50, s0, w0);
					weight += DF;
				}
				weight += tail;
			}
			svst1_f32(mask, dst0 + 0 * F, Activate0<type>(d00, param0, param1, mask));
			svst1_f32(mask, dst0 + 1 * F, Activate0<type>(d10, param0, param1, mask));
			svst1_f32(mask, dst0 + 2 * F, Activate0<type>(d20, param0, param1, mask));
			svst1_f32(mask, dst0 + 3 * F, Activate0<type>(d30, param0, param1, mask));
			svst1_f32(mask, dst0 + 4 * F, Activate0<type>(d40, param0, param1, mask));
			svst1_f32(mask, dst0 + 5 * F, Activate0<type>(d50, param0, param1, mask));
		}

		template<SimdConvolutionActivationType type> void InputConvolution(const float* src, const SimdConvolutionParameters& p,
			size_t dstC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
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

			svfloat32_t param0 = svdup_n_f32(params[0]), param1 = svdup_n_f32(0.0f);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				param1 = svdup_n_f32(params[1]);

			size_t dc = 0;
			for (; dc < dstCDF; dc += DF)
			{
				svfloat32_t bias0 = bias ? svld1_f32(mask, bias + dc + 0) : svdup_n_f32(0.0f);
				svfloat32_t bias1 = bias ? svld1_f32(mask, bias + dc + F) : svdup_n_f32(0.0f);
				if (type == ::SimdConvolutionActivationPrelu)
				{
					param0 = svld1_f32(mask, params + dc + 0);
					param1 = svld1_f32(mask, params + dc + F);
				}
				size_t dy = yBeg, sy = dy * strideY;
				for (; sy < noseH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS, * dst1 = dst0 + dstS;
					size_t sx = 0;
					const float* s = src;
					const float* w = weight + (noseH - sy) * kernelX * DF * srcC;
					for (; sx < noseW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s, p, kY + sy, kX + sx, w + (noseW - sx) * srcC * DF, bias0, bias1, param0, param1, dst0, dst1);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F, dst1 += 6 * F)
						InputConvolution_2x6<type>(s + (sx - noseW) * srcC, p, kY + sy, kernelX, w, bias0, bias1, param0, param1, dst0, dst1);
					for (; sx < bodyW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kY + sy, kernelX, w, bias0, bias1, param0, param1, dst0, dst1);
					for (; sx < tailW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kY + sy, kW - sx, w, bias0, bias1, param0, param1, dst0, dst1);
				}
				for (; sy < bodyH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS, * dst1 = dst0 + dstS;
					size_t sx = 0;
					const float* s = src + (sy - noseH) * srcW * srcC;
					const float* w = weight;
					for (; sx < noseW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s, p, kernelY, kX + sx, w + (noseW - sx) * srcC * DF, bias0, bias1, param0, param1, dst0, dst1);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F, dst1 += 6 * F)
						InputConvolution_2x6<type>(s + (sx - noseW) * srcC, p, kernelY, kernelX, w, bias0, bias1, param0, param1, dst0, dst1);
					for (; sx < bodyW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kernelY, kernelX, w, bias0, bias1, param0, param1, dst0, dst1);
					for (; sx < tailW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kernelY, kW - sx, w, bias0, bias1, param0, param1, dst0, dst1);
				}
				for (; sy < tailH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS, * dst1 = dst0 + dstS;
					size_t sx = 0;
					const float* s = src + (sy - noseH) * srcW * srcC;
					const float* w = weight;
					for (; sx < noseW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s, p, kH - sy, kX + sx, w + (noseW - sx) * srcC * DF, bias0, bias1, param0, param1, dst0, dst1);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F, dst1 += 6 * F)
						InputConvolution_2x6<type>(s + (sx - noseW) * srcC, p, kH - sy, kernelX, w, bias0, bias1, param0, param1, dst0, dst1);
					for (; sx < bodyW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kH - sy, kernelX, w, bias0, bias1, param0, param1, dst0, dst1);
					for (; sx < tailW; sx += strideX, dst0 += F, dst1 += F)
						InputConvolution_2x1<type>(s + (sx - noseW) * srcC, p, kH - sy, kW - sx, w, bias0, bias1, param0, param1, dst0, dst1);
				}
				weight += kernelY * kernelX * srcC * DF;
			}
			if (dc < dstC)
			{
				svfloat32_t bias0 = bias ? svld1_f32(mask, bias + dc) : svdup_n_f32(0.0f);
				if (type == ::SimdConvolutionActivationPrelu)
					param0 = svld1_f32(mask, params + dc);
				size_t dy = yBeg, sy = dy * strideY;
				for (; sy < noseH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS;
					size_t sx = 0;
					const float* s = src;
					const float* w = weight + (noseH - sy) * kernelX * DF * srcC;
					for (; sx < noseW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s, p, kY + sy, kX + sx, w + (noseW - sx) * srcC * DF, bias0, param0, param1, dst0);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F)
						InputConvolution_1x6<type>(s + (sx - noseW) * srcC, p, kY + sy, kernelX, w, bias0, param0, param1, dst0);
					for (; sx < bodyW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kY + sy, kernelX, w, bias0, param0, param1, dst0);
					for (; sx < tailW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kY + sy, kW - sx, w, bias0, param0, param1, dst0);
				}
				for (; sy < bodyH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS;
					size_t sx = 0;
					const float* s = src + (sy - noseH) * srcW * srcC;
					const float* w = weight;
					for (; sx < noseW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s, p, kernelY, kX + sx, w + (noseW - sx) * srcC * DF, bias0, param0, param1, dst0);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F)
						InputConvolution_1x6<type>(s + (sx - noseW) * srcC, p, kernelY, kernelX, w, bias0, param0, param1, dst0);
					for (; sx < bodyW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kernelY, kernelX, w, bias0, param0, param1, dst0);
					for (; sx < tailW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kernelY, kW - sx, w, bias0, param0, param1, dst0);
				}
				for (; sy < tailH && dy < yEnd; sy += strideY, dy++)
				{
					float* dst0 = dst + (dy & dstM) * dstW * F + (dc / F) * dstS;
					size_t sx = 0;
					const float* s = src + (sy - noseH) * srcW * srcC;
					const float* w = weight;
					for (; sx < noseW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s, p, kH - sy, kX + sx, w + (noseW - sx) * srcC * DF, bias0, param0, param1, dst0);
					for (; sx < bodyW6; sx += 6 * strideX, dst0 += 6 * F)
						InputConvolution_1x6<type>(s + (sx - noseW) * srcC, p, kH - sy, kernelX, w, bias0, param0, param1, dst0);
					for (; sx < bodyW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kH - sy, kernelX, w, bias0, param0, param1, dst0);
					for (; sx < tailW; sx += strideX, dst0 += F)
						InputConvolution_1x1<type>(s + (sx - noseW) * srcC, p, kH - sy, kW - sx, w, bias0, param0, param1, dst0);
				}
			}
		}

		//-------------------------------------------------------------------------------------------------------

		template <SimdConvolutionActivationType type> void SetInput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			if (p.kernelY == 1 && p.strideY == 1)
				convolution[0] = InputConvolution1x1<type>;
			else
				convolution[0] = InputConvolution<type>;
		}

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
