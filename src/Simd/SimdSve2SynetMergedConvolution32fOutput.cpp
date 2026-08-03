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
		SIMD_INLINE svfloat32_t Exp2(const svbool_t& mask, svfloat32_t x)
		{
			x = svmax_f32_x(mask, svmin_f32_x(mask, x, svdup_n_f32(126.99999f)), svdup_n_f32(-126.99999f));
			svint32_t ipart = svcvt_s32_f32_x(mask, svsub_n_f32_x(mask, x, 0.5f));
			svfloat32_t fpart = svsub_f32_x(mask, x, svcvt_f32_s32_x(mask, ipart));
			svfloat32_t expipart = svreinterpret_f32_s32(svlsl_n_s32_x(mask, svadd_n_s32_x(mask, ipart, 127), 23));
			svfloat32_t p = svdup_n_f32(1.8775767e-3f);
			p = svmla_f32_x(mask, svdup_n_f32(8.9893397e-3f), fpart, p);
			p = svmla_f32_x(mask, svdup_n_f32(5.5826318e-2f), fpart, p);
			p = svmla_f32_x(mask, svdup_n_f32(2.4015361e-1f), fpart, p);
			p = svmla_f32_x(mask, svdup_n_f32(6.9315308e-1f), fpart, p);
			p = svmla_f32_x(mask, svdup_n_f32(9.9999994e-1f), fpart, p);
			return svmul_f32_x(mask, expipart, p);
		}

		SIMD_INLINE svfloat32_t Exp(const svbool_t& mask, svfloat32_t value)
		{
			return Exp2(mask, svmul_n_f32_x(mask, value, 1.44269504f));
		}

		SIMD_INLINE svfloat32_t Log2(const svbool_t& mask, svfloat32_t x)
		{
			svuint32_t i = svreinterpret_u32_f32(x);
			svint32_t e32 = svsub_n_s32_x(mask, svreinterpret_s32_u32(svlsr_n_u32_x(mask, svand_n_u32_x(mask, i, 0x7F800000), 23)), 127);
			svfloat32_t e = svcvt_f32_s32_x(mask, e32);
			svfloat32_t one = svdup_n_f32(1.0f);
			svfloat32_t m = svreinterpret_f32_u32(svorr_u32_x(mask, svand_n_u32_x(mask, i, 0x007FFFFF), svreinterpret_u32_f32(one)));
			svfloat32_t p = svdup_n_f32(-3.4436006e-2f);
			p = svmla_f32_x(mask, svdup_n_f32(3.1821337e-1f), m, p);
			p = svmla_f32_x(mask, svdup_n_f32(-1.2315303f), m, p);
			p = svmla_f32_x(mask, svdup_n_f32(2.5988452f), m, p);
			p = svmla_f32_x(mask, svdup_n_f32(-3.3241990f), m, p);
			p = svmla_f32_x(mask, svdup_n_f32(3.1157899f), m, p);
			return svmla_f32_x(mask, e, p, svsub_f32_x(mask, m, one));
		}

		SIMD_INLINE svfloat32_t Log(const svbool_t& mask, svfloat32_t value)
		{
			return svmul_n_f32_x(mask, Log2(mask, value), 0.693147181f);
		}

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

		SIMD_INLINE svfloat32_t Tanh(const svbool_t& mask, svfloat32_t x)
		{
			svfloat32_t e = Exp(mask, svmul_n_f32_x(mask, x, -2.0f));
			return svsub_n_f32_x(mask, svdiv_f32_x(mask, svdup_n_f32(2.0f), svadd_n_f32_x(mask, e, 1.0f)), 1.0f);
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
			svfloat32_t neg = svmul_f32_x(mask, param0, svsub_n_f32_x(mask, Exp(mask, value), 1.0f));
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
			svfloat32_t exp = svmin_f32_x(mask, Exp(mask, value), param0);
			return svmul_f32_x(mask, value, Tanh(mask, Log(mask, svadd_n_f32_x(mask, exp, 1.0f))));
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationHardSigmoid>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return svmax_n_f32_x(mask, svmin_n_f32_x(mask, svmla_f32_x(mask, param1, value, param0), 1.0f), 0.0f);
		}

		template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationSwish>(svfloat32_t value, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
		{
			return svdiv_f32_x(mask, value, svadd_n_f32_x(mask, Exp(mask, svmul_f32_x(mask, svneg_f32_x(mask, value), param0)), 1.0f));
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

		
		template <TermType term> struct Term
		{
			template<SimdConvolutionActivationType type> static SIMD_INLINE void Save0(float* ptr, svfloat32_t value, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask);
			template<SimdConvolutionActivationType type> static SIMD_INLINE void Save1(float* ptr, svfloat32_t value, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask);
		};

		template <> struct Term<TermLast>
		{
			template<SimdConvolutionActivationType type> static SIMD_INLINE void Save0(float* ptr, svfloat32_t value, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
			{
				svst1_f32(mask, ptr, Activate0<type>(svadd_f32_x(mask, value, bias), param0, param1, mask));
			}

			template<SimdConvolutionActivationType type> static SIMD_INLINE void Save1(float* ptr, svfloat32_t value, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
			{
				svst1_f32(mask, ptr, Activate1<type>(svadd_f32_x(mask, value, bias), param0, param1, mask));
			}
		};

		template <> struct Term<TermInterim>
		{
			template<SimdConvolutionActivationType type> static SIMD_INLINE void Save0(float* ptr, svfloat32_t value, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
			{
				svst1_f32(mask, ptr, value);
			}

			template<SimdConvolutionActivationType type> static SIMD_INLINE void Save1(float* ptr, svfloat32_t value, svfloat32_t bias, svfloat32_t param0, svfloat32_t param1, const svbool_t& mask)
			{
				svst1_f32(mask, ptr, value);
			}
		};

		//-------------------------------------------------------------------------------------------------------

		template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x6(const float* src, size_t srcC, size_t srcS,
			const float* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst, size_t dstC, size_t tail, int first)
		{
			const size_t F = svcntw(), DF = 2 * F;
			svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
			if (tail > F)
			{
				svbool_t mask0 = svptrue_b32();
				svbool_t mask1 = svwhilelt_b32((uint64_t)0, (uint64_t)(tail == DF ? F : (tail - F)));
				if (first)
				{
					d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
					d10 = svdup_n_f32(0.0f), d11 = svdup_n_f32(0.0f);
					d20 = svdup_n_f32(0.0f), d21 = svdup_n_f32(0.0f);
					d30 = svdup_n_f32(0.0f), d31 = svdup_n_f32(0.0f);
					d40 = svdup_n_f32(0.0f), d41 = svdup_n_f32(0.0f);
					d50 = svdup_n_f32(0.0f), d51 = svdup_n_f32(0.0f);
				}
				else
				{
					d00 = svld1_f32(mask0, dst + 0 * dstC + 0), d01 = svld1_f32(mask1, dst + 0 * dstC + F);
					d10 = svld1_f32(mask0, dst + 1 * dstC + 0), d11 = svld1_f32(mask1, dst + 1 * dstC + F);
					d20 = svld1_f32(mask0, dst + 2 * dstC + 0), d21 = svld1_f32(mask1, dst + 2 * dstC + F);
					d30 = svld1_f32(mask0, dst + 3 * dstC + 0), d31 = svld1_f32(mask1, dst + 3 * dstC + F);
					d40 = svld1_f32(mask0, dst + 4 * dstC + 0), d41 = svld1_f32(mask1, dst + 4 * dstC + F);
					d50 = svld1_f32(mask0, dst + 5 * dstC + 0), d51 = svld1_f32(mask1, dst + 5 * dstC + F);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = svld1_f32(mask0, weight + 0);
						w1 = svld1_f32(mask1, weight + F);
						s0 = svdup_n_f32(src[i + 0 * F]);
						d00 = svmla_f32_x(mask0, d00, s0, w0);
						d01 = svmla_f32_x(mask1, d01, s0, w1);
						s0 = svdup_n_f32(src[i + 1 * F]);
						d10 = svmla_f32_x(mask0, d10, s0, w0);
						d11 = svmla_f32_x(mask1, d11, s0, w1);
						s0 = svdup_n_f32(src[i + 2 * F]);
						d20 = svmla_f32_x(mask0, d20, s0, w0);
						d21 = svmla_f32_x(mask1, d21, s0, w1);
						s0 = svdup_n_f32(src[i + 3 * F]);
						d30 = svmla_f32_x(mask0, d30, s0, w0);
						d31 = svmla_f32_x(mask1, d31, s0, w1);
						s0 = svdup_n_f32(src[i + 4 * F]);
						d40 = svmla_f32_x(mask0, d40, s0, w0);
						d41 = svmla_f32_x(mask1, d41, s0, w1);
						s0 = svdup_n_f32(src[i + 5 * F]);
						d50 = svmla_f32_x(mask0, d50, s0, w0);
						d51 = svmla_f32_x(mask1, d51, s0, w1);
					}
					src += srcS;
				}
				if (tail == DF)
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d01, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d10, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d11, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d20, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d21, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d30, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d31, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d40, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d41, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d50, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d51, bias1, param0, param1, mask1);
				}
				else
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d01, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d10, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d11, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d20, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d21, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d30, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d31, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d40, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d41, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d50, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d51, bias1, param0, param1, mask1);
				}
			}
			else
			{
				svbool_t mask0 = svwhilelt_b32((uint64_t)0, (uint64_t)tail);
				if (first)
				{
					d00 = svdup_n_f32(0.0f);
					d10 = svdup_n_f32(0.0f);
					d20 = svdup_n_f32(0.0f);
					d30 = svdup_n_f32(0.0f);
					d40 = svdup_n_f32(0.0f);
					d50 = svdup_n_f32(0.0f);
				}
				else
				{
					d00 = svld1_f32(mask0, dst + 0 * dstC + 0);
					d10 = svld1_f32(mask0, dst + 1 * dstC + 0);
					d20 = svld1_f32(mask0, dst + 2 * dstC + 0);
					d30 = svld1_f32(mask0, dst + 3 * dstC + 0);
					d40 = svld1_f32(mask0, dst + 4 * dstC + 0);
					d50 = svld1_f32(mask0, dst + 5 * dstC + 0);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = svld1_f32(mask0, weight + 0);
						s0 = svdup_n_f32(src[i + 0 * F]);
						d00 = svmla_f32_x(mask0, d00, s0, w0);
						s0 = svdup_n_f32(src[i + 1 * F]);
						d10 = svmla_f32_x(mask0, d10, s0, w0);
						s0 = svdup_n_f32(src[i + 2 * F]);
						d20 = svmla_f32_x(mask0, d20, s0, w0);
						s0 = svdup_n_f32(src[i + 3 * F]);
						d30 = svmla_f32_x(mask0, d30, s0, w0);
						s0 = svdup_n_f32(src[i + 4 * F]);
						d40 = svmla_f32_x(mask0, d40, s0, w0);
						s0 = svdup_n_f32(src[i + 5 * F]);
						d50 = svmla_f32_x(mask0, d50, s0, w0);
					}
					src += srcS;
				}
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d10, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d20, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d30, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d40, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d50, bias0, param0, param1, mask0);
				}
			}
		}

		template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x4(const float* src, size_t srcC, size_t srcS,
			const float* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst, size_t dstC, size_t tail, int first)
		{
			const size_t F = svcntw(), DF = 2 * F;
			svfloat32_t d00, d01, d10, d11, d20, d21, d30, d31, s0, w0, w1;
			if (tail > F)
			{
				svbool_t mask0 = svptrue_b32();
				svbool_t mask1 = svwhilelt_b32((uint64_t)0, (uint64_t)(tail == DF ? F : (tail - F)));
				if (first)
				{
					d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
					d10 = svdup_n_f32(0.0f), d11 = svdup_n_f32(0.0f);
					d20 = svdup_n_f32(0.0f), d21 = svdup_n_f32(0.0f);
					d30 = svdup_n_f32(0.0f), d31 = svdup_n_f32(0.0f);
				}
				else
				{
					d00 = svld1_f32(mask0, dst + 0 * dstC + 0), d01 = svld1_f32(mask1, dst + 0 * dstC + F);
					d10 = svld1_f32(mask0, dst + 1 * dstC + 0), d11 = svld1_f32(mask1, dst + 1 * dstC + F);
					d20 = svld1_f32(mask0, dst + 2 * dstC + 0), d21 = svld1_f32(mask1, dst + 2 * dstC + F);
					d30 = svld1_f32(mask0, dst + 3 * dstC + 0), d31 = svld1_f32(mask1, dst + 3 * dstC + F);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = svld1_f32(mask0, weight + 0);
						w1 = svld1_f32(mask1, weight + F);
						s0 = svdup_n_f32(src[i + 0 * F]);
						d00 = svmla_f32_x(mask0, d00, s0, w0);
						d01 = svmla_f32_x(mask1, d01, s0, w1);
						s0 = svdup_n_f32(src[i + 1 * F]);
						d10 = svmla_f32_x(mask0, d10, s0, w0);
						d11 = svmla_f32_x(mask1, d11, s0, w1);
						s0 = svdup_n_f32(src[i + 2 * F]);
						d20 = svmla_f32_x(mask0, d20, s0, w0);
						d21 = svmla_f32_x(mask1, d21, s0, w1);
						s0 = svdup_n_f32(src[i + 3 * F]);
						d30 = svmla_f32_x(mask0, d30, s0, w0);
						d31 = svmla_f32_x(mask1, d31, s0, w1);
					}
					src += srcS;
				}
				if (tail == DF)
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d01, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d10, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d11, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d20, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d21, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d30, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d31, bias1, param0, param1, mask1);
				}
				else
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d01, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d10, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d11, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d20, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d21, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d30, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d31, bias1, param0, param1, mask1);
				}
			}
			else
			{
				svbool_t mask0 = svwhilelt_b32((uint64_t)0, (uint64_t)tail);
				if (first)
				{
					d00 = svdup_n_f32(0.0f);
					d10 = svdup_n_f32(0.0f);
					d20 = svdup_n_f32(0.0f);
					d30 = svdup_n_f32(0.0f);
				}
				else
				{
					d00 = svld1_f32(mask0, dst + 0 * dstC + 0);
					d10 = svld1_f32(mask0, dst + 1 * dstC + 0);
					d20 = svld1_f32(mask0, dst + 2 * dstC + 0);
					d30 = svld1_f32(mask0, dst + 3 * dstC + 0);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = svld1_f32(mask0, weight + 0);
						s0 = svdup_n_f32(src[i + 0 * F]);
						d00 = svmla_f32_x(mask0, d00, s0, w0);
						s0 = svdup_n_f32(src[i + 1 * F]);
						d10 = svmla_f32_x(mask0, d10, s0, w0);
						s0 = svdup_n_f32(src[i + 2 * F]);
						d20 = svmla_f32_x(mask0, d20, s0, w0);
						s0 = svdup_n_f32(src[i + 3 * F]);
						d30 = svmla_f32_x(mask0, d30, s0, w0);
					}
					src += srcS;
				}
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d10, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d20, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d30, bias0, param0, param1, mask0);
				}
			}
		}

		template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x3(const float* src, size_t srcC, size_t srcS,
			const float* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst, size_t dstC, size_t tail, int first)
		{
			const size_t F = svcntw(), DF = 2 * F;
			svfloat32_t d00, d01, d10, d11, d20, d21, s0, w0, w1;
			if (tail > F)
			{
				svbool_t mask0 = svptrue_b32();
				svbool_t mask1 = svwhilelt_b32((uint64_t)0, (uint64_t)(tail == DF ? F : (tail - F)));
				if (first)
				{
					d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
					d10 = svdup_n_f32(0.0f), d11 = svdup_n_f32(0.0f);
					d20 = svdup_n_f32(0.0f), d21 = svdup_n_f32(0.0f);
				}
				else
				{
					d00 = svld1_f32(mask0, dst + 0 * dstC + 0), d01 = svld1_f32(mask1, dst + 0 * dstC + F);
					d10 = svld1_f32(mask0, dst + 1 * dstC + 0), d11 = svld1_f32(mask1, dst + 1 * dstC + F);
					d20 = svld1_f32(mask0, dst + 2 * dstC + 0), d21 = svld1_f32(mask1, dst + 2 * dstC + F);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = svld1_f32(mask0, weight + 0);
						w1 = svld1_f32(mask1, weight + F);
						s0 = svdup_n_f32(src[i + 0 * F]);
						d00 = svmla_f32_x(mask0, d00, s0, w0);
						d01 = svmla_f32_x(mask1, d01, s0, w1);
						s0 = svdup_n_f32(src[i + 1 * F]);
						d10 = svmla_f32_x(mask0, d10, s0, w0);
						d11 = svmla_f32_x(mask1, d11, s0, w1);
						s0 = svdup_n_f32(src[i + 2 * F]);
						d20 = svmla_f32_x(mask0, d20, s0, w0);
						d21 = svmla_f32_x(mask1, d21, s0, w1);
					}
					src += srcS;
				}
				if (tail == DF)
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d01, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d10, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d11, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d20, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d21, bias1, param0, param1, mask1);
				}
				else
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d01, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d10, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d11, bias1, param0, param1, mask1);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d20, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d21, bias1, param0, param1, mask1);
				}
			}
			else
			{
				svbool_t mask0 = svwhilelt_b32((uint64_t)0, (uint64_t)tail);
				if (first)
				{
					d00 = svdup_n_f32(0.0f);
					d10 = svdup_n_f32(0.0f);
					d20 = svdup_n_f32(0.0f);
				}
				else
				{
					d00 = svld1_f32(mask0, dst + 0 * dstC + 0);
					d10 = svld1_f32(mask0, dst + 1 * dstC + 0);
					d20 = svld1_f32(mask0, dst + 2 * dstC + 0);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = svld1_f32(mask0, weight + 0);
						s0 = svdup_n_f32(src[i + 0 * F]);
						d00 = svmla_f32_x(mask0, d00, s0, w0);
						s0 = svdup_n_f32(src[i + 1 * F]);
						d10 = svmla_f32_x(mask0, d10, s0, w0);
						s0 = svdup_n_f32(src[i + 2 * F]);
						d20 = svmla_f32_x(mask0, d20, s0, w0);
					}
					src += srcS;
				}
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d10, bias0, param0, param1, mask0);
					dst += dstC;
					Term<term>::template Save0<type>(dst + 0, d20, bias0, param0, param1, mask0);
				}
			}
		}

		template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x1(const float* src, size_t srcC, size_t srcS,
			const float* weight, svfloat32_t bias0, svfloat32_t bias1, svfloat32_t param0, svfloat32_t param1, float* dst, size_t dstC, size_t tail, int first)
		{
			const size_t F = svcntw(), DF = 2 * F;
			svfloat32_t d00, d01, s0, w0, w1;
			if (tail > F)
			{
				svbool_t mask0 = svptrue_b32();
				svbool_t mask1 = svwhilelt_b32((uint64_t)0, (uint64_t)(tail == DF ? F : (tail - F)));
				if (first)
				{
					d00 = svdup_n_f32(0.0f), d01 = svdup_n_f32(0.0f);
				}
				else
				{
					d00 = svld1_f32(mask0, dst + 0 * dstC + 0), d01 = svld1_f32(mask1, dst + 0 * dstC + F);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = svld1_f32(mask0, weight + 0);
						w1 = svld1_f32(mask1, weight + F);
						s0 = svdup_n_f32(src[i + 0 * F]);
						d00 = svmla_f32_x(mask0, d00, s0, w0);
						d01 = svmla_f32_x(mask1, d01, s0, w1);
					}
					src += srcS;
				}
				if (tail == DF)
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d01, bias1, param0, param1, mask1);
				}
				else
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
					Term<term>::template Save1<type>(dst + F, d01, bias1, param0, param1, mask1);
				}
			}
			else
			{
				svbool_t mask0 = svwhilelt_b32((uint64_t)0, (uint64_t)tail);
				if (first)
				{
					d00 = svdup_n_f32(0.0f);
				}
				else
				{
					d00 = svld1_f32(mask0, dst + 0 * dstC + 0);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = svld1_f32(mask0, weight + 0);
						s0 = svdup_n_f32(src[i + 0 * F]);
						d00 = svmla_f32_x(mask0, d00, s0, w0);
					}
					src += srcS;
				}
				{
					Term<term>::template Save0<type>(dst + 0, d00, bias0, param0, param1, mask0);
				}
			}
		}

		template<TermType term, SimdConvolutionActivationType type> void OutputConvolution(const float* src, const SimdConvolutionParameters& p,
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			assert(p.group == 1 && p.kernelY == 1 && p.strideY == 1);
			const size_t F = svcntw(), DF = 2 * F;
			const svbool_t mask = svptrue_b32();
			size_t srcH = p.srcH, srcW = p.srcW, dstW = p.dstW, dstC = p.dstC;
			size_t srcM = (bufH[1] - 1), srcS = bufH[1] * srcW * F;
			size_t dstW3 = AlignLoAny(dstW, 3), dstW6 = AlignLoAny(dstW, 6);
			svfloat32_t param0 = svdup_n_f32(params[0]), param1 = svdup_n_f32(0.0f);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				param1 = svdup_n_f32(params[1]);

			dst += yBeg * p.dstW * p.dstC;
			size_t dc = 0;
			for (; dc < dstC; dc += DF)
			{
				size_t tail = Simd::Min(DF, dstC - dc);
				svfloat32_t bias0 = svld1_f32(mask, bias + dc + 0);
				svfloat32_t bias1 = svld1_f32(mask, bias + dc + F);
				if (type == ::SimdConvolutionActivationPrelu)
				{
					param0 = svld1_f32(mask, params + dc + 0);
					param1 = svld1_f32(mask, params + dc + F);
				}
				float* pDst = dst + dc;
				for (size_t y = yBeg; y < yEnd; ++y)
				{
					const float* pSrc = src + (y & srcM) * srcW * F;
					size_t x = 0;
					for (; x < dstW6; x += 6, pDst += 6 * dstC, pSrc += 6 * F)
						OutputConvolution_2x6<term, type>(pSrc, srcC, srcS, weight, bias0, bias1, param0, param1, pDst, dstC, tail, first);
					if (dstW - dstW6 == 4)
						OutputConvolution_2x4<term, type>(pSrc, srcC, srcS, weight, bias0, bias1, param0, param1, pDst, dstC, tail, first), pDst += 4 * dstC;
					else
					{
						for (; x < dstW3; x += 3, pDst += 3 * dstC, pSrc += 3 * F)
							OutputConvolution_2x3<term, type>(pSrc, srcC, srcS, weight, bias0, bias1, param0, param1, pDst, dstC, tail, first);
						for (; x < dstW; ++x, pDst += dstC, pSrc += F)
							OutputConvolution_2x1<term, type>(pSrc, srcC, srcS, weight, bias0, bias1, param0, param1, pDst, dstC, tail, first);
					}
				}
				weight += srcC * DF;
			}
		}

		//-------------------------------------------------------------------------------------------------------

		template <SimdConvolutionActivationType type> void SetOutput(const ConvParam& p, Base::SynetMergedConvolution32f::ConvolutionPtr* convolution)
		{
			convolution[0] = OutputConvolution<TermLast, type>;
			convolution[1] = OutputConvolution<TermInterim, SimdConvolutionActivationIdentity>;
		}

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
