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
		template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x6(const float* src, size_t srcC, size_t srcS,
			const float* weight, const float32x4_t* bias, const float32x4_t* params, float* dst, size_t dstC, size_t tail, int first)
		{
			float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
			if (tail > F)
			{
				if (first)
				{
					d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
					d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
					d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
					d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
					d40 = vdupq_n_f32(0.0f), d41 = vdupq_n_f32(0.0f);
					d50 = vdupq_n_f32(0.0f), d51 = vdupq_n_f32(0.0f);
				}
				else
				{
					d00 = Load<false>(dst + 0 * dstC + 0), d01 = Load<false>(dst + 0 * dstC + F);
					d10 = Load<false>(dst + 1 * dstC + 0), d11 = Load<false>(dst + 1 * dstC + F);
					d20 = Load<false>(dst + 2 * dstC + 0), d21 = Load<false>(dst + 2 * dstC + F);
					d30 = Load<false>(dst + 3 * dstC + 0), d31 = Load<false>(dst + 3 * dstC + F);
					d40 = Load<false>(dst + 4 * dstC + 0), d41 = Load<false>(dst + 4 * dstC + F);
					d50 = Load<false>(dst + 5 * dstC + 0), d51 = Load<false>(dst + 5 * dstC + F);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = Load<false>(weight + 0);
						w1 = Load<false>(weight + F);
						s0 = vld1q_dup_f32(src + i + 0 * F);
						d00 = vmlaq_f32(d00, s0, w0);
						d01 = vmlaq_f32(d01, s0, w1);
						s0 = vld1q_dup_f32(src + i + 1 * F);
						d10 = vmlaq_f32(d10, s0, w0);
						d11 = vmlaq_f32(d11, s0, w1);
						s0 = vld1q_dup_f32(src + i + 2 * F);
						d20 = vmlaq_f32(d20, s0, w0);
						d21 = vmlaq_f32(d21, s0, w1);
						s0 = vld1q_dup_f32(src + i + 3 * F);
						d30 = vmlaq_f32(d30, s0, w0);
						d31 = vmlaq_f32(d31, s0, w1);
						s0 = vld1q_dup_f32(src + i + 4 * F);
						d40 = vmlaq_f32(d40, s0, w0);
						d41 = vmlaq_f32(d41, s0, w1);
						s0 = vld1q_dup_f32(src + i + 5 * F);
						d50 = vmlaq_f32(d50, s0, w0);
						d51 = vmlaq_f32(d51, s0, w1);
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
				if (first)
				{
					d00 = vdupq_n_f32(0.0f);
					d10 = vdupq_n_f32(0.0f);
					d20 = vdupq_n_f32(0.0f);
					d30 = vdupq_n_f32(0.0f);
					d40 = vdupq_n_f32(0.0f);
					d50 = vdupq_n_f32(0.0f);
				}
				else
				{
					d00 = Load<false>(dst + 0 * dstC + 0);
					d10 = Load<false>(dst + 1 * dstC + 0);
					d20 = Load<false>(dst + 2 * dstC + 0);
					d30 = Load<false>(dst + 3 * dstC + 0);
					d40 = Load<false>(dst + 4 * dstC + 0);
					d50 = Load<false>(dst + 5 * dstC + 0);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = Load<false>(weight + 0);
						s0 = vld1q_dup_f32(src + i + 0 * F);
						d00 = vmlaq_f32(d00, s0, w0);
						s0 = vld1q_dup_f32(src + i + 1 * F);
						d10 = vmlaq_f32(d10, s0, w0);
						s0 = vld1q_dup_f32(src + i + 2 * F);
						d20 = vmlaq_f32(d20, s0, w0);
						s0 = vld1q_dup_f32(src + i + 3 * F);
						d30 = vmlaq_f32(d30, s0, w0);
						s0 = vld1q_dup_f32(src + i + 4 * F);
						d40 = vmlaq_f32(d40, s0, w0);
						s0 = vld1q_dup_f32(src + i + 5 * F);
						d50 = vmlaq_f32(d50, s0, w0);
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
			const float* weight, const float32x4_t* bias, const float32x4_t* params, float* dst, size_t dstC, size_t tail, int first)
		{
			float32x4_t d00, d01, d10, d11, d20, d21, d30, d31, s0, w0, w1;
			if (tail > F)
			{
				if (first)
				{
					d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
					d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
					d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
					d30 = vdupq_n_f32(0.0f), d31 = vdupq_n_f32(0.0f);
				}
				else
				{
					d00 = Load<false>(dst + 0 * dstC + 0), d01 = Load<false>(dst + 0 * dstC + F);
					d10 = Load<false>(dst + 1 * dstC + 0), d11 = Load<false>(dst + 1 * dstC + F);
					d20 = Load<false>(dst + 2 * dstC + 0), d21 = Load<false>(dst + 2 * dstC + F);
					d30 = Load<false>(dst + 3 * dstC + 0), d31 = Load<false>(dst + 3 * dstC + F);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = Load<false>(weight + 0);
						w1 = Load<false>(weight + F);
						s0 = vld1q_dup_f32(src + i + 0 * F);
						d00 = vmlaq_f32(d00, s0, w0);
						d01 = vmlaq_f32(d01, s0, w1);
						s0 = vld1q_dup_f32(src + i + 1 * F);
						d10 = vmlaq_f32(d10, s0, w0);
						d11 = vmlaq_f32(d11, s0, w1);
						s0 = vld1q_dup_f32(src + i + 2 * F);
						d20 = vmlaq_f32(d20, s0, w0);
						d21 = vmlaq_f32(d21, s0, w1);
						s0 = vld1q_dup_f32(src + i + 3 * F);
						d30 = vmlaq_f32(d30, s0, w0);
						d31 = vmlaq_f32(d31, s0, w1);
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
				if (first)
				{
					d00 = vdupq_n_f32(0.0f);
					d10 = vdupq_n_f32(0.0f);
					d20 = vdupq_n_f32(0.0f);
					d30 = vdupq_n_f32(0.0f);
				}
				else
				{
					d00 = Load<false>(dst + 0 * dstC + 0);
					d10 = Load<false>(dst + 1 * dstC + 0);
					d20 = Load<false>(dst + 2 * dstC + 0);
					d30 = Load<false>(dst + 3 * dstC + 0);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = Load<false>(weight + 0);
						s0 = vld1q_dup_f32(src + i + 0 * F);
						d00 = vmlaq_f32(d00, s0, w0);
						s0 = vld1q_dup_f32(src + i + 1 * F);
						d10 = vmlaq_f32(d10, s0, w0);
						s0 = vld1q_dup_f32(src + i + 2 * F);
						d20 = vmlaq_f32(d20, s0, w0);
						s0 = vld1q_dup_f32(src + i + 3 * F);
						d30 = vmlaq_f32(d30, s0, w0);
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
			const float* weight, const float32x4_t* bias, const float32x4_t* params, float* dst, size_t dstC, size_t tail, int first)
		{
			float32x4_t d00, d01, d10, d11, d20, d21, s0, w0, w1;
			if (tail > F)
			{
				if (first)
				{
					d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
					d10 = vdupq_n_f32(0.0f), d11 = vdupq_n_f32(0.0f);
					d20 = vdupq_n_f32(0.0f), d21 = vdupq_n_f32(0.0f);
				}
				else
				{
					d00 = Load<false>(dst + 0 * dstC + 0), d01 = Load<false>(dst + 0 * dstC + F);
					d10 = Load<false>(dst + 1 * dstC + 0), d11 = Load<false>(dst + 1 * dstC + F);
					d20 = Load<false>(dst + 2 * dstC + 0), d21 = Load<false>(dst + 2 * dstC + F);
				}					
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = Load<false>(weight + 0);
						w1 = Load<false>(weight + F);
						s0 = vld1q_dup_f32(src + i + 0 * F);
						d00 = vmlaq_f32(d00, s0, w0);
						d01 = vmlaq_f32(d01, s0, w1);
						s0 = vld1q_dup_f32(src + i + 1 * F);
						d10 = vmlaq_f32(d10, s0, w0);
						d11 = vmlaq_f32(d11, s0, w1);
						s0 = vld1q_dup_f32(src + i + 2 * F);
						d20 = vmlaq_f32(d20, s0, w0);
						d21 = vmlaq_f32(d21, s0, w1);
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
				if (first)
				{
					d00 = vdupq_n_f32(0.0f);
					d10 = vdupq_n_f32(0.0f);
					d20 = vdupq_n_f32(0.0f);
				}
				else
				{
					d00 = Load<false>(dst + 0 * dstC + 0);
					d10 = Load<false>(dst + 1 * dstC + 0);
					d20 = Load<false>(dst + 2 * dstC + 0);
				}
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = Load<false>(weight + 0);
						s0 = vld1q_dup_f32(src + i + 0 * F);
						d00 = vmlaq_f32(d00, s0, w0);
						s0 = vld1q_dup_f32(src + i + 1 * F);
						d10 = vmlaq_f32(d10, s0, w0);
						s0 = vld1q_dup_f32(src + i + 2 * F);
						d20 = vmlaq_f32(d20, s0, w0);
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
			const float* weight, const float32x4_t* bias, const float32x4_t* params, float* dst, size_t dstC, size_t tail, int first)
		{
			float32x4_t d00, d01, s0, w0, w1;
			if (tail > F)
			{
				if (first)
					d00 = vdupq_n_f32(0.0f), d01 = vdupq_n_f32(0.0f);
				else
					d00 = Load<false>(dst + 0 * dstC + 0), d01 = Load<false>(dst + 0 * dstC + F);
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = Load<false>(weight + 0);
						w1 = Load<false>(weight + F);
						s0 = vld1q_dup_f32(src + i + 0 * F);
						d00 = vmlaq_f32(d00, s0, w0);
						d01 = vmlaq_f32(d01, s0, w1);
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
				if (first)
					d00 = vdupq_n_f32(0.0f);
				else
					d00 = Load<false>(dst + 0 * dstC + 0);
				for (size_t c = 0; c < srcC; c += F)
				{
					size_t n = Simd::Min(F, srcC - c);
					for (size_t i = 0; i < n; ++i, weight += DF)
					{
						w0 = Load<false>(weight + 0);
						s0 = vld1q_dup_f32(src + i + 0 * F);
						d00 = vmlaq_f32(d00, s0, w0);
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
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			assert(p.group == 1 && p.kernelY == 1 && p.strideY == 1);
			size_t srcH = p.srcH, srcW = p.srcW, dstW = p.dstW, dstC = p.dstC;
			size_t srcM = (bufH[1] - 1), srcS = bufH[1] * srcW * F;
			size_t dstW3 = AlignLoAny(dstW, 3), dstW6 = AlignLoAny(dstW, 6);
			float32x4_t _params[2], _bias[2];
			_params[0] = vdupq_n_f32(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
				_params[1] = vdupq_n_f32(params[1]);

			dst += yBeg * p.dstW * p.dstC;
			size_t dc = 0;
			for (; dc < dstC; dc += DF)
			{
				size_t tail = Simd::Min(DF, dstC - dc);
				_bias[0] = Load<false>(bias + dc + 0);
				_bias[1] = Load<false>(bias + dc + F);
				if (type == ::SimdConvolutionActivationPrelu)
				{
					_params[0] = Load<false>(params + dc + 0);
					_params[1] = Load<false>(params + dc + F);
				}
				float* pDst = dst + dc;
				for (size_t y = yBeg; y < yEnd; ++y)
				{
					const float* pSrc = src + (y & srcM) * srcW * F;
					size_t x = 0;
					for (; x < dstW6; x += 6, pDst += 6 * dstC, pSrc += 6 * F)
						OutputConvolution_2x6<term, type>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail, first);
					if (dstW - dstW6 == 4)
						OutputConvolution_2x4<term, type>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail, first), pDst += 4 * dstC;
					else
					{
						for (; x < dstW3; x += 3, pDst += 3 * dstC, pSrc += 3 * F)
							OutputConvolution_2x3<term, type>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail, first);
						for (; x < dstW; ++x, pDst += dstC, pSrc += F)
							OutputConvolution_2x1<term, type>(pSrc, srcC, srcS, weight, _bias, _params, pDst, dstC, tail, first);
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
