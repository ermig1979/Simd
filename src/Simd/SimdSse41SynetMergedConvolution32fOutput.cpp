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
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Sse41
	{
		template<TermType term, SimdConvolutionActivationType type> void OutputConvolution_2x6(const float* src, size_t srcC, size_t srcS,
			const float* weight, const __m128* bias, const __m128* params, float* dst, size_t dstC, size_t tail, int first)
		{
			__m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, s0, w0, w1;
			if (tail > F)
			{
				if (first)
				{
					d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
					d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
					d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
					d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
					d40 = _mm_setzero_ps(), d41 = _mm_setzero_ps();
					d50 = _mm_setzero_ps(), d51 = _mm_setzero_ps();
				}
				else
				{
					d00 = _mm_loadu_ps(dst + 0 * dstC + 0), d01 = _mm_loadu_ps(dst + 0 * dstC + F);
					d10 = _mm_loadu_ps(dst + 1 * dstC + 0), d11 = _mm_loadu_ps(dst + 1 * dstC + F);
					d20 = _mm_loadu_ps(dst + 2 * dstC + 0), d21 = _mm_loadu_ps(dst + 2 * dstC + F);
					d30 = _mm_loadu_ps(dst + 3 * dstC + 0), d31 = _mm_loadu_ps(dst + 3 * dstC + F);
					d40 = _mm_loadu_ps(dst + 4 * dstC + 0), d41 = _mm_loadu_ps(dst + 4 * dstC + F);
					d50 = _mm_loadu_ps(dst + 5 * dstC + 0), d51 = _mm_loadu_ps(dst + 5 * dstC + F);
				}
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
				if (first)
				{
					d00 = _mm_setzero_ps();
					d10 = _mm_setzero_ps();
					d20 = _mm_setzero_ps();
					d30 = _mm_setzero_ps();
					d40 = _mm_setzero_ps();
					d50 = _mm_setzero_ps();
				}
				else
				{
					d00 = _mm_loadu_ps(dst + 0 * dstC + 0);
					d10 = _mm_loadu_ps(dst + 1 * dstC + 0);
					d20 = _mm_loadu_ps(dst + 2 * dstC + 0);
					d30 = _mm_loadu_ps(dst + 3 * dstC + 0);
					d40 = _mm_loadu_ps(dst + 4 * dstC + 0);
					d50 = _mm_loadu_ps(dst + 5 * dstC + 0);
				}
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
			const float* weight, const __m128* bias, const __m128* params, float* dst, size_t dstC, size_t tail, int first)
		{
			__m128 d00, d01, d10, d11, d20, d21, d30, d31, s0, w0, w1;
			if (tail > F)
			{
				if (first)
				{
					d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
					d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
					d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
					d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
				}
				else
				{
					d00 = _mm_loadu_ps(dst + 0 * dstC + 0), d01 = _mm_loadu_ps(dst + 0 * dstC + F);
					d10 = _mm_loadu_ps(dst + 1 * dstC + 0), d11 = _mm_loadu_ps(dst + 1 * dstC + F);
					d20 = _mm_loadu_ps(dst + 2 * dstC + 0), d21 = _mm_loadu_ps(dst + 2 * dstC + F);
					d30 = _mm_loadu_ps(dst + 3 * dstC + 0), d31 = _mm_loadu_ps(dst + 3 * dstC + F);
				}
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
				if (first)
				{
					d00 = _mm_setzero_ps();
					d10 = _mm_setzero_ps();
					d20 = _mm_setzero_ps();
					d30 = _mm_setzero_ps();
				}
				else
				{
					d00 = _mm_loadu_ps(dst + 0 * dstC + 0);
					d10 = _mm_loadu_ps(dst + 1 * dstC + 0);
					d20 = _mm_loadu_ps(dst + 2 * dstC + 0);
					d30 = _mm_loadu_ps(dst + 3 * dstC + 0);
				}
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
			const float* weight, const __m128* bias, const __m128* params, float* dst, size_t dstC, size_t tail, int first)
		{
			__m128 d00, d01, d10, d11, d20, d21, s0, w0, w1;
			if (tail > F)
			{
				if (first)
				{
					d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
					d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
					d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
				}
				else
				{
					d00 = _mm_loadu_ps(dst + 0 * dstC + 0), d01 = _mm_loadu_ps(dst + 0 * dstC + F);
					d10 = _mm_loadu_ps(dst + 1 * dstC + 0), d11 = _mm_loadu_ps(dst + 1 * dstC + F);
					d20 = _mm_loadu_ps(dst + 2 * dstC + 0), d21 = _mm_loadu_ps(dst + 2 * dstC + F);
				}
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
				if (first)
				{
					d00 = _mm_setzero_ps();
					d10 = _mm_setzero_ps();
					d20 = _mm_setzero_ps();
				}
				else
				{
					d00 = _mm_loadu_ps(dst + 0 * dstC + 0);
					d10 = _mm_loadu_ps(dst + 1 * dstC + 0);
					d20 = _mm_loadu_ps(dst + 2 * dstC + 0);
				}
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
			const float* weight, const __m128* bias, const __m128* params, float* dst, size_t dstC, size_t tail, int first)
		{
			__m128 d00, d01, s0, w0, w1;
			if (tail > F)
			{
				if (first)
					d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
				else
					d00 = _mm_loadu_ps(dst + 0 * dstC + 0), d01 = _mm_loadu_ps(dst + 0 * dstC + F);
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
				if (first)
					d00 = _mm_setzero_ps();
				else
					d00 = _mm_loadu_ps(dst + 0 * dstC + 0);
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
			size_t srcC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
		{
			assert(p.group == 1 && p.kernelY == 1 && p.strideY == 1);
			size_t srcH = p.srcH, srcW = p.srcW, dstW = p.dstW, dstC = p.dstC;
			size_t srcM = (bufH[1] - 1), srcS = bufH[1] * srcW * F;
			size_t dstW3 = AlignLoAny(dstW, 3), dstW6 = AlignLoAny(dstW, 6);
			__m128 _params[2], _bias[2];
			_params[0] = _mm_set1_ps(params[0]);
			if (type == SimdConvolutionActivationRestrictRange ||
				type == SimdConvolutionActivationHswish ||
				type == SimdConvolutionActivationHardSigmoid)
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
