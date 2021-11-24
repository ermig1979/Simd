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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) && defined(SIMD_INT8_DEBUG_ENABLE) 
    namespace Avx512bw
    {
		using AlgParam = SynetConvolution8iNhwcDepthwise::AlgParam;
		using ConvolutionPtr = SynetConvolution8iNhwcDepthwise::ConvolutionPtr;

		SIMD_INLINE __m512i LoadAs32i(const uint8_t* src, __mmask16 tail = -1)
		{
			return _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, src));
		}

		SIMD_INLINE __m512i LoadAs32i(const int8_t* src, __mmask16 tail = -1)
		{
			return _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(tail, src));
		}

		SIMD_INLINE void Madd1(__m512i& i32, __m512i u8, __m512i i8)
		{
			i32 = _mm512_add_epi32(i32, _mm512_madd_epi16(u8, i8));
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
									Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
									Madd1(d01, LoadAs32i(src + os + 1 * F), w1);
									Madd1(d02, LoadAs32i(src + os + 2 * F), w2);
									Madd1(d03, LoadAs32i(src + os + 3 * F), w3);
								}
								else
								{
									Madd1(d00, zero, w0);
									Madd1(d01, zero, w1);
									Madd1(d02, zero, w2);
									Madd1(d03, zero, w3);
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
									Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
									Madd1(d01, LoadAs32i(src + os + 1 * F), w1);
								}
								else
								{
									Madd1(d00, zero, w0);
									Madd1(d01, zero, w1);
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
								Madd1(d00, s0, w0);
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
								Madd1(d00, s0, w0);
							}
						}
						Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, i, tail);
					}
					dst += p.dstC * a.size;
				}
			}
		}

		//---------------------------------------------------------------------

		template<Term8iType term, SimdConvolutionActivationType activation, bool nofma> SIMD_INLINE void ConvolutionNhwcDepthwise3x3Edge(
			const uint8_t* src, const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, const int8_t* weight,
			const float* norm, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
		{
			__m512i zero = _mm512_set1_epi32(a.zero);
			__m128i upper = _mm_set1_epi32(a.upper);
			__m512i d00, d01, d02, d03, w0, w1, w2, w3, s0;
			size_t srcC = p.srcC;
			size_t srcCF = AlignLo(srcC, F);
			size_t srcCF2 = AlignLo(srcC, F * 2);
			size_t srcCF4 = AlignLo(srcC, F * 4);
			__mmask16 tail = TailMask16(srcC - srcCF);
			size_t c = 0;
			for (; c < srcCF4; c += F * 4)
			{
				d00 = _mm512_setzero_si512();
				d01 = _mm512_setzero_si512();
				d02 = _mm512_setzero_si512();
				d03 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					size_t sy = dy * p.strideY + ky - p.padY;
					for (size_t kx = 0; kx < 3; ++kx)
					{
						size_t sx = dx * p.strideX + kx - p.padX;
						size_t ow = (ky * 3 + kx) * srcC + c;
						w0 = LoadAs32i(weight + ow + 0 * F);
						w1 = LoadAs32i(weight + ow + 1 * F);
						w2 = LoadAs32i(weight + ow + 2 * F);
						w3 = LoadAs32i(weight + ow + 3 * F);
						if (sy < p.srcH && sx < p.srcW)
						{
							size_t os = (sy * p.srcW + sx) * srcC + c;
							Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
							Madd1(d01, LoadAs32i(src + os + 1 * F), w1);
							Madd1(d02, LoadAs32i(src + os + 2 * F), w2);
							Madd1(d03, LoadAs32i(src + os + 3 * F), w3);
						}
						else
						{
							Madd1(d00, zero, w0);
							Madd1(d01, zero, w1);
							Madd1(d02, zero, w2);
							Madd1(d03, zero, w3);
						}
					}
				}
				Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst, d01, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst, d02, norm, bias, params, scale, shift, upper, c + F * 2);
				Save<term, activation, nofma>(dst, d03, norm, bias, params, scale, shift, upper, c + F * 3);
			}
			for (; c < srcCF2; c += F * 2)
			{
				d00 = _mm512_setzero_si512();
				d01 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					size_t sy = dy * p.strideY + ky - p.padY;
					for (size_t kx = 0; kx < 3; ++kx)
					{
						size_t sx = dx * p.strideX + kx - p.padX;
						size_t ow = (ky * 3 + kx) * srcC + c;
						w0 = LoadAs32i(weight + ow + 0 * F);
						w1 = LoadAs32i(weight + ow + 1 * F);
						if (sy < p.srcH && sx < p.srcW)
						{
							size_t os = (sy * p.srcW + sx) * srcC + c;
							Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
							Madd1(d01, LoadAs32i(src + os + 1 * F), w1);
						}
						else
						{
							Madd1(d00, zero, w0);
							Madd1(d01, zero, w1);
						}
					}
				}
				Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst, d01, norm, bias, params, scale, shift, upper, c + F * 1);
			}
			for (; c < srcCF; c += F)
			{
				d00 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					size_t sy = dy * p.strideY + ky - p.padY;
					for (size_t kx = 0; kx < 3; ++kx)
					{
						size_t sx = dx * p.strideX + kx - p.padX;
						w0 = LoadAs32i(weight + (ky * 3 + kx) * srcC + c);
						if (sy < p.srcH && sx < p.srcW)
							s0 = LoadAs32i(src + (sy * p.srcW + sx) * srcC + c);
						else
							s0 = zero;
						Madd1(d00, s0, w0);
					}
				}
				Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, c);
			}
			for (; c < srcC; c += F)
			{
				d00 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					size_t sy = dy * p.strideY + ky - p.padY;
					for (size_t kx = 0; kx < 3; ++kx)
					{
						size_t sx = dx * p.strideX + kx - p.padX;
						w0 = LoadAs32i(weight + (ky * 3 + kx) * srcC + c, tail);
						if (sy < p.srcH && sx < p.srcW)
							s0 = LoadAs32i(src + (sy * p.srcW + sx) * srcC + c, tail);
						else
							s0 = zero;
						Madd1(d00, s0, w0);
					}
				}
				Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, c, tail);
			}
		}

		template<Term8iType term, SimdConvolutionActivationType activation, bool nofma> SIMD_INLINE void ConvolutionNhwcDepthwise3x3Main1(
			const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight,
			const float* norm, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
		{
			__m512i zero = _mm512_set1_epi32(a.zero);
			__m128i upper = _mm_set1_epi32(a.upper);
			__m512i d00, d01, d02, d03, w0, s0;
			size_t srcC = p.srcC;
			size_t srcCF = AlignLo(srcC, F);
			size_t srcCF2 = AlignLo(srcC, F * 2);
			size_t srcCF4 = AlignLo(srcC, F * 4);
			size_t srcS = srcC * p.srcW;
			__mmask16 tail = TailMask16(srcC - srcCF);
			size_t c = 0;
			for (; c < srcCF4; c += F * 4)
			{
				d00 = _mm512_setzero_si512();
				d01 = _mm512_setzero_si512();
				d02 = _mm512_setzero_si512();
				d03 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						Madd1(d00, LoadAs32i(ps + 0 * F), LoadAs32i(pw + 0 * F));
						Madd1(d01, LoadAs32i(ps + 1 * F), LoadAs32i(pw + 1 * F));
						Madd1(d02, LoadAs32i(ps + 2 * F), LoadAs32i(pw + 2 * F));
						Madd1(d03, LoadAs32i(ps + 3 * F), LoadAs32i(pw + 3 * F));
					}
				}
				Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst, d01, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst, d02, norm, bias, params, scale, shift, upper, c + F * 2);
				Save<term, activation, nofma>(dst, d03, norm, bias, params, scale, shift, upper, c + F * 3);
			}
			for (; c < srcCF2; c += F * 2)
			{
				d00 = _mm512_setzero_si512();
				d01 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						Madd1(d00, LoadAs32i(ps + 0 * F), LoadAs32i(pw + 0 * F));
						Madd1(d01, LoadAs32i(ps + 1 * F), LoadAs32i(pw + 1 * F));
					}
				}
				Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst, d01, norm, bias, params, scale, shift, upper, c + F * 1);
			}
			for (; c < srcCF; c += F)
			{
				d00 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx)
					{
						Madd1(d00, LoadAs32i(ps + 0 * F), LoadAs32i(pw + 0 * F));
					}
				}
				Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, c);
			}
			for (; c < srcC; c += F)
			{
				d00 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx)
					{
						w0 = LoadAs32i(pw + kx * srcC, tail);
						s0 = LoadAs32i(ps + kx * srcC, tail);
						Madd1(d00, s0, w0);
					}
				}
				Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, c, tail);
			}
		}

		template<Term8iType term, SimdConvolutionActivationType activation, bool nofma> SIMD_INLINE void ConvolutionNhwcDepthwise3x3Main2(
			const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight,
			const float* norm, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
		{
			__m512i zero = _mm512_set1_epi32(a.zero);
			__m128i upper = _mm_set1_epi32(a.upper);
			__m512i d00, d01, d02, d03, d10, d11, d12, d13, w0;
			size_t srcC = p.srcC;
			size_t srcCF = AlignLo(srcC, F);
			size_t srcCF2 = AlignLo(srcC, F * 2);
			size_t srcCF4 = AlignLo(srcC, F * 4);
			size_t srcS = srcC * p.srcW;
			size_t srcX = srcC * p.strideX;
			__mmask16 tail = TailMask16(srcC - srcCF);
			size_t c = 0;
			for (; c < srcCF4; c += F * 4)
			{
				d00 = _mm512_setzero_si512();
				d01 = _mm512_setzero_si512();
				d02 = _mm512_setzero_si512();
				d03 = _mm512_setzero_si512();
				d10 = _mm512_setzero_si512();
				d11 = _mm512_setzero_si512();
				d12 = _mm512_setzero_si512();
				d13 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						w0 = LoadAs32i(pw + 0 * F);
						Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX), w0);
						Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX), w0);
						w0 = LoadAs32i(pw + 1 * F);
						Madd1(d01, LoadAs32i(ps + 1 * F + 0 * srcX), w0);
						Madd1(d11, LoadAs32i(ps + 1 * F + 1 * srcX), w0);
						w0 = LoadAs32i(pw + 2 * F);
						Madd1(d02, LoadAs32i(ps + 2 * F + 0 * srcX), w0);
						Madd1(d12, LoadAs32i(ps + 2 * F + 1 * srcX), w0);
						w0 = LoadAs32i(pw + 3 * F);
						Madd1(d03, LoadAs32i(ps + 3 * F + 0 * srcX), w0);
						Madd1(d13, LoadAs32i(ps + 3 * F + 1 * srcX), w0);
					}
				}
				Save<term, activation, nofma>(dst + 0 * srcC, d00, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 0 * srcC, d01, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 0 * srcC, d02, norm, bias, params, scale, shift, upper, c + F * 2);
				Save<term, activation, nofma>(dst + 0 * srcC, d03, norm, bias, params, scale, shift, upper, c + F * 3);
				Save<term, activation, nofma>(dst + 1 * srcC, d10, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 1 * srcC, d11, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 1 * srcC, d12, norm, bias, params, scale, shift, upper, c + F * 2);
				Save<term, activation, nofma>(dst + 1 * srcC, d13, norm, bias, params, scale, shift, upper, c + F * 3);
			}
			for (; c < srcCF2; c += F * 2)
			{
				d00 = _mm512_setzero_si512();
				d01 = _mm512_setzero_si512();
				d10 = _mm512_setzero_si512();
				d11 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						w0 = LoadAs32i(pw + 0 * F);
						Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX), w0);
						Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX), w0);
						w0 = LoadAs32i(pw + 1 * F);
						Madd1(d01, LoadAs32i(ps + 1 * F + 0 * srcX), w0);
						Madd1(d11, LoadAs32i(ps + 1 * F + 1 * srcX), w0);
					}
				}
				Save<term, activation, nofma>(dst + 0 * srcC, d00, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 0 * srcC, d01, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 1 * srcC, d10, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 1 * srcC, d11, norm, bias, params, scale, shift, upper, c + F * 1);
			}
			for (; c < srcCF; c += F)
			{
				d00 = _mm512_setzero_si512();
				d10 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						w0 = LoadAs32i(pw + 0 * F);
						Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX), w0);
						Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX), w0);
					}
				}
				Save<term, activation, nofma>(dst + 0 * srcC, d00, norm, bias, params, scale, shift, upper, c);
				Save<term, activation, nofma>(dst + 1 * srcC, d10, norm, bias, params, scale, shift, upper, c);
			}
			for (; c < srcC; c += F)
			{
				d00 = _mm512_setzero_si512();
				d10 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						w0 = LoadAs32i(pw + 0 * F, tail);
						Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX, tail), w0);
						Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX, tail), w0);
					}
				}
				Save<term, activation, nofma>(dst + 0 * srcC, d00, norm, bias, params, scale, shift, upper, c, tail);
				Save<term, activation, nofma>(dst + 1 * srcC, d10, norm, bias, params, scale, shift, upper, c, tail);
			}
		}

		template<Term8iType term, SimdConvolutionActivationType activation, bool nofma> SIMD_INLINE void ConvolutionNhwcDepthwise3x3Main4(
			const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight,
			const float* norm, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
		{
			__m512i zero = _mm512_set1_epi32(a.zero);
			__m128i upper = _mm_set1_epi32(a.upper);
			__m512i d00, d01, d02, d03, d10, d11, d12, d13, d20, d21, d22, d23, d30, d31, d32, d33, w0;
			size_t srcC = p.srcC;
			size_t srcCF = AlignLo(srcC, F);
			size_t srcCF2 = AlignLo(srcC, F * 2);
			size_t srcCF4 = AlignLo(srcC, F * 4);
			size_t srcS = srcC * p.srcW;
			size_t srcX = srcC * p.strideX;
			__mmask16 tail = TailMask16(srcC - srcCF);
			size_t c = 0;
			for (; c < srcCF4; c += F * 4)
			{
				d00 = _mm512_setzero_si512();
				d01 = _mm512_setzero_si512();
				d02 = _mm512_setzero_si512();
				d03 = _mm512_setzero_si512();
				d10 = _mm512_setzero_si512();
				d11 = _mm512_setzero_si512();
				d12 = _mm512_setzero_si512();
				d13 = _mm512_setzero_si512();
				d20 = _mm512_setzero_si512();
				d21 = _mm512_setzero_si512();
				d22 = _mm512_setzero_si512();
				d23 = _mm512_setzero_si512();
				d30 = _mm512_setzero_si512();
				d31 = _mm512_setzero_si512();
				d32 = _mm512_setzero_si512();
				d33 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						w0 = LoadAs32i(pw + 0 * F);
						Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX), w0);
						Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX), w0);
						Madd1(d20, LoadAs32i(ps + 0 * F + 2 * srcX), w0);
						Madd1(d30, LoadAs32i(ps + 0 * F + 3 * srcX), w0);
						w0 = LoadAs32i(pw + 1 * F);
						Madd1(d01, LoadAs32i(ps + 1 * F + 0 * srcX), w0);
						Madd1(d11, LoadAs32i(ps + 1 * F + 1 * srcX), w0);
						Madd1(d21, LoadAs32i(ps + 1 * F + 2 * srcX), w0);
						Madd1(d31, LoadAs32i(ps + 1 * F + 3 * srcX), w0);
						w0 = LoadAs32i(pw + 2 * F);
						Madd1(d02, LoadAs32i(ps + 2 * F + 0 * srcX), w0);
						Madd1(d12, LoadAs32i(ps + 2 * F + 1 * srcX), w0);
						Madd1(d22, LoadAs32i(ps + 2 * F + 2 * srcX), w0);
						Madd1(d32, LoadAs32i(ps + 2 * F + 3 * srcX), w0);
						w0 = LoadAs32i(pw + 3 * F);
						Madd1(d03, LoadAs32i(ps + 3 * F + 0 * srcX), w0);
						Madd1(d13, LoadAs32i(ps + 3 * F + 1 * srcX), w0);
						Madd1(d23, LoadAs32i(ps + 3 * F + 2 * srcX), w0);
						Madd1(d33, LoadAs32i(ps + 3 * F + 3 * srcX), w0);
					}
				}
				Save<term, activation, nofma>(dst + 0 * srcC, d00, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 0 * srcC, d01, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 0 * srcC, d02, norm, bias, params, scale, shift, upper, c + F * 2);
				Save<term, activation, nofma>(dst + 0 * srcC, d03, norm, bias, params, scale, shift, upper, c + F * 3);
				Save<term, activation, nofma>(dst + 1 * srcC, d10, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 1 * srcC, d11, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 1 * srcC, d12, norm, bias, params, scale, shift, upper, c + F * 2);
				Save<term, activation, nofma>(dst + 1 * srcC, d13, norm, bias, params, scale, shift, upper, c + F * 3);
				Save<term, activation, nofma>(dst + 2 * srcC, d20, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 2 * srcC, d21, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 2 * srcC, d22, norm, bias, params, scale, shift, upper, c + F * 2);
				Save<term, activation, nofma>(dst + 2 * srcC, d23, norm, bias, params, scale, shift, upper, c + F * 3);
				Save<term, activation, nofma>(dst + 3 * srcC, d30, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 3 * srcC, d31, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 3 * srcC, d32, norm, bias, params, scale, shift, upper, c + F * 2);
				Save<term, activation, nofma>(dst + 3 * srcC, d33, norm, bias, params, scale, shift, upper, c + F * 3);
			}
			for (; c < srcCF2; c += F * 2)
			{
				d00 = _mm512_setzero_si512();
				d01 = _mm512_setzero_si512();
				d10 = _mm512_setzero_si512();
				d11 = _mm512_setzero_si512();
				d20 = _mm512_setzero_si512();
				d21 = _mm512_setzero_si512();
				d30 = _mm512_setzero_si512();
				d31 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						w0 = LoadAs32i(pw + 0 * F);
						Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX), w0);
						Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX), w0);
						Madd1(d20, LoadAs32i(ps + 0 * F + 2 * srcX), w0);
						Madd1(d30, LoadAs32i(ps + 0 * F + 3 * srcX), w0);
						w0 = LoadAs32i(pw + 1 * F);
						Madd1(d01, LoadAs32i(ps + 1 * F + 0 * srcX), w0);
						Madd1(d11, LoadAs32i(ps + 1 * F + 1 * srcX), w0);
						Madd1(d21, LoadAs32i(ps + 1 * F + 2 * srcX), w0);
						Madd1(d31, LoadAs32i(ps + 1 * F + 3 * srcX), w0);
					}
				}
				Save<term, activation, nofma>(dst + 0 * srcC, d00, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 0 * srcC, d01, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 1 * srcC, d10, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 1 * srcC, d11, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 2 * srcC, d20, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 2 * srcC, d21, norm, bias, params, scale, shift, upper, c + F * 1);
				Save<term, activation, nofma>(dst + 3 * srcC, d30, norm, bias, params, scale, shift, upper, c + F * 0);
				Save<term, activation, nofma>(dst + 3 * srcC, d31, norm, bias, params, scale, shift, upper, c + F * 1);
			}
			for (; c < srcCF; c += F)
			{
				d00 = _mm512_setzero_si512();
				d10 = _mm512_setzero_si512();
				d20 = _mm512_setzero_si512();
				d30 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						w0 = LoadAs32i(pw + 0 * F);
						Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX), w0);
						Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX), w0);
						Madd1(d20, LoadAs32i(ps + 0 * F + 2 * srcX), w0);
						Madd1(d30, LoadAs32i(ps + 0 * F + 3 * srcX), w0);
					}
				}
				Save<term, activation, nofma>(dst + 0 * srcC, d00, norm, bias, params, scale, shift, upper, c);
				Save<term, activation, nofma>(dst + 1 * srcC, d10, norm, bias, params, scale, shift, upper, c);
				Save<term, activation, nofma>(dst + 2 * srcC, d20, norm, bias, params, scale, shift, upper, c);
				Save<term, activation, nofma>(dst + 3 * srcC, d30, norm, bias, params, scale, shift, upper, c);
			}
			for (; c < srcC; c += F)
			{
				d00 = _mm512_setzero_si512();
				d10 = _mm512_setzero_si512();
				d20 = _mm512_setzero_si512();
				d30 = _mm512_setzero_si512();
				for (size_t ky = 0; ky < 3; ++ky)
				{
					const uint8_t* ps = src + ky * srcS + c;
					const int8_t* pw = weight + ky * 3 * srcC + c;
					for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
					{
						w0 = LoadAs32i(pw + 0 * F, tail);
						Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX, tail), w0);
						Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX, tail), w0);
						Madd1(d20, LoadAs32i(ps + 0 * F + 2 * srcX, tail), w0);
						Madd1(d30, LoadAs32i(ps + 0 * F + 3 * srcX, tail), w0);
					}
				}
				Save<term, activation, nofma>(dst + 0 * srcC, d00, norm, bias, params, scale, shift, upper, c, tail);
				Save<term, activation, nofma>(dst + 1 * srcC, d10, norm, bias, params, scale, shift, upper, c, tail);
				Save<term, activation, nofma>(dst + 2 * srcC, d20, norm, bias, params, scale, shift, upper, c, tail);
				Save<term, activation, nofma>(dst + 3 * srcC, d30, norm, bias, params, scale, shift, upper, c, tail);
			}
		}

		template<Term8iType term, SimdConvolutionActivationType activation, bool nofma> SIMD_INLINE void ConvolutionNhwcDepthwise3x3(
			const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight, const float* norm,
			const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
		{
			size_t srcS = p.srcC * p.srcW;
			size_t srcX = p.srcC * p.strideX;
			size_t dstH = p.dstH - p.padH;
			size_t dstW = p.dstW - p.padW;
			size_t dstC = p.dstC * a.size;
			size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
			size_t dstW4 = AlignLo(dstW - p.padX, 4) + p.padX;
			size_t dy = 0;
			for (; dy < p.padY; ++dy)
				for (size_t dx = 0; dx < p.dstW; ++dx)
					ConvolutionNhwcDepthwise3x3Edge<term, activation, nofma>(src, p, a, dy, dx, weight, norm, bias, params, scale, shift, dst), dst += dstC;
			for (; dy < dstH; ++dy)
			{
				size_t dx = 0;
				for (; dx < p.padX; ++dx)
					ConvolutionNhwcDepthwise3x3Edge<term, activation, nofma>(src, p, a, dy, dx, weight, norm, bias, params, scale, shift, dst), dst += dstC;
				size_t offset = ((dy * p.strideY - p.padY) * p.srcW + dx * p.strideX - p.padX) * p.srcC;
				for (; dx < dstW4; dx += 4)
					ConvolutionNhwcDepthwise3x3Main4<term, activation, nofma>(src + offset, p, a, weight, norm, bias, params, scale, shift, dst), dst += dstC * 4, offset += srcX * 4;
				for (; dx < dstW2; dx += 2)
					ConvolutionNhwcDepthwise3x3Main2<term, activation, nofma>(src + offset, p, a, weight, norm, bias, params, scale, shift, dst), dst += dstC * 2, offset += srcX * 2;
				for (; dx < dstW; dx += 1)
					ConvolutionNhwcDepthwise3x3Main1<term, activation, nofma>(src + offset, p, a, weight, norm, bias, params, scale, shift, dst), dst += dstC, offset += srcX;
				for (; dx < p.dstW; ++dx)
					ConvolutionNhwcDepthwise3x3Edge<term, activation, nofma>(src, p, a, dy, dx, weight, norm, bias, params, scale, shift, dst), dst += dstC;
			}
			for (; dy < p.dstH; ++dy)
				for (size_t dx = 0; dx < p.dstW; ++dx)
					ConvolutionNhwcDepthwise3x3Edge<term, activation, nofma>(src, p, a, dy, dx, weight, norm, bias, params, scale, shift, dst), dst += dstC;
		}

        //---------------------------------------------------------------------

		template <Term8iType term, SimdConvolutionActivationType activation, bool nofma> void Set(const ConvParam8i& p, ConvolutionPtr& d)
		{
			if (p.IsKernel(3) && p.IsDilation(1))
				d = ConvolutionNhwcDepthwise3x3<term, activation, nofma>;
			else
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
				Set<Term8iLast8u, activation>(p, d);
			else
				Set<Term8iLast32f, activation>(p, d);
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
			case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, d); break;
			case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, d); break;
			case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, d); break;
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
