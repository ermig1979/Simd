/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdAllocator.hpp"

#include <vector>

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
	namespace Avx512bw
	{
		namespace
		{
			struct Buffer
			{
				const int size;
				__m512 * cos, *sin;
				__m512i * pos, *neg;
				int * index;
				float * value;

				Buffer(size_t width, size_t quantization)
					: size((int)quantization / 2)
				{
					width = AlignHi(width, A / sizeof(float));
					_p = Allocate(width*(sizeof(int) + sizeof(float)) + (sizeof(__m512i) + sizeof(__m512)) * 2 * size);
					index = (int*)_p - 1;
					value = (float*)index + width;
					cos = (__m512*)(value + width + 1);
					sin = cos + size;
					pos = (__m512i*)(sin + size);
					neg = pos + size;
					for (int i = 0; i < size; ++i)
					{
						cos[i] = _mm512_set1_ps((float)::cos(i*M_PI / size));
						sin[i] = _mm512_set1_ps((float)::sin(i*M_PI / size));
						pos[i] = _mm512_set1_epi32(i);
						neg[i] = _mm512_set1_epi32(size + i);
					}
				}

				~Buffer()
				{
					Free(_p);
				}

			private:
				void *_p;
			};
		}

		template <bool align> SIMD_INLINE void HogDirectionHistograms(const __m512 & dx, const __m512 & dy, Buffer & buffer, size_t col)
		{
			__m512 bestDot = _mm512_setzero_ps();
			__m512i bestIndex = _mm512_setzero_si512();
			for (int i = 0; i < buffer.size; ++i)
			{
				__m512 dot = _mm512_fmadd_ps(dx, buffer.cos[i], _mm512_mul_ps(dy, buffer.sin[i]));
				bestIndex = _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(dot, bestDot, _CMP_GT_OS), bestIndex, buffer.pos[i]);
				bestDot = _mm512_max_ps(dot, bestDot);

				dot = _mm512_sub_ps(_mm512_setzero_ps(), dot);
				bestIndex = _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(dot, bestDot, _CMP_GT_OS), bestIndex, buffer.neg[i]);
				bestDot = _mm512_max_ps(dot, bestDot);
			}
			Store<align>(buffer.index + col, bestIndex);
			Avx512f::Store<align>(buffer.value + col, _mm512_sqrt_ps(_mm512_fmadd_ps(dx, dx, _mm512_mul_ps(dy, dy))));
		}

		template <int part> SIMD_INLINE __m512 CovertDifference(const __m256i & a, const __m256i & b)
		{
			return _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(Avx2::SubUnpackedU8<part>(a, b)));
		}

		template <bool align> SIMD_INLINE void HogDirectionHistograms(const uint8_t * src, size_t stride, Buffer & buffer, size_t col)
		{
			const uint8_t * s = src + col;
			__m256i t = Avx2::LoadPermuted<false>((__m256i*)(s - stride));
			__m256i l = Avx2::LoadPermuted<false>((__m256i*)(s - 1));
			__m256i r = Avx2::LoadPermuted<false>((__m256i*)(s + 1));
			__m256i b = Avx2::LoadPermuted<false>((__m256i*)(s + stride));
			HogDirectionHistograms<align>(CovertDifference<0>(r, l), CovertDifference<0>(b, t), buffer, col + 0);
			HogDirectionHistograms<align>(CovertDifference<1>(r, l), CovertDifference<1>(b, t), buffer, col + F);
		}

		namespace Custom_8x8_18
		{
			struct Buffer
			{
				__m512i pos[5];
				__m512 cos[5], sin[5];
				__m128 kx[8], ky[8];

				int * index;
				float * value;
				__m128 * hist;
				size_t hs;

				Buffer(size_t width)
				{
					width = AlignHi(width, A / sizeof(float));
					hs = (width / 8 + 1) * 18 * sizeof(__m128);
					_p = Allocate(width*(sizeof(int) + sizeof(float)) + hs);
					index = (int*)_p - 1;
					value = (float*)index + width;
					hist = (__m128*)(value + width + 1);

					for (int i = 0; i < 5; ++i)
					{
						cos[i] = _mm512_set1_ps((float)::cos(i*M_PI / 9));
						sin[i] = _mm512_set1_ps((float)::sin(i*M_PI / 9));
						pos[i] = _mm512_set1_epi32(i);
					}
					for (int i = 0; i < 8; ++i)
					{
						float k0 = float((15 - i * 2) / 16.0f);
						float k1 = 1.0f - k0;
						kx[i] = _mm_setr_ps(k0, k1, k0, k1);
						ky[i] = _mm_setr_ps(k0, k0, k1, k1);
					}
					ClearHist();
				}

				~Buffer()
				{
					Free(_p);
				}

				void ClearHist()
				{
					memset(hist, 0, hs);
				}

			private:
				void *_p;
			};

			const __m512i K32_9 = SIMD_MM512_SET1_EPI32(9);
			const __m512i K32_18 = SIMD_MM512_SET1_EPI32(18);

			template <bool align> SIMD_INLINE void HogDirectionHistograms(const __m512 & dx, const __m512 & dy, Buffer & buffer, size_t col)
			{
				__m512 bestDot = _mm512_setzero_ps();
				__m512i bestIndex = _mm512_setzero_si512();
				__m512 _0 = _mm512_set1_ps(-0.0f);
				__m512 adx = _mm512_andnot_ps(_0, dx);
				__m512 ady = _mm512_andnot_ps(_0, dy);
				for (int i = 0; i < 5; ++i)
				{
					__m512 dot = _mm512_fmadd_ps(adx, buffer.cos[i], _mm512_mul_ps(ady, buffer.sin[i]));
					bestIndex = _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(dot, bestDot, _CMP_GT_OS), bestIndex, buffer.pos[i]);
					bestDot = _mm512_max_ps(dot, bestDot);
				}
				bestIndex = _mm512_mask_sub_epi32(bestIndex, _mm512_cmp_ps_mask(dx, _0, _CMP_LT_OS), K32_9, bestIndex);

				__m512i corr = _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(adx, _mm512_setzero_ps(), _CMP_EQ_OS), 1);
				bestIndex = _mm512_mask_sub_epi32(bestIndex, _mm512_cmp_ps_mask(dy, _0, _CMP_LT_OS), K32_18, _mm512_add_epi32(bestIndex, corr));

				bestIndex = _mm512_mask_set1_epi32(bestIndex, _mm512_cmpeq_epi32_mask(bestIndex, K32_18), 0);

				Store<align>(buffer.index + col, bestIndex);
				Avx512f::Store<align>(buffer.value + col, _mm512_sqrt_ps(_mm512_fmadd_ps(adx, adx, _mm512_mul_ps(ady, ady))));
			}

			template <bool align> SIMD_INLINE void HogDirectionHistograms(const uint8_t * src, size_t stride, Buffer & buffer, size_t col)
			{
				const uint8_t * s = src + col;
				__m256i t = Avx2::LoadPermuted<false>((__m256i*)(s - stride));
				__m256i l = Avx2::LoadPermuted<false>((__m256i*)(s - 1));
				__m256i r = Avx2::LoadPermuted<false>((__m256i*)(s + 1));
				__m256i b = Avx2::LoadPermuted<false>((__m256i*)(s + stride));
				HogDirectionHistograms<align>(CovertDifference<0>(r, l), CovertDifference<0>(b, t), buffer, col + 0);
				HogDirectionHistograms<align>(CovertDifference<1>(r, l), CovertDifference<1>(b, t), buffer, col + F);
			}

			void AddRowToBuffer(const uint8_t * src, size_t stride, Buffer & buffer, size_t row, size_t width, size_t aligned)
			{
				const uint8_t * s = src + stride*row;
				for (size_t col = 1; col < aligned; col += HA)
					HogDirectionHistograms<true>(s, stride, buffer, col);
				HogDirectionHistograms<false>(s, stride, buffer, width - 1 - HA);

				__m128 ky = buffer.ky[(row + 4) & 7];
				__m128 * hist = buffer.hist;
				size_t cellEnd = width / 8;

				for (size_t col = 1; col < 4; ++col)
				{
					int index = buffer.index[col];
					__m128 value = _mm_set1_ps(buffer.value[col]);
					__m128 kx = buffer.kx[(col + 4) & 7];
					hist[index] = _mm_fmadd_ps(_mm_mul_ps(ky, kx), value, hist[index]);
				}
				hist += 18;

				for (size_t cell = 1, col = 4; cell < cellEnd; ++cell)
				{
					for (size_t i = 0; i < 8; ++i, ++col)
					{
						int index = buffer.index[col];
						__m128 value = _mm_set1_ps(buffer.value[col]);
						__m128 kx = buffer.kx[i];
						hist[index] = _mm_fmadd_ps(_mm_mul_ps(ky, kx), value, hist[index]);
					}
					hist += 18;
				}

				for (size_t col = width - 4; col < width - 1; ++col)
				{
					int index = buffer.index[col];
					__m128 value = _mm_set1_ps(buffer.value[col]);
					__m128 kx = buffer.kx[(col + 4) & 7];
					hist[index] = _mm_fmadd_ps(_mm_mul_ps(ky, kx), value, hist[index]);
				}
			}

			void AddToHistogram(Buffer & buffer, size_t row, size_t width, size_t height, float * histograms)
			{
				typedef float f18_t[18];

				float * src = (float*)buffer.hist;
				f18_t * h0 = (f18_t*)histograms + row*width - width - 1;
				f18_t * h1 = h0 + width;

				if (row == 0)
				{
					for (size_t i = 0; i < 18; ++i)
						h1[1][i] += src[i * 4 + 3];
					h1++;
					src += 72;
					for (size_t cell = 1; cell < width; ++cell)
					{
						for (size_t i = 0; i < 18; ++i)
						{
							h1[0][i] += src[i * 4 + 2];
							h1[1][i] += src[i * 4 + 3];
						}
						h1++;
						src += 72;
					}
					for (size_t i = 0; i < 18; ++i)
						h1[0][i] += src[i * 4 + 2];
				}
				else if (row == height)
				{
					for (size_t i = 0; i < 18; ++i)
						h0[1][i] += src[i * 4 + 1];
					h0++;
					src += 72;
					for (size_t cell = 1; cell < width; ++cell)
					{
						for (size_t i = 0; i < 18; ++i)
						{
							h0[0][i] += src[i * 4 + 0];
							h0[1][i] += src[i * 4 + 1];
						}
						h0++;
						src += 72;
					}
					for (size_t i = 0; i < 18; ++i)
						h0[0][i] += src[i * 4 + 0];
				}
				else
				{
					for (size_t i = 0; i < 18; ++i)
					{
						h0[1][i] += src[i * 4 + 1];
						h1[1][i] += src[i * 4 + 3];
					}
					h0++;
					h1++;
					src += 72;
					for (size_t cell = 1; cell < width; ++cell)
					{
						__m512 a0 = Load<true>(src + 0x00, src + 0x10, src + 0x20, src + 0x30);
						__m512 a1 = Load<true>(src + 0x04, src + 0x14, src + 0x24, src + 0x34);
						__m512 a2 = Load<true>(src + 0x08, src + 0x18, src + 0x28, src + 0x38);
						__m512 a3 = Load<true>(src + 0x0C, src + 0x1C, src + 0x2C, src + 0x3C);
						__m512 b0 = _mm512_unpacklo_ps(a0, a2);
						__m512 b1 = _mm512_unpackhi_ps(a0, a2);
						__m512 b2 = _mm512_unpacklo_ps(a1, a3);
						__m512 b3 = _mm512_unpackhi_ps(a1, a3);
						Avx512f::Store<false>(h0[0], _mm512_add_ps(Avx512f::Load<false>(h0[0]), _mm512_unpacklo_ps(b0, b2)));
						Avx512f::Store<false>(h0[1], _mm512_add_ps(Avx512f::Load<false>(h0[1]), _mm512_unpackhi_ps(b0, b2)));
						Avx512f::Store<false>(h1[0], _mm512_add_ps(Avx512f::Load<false>(h1[0]), _mm512_unpacklo_ps(b1, b3)));
						Avx512f::Store<false>(h1[1], _mm512_add_ps(Avx512f::Load<false>(h1[1]), _mm512_unpackhi_ps(b1, b3)));
						for (size_t i = 16; i < 18; ++i)
						{
							h0[0][i] += src[i * 4 + 0];
							h0[1][i] += src[i * 4 + 1];
							h1[0][i] += src[i * 4 + 2];
							h1[1][i] += src[i * 4 + 3];
						}						
						h0++;
						h1++;
						src += 72;
					}
					for (size_t i = 0; i < 18; ++i)
					{
						h0[0][i] += src[i * 4 + 0];
						h1[0][i] += src[i * 4 + 2];
					}
				}
				buffer.ClearHist();
			}

			void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height, float * histograms)
			{
				const size_t quantization = 18;

				size_t sizeX = width / 8, sizeY = height / 8;

				memset(histograms, 0, quantization*sizeX*sizeY * sizeof(float));

				Buffer buffer(width);

				size_t aligned = AlignLo(width - 2, HA) + 1;

				for (size_t row = 1; row < 4; ++row)
					AddRowToBuffer(src, stride, buffer, row, width, aligned);
				AddToHistogram(buffer, 0, sizeX, sizeY, histograms);
				for (size_t row = 4, cell = 1; row < height - 4; ++row)
				{
					AddRowToBuffer(src, stride, buffer, row, width, aligned);
					if ((row & 7) == 3)
						AddToHistogram(buffer, cell++, sizeX, sizeY, histograms);
				}
				for (size_t row = height - 4; row < height - 1; ++row)
					AddRowToBuffer(src, stride, buffer, row, width, aligned);
				AddToHistogram(buffer, sizeY, sizeX, sizeY, histograms);
			}
		}

		void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height,
			size_t cellX, size_t cellY, size_t quantization, float * histograms)
		{
			assert(width%cellX == 0 && height%cellY == 0 && quantization % 2 == 0);

			if (cellX == 8 && cellY == 8 && quantization == 18)
				Custom_8x8_18::HogDirectionHistograms(src, stride, width, height, histograms);
			else
			{
				memset(histograms, 0, quantization*(width / cellX)*(height / cellY) * sizeof(float));

				Buffer buffer(width, quantization);

				size_t alignedWidth = AlignLo(width - 2, HA) + 1;

				for (size_t row = 1; row < height - 1; ++row)
				{
					const uint8_t * s = src + stride*row;
					for (size_t col = 1; col < alignedWidth; col += HA)
						HogDirectionHistograms<true>(s, stride, buffer, col);
					HogDirectionHistograms<false>(s, stride, buffer, width - 1 - HA);
					Base::AddRowToHistograms(buffer.index, buffer.value, row, width, height, cellX, cellY, quantization, histograms);
				}
			}
		}

		SIMD_INLINE void HogDeinterleave(const float * src, size_t count, float ** dst, size_t offset, size_t i)
		{
			src += i;
			__m512 a0 = Load<false>(src + 0x0 * count, src + 0x4 * count, src + 0x8 * count, src + 0xC * count);
			__m512 a1 = Load<false>(src + 0x1 * count, src + 0x5 * count, src + 0x9 * count, src + 0xD * count);
			__m512 a2 = Load<false>(src + 0x2 * count, src + 0x6 * count, src + 0xA * count, src + 0xE * count);
			__m512 a3 = Load<false>(src + 0x3 * count, src + 0x7 * count, src + 0xB * count, src + 0xF * count);
			__m512 b0 = _mm512_unpacklo_ps(a0, a2);
			__m512 b1 = _mm512_unpackhi_ps(a0, a2);
			__m512 b2 = _mm512_unpacklo_ps(a1, a3);
			__m512 b3 = _mm512_unpackhi_ps(a1, a3);
			Avx512f::Store<false>(dst[i + 0] + offset, _mm512_unpacklo_ps(b0, b2));
			Avx512f::Store<false>(dst[i + 1] + offset, _mm512_unpackhi_ps(b0, b2));
			Avx512f::Store<false>(dst[i + 2] + offset, _mm512_unpacklo_ps(b1, b3));
			Avx512f::Store<false>(dst[i + 3] + offset, _mm512_unpackhi_ps(b1, b3));
		}

        void HogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride)
        {
            assert(width >= F && count >= Sse::F);

            size_t alignedCount = AlignLo(count, Sse::F);
            size_t alignedWidth = AlignLo(width, F);

            for (size_t row = 0; row < height; ++row)
            {
                size_t rowOffset = row*dstStride;
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    const float * s = src + count*col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += Sse::F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - Sse::F);
                }
                if (alignedWidth != width)
                {
                    size_t col = width - F;
                    const float * s = src + count*col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += Sse::F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - Sse::F);
                }
                src += srcStride;
            }
        }
	}
#endif
}
