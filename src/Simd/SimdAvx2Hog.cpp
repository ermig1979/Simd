/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
	namespace Avx2
	{
        namespace
        {
            struct Buffer
            {
                const int size;
                __m256 * cos, * sin;
                __m256i * pos, * neg; 
                int * index;
                float * value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization/2)
                {
                    width = AlignHi(width, A/sizeof(float));
                    _p = Allocate(width*(sizeof(int) + sizeof(float)) + (sizeof(__m256i) + sizeof(__m256))*2*size);
                    index = (int*)_p - 1;
                    value = (float*)index + width;
                    cos = (__m256*)(value + width + 1);
                    sin = cos + size;
                    pos = (__m256i*)(sin + size);
                    neg = pos + size;
                    for(int i = 0; i < size; ++i)
                    {
                        cos[i] = _mm256_set1_ps((float)::cos(i*M_PI/size));
                        sin[i] = _mm256_set1_ps((float)::sin(i*M_PI/size));
                        pos[i] = _mm256_set1_epi32(i);
                        neg[i] = _mm256_set1_epi32(size + i);
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

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const __m256 & dx, const __m256 & dy, Buffer & buffer, size_t col)
        {
            __m256 bestDot = _mm256_setzero_ps();
            __m256i bestIndex = _mm256_setzero_si256();
            for(int i = 0; i < buffer.size; ++i)
            {
                __m256 dot = _mm256_add_ps(_mm256_mul_ps(dx, buffer.cos[i]), _mm256_mul_ps(dy, buffer.sin[i]));
                __m256 mask = _mm256_cmp_ps(dot, bestDot, _CMP_GT_OS);
                bestDot = _mm256_max_ps(dot, bestDot);
                bestIndex = Combine(_mm256_castps_si256(mask), buffer.pos[i], bestIndex);

                dot = _mm256_sub_ps(_mm256_setzero_ps(), dot);
                mask = _mm256_cmp_ps(dot, bestDot, _CMP_GT_OS);
                bestDot = _mm256_max_ps(dot, bestDot);
                bestIndex = Combine(_mm256_castps_si256(mask), buffer.neg[i], bestIndex);
            }
            Store<align>((__m256i*)(buffer.index + col), bestIndex);
            Avx::Store<align>(buffer.value + col, _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy))));
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const __m256i & t, const __m256i & l, const __m256i & r, const __m256i & b, Buffer & buffer, size_t col)
        {
            HogDirectionHistograms<align>(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_unpacklo_epi16(r, K_ZERO), _mm256_unpacklo_epi16(l, K_ZERO))), 
                _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_unpacklo_epi16(b, K_ZERO), _mm256_unpacklo_epi16(t, K_ZERO))), 
                buffer, col + 0);
            HogDirectionHistograms<align>(
                _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_unpackhi_epi16(r, K_ZERO), _mm256_unpackhi_epi16(l, K_ZERO))), 
                _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_unpackhi_epi16(b, K_ZERO), _mm256_unpackhi_epi16(t, K_ZERO))), 
                buffer, col + 8);
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const uint8_t * src, size_t stride, Buffer & buffer, size_t col)
        {
            const uint8_t * s = src + col;
            __m256i t = LoadPermuted<false>((__m256i*)(s - stride));
            __m256i l = LoadPermuted<false>((__m256i*)(s - 1));
            __m256i r = LoadPermuted<false>((__m256i*)(s + 1));
            __m256i b = LoadPermuted<false>((__m256i*)(s + stride));
            HogDirectionHistograms<align>(PermutedUnpackLoU8(t), PermutedUnpackLoU8(l), PermutedUnpackLoU8(r), PermutedUnpackLoU8(b), buffer, col + 0);
            HogDirectionHistograms<align>(PermutedUnpackHiU8(t), PermutedUnpackHiU8(l), PermutedUnpackHiU8(r), PermutedUnpackHiU8(b), buffer, col + 16);
        }

        void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height, 
            size_t cellX, size_t cellY, size_t quantization, float * histograms)
        {
            assert(width%cellX == 0 && height%cellY == 0 && quantization%2 == 0);

            Buffer buffer(width, quantization);

            memset(histograms, 0, quantization*(width/cellX)*(height/cellY)*sizeof(float));

            size_t alignedWidth = AlignLo(width - 2, A) + 1;

            for (size_t row = 1; row < height - 1; ++row) 
            {
                const uint8_t * s = src + stride*row;
                for (size_t col = 1; col < alignedWidth; col += A)
                    HogDirectionHistograms<true>(s, stride, buffer, col);
                HogDirectionHistograms<false>(s, stride, buffer, width - 1 - A);
                Base::AddRowToHistograms(buffer.index, buffer.value, row, width, height, cellX, cellY, quantization, histograms);
            }
        }
	}
#endif
}