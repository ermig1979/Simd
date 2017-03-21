/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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

#include <vector>

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
	namespace Sse41
	{
        namespace
        {
            const int K_COSSIN = INT16_MAX;
            const int K_SHIFT = 6;

            struct Buffer
            {
                const int size;
                __m128i * cos, * sin;
                __m128i * pos, * neg; 
                int * index;
                float * value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization/2)
                {
                    width = AlignHi(width, A/sizeof(float));
                    _p = Allocate(width*(sizeof(int) + sizeof(float)) + (sizeof(__m128i) + sizeof(__m128i))*2*size);
                    index = (int*)_p - 1;
                    value = (float*)index + width;
                    cos = (__m128i*)(value + width + 1);
                    sin = cos + size;
                    pos = (__m128i*)(sin + size);
                    neg = pos + size;
                    for(int i = 0; i < size; ++i)
                    {
                        cos[i] = _mm_set1_epi16(int16_t(::cos(i*M_PI/size)*K_COSSIN));
                        sin[i] = _mm_set1_epi16(int16_t(::sin(i*M_PI/size)*K_COSSIN));
                        pos[i] = _mm_set1_epi16(i);
                        neg[i] = _mm_set1_epi16(size + i);
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

        template <bool align, int part> SIMD_INLINE void Store(const __m128i & dx, const __m128i & dy, const __m128i & index, Buffer & buffer, size_t col)
        {
            Sse2::Store<align>((__m128i*)(buffer.index + col), UnpackU16<part>(index));
            __m128i xy = UnpackU16<part>(dx, dy);
            Sse::Store<align>(buffer.value + col, Sse::Sqrt<1>(_mm_cvtepi32_ps(_mm_madd_epi16(xy, xy))));
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const __m128i & dx, const __m128i & dy, Buffer & buffer, size_t col)
        {
            __m128i bestDot = _mm_setzero_si128();
            __m128i bestIndex = _mm_setzero_si128();
            __m128i sdx = _mm_slli_epi16(dx, K_SHIFT);
            __m128i sdy = _mm_slli_epi16(dy, K_SHIFT);
            for(int i = 0; i < buffer.size; ++i)
            {
                __m128i dot = _mm_add_epi16(_mm_mulhi_epi16(sdx, buffer.cos[i]), _mm_mulhi_epi16(sdy, buffer.sin[i]));
                __m128i mask = _mm_cmpgt_epi16(dot, bestDot);
                bestDot = _mm_max_epi16(dot, bestDot);
                bestIndex = _mm_blendv_epi8(bestIndex, buffer.pos[i], mask);

                dot = _mm_sub_epi16(_mm_setzero_si128(), dot);
                mask = _mm_cmpgt_epi16(dot, bestDot);
                bestDot = _mm_max_epi16(dot, bestDot);
                bestIndex = _mm_blendv_epi8(bestIndex, buffer.neg[i], mask);
            }
            Store<align, 0>(dx, dy, bestIndex, buffer, col + 0);
            Store<align, 1>(dx, dy, bestIndex, buffer, col + 4);
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const uint8_t * src, size_t stride, Buffer & buffer, size_t col)
        {
            const uint8_t * s = src + col;
            __m128i t = Load<false>((__m128i*)(s - stride));
            __m128i l = Load<false>((__m128i*)(s - 1));
            __m128i r = Load<false>((__m128i*)(s + 1));
            __m128i b = Load<false>((__m128i*)(s + stride));
            HogDirectionHistograms<align>(SubUnpackedU8<0>(r, l), SubUnpackedU8<0>(b, t), buffer, col + 0);
            HogDirectionHistograms<align>(SubUnpackedU8<1>(r, l), SubUnpackedU8<1>(b, t), buffer, col + 8);
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
#endif// SIMD_SSE41_ENABLE
}