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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
	namespace Neon
	{
        namespace
        {
            struct Buffer
            {
                const int size;
                float32x4_t * cos, * sin;
                int32x4_t * pos, * neg; 
                int * index;
                float * value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization/2)
                {
                    width = AlignHi(width, A/sizeof(float));
                    _p = Allocate(width*(sizeof(int) + sizeof(float)) + (sizeof(int32x4_t) + sizeof(float32x4_t))*2*size);
                    index = (int*)_p - 1;
                    value = (float*)index + width;
                    cos = (float32x4_t*)(value + width + 1);
                    sin = cos + size;
                    pos = (int32x4_t*)(sin + size);
                    neg = pos + size;
                    for(int i = 0; i < size; ++i)
                    {
                        cos[i] = vdupq_n_f32((float)::cos(i*M_PI/size));
                        sin[i] = vdupq_n_f32((float)::sin(i*M_PI/size));
                        pos[i] = vdupq_n_s32(i);
                        neg[i] = vdupq_n_s32(size + i);
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

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const float32x4_t & dx, const float32x4_t & dy, Buffer & buffer, size_t col)
        {
            float32x4_t bestDot = vdupq_n_f32(0);
            int32x4_t bestIndex = vdupq_n_s32(0);
            for(int i = 0; i < buffer.size; ++i)
            {
                float32x4_t dot = vaddq_f32(vmulq_f32(dx, buffer.cos[i]), vmulq_f32(dy, buffer.sin[i]));
                uint32x4_t mask = vcgtq_f32(dot, bestDot);
                bestDot = vmaxq_f32(dot, bestDot);
                bestIndex = vbslq_s32(mask, buffer.pos[i], bestIndex);

                dot = vnegq_f32(dot);
                mask = vcgtq_f32(dot, bestDot);
                bestDot = vmaxq_f32(dot, bestDot);
                bestIndex = vbslq_s32(mask, buffer.neg[i], bestIndex);
            }
            Store<align>(buffer.index + col, bestIndex);
            Store<align>(buffer.value + col, Sqrt<SIMD_NEON_RCP_ITER>(vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy))));
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const int16x8_t & dx, const int16x8_t & dy, Buffer & buffer, size_t col)
        {
            HogDirectionHistograms<align>(ToFloat<0>(dx), ToFloat<0>(dy), buffer, col + 0);
            HogDirectionHistograms<align>(ToFloat<1>(dx), ToFloat<1>(dy), buffer, col + 4);
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const uint8_t * src, size_t stride, Buffer & buffer, size_t col)
        {
            const uint8_t * s = src + col;
            uint8x16_t t = Load<false>(s - stride);
            uint8x16_t l = Load<false>(s - 1);
            uint8x16_t r = Load<false>(s + 1);
            uint8x16_t b = Load<false>(s + stride);
            HogDirectionHistograms<align>(Sub<0>(r, l), Sub<0>(b, t), buffer, col + 0);
            HogDirectionHistograms<align>(Sub<1>(r, l), Sub<1>(b, t), buffer, col + 8);
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
#endif// SIMD_NEON_ENABLE
}