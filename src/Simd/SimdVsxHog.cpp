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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE    
    namespace Vsx
    {
        namespace
        {
            struct Buffer
            {
                const int size;
                v128_f32 * cos, *sin;
                v128_s32 * pos, *neg;
                int * index;
                float * value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization / 2)
                {
                    width = AlignHi(width, A / sizeof(float));
                    _p = Allocate(width*(sizeof(int) + sizeof(float)) + (sizeof(v128_s32) + sizeof(v128_f32)) * 2 * size);
                    index = (int*)_p - 1;
                    value = (float*)index + width;
                    cos = (v128_f32*)(value + width + 1);
                    sin = cos + size;
                    pos = (v128_s32*)(sin + size);
                    neg = pos + size;
                    for (int i = 0; i < size; ++i)
                    {
                        cos[i] = SetF32((float)::cos(i*M_PI / size));
                        sin[i] = SetF32((float)::sin(i*M_PI / size));
                        pos[i] = SetI32(i);
                        neg[i] = SetI32(size + i);
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

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const v128_f32 & dx, const v128_f32 & dy, Buffer & buffer, size_t col)
        {
            v128_f32 bestDot = K_0_0f;
            v128_s32 bestIndex = (v128_s32)K32_00000000;
            for (int i = 0; i < buffer.size; ++i)
            {
                v128_f32 dot = vec_add(vec_mul(dx, buffer.cos[i]), vec_mul(dy, buffer.sin[i]));
                v128_u32 mask = (v128_u32)vec_cmpgt(dot, bestDot);
                bestDot = vec_max(dot, bestDot);
                bestIndex = vec_sel(bestIndex, buffer.pos[i], mask);

                dot = vec_sub(K_0_0f, dot);
                mask = (v128_u32)vec_cmpgt(dot, bestDot);
                bestDot = vec_max(dot, bestDot);
                bestIndex = vec_sel(bestIndex, buffer.neg[i], mask);
            }
            Store<align>(buffer.index + col, bestIndex);
            Store<align>(buffer.value + col, vec_sqrt(vec_add(vec_mul(dx, dx), vec_mul(dy, dy))));
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const v128_u16 & t, const v128_u16 & l, const v128_u16 & r, const v128_u16 & b, Buffer & buffer, size_t col)
        {
            HogDirectionHistograms<align>(vec_ctf((v128_s32)vec_sub(UnpackLoU16(r), UnpackLoU16(l)), 0),
                vec_ctf((v128_s32)vec_sub(UnpackLoU16(b), UnpackLoU16(t)), 0), buffer, col + 0);
            HogDirectionHistograms<align>(vec_ctf((v128_s32)vec_sub(UnpackHiU16(r), UnpackHiU16(l)), 0),
                vec_ctf((v128_s32)vec_sub(UnpackHiU16(b), UnpackHiU16(t)), 0), buffer, col + 4);
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const uint8_t * src, size_t stride, Buffer & buffer, size_t col)
        {
            const uint8_t * s = src + col;
            v128_u8 t = Load<false>(s - stride);
            v128_u8 l = Load<false>(s - 1);
            v128_u8 r = Load<false>(s + 1);
            v128_u8 b = Load<false>(s + stride);
            HogDirectionHistograms<align>(UnpackLoU8(t), UnpackLoU8(l), UnpackLoU8(r), UnpackLoU8(b), buffer, col + 0);
            HogDirectionHistograms<align>(UnpackHiU8(t), UnpackHiU8(l), UnpackHiU8(r), UnpackHiU8(b), buffer, col + 8);
        }

        void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height,
            size_t cellX, size_t cellY, size_t quantization, float * histograms)
        {
            assert(width%cellX == 0 && height%cellY == 0 && quantization % 2 == 0);

            Buffer buffer(width, quantization);

            memset(histograms, 0, quantization*(width / cellX)*(height / cellY) * sizeof(float));

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
