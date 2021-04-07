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
#ifndef __SimdImageSaveJpeg_h__
#define __SimdImageSaveJpeg_h__

#include "Simd/SimdImageSave.h"

#define SIMD_JPEG_CALC_BITS_TABLE

namespace Simd
{
    namespace Base
    {
        extern const uint8_t JpegZigZagD[64];
        extern const uint8_t JpegZigZagT[64];

        extern const uint16_t HuffmanYdc[256][2];
        extern const uint16_t HuffmanUVdc[256][2];
        extern const uint16_t HuffmanYac[256][2];
        extern const uint16_t HuffmanUVac[256][2];

#if defined(SIMD_JPEG_CALC_BITS_TABLE)
        extern int JpegCalcBitsTable[1024];
        SIMD_INLINE void JpegCalcBits(int val, uint16_t bits[2])
        {
            int tmp = val < 0 ? -val : val;
            val = val < 0 ? val - 1 : val;
            assert(tmp < 1024);
            bits[1] = JpegCalcBitsTable[tmp];
            bits[0] = val & ((1 << bits[1]) - 1);
        }
#else
        SIMD_INLINE void JpegCalcBits(int val, uint16_t bits[2])
        {
            int tmp = val < 0 ? -val : val;
            val = val < 0 ? val - 1 : val;
            bits[1] = 1;
            while (tmp >>= 1)
                ++bits[1];
            bits[0] = val & ((1 << bits[1]) - 1);
        }
#endif

        SIMD_INLINE void RgbToYuv(const uint8_t* r, const uint8_t* g, const uint8_t* b, int stride, int height, int width, float* y, float* u, float* v, int size)
        {
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += 1)
                {
                    int offs = (col < width ? col : width - 1);
                    float _r = r[offs], _g = g[offs], _b = b[offs];
                    y[col] = +0.29900f * _r + 0.58700f * _g + 0.11400f * _b - 128.000f;
                    u[col] = -0.16874f * _r - 0.33126f * _g + 0.50000f * _b;
                    v[col] = +0.50000f * _r - 0.41869f * _g - 0.08131f * _b;
                }
                if (++row < height)
                    r += stride, g += stride, b += stride;
                y += size, u += size, v += size;
            }
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
    }
#endif// SIMD_SSE41_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
    }
#endif// SIMD_AVX512BW_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif// SIMD_NEON_ENABLE
}

#endif//__SimdImageSaveJpeg_h__
