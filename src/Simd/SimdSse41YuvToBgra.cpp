/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdConversion.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <class T> SIMD_INLINE void YuvaToBgra16(__m128i y16, __m128i u16, __m128i v16, const __m128i& a16, __m128i* bgra)
        {
            const __m128i b16 = YuvToBlue16<T>(y16, u16);
            const __m128i g16 = YuvToGreen16<T>(y16, u16, v16);
            const __m128i r16 = YuvToRed16<T>(y16, v16);
            const __m128i bg8 = _mm_or_si128(b16, _mm_slli_si128(g16, 1));
            const __m128i ra8 = _mm_or_si128(r16, _mm_slli_si128(a16, 1));
            _mm_storeu_si128(bgra + 0, _mm_unpacklo_epi16(bg8, ra8));
            _mm_storeu_si128(bgra + 1, _mm_unpackhi_epi16(bg8, ra8));
        }

        template <class T> SIMD_INLINE void Yuva444pToBgraV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, const uint8_t* a, uint8_t* bgra)
        {
            __m128i _y = _mm_loadu_si128((__m128i*)y);
            __m128i _u = _mm_loadu_si128((__m128i*)u);
            __m128i _v = _mm_loadu_si128((__m128i*)v);
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            YuvaToBgra16<T>(UnpackY<T, 0>(_y), UnpackUV<T, 0>(_u), UnpackUV<T, 0>(_v), UnpackU8<0>(_a), (__m128i*)bgra + 0);
            YuvaToBgra16<T>(UnpackY<T, 1>(_y), UnpackUV<T, 1>(_u), UnpackUV<T, 1>(_v), UnpackU8<1>(_a), (__m128i*)bgra + 2);
        }

        template <class T> void Yuva444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride,
            const uint8_t* v, size_t vStride, const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            assert((width >= A));

            size_t widthA = AlignLo(width, A);
            size_t tail = width - widthA;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthA; col += A)
                    Yuva444pToBgraV2<T>(y + col, u + col, v + col, a + col, bgra + col * 4);
                if (tail)
                {
                    size_t col = width - A;
                    Yuva444pToBgraV2<T>(y + col, u + col, v + col, a + col, bgra + col * 4);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                a += aStride;
                bgra += bgraStride;
            }
        }

        void Yuva444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuva444pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt709: Yuva444pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt2020: Yuva444pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvTrect871: Yuva444pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            default:
                assert(0);
            }
        }
    }
#endif// SIMD_SSE2_ENABLE
}
