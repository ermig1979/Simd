/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <class T, bool mask> SIMD_INLINE void YuvToBgra(const __m512i& y, const __m512i& u, const __m512i& v, const __m512i& a, uint8_t* bgra, const __mmask64* tails)
        {
            __m512i b = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, YuvToBlue<T>(y, u));
            __m512i g = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, YuvToGreen<T>(y, u, v));
            __m512i r = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, YuvToRed<T>(y, v));
            __m512i _a = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, a);
            __m512i bg0 = UnpackU8<0>(b, g);
            __m512i bg1 = UnpackU8<1>(b, g);
            __m512i ra0 = UnpackU8<0>(r, _a);
            __m512i ra1 = UnpackU8<1>(r, _a);
            Store<false, mask>(bgra + 0 * A, UnpackU16<0>(bg0, ra0), tails[0]);
            Store<false, mask>(bgra + 1 * A, UnpackU16<1>(bg0, ra0), tails[1]);
            Store<false, mask>(bgra + 2 * A, UnpackU16<0>(bg1, ra1), tails[2]);
            Store<false, mask>(bgra + 3 * A, UnpackU16<1>(bg1, ra1), tails[3]);
        }

        template <class T, bool mask> SIMD_YUV_TO_BGR_INLINE void Yuva420pToBgraV2(const uint8_t* y0, size_t yStride, const uint8_t* u, const uint8_t* v,
            const uint8_t* a0, size_t aStride, uint8_t* bgra0, size_t bgraStride, const __mmask64* tails)
        {
            const uint8_t* y1 = y0 + yStride;
            const uint8_t* a1 = a0 + aStride;
            uint8_t* bgra1 = bgra0 + bgraStride;
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, mask>(u, tails[0])));
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, mask>(v, tails[0])));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i v0 = UnpackU8<0>(_v, _v);
            YuvToBgra<T, mask>(Load<false, mask>(y0 + 0, tails[1]), u0, v0, Load<false, mask>(a0 + 0, tails[1]), bgra0 + 00, tails + 3);
            YuvToBgra<T, mask>(Load<false, mask>(y1 + 0, tails[1]), u0, v0, Load<false, mask>(a1 + 0, tails[1]), bgra1 + 00, tails + 3);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToBgra<T, mask>(Load<false, mask>(y0 + A, tails[1]), u1, v1, Load<false, mask>(a0 + A, tails[1]), bgra0 + QA, tails + 7);
            YuvToBgra<T, mask>(Load<false, mask>(y1 + A, tails[1]), u1, v1, Load<false, mask>(a1 + A, tails[1]), bgra1 + QA, tails + 7);
        }

        template <class T> void Yuva420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[11];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[1 + i] = TailMask64(tail * 2 - A * i);
            for (size_t i = 0; i < 8; ++i)
                tailMasks[3 + i] = TailMask64(tail * 8 - A * i);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuva420pToBgraV2<T, false>(y + col * 2, yStride, u + col, v + col, a + col * 2, aStride, bgra + col * 8, bgraStride, tailMasks);
                if (col < width)
                    Yuva420pToBgraV2<T, true>(y + col * 2, yStride, u + col, v + col, a + col * 2, aStride, bgra + col * 8, bgraStride, tailMasks);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuva420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuva420pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt709: Yuva420pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt2020: Yuva420pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvTrect871: Yuva420pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool mask> SIMD_YUV_TO_BGR_INLINE void Yuva422pToBgraV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, const uint8_t* a, uint8_t* bgra, const __mmask64* tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, mask>(u, tails[0])));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, mask>(v, tails[0])));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToBgra<T, mask>(Load<false, mask>(y + 0, tails[1]), u0, v0, Load<false, mask>(a + 0, tails[1]), bgra + 00, tails + 3);
            YuvToBgra<T, mask>(Load<false, mask>(y + A, tails[2]), u1, v1, Load<false, mask>(a + A, tails[2]), bgra + QA, tails + 7);
        }

        template <class T> void Yuva422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            assert(width % 2 == 0);

            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[11];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[1 + i] = TailMask64(tail * 2 - A * i);
            for (size_t i = 0; i < 8; ++i)
                tailMasks[3 + i] = TailMask64(tail * 8 - A * i);
            for (size_t row = 0; row < height; row += 1)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuva422pToBgraV2<T, false>(y + col * 2, u + col, v + col, a + col * 2, bgra + col * 8, tailMasks);
                if (col < width)
                    Yuva422pToBgraV2<T, true>(y + col * 2, u + col, v + col, a + col * 2, bgra + col * 8, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                a += aStride;
                bgra += bgraStride;
            }
        }

        void Yuva422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuva422pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt709: Yuva422pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt2020: Yuva422pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvTrect871: Yuva422pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void YuvaToBgra16(__m512i y16, __m512i u16, __m512i v16, const __m512i& a16, __m512i* bgra, __mmask64 tail)
        {
            const __m512i b16 = YuvToBlue16<T>(y16, u16);
            const __m512i g16 = YuvToGreen16<T>(y16, u16, v16);
            const __m512i r16 = YuvToRed16<T>(y16, v16);
            const __m512i bg8 = _mm512_or_si512(b16, _mm512_slli_epi16(g16, 8));
            const __m512i ra8 = _mm512_or_si512(r16, _mm512_slli_epi16(a16, 8));
            __m512i bgra0 = _mm512_unpacklo_epi16(bg8, ra8);
            __m512i bgra1 = _mm512_unpackhi_epi16(bg8, ra8);
            _mm512_mask_storeu_epi32(bgra + 0, __mmask16(tail >> 0 * 16), bgra0);
            _mm512_mask_storeu_epi32(bgra + 1, __mmask16(tail >> 1 * 16), bgra1);
        }

        template <class T> SIMD_YUV_TO_BGR_INLINE void Yuva444pToBgraV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, const uint8_t* a, uint8_t* bgra, __mmask64 tail = __mmask64(-1))
        {
            __m512i _y = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_maskz_loadu_epi8(tail, y));
            __m512i _u = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_maskz_loadu_epi8(tail, u));
            __m512i _v = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_maskz_loadu_epi8(tail, v));
            __m512i _a = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_maskz_loadu_epi8(tail, a));
            YuvaToBgra16<T>(UnpackY<T, 0>(_y), UnpackUV<T, 0>(_u), UnpackUV<T, 0>(_v), UnpackU8<0>(_a), (__m512i*)bgra + 0, tail >> 0 * 16);
            YuvaToBgra16<T>(UnpackY<T, 1>(_y), UnpackUV<T, 1>(_u), UnpackUV<T, 1>(_v), UnpackU8<1>(_a), (__m512i*)bgra + 2, tail >> 2 * 16);
        }

        template <class T> void Yuva444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride,
            const uint8_t* v, size_t vStride, const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            size_t widthA = AlignLo(width, A);
            __mmask64 tail = TailMask64(width - widthA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    Yuva444pToBgraV2<T>(y + col, u + col, v + col, a + col, bgra + col * 4);
                if (tail)
                    Yuva444pToBgraV2<T>(y + col, u + col, v + col, a + col, bgra + col * 4, tail);
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
#endif
}
