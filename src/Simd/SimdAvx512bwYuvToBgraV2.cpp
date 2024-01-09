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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool mask, class T> SIMD_INLINE void Yuv444pToBgraV2(const uint8_t* y,
            const uint8_t* u, const uint8_t* v, const __m512i& a, uint8_t* bgra, const __mmask64* tails)
        {
            YuvToBgra<false, mask, T>(Load<false, mask>(y, tails[0]), Load<false, mask>(u, tails[0]), Load<false, mask>(v, tails[0]), a, bgra, tails + 1);
        }

        template <class T> void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);

            __m512i a = _mm512_set1_epi8(alpha);
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[5];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 4; ++i)
                tailMasks[1 + i] = TailMask64(tail * 4 - A * i);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv444pToBgraV2<false, T>(y + col, u + col, v + col, a, bgra + col * 4, tailMasks);
                if (col < width)
                    Yuv444pToBgraV2<true, T>(y + col, u + col, v + col, a, bgra + col * 4, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }

        }

        void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv444pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv444pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv444pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <bool mask, class T> SIMD_YUV_TO_BGR_INLINE void Yuv422pToBgraV2(const uint8_t* y,
            const uint8_t* u, const uint8_t* v, const __m512i& a, uint8_t* bgra, const __mmask64* tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, mask>(u, tails[0])));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, mask>(v, tails[0])));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToBgra<false, mask, T>(Load<false, mask>(y + 0, tails[1]), u0, v0, a, bgra + 00, tails + 3);
            YuvToBgra<false, mask, T>(Load<false, mask>(y + A, tails[2]), u1, v1, a, bgra + QA, tails + 7);
        }

        template <class T> void Yuv422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width % 2 == 0);

            __m512i a = _mm512_set1_epi8(alpha);
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
                    Yuv422pToBgraV2<false, T>(y + col * 2, u + col, v + col, a, bgra + col * 8, tailMasks);
                if (col < width)
                    Yuv422pToBgraV2<true, T>(y + col * 2, u + col, v + col, a, bgra + col * 8, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv422pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv422pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv422pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv422pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <bool mask, class T> SIMD_YUV_TO_BGR_INLINE void Yuv420pToBgraV2(const uint8_t* y0, const uint8_t* y1, 
            const uint8_t* u, const uint8_t* v, const __m512i& a, uint8_t* bgra0, uint8_t* bgra1, const __mmask64* tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, mask>(u, tails[0])));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, mask>(v, tails[0])));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToBgra<false, mask, T>(Load<false, mask>(y0 + 0, tails[1]), u0, v0, a, bgra0 + 00, tails + 3);
            YuvToBgra<false, mask, T>(Load<false, mask>(y0 + A, tails[2]), u1, v1, a, bgra0 + QA, tails + 7);
            YuvToBgra<false, mask, T>(Load<false, mask>(y1 + 0, tails[1]), u0, v0, a, bgra1 + 00, tails + 3);
            YuvToBgra<false, mask, T>(Load<false, mask>(y1 + A, tails[2]), u1, v1, a, bgra1 + QA, tails + 7);
        }

        template <class T> void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            __m512i a = _mm512_set1_epi8(alpha);
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
                    Yuv420pToBgraV2<false, T>(y + col * 2, y + yStride + col * 2, u + col, v + col, a, bgra + col * 8, bgra + bgraStride + col * 8, tailMasks);
                if (col < width)
                    Yuv420pToBgraV2<true, T>(y + col * 2, y + yStride + col * 2, u + col, v + col, a, bgra + col * 8, bgra + bgraStride + col * 8, tailMasks);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv420pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv420pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv420pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
