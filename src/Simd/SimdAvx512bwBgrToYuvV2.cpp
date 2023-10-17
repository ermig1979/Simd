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
#include "Simd/SimdConversion.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask> SIMD_INLINE void LoadPreparedBgr16(const uint8_t * bgr, __m512i & b16_r16, __m512i & g16_1, const __mmask64 * ms)
        {
            __m512i _bgr = Load<align, mask>(bgr, ms[0]);
            __m512i bgr1 = _mm512_permutex2var_epi32(_bgr, K32_PERMUTE_BGR_TO_BGRA, K8_01);
            b16_r16 = _mm512_shuffle_epi8(bgr1, K8_SUFFLE_BGR_TO_B0R0);
            g16_1 = _mm512_shuffle_epi8(bgr1, K8_SUFFLE_BGR_TO_G010);
        }

        SIMD_INLINE void Average16(__m512i & a, const __m512i & b)
        {
            a = _mm512_srli_epi16(_mm512_add_epi16(_mm512_add_epi16(a, b), K16_0002), 2);
        }

        SIMD_INLINE void Average16(__m512i a[2][2])
        {
            a[0][0] = _mm512_srli_epi16(_mm512_add_epi16(a[0][0], K16_0001), 1);
            a[0][1] = _mm512_srli_epi16(_mm512_add_epi16(a[0][1], K16_0001), 1);
            a[1][0] = _mm512_srli_epi16(_mm512_add_epi16(a[1][0], K16_0001), 1);
            a[1][1] = _mm512_srli_epi16(_mm512_add_epi16(a[1][1], K16_0001), 1);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool mask> SIMD_INLINE __m512i LoadAndConvertBgrToY16V2(const uint8_t* bgr, __m512i& b16_r16, __m512i& g16_1, const __mmask64* tails)
        {
            __m512i _b16_r16[2], _g16_1[2];
            LoadPreparedBgr16<false, mask>(bgr + 00, _b16_r16[0], _g16_1[0], tails + 0);
            LoadPreparedBgr16<false, mask>(bgr + 48, _b16_r16[1], _g16_1[1], tails + 1);
            b16_r16 = Hadd32(_b16_r16[0], _b16_r16[1]);
            g16_1 = Hadd32(_g16_1[0], _g16_1[1]);
            return BgrToY16<T>(_b16_r16, _g16_1);
        }

        template <class T, bool mask> SIMD_INLINE __m512i LoadAndConvertBgrToY8V2(const uint8_t* bgr, __m512i b16_r16[2], __m512i g16_1[2], const __mmask64* tails)
        {
            __m512i lo = LoadAndConvertBgrToY16V2<T, mask>(bgr + 00, b16_r16[0], g16_1[0], tails + 0);
            __m512i hi = LoadAndConvertBgrToY16V2<T, mask>(bgr + 96, b16_r16[1], g16_1[1], tails + 2);
            return Permuted2Pack16iTo8u(lo, hi);
        }

        template <class T, bool mask> SIMD_INLINE void BgrToYuv420pV2(const uint8_t* bgr0, size_t bgrStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v, const __mmask64* tails)
        {
            const uint8_t* bgr1 = bgr0 + bgrStride;
            uint8_t* y1 = y0 + yStride;

            __m512i _b16_r16[2][2][2], _g16_1[2][2][2];
            Store<false, mask>(y0 + 0, LoadAndConvertBgrToY8V2<T, mask>(bgr0 + 0 * A, _b16_r16[0][0], _g16_1[0][0], tails + 0), tails[8]);
            Store<false, mask>(y0 + A, LoadAndConvertBgrToY8V2<T, mask>(bgr0 + 3 * A, _b16_r16[0][1], _g16_1[0][1], tails + 4), tails[9]);
            Store<false, mask>(y1 + 0, LoadAndConvertBgrToY8V2<T, mask>(bgr1 + 0 * A, _b16_r16[1][0], _g16_1[1][0], tails + 0), tails[8]);
            Store<false, mask>(y1 + A, LoadAndConvertBgrToY8V2<T, mask>(bgr1 + 3 * A, _b16_r16[1][1], _g16_1[1][1], tails + 4), tails[9]);

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            Store<false, mask>(u, Permuted2Pack16iTo8u(BgrToU16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToU16<T>(_b16_r16[0][1], _g16_1[0][1])), tails[10]);
            Store<false, mask>(v, Permuted2Pack16iTo8u(BgrToV16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToV16<T>(_b16_r16[0][1], _g16_1[0][1])), tails[10]);
        }

        template <class T>  void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));

            width /= 2;
            size_t widthA = AlignLo(width - 1, A);
            size_t tail = width - widthA;
            __mmask64 tails[11];
            for (size_t i = 0; i < 8; ++i)
                tails[i] = TailMask64(tail * 6 - 48 * i) & 0x0000FFFFFFFFFFFF;
            for (size_t i = 0; i < 2; ++i)
                tails[8 + i] = TailMask64(tail * 2 - A * i);
            tails[10] = TailMask64(tail);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BgrToYuv420pV2<T, false>(bgr + col * 6, bgrStride, y + col * 2, yStride, u + col, v + col, tails);
                if (tail)
                    BgrToYuv420pV2<T, true>(bgr + col * 6, bgrStride, y + col * 2, yStride, u + col, v + col, tails);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        void BgrToYuv420pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv420pV2<Base::Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv420pV2<Base::Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv420pV2<Base::Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv420pV2<Base::Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool mask> SIMD_INLINE void BgrToYuv422pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v, const __mmask64* tails)
        {
            __m512i _b16_r16[2][2], _g16_1[2][2];
            Store<false, mask>(y + 0, LoadAndConvertBgrToY8V2<T, mask>(bgr + 0 * A, _b16_r16[0], _g16_1[0], tails + 0), tails[8]);
            Store<false, mask>(y + A, LoadAndConvertBgrToY8V2<T, mask>(bgr + 3 * A, _b16_r16[1], _g16_1[1], tails + 4), tails[9]);

            Average16(_b16_r16);
            Average16(_g16_1);

            Store<false, mask>(u, Permuted2Pack16iTo8u(BgrToU16<T>(_b16_r16[0], _g16_1[0]), BgrToU16<T>(_b16_r16[1], _g16_1[1])), tails[10]);
            Store<false, mask>(v, Permuted2Pack16iTo8u(BgrToV16<T>(_b16_r16[0], _g16_1[0]), BgrToV16<T>(_b16_r16[1], _g16_1[1])), tails[10]);
        }

        template <class T>  void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert(width % 2 == 0);

            width /= 2;
            size_t widthA = AlignLo(width - 1, A);
            size_t tail = width - widthA;
            __mmask64 tails[11];
            for (size_t i = 0; i < 8; ++i)
                tails[i] = TailMask64(tail * 6 - 48 * i) & 0x0000FFFFFFFFFFFF;
            for (size_t i = 0; i < 2; ++i)
                tails[8 + i] = TailMask64(tail * 2 - A * i);
            tails[10] = TailMask64(tail);
            for (size_t row = 0; row < height; row += 1)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BgrToYuv422pV2<T, false>(bgr + col * 6, y + col * 2, u + col, v + col, tails);
                if (tail)
                    BgrToYuv422pV2<T, true>(bgr + col * 6, y + col * 2, u + col, v + col, tails);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void BgrToYuv422pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv422pV2<Base::Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv422pV2<Base::Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv422pV2<Base::Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv422pV2<Base::Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool mask> SIMD_INLINE void BgrToYuv444pV2(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v, const __mmask64* tails)
        {
            __m512i _b16_r16[2][2], _g16_1[2][2];
            LoadPreparedBgr16<false, mask>(bgr + 0x00, _b16_r16[0][0], _g16_1[0][0], tails + 0);
            LoadPreparedBgr16<false, mask>(bgr + 0x30, _b16_r16[0][1], _g16_1[0][1], tails + 1);
            LoadPreparedBgr16<false, mask>(bgr + 0x60, _b16_r16[1][0], _g16_1[1][0], tails + 2);
            LoadPreparedBgr16<false, mask>(bgr + 0x90, _b16_r16[1][1], _g16_1[1][1], tails + 3);
            Store<false, mask>(y, Permuted2Pack16iTo8u(BgrToY16<T>(_b16_r16[0], _g16_1[0]), BgrToY16<T>(_b16_r16[1], _g16_1[1])), tails[4]);
            Store<false, mask>(u, Permuted2Pack16iTo8u(BgrToU16<T>(_b16_r16[0], _g16_1[0]), BgrToU16<T>(_b16_r16[1], _g16_1[1])), tails[4]);
            Store<false, mask>(v, Permuted2Pack16iTo8u(BgrToV16<T>(_b16_r16[0], _g16_1[0]), BgrToV16<T>(_b16_r16[1], _g16_1[1])), tails[4]);
        }

        template <class T>  void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            size_t widthA = AlignLo(width - 1, A);
            size_t tail = width - widthA;
            __mmask64 tails[5];
            for (size_t i = 0; i < 4; ++i)
                tails[i] = TailMask64(tail * 3 - 48 * i) & 0x0000FFFFFFFFFFFF;
            tails[4] = TailMask64(tail);
            for (size_t row = 0; row < height; row += 1)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    BgrToYuv444pV2<T, false>(bgr + col * 3, y + col, u + col, v + col, tails);
                if (tail)
                    BgrToYuv444pV2<T, true>(bgr + col * 3, y + col, u + col, v + col, tails);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void BgrToYuv444pV2(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: BgrToYuv444pV2<Base::Bt601>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgrToYuv444pV2<Base::Bt709>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgrToYuv444pV2<Base::Bt2020>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgrToYuv444pV2<Base::Trect871>(bgr, bgrStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
