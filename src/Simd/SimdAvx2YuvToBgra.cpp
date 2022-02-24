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
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void AdjustedYuv16ToBgra(__m256i y16, __m256i u16, __m256i v16,
            const __m256i & a_0, __m256i * bgra)
        {
            const __m256i b16 = AdjustedYuvToBlue16(y16, u16);
            const __m256i g16 = AdjustedYuvToGreen16(y16, u16, v16);
            const __m256i r16 = AdjustedYuvToRed16(y16, v16);
            const __m256i bg8 = _mm256_or_si256(b16, _mm256_slli_si256(g16, 1));
            const __m256i ra8 = _mm256_or_si256(r16, a_0);
            __m256i bgra0 = _mm256_unpacklo_epi16(bg8, ra8);
            __m256i bgra1 = _mm256_unpackhi_epi16(bg8, ra8);
            Permute2x128(bgra0, bgra1);
            Store<align>(bgra + 0, bgra0);
            Store<align>(bgra + 1, bgra1);
        }

        template <bool align> SIMD_INLINE void Yuv16ToBgra(__m256i y16, __m256i u16, __m256i v16,
            const __m256i & a_0, __m256i * bgra)
        {
            AdjustedYuv16ToBgra<align>(AdjustY16(y16), AdjustUV16(u16), AdjustUV16(v16), a_0, bgra);
        }

        template <bool align> SIMD_INLINE void Yuva8ToBgra(__m256i y8, __m256i u8, __m256i v8, const __m256i & a8, __m256i * bgra)
        {
            Yuv16ToBgra<align>(_mm256_unpacklo_epi8(y8, K_ZERO), _mm256_unpacklo_epi8(u8, K_ZERO),
                _mm256_unpacklo_epi8(v8, K_ZERO), _mm256_unpacklo_epi8(K_ZERO, a8), bgra + 0);
            Yuv16ToBgra<align>(_mm256_unpackhi_epi8(y8, K_ZERO), _mm256_unpackhi_epi8(u8, K_ZERO),
                _mm256_unpackhi_epi8(v8, K_ZERO), _mm256_unpackhi_epi8(K_ZERO, a8), bgra + 2);
        }

        template <bool align> SIMD_INLINE void Yuva422pToBgra(const uint8_t * y, const __m256i & u, const __m256i & v,
            const uint8_t * a, uint8_t * bgra)
        {
            Yuva8ToBgra<align>(LoadPermuted<align>((__m256i*)y + 0), _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(u, u), 0xD8), 
                _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(v, v), 0xD8), LoadPermuted<align>((__m256i*)a + 0), (__m256i*)bgra + 0);
            Yuva8ToBgra<align>(LoadPermuted<align>((__m256i*)y + 1), _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(u, u), 0xD8),
                _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(v, v), 0xD8), LoadPermuted<align>((__m256i*)a + 1), (__m256i*)bgra + 4);
        }

        template <bool align> void Yuva420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));
                assert(Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
                {
                    __m256i u_ = LoadPermuted<align>((__m256i*)(u + colUV));
                    __m256i v_ = LoadPermuted<align>((__m256i*)(v + colUV));
                    Yuva422pToBgra<align>(y + colY, u_, v_, a + colY, bgra + colBgra);
                    Yuva422pToBgra<align>(y + colY + yStride, u_, v_, a + colY + aStride, bgra + colBgra + bgraStride);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m256i u_ = LoadPermuted<false>((__m256i*)(u + offset / 2));
                    __m256i v_ = LoadPermuted<false>((__m256i*)(v + offset / 2));
                    Yuva422pToBgra<false>(y + offset, u_, v_, a + offset, bgra + 4 * offset);
                    Yuva422pToBgra<false>(y + offset + yStride, u_, v_, a + offset + aStride, bgra + 4 * offset + bgraStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuva420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride)
                && Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuva420pToBgra<true>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride);
            else
                Yuva420pToBgra<false>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride);
        }

        template <bool align> SIMD_INLINE void Yuv8ToBgra(__m256i y8, __m256i u8, __m256i v8, const __m256i & a_0, __m256i * bgra)
        {
            Yuv16ToBgra<align>(_mm256_unpacklo_epi8(y8, K_ZERO), _mm256_unpacklo_epi8(u8, K_ZERO),
                _mm256_unpacklo_epi8(v8, K_ZERO), a_0, bgra + 0);
            Yuv16ToBgra<align>(_mm256_unpackhi_epi8(y8, K_ZERO), _mm256_unpackhi_epi8(u8, K_ZERO),
                _mm256_unpackhi_epi8(v8, K_ZERO), a_0, bgra + 2);
        }

        template <bool align> SIMD_INLINE void Yuv444pToBgra(const uint8_t * y, const uint8_t * u,
            const uint8_t * v, const __m256i & a_0, uint8_t * bgra)
        {
            Yuv8ToBgra<align>(LoadPermuted<align>((__m256i*)y), LoadPermuted<align>((__m256i*)u), LoadPermuted<align>((__m256i*)v), a_0, (__m256i*)bgra);
        }

        template <bool align> void Yuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            __m256i a_0 = _mm256_slli_si256(_mm256_set1_epi16(alpha), 1);
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colYuv = 0, colBgra = 0; colYuv < bodyWidth; colYuv += A, colBgra += QA)
                {
                    Yuv444pToBgra<align>(y + colYuv, u + colYuv, v + colYuv, a_0, bgra + colBgra);
                }
                if (tail)
                {
                    size_t col = width - A;
                    Yuv444pToBgra<false>(y + col, u + col, v + col, a_0, bgra + 4 * col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv444pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
            else
                Yuv444pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
        }

        template <bool align> SIMD_INLINE void Yuv422pToBgra(const uint8_t * y, const __m256i & u, const __m256i & v,
            const __m256i & a_0, uint8_t * bgra)
        {
            Yuv8ToBgra<align>(LoadPermuted<align>((__m256i*)y + 0),
                _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(u, u), 0xD8),
                _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(v, v), 0xD8), a_0, (__m256i*)bgra + 0);
            Yuv8ToBgra<align>(LoadPermuted<align>((__m256i*)y + 1),
                _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(u, u), 0xD8),
                _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(v, v), 0xD8), a_0, (__m256i*)bgra + 4);
        }

        template <bool align> void Yuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            __m256i a_0 = _mm256_slli_si256(_mm256_set1_epi16(alpha), 1);
            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
                {
                    __m256i u_ = LoadPermuted<align>((__m256i*)(u + colUV));
                    __m256i v_ = LoadPermuted<align>((__m256i*)(v + colUV));
                    Yuv422pToBgra<align>(y + colY, u_, v_, a_0, bgra + colBgra);
                    Yuv422pToBgra<align>(y + colY + yStride, u_, v_, a_0, bgra + colBgra + bgraStride);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m256i u_ = LoadPermuted<false>((__m256i*)(u + offset / 2));
                    __m256i v_ = LoadPermuted<false>((__m256i*)(v + offset / 2));
                    Yuv422pToBgra<false>(y + offset, u_, v_, a_0, bgra + 4 * offset);
                    Yuv422pToBgra<false>(y + offset + yStride, u_, v_, a_0, bgra + 4 * offset + bgraStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv420pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
            else
                Yuv420pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
        }

        template <bool align> SIMD_INLINE void Yuv422pToBgra(const uint8_t * y, const uint8_t * u, const uint8_t * v, const __m256i & a_0, uint8_t * bgra)
        {
            Yuv422pToBgra<align>(y, LoadPermuted<align>((__m256i*)u), LoadPermuted<align>((__m256i*)v), a_0, bgra);
        }

        template <bool align> void Yuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            __m256i a_0 = _mm256_slli_si256(_mm256_set1_epi16(alpha), 1);
            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
                    Yuv422pToBgra<align>(y + colY, u + colUV, v + colUV, a_0, bgra + colBgra);
                if (tail)
                {
                    size_t offset = width - DA;
                    Yuv422pToBgra<false>(y + offset, u + offset / 2, v + offset / 2, a_0, bgra + 4 * offset);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv422pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
            else
                Yuv422pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, class T> SIMD_INLINE void YuvToBgra16(__m256i y16, __m256i u16, __m256i v16, const __m256i& a_0, __m256i* bgra)
        {
            const __m256i b16 = YuvToBlue16<T>(y16, u16);
            const __m256i g16 = YuvToGreen16<T>(y16, u16, v16);
            const __m256i r16 = YuvToRed16<T>(y16, v16);
            const __m256i bg8 = _mm256_or_si256(b16, _mm256_slli_si256(g16, 1));
            const __m256i ra8 = _mm256_or_si256(r16, a_0);
            __m256i bgra0 = _mm256_unpacklo_epi16(bg8, ra8);
            __m256i bgra1 = _mm256_unpackhi_epi16(bg8, ra8);
            Permute2x128(bgra0, bgra1);
            Store<align>(bgra + 0, bgra0);
            Store<align>(bgra + 1, bgra1);
        }

        template <bool align, class T> SIMD_INLINE void YuvToBgra(__m256i y8, __m256i u8, __m256i v8, const __m256i& a_0, __m256i* bgra)
        {
            YuvToBgra16<align, T>(UnpackY<T, 0>(y8), UnpackUV<T, 0>(u8), UnpackUV<T, 0>(v8), a_0, bgra + 0);
            YuvToBgra16<align, T>(UnpackY<T, 1>(y8), UnpackUV<T, 1>(u8), UnpackUV<T, 1>(v8), a_0, bgra + 2);
        }

        template <bool align, class T> SIMD_INLINE void Yuv444pToBgraV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, const __m256i& a_0, uint8_t* bgra)
        {
            YuvToBgra<align, T>(LoadPermuted<align>((__m256i*)y), LoadPermuted<align>((__m256i*)u), LoadPermuted<align>((__m256i*)v), a_0, (__m256i*)bgra);
        }

        template <bool align, class T> void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            __m256i a_0 = _mm256_slli_si256(_mm256_set1_epi16(alpha), 1);
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colYuv = 0, colBgra = 0; colYuv < bodyWidth; colYuv += A, colBgra += QA)
                {
                    Yuv444pToBgraV2<align, T>(y + colYuv, u + colYuv, v + colYuv, a_0, bgra + colBgra);
                }
                if (tail)
                {
                    size_t col = width - A;
                    Yuv444pToBgraV2<false, T>(y + col, u + col, v + col, a_0, bgra + 4 * col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        template <bool align> void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToBgraV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv444pToBgraV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv444pToBgraV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv444pToBgraV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv444pToBgraV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
            else
                Yuv444pToBgraV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, class T> SIMD_INLINE void Yuv422pToBgraV2(const uint8_t* y, const __m256i& u, const __m256i& v,
            const __m256i& a_0, uint8_t* bgra)
        {
            YuvToBgra<align, T>(LoadPermuted<align>((__m256i*)y + 0),
                _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(u, u), 0xD8), 
                _mm256_permute4x64_epi64(_mm256_unpacklo_epi8(v, v), 0xD8), a_0, (__m256i*)bgra + 0);
            YuvToBgra<align, T>(LoadPermuted<align>((__m256i*)y + 1),
                _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(u, u), 0xD8),
                _mm256_permute4x64_epi64(_mm256_unpackhi_epi8(v, v), 0xD8), a_0, (__m256i*)bgra + 4);
        }

        template <bool align, class T> void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            __m256i a_0 = _mm256_slli_si256(_mm256_set1_epi16(alpha), 1);
            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
                {
                    __m256i u_ = LoadPermuted<align>((__m256i*)(u + colUV));
                    __m256i v_ = LoadPermuted<align>((__m256i*)(v + colUV));
                    Yuv422pToBgraV2<align, T>(y + colY, u_, v_, a_0, bgra + colBgra);
                    Yuv422pToBgraV2<align, T>(y + colY + yStride, u_, v_, a_0, bgra + colBgra + bgraStride);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m256i u_ = LoadPermuted<false>((__m256i*)(u + offset / 2));
                    __m256i v_ = LoadPermuted<false>((__m256i*)(v + offset / 2));
                    Yuv422pToBgraV2<false, T>(y + offset, u_, v_, a_0, bgra + 4 * offset);
                    Yuv422pToBgraV2<false, T>(y + offset + yStride, u_, v_, a_0, bgra + 4 * offset + bgraStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        template <bool align> void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToBgraV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv420pToBgraV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv420pToBgraV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv420pToBgraV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv420pToBgraV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
            else
                Yuv420pToBgraV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
