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
#include "Simd/SimdSse41.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <class T> SIMD_INLINE void YuvaToBgra16(__m256i y16, __m256i u16, __m256i v16, const __m256i& a16, __m256i* bgra)
        {
            const __m256i b16 = YuvToBlue16<T>(y16, u16);
            const __m256i g16 = YuvToGreen16<T>(y16, u16, v16);
            const __m256i r16 = YuvToRed16<T>(y16, v16);
            const __m256i bg8 = _mm256_or_si256(b16, _mm256_slli_si256(g16, 1));
            const __m256i ra8 = _mm256_or_si256(r16, _mm256_slli_si256(a16, 1));
            __m256i bgra0 = _mm256_unpacklo_epi16(bg8, ra8);
            __m256i bgra1 = _mm256_unpackhi_epi16(bg8, ra8);
            _mm256_storeu_si256(bgra + 0, bgra0);
            _mm256_storeu_si256(bgra + 1, bgra1);
        }

        template <class T, int part> SIMD_INLINE void Yuva420pToBgraV2(const uint8_t* y0, const uint8_t* y1, __m256i u, __m256i v, const uint8_t* a0, const uint8_t* a1, uint8_t* bgra0, uint8_t* bgra1)
        {
            static const __m256i K32_PERMUTE = SIMD_MM256_SETR_EPI32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i y00 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)y0 + part), K32_PERMUTE);
            __m256i a00 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)a0 + part), K32_PERMUTE);
            __m256i y10 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)y1 + part), K32_PERMUTE);
            __m256i a10 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)a1 + part), K32_PERMUTE);
            u = _mm256_permutevar8x32_epi32(u, K32_PERMUTE);
            v = _mm256_permutevar8x32_epi32(v, K32_PERMUTE);
            __m256i u0 = UnpackUV<T, 0>(u);
            __m256i v0 = UnpackUV<T, 0>(v);
            YuvaToBgra16<T>(UnpackY<T, 0>(y00), u0, v0, UnpackU8<0>(a00), (__m256i*)bgra0 + 4 * part + 0);
            YuvaToBgra16<T>(UnpackY<T, 0>(y10), u0, v0, UnpackU8<0>(a10), (__m256i*)bgra1 + 4 * part + 0);
            __m256i u1 = UnpackUV<T, 1>(u);
            __m256i v1 = UnpackUV<T, 1>(v);
            YuvaToBgra16<T>(UnpackY<T, 1>(y00), u1, v1, UnpackU8<1>(a00), (__m256i*)bgra0 + 4 * part + 2);
            YuvaToBgra16<T>(UnpackY<T, 1>(y10), u1, v1, UnpackU8<1>(a10), (__m256i*)bgra1 + 4 * part + 2);
        }

        template <class T> SIMD_INLINE void Yuva420pToBgraV2(const uint8_t* y0, size_t yStride, const uint8_t* u, const uint8_t* v, const uint8_t* a0, size_t aStride, uint8_t* bgra0, size_t bgraStride)
        {
            const uint8_t* y1 = y0 + yStride;
            const uint8_t* a1 = a0 + aStride;
            uint8_t* bgra1 = bgra0 + bgraStride;
            __m256i _u = LoadPermuted<false>((__m256i*)u);
            __m256i _v = LoadPermuted<false>((__m256i*)v);
            Yuva420pToBgraV2<T, 0>(y0, y1, UnpackU8<0>(_u, _u), UnpackU8<0>(_v, _v), a0, a1, bgra0, bgra1);
            Yuva420pToBgraV2<T, 1>(y0, y1, UnpackU8<1>(_u, _u), UnpackU8<1>(_v, _v), a0, a1, bgra0, bgra1);
        }

        template <class T> void Yuva420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride,
            const uint8_t* v, size_t vStride, const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));

            width /= 2;
            size_t widthA = AlignLo(width, A);
            size_t tail = width - widthA;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0; col < widthA; col += A)
                    Yuva420pToBgraV2<T>(y + 2 * col, yStride, u + col, v + col, a + 2 * col, aStride, bgra + 8 * col, bgraStride);
                if (tail)
                {
                    size_t col = width - A;
                    Yuva420pToBgraV2<T>(y + 2 * col, yStride, u + col, v + col, a + 2 * col, aStride, bgra + 8 * col, bgraStride);
                }
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

        template <class T> SIMD_INLINE void Yuva422pToBgraV2(const uint8_t* y, __m256i u, __m256i v, const uint8_t* a, uint8_t* bgra)
        {
            static const __m256i K32_PERMUTE = SIMD_MM256_SETR_EPI32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i y0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)y + 0), K32_PERMUTE);
            __m256i u0 = _mm256_permutevar8x32_epi32(UnpackU8<0>(u, u), K32_PERMUTE);
            __m256i v0 = _mm256_permutevar8x32_epi32(UnpackU8<0>(v, v), K32_PERMUTE);
            __m256i a0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)a + 0), K32_PERMUTE);
            YuvaToBgra16<T>(UnpackY<T, 0>(y0), UnpackUV<T, 0>(u0), UnpackUV<T, 0>(v0), UnpackU8<0>(a0), (__m256i*)bgra + 0);
            YuvaToBgra16<T>(UnpackY<T, 1>(y0), UnpackUV<T, 1>(u0), UnpackUV<T, 1>(v0), UnpackU8<1>(a0), (__m256i*)bgra + 2);
            __m256i y1 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)y + 1), K32_PERMUTE);
            __m256i u1 = _mm256_permutevar8x32_epi32(UnpackU8<1>(u, u), K32_PERMUTE);
            __m256i v1 = _mm256_permutevar8x32_epi32(UnpackU8<1>(v, v), K32_PERMUTE);
            __m256i a1 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)a + 1), K32_PERMUTE);
            YuvaToBgra16<T>(UnpackY<T, 0>(y1), UnpackUV<T, 0>(u1), UnpackUV<T, 0>(v1), UnpackU8<0>(a1), (__m256i*)bgra + 4);
            YuvaToBgra16<T>(UnpackY<T, 1>(y1), UnpackUV<T, 1>(u1), UnpackUV<T, 1>(v1), UnpackU8<1>(a1), (__m256i*)bgra + 6);
        }

        template <class T> void Yuva422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride,
            const uint8_t* v, size_t vStride, const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride)
        {
            assert((width % 2 == 0) && (width >= DA));

            size_t widthDA = AlignLo(width, DA);
            size_t tail = width - widthDA;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colYA = 0, colBgra = 0; colYA < widthDA; colYA += DA, colUV += A, colBgra += OA)
                {
                    __m256i u_ = LoadPermuted<false>((__m256i*)(u + colUV));
                    __m256i v_ = LoadPermuted<false>((__m256i*)(v + colUV));

                    Yuva422pToBgraV2<T>(y + colYA, u_, v_, a + colYA, bgra + colBgra);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m256i u_ = LoadPermuted<false>((__m256i*)(u + offset / 2));
                    __m256i v_ = LoadPermuted<false>((__m256i*)(v + offset / 2));
                    Yuva422pToBgraV2<T>(y + offset, u_, v_, a + offset, bgra + 4 * offset);
                }
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


        template <class T> SIMD_INLINE void Yuva444pToBgraV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, const uint8_t* a, uint8_t* bgra)
        {
            static const __m256i K32_PERMUTE = SIMD_MM256_SETR_EPI32(0, 2, 4, 6, 1, 3, 5, 7);
            __m256i _y = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)y), K32_PERMUTE);
            __m256i _u = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)u), K32_PERMUTE);
            __m256i _v = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)v), K32_PERMUTE);
            __m256i _a = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)a), K32_PERMUTE);
            YuvaToBgra16<T>(UnpackY<T, 0>(_y), UnpackUV<T, 0>(_u), UnpackUV<T, 0>(_v), UnpackU8<0>(_a), (__m256i*)bgra + 0);
            YuvaToBgra16<T>(UnpackY<T, 1>(_y), UnpackUV<T, 1>(_u), UnpackUV<T, 1>(_v), UnpackU8<1>(_a), (__m256i*)bgra + 2);
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
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Sse41::Yuva444pToBgraV2(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: Yuva444pToBgraV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt709: Yuva444pToBgraV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvBt2020: Yuva444pToBgraV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            case SimdYuvTrect871: Yuva444pToBgraV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride); break;
            default:
                assert(0);
            }
#endif
        }

        //-------------------------------------------------------------------------------------------------

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
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Sse41::Yuv444pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
#else
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv444pToBgraV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
            else
                Yuv444pToBgraV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
#endif
        }

        //-------------------------------------------------------------------------------------------------

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

        template <bool align, class T> void Yuv422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
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
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
                {
                    __m256i u_ = LoadPermuted<align>((__m256i*)(u + colUV));
                    __m256i v_ = LoadPermuted<align>((__m256i*)(v + colUV));
                    Yuv422pToBgraV2<align, T>(y + colY, u_, v_, a_0, bgra + colBgra);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m256i u_ = LoadPermuted<false>((__m256i*)(u + offset / 2));
                    __m256i v_ = LoadPermuted<false>((__m256i*)(v + offset / 2));
                    Yuv422pToBgraV2<false, T>(y + offset, u_, v_, a_0, bgra + 4 * offset);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        template <bool align> void Yuv422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv422pToBgraV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv422pToBgraV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv422pToBgraV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv422pToBgraV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        void Yuv422pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Sse41::Yuv422pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
#else
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv422pToBgraV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
            else
                Yuv422pToBgraV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
#endif
        }

        //-------------------------------------------------------------------------------------------------

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
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Sse41::Yuv420pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
#else
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv420pToBgraV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
            else
                Yuv420pToBgraV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align, class T> SIMD_INLINE void YuvToRgba16(__m256i y16, __m256i u16, __m256i v16, const __m256i& a_0, __m256i* rgba)
        {
            const __m256i b16 = YuvToBlue16<T>(y16, u16);
            const __m256i g16 = YuvToGreen16<T>(y16, u16, v16);
            const __m256i r16 = YuvToRed16<T>(y16, v16);
            const __m256i rg8 = _mm256_or_si256(r16, _mm256_slli_si256(g16, 1));
            const __m256i ba8 = _mm256_or_si256(b16, a_0);
            __m256i rgba0 = _mm256_unpacklo_epi16(rg8, ba8);
            __m256i rgba1 = _mm256_unpackhi_epi16(rg8, ba8);
            Permute2x128(rgba0, rgba1);
            Store<align>(rgba + 0, rgba0);
            Store<align>(rgba + 1, rgba1);
        }

        template <bool align, class T> SIMD_INLINE void YuvToRgba(__m256i y8, __m256i u8, __m256i v8, const __m256i& a_0, __m256i* rgba)
        {
            YuvToRgba16<align, T>(UnpackY<T, 0>(y8), UnpackUV<T, 0>(u8), UnpackUV<T, 0>(v8), a_0, rgba + 0);
            YuvToRgba16<align, T>(UnpackY<T, 1>(y8), UnpackUV<T, 1>(u8), UnpackUV<T, 1>(v8), a_0, rgba + 2);
        }

        template <bool align, class T> SIMD_INLINE void Yuv444pToRgbaV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, const __m256i& a_0, uint8_t* rgba)
        {
            YuvToRgba<align, T>(LoadPermuted<align>((__m256i*)y), LoadPermuted<align>((__m256i*)u), LoadPermuted<align>((__m256i*)v), a_0, (__m256i*)rgba);
        }

        template <bool align, class T> void Yuv444pToRgbaV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(rgba) && Aligned(rgbaStride));
            }

            __m256i a_0 = _mm256_slli_si256(_mm256_set1_epi16(alpha), 1);
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colYuv = 0, colRgba = 0; colYuv < bodyWidth; colYuv += A, colRgba += QA)
                {
                    Yuv444pToRgbaV2<align, T>(y + colYuv, u + colYuv, v + colYuv, a_0, rgba + colRgba);
                }
                if (tail)
                {
                    size_t col = width - A;
                    Yuv444pToRgbaV2<false, T>(y + col, u + col, v + col, a_0, rgba + 4 * col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                rgba += rgbaStride;
            }
        }

        template <bool align> void Yuv444pToRgbaV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToRgbaV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            case SimdYuvBt709: Yuv444pToRgbaV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            case SimdYuvBt2020: Yuv444pToRgbaV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            case SimdYuvTrect871: Yuv444pToRgbaV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            default:
                assert(0);
            }
        }

        void Yuv444pToRgbaV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Sse41::Yuv444pToRgbaV2(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha, yuvType);
#else
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(rgba) && Aligned(rgbaStride))
                Yuv444pToRgbaV2<true>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha, yuvType);
            else
                Yuv444pToRgbaV2<false>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha, yuvType);
#endif
        }
    }
#endif
}
