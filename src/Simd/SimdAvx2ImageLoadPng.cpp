/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdImageSavePng.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE)
    namespace Sse41
    {
        void DecodeLine3(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst);
        void DecodeLine4(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst);
        void DecodeLine5(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst);
    }

    namespace Avx2
    {
        template<size_t pixelSize> SIMD_INLINE void Copy(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            Base::Copy(src, srcStride, width, height, pixelSize, dst, dstStride);
        }

        SIMD_INLINE void GrayToBgra(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            GrayToBgra(src, width, height, srcStride, dst, dstStride, 0xFF);
        }

        SIMD_INLINE void BgrToBgra(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            BgrToBgra(src, width, height, srcStride, dst, dstStride, 0xFF);
        }

        SIMD_INLINE void RgbToBgra(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            RgbToBgra(src, width, height, srcStride, dst, dstStride, 0xFF);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void FillExtra(uint8_t* dst, int srcN, int dstN)
        {
            for (int i = srcN; i < dstN; ++i)
                dst[i] = 0xFF;
        }

        const __m128i K8_SHUF_RGB_TO_RGBA = SIMD_MM_SETR_EPI8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
        const __m256i K32_FFFF0000 = SIMD_MM256_SET1_EPI32(0xFFFF0000);

        enum PngOp
        {
            PngNone = 0,
            PngSub = 1,
            PngUp = 2,
            PngAvg = 3,
            PngPaeth = 4,
            PngAvg0 = 5
        };

        SIMD_INLINE __m256i Combine(const __m128i& lo, const __m128i& hi)
        {
            return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        }

        //-------------------------------------------------------------------------------------------------
        // 128-bit reconstruction, matching Sse41. Used for pixel sizes that do not divide 16
        // and for the serial Average / Paeth steps (each pixel depends on the previous one).

        template<int n> SIMD_INLINE __m128i PrefixSum128(__m128i x);

        template<> SIMD_INLINE __m128i PrefixSum128<1>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum128<2>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum128<3>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 3));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 6));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 12));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum128<4>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum128<6>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 6));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 12));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum128<8>(__m128i x)
        {
            return _mm_add_epi8(x, _mm_slli_si128(x, 8));
        }

        template<int n> SIMD_INLINE __m128i FirstN128()
        {
            return _mm_srli_si128(Sse41::K_INV_ZERO, 16 - n);
        }


        template<int n> SIMD_INLINE __m128i SubVector128(__m128i curr, __m128i& carry)
        {
            enum { step = n * (16 / n) };
            __m128i x = PrefixSum128<n>(_mm_add_epi8(curr, carry));
            carry = _mm_and_si128(_mm_srli_si128(x, step - n), FirstN128<n>());
            return x;
        }


        //-------------------------------------------------------------------------------------------------
        // 256-bit Sub. _mm256_slli_si256 stays inside a 16-byte lane, so both lanes are prefix-summed
        // together and the low lane's last pixel (the running total of each channel) is then added
        // to every byte of the high lane.

        SIMD_INLINE __m256i SubVector1(__m256i curr, __m256i& carry)
        {
            __m256i x = _mm256_add_epi8(curr, carry);
            x = _mm256_add_epi8(x, _mm256_slli_si256(x, 1));
            x = _mm256_add_epi8(x, _mm256_slli_si256(x, 2));
            x = _mm256_add_epi8(x, _mm256_slli_si256(x, 4));
            x = _mm256_add_epi8(x, _mm256_slli_si256(x, 8));
            __m128i last = _mm_shuffle_epi8(_mm_srli_si128(_mm256_castsi256_si128(x), 15), _mm_setzero_si128());
            x = _mm256_add_epi8(x, _mm256_inserti128_si256(_mm256_setzero_si256(), last, 1));
            __m128i next = _mm_and_si128(_mm_srli_si128(_mm256_extracti128_si256(x, 1), 15), FirstN128<1>());
            carry = _mm256_inserti128_si256(_mm256_setzero_si256(), next, 0);
            return x;
        }


        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void TailSub(const uint8_t* curr, int n, int i, int size, uint8_t* dst)
        {
            if (i == 0)
            {
                for (; i < n && i < size; ++i)
                    dst[i] = curr[i];
            }
            for (; i < size; ++i)
                dst[i] = curr[i] + dst[i - n];
        }

        template<int n> void DecodeSubRowNarrow(const uint8_t* curr, int width, uint8_t* dst)
        {
            enum { step = n * (16 / n) };
            int size = width * n, i = 0;
            __m128i carry = _mm_setzero_si128();
            for (; i + 16 <= size; i += step)
            {
                __m128i x = SubVector128<n>(_mm_loadu_si128((__m128i*)(curr + i)), carry);
                _mm_storeu_si128((__m128i*)(dst + i), x);
            }
            TailSub(curr, n, i, size, dst);
        }

        void DecodeSubRowWide1(const uint8_t* curr, int width, uint8_t* dst)
        {
            int size = width, i = 0;
            __m256i carry = _mm256_setzero_si256();
            for (; i + 32 <= size; i += 32)
            {
                __m256i x = SubVector1(_mm256_loadu_si256((__m256i*)(curr + i)), carry);
                _mm256_storeu_si256((__m256i*)(dst + i), x);
            }
            if (i + 16 <= size)
            {
                __m128i c = _mm256_castsi256_si128(carry);
                _mm_storeu_si128((__m128i*)(dst + i), SubVector128<1>(_mm_loadu_si128((__m128i*)(curr + i)), c));
                i += 16;
            }
            TailSub(curr, 1, i, size, dst);
        }

        template<int n> void DecodeSubRow(const uint8_t* curr, int width, uint8_t* dst)
        {
            DecodeSubRowNarrow<n>(curr, width, dst);
        }

        template<> void DecodeSubRow<1>(const uint8_t* curr, int width, uint8_t* dst)
        {
            DecodeSubRowWide1(curr, width, dst);
        }


        static void ScalarLine(int op, const uint8_t* curr, const uint8_t* prev, int x, int width, int srcN, int dstN, uint8_t* dst)
        {
            for (; x < width; ++x)
            {
                int i = 0;
                for (; i < srcN; ++i)
                {
                    int left = x ? dst[i - dstN] : 0;
                    int up = 0, ul = 0;
                    if (op == PngUp || op == PngAvg || op == PngPaeth)
                        up = prev[i];
                    if (op == PngPaeth && x)
                        ul = prev[i - dstN];
                    int add = 0;
                    if (op == PngSub)
                        add = left;
                    else if (op == PngUp)
                        add = up;
                    else if (op == PngAvg)
                        add = (up + left) >> 1;
                    else if (op == PngPaeth)
                        add = Base::Paeth(left, up, ul);
                    else if (op == PngAvg0)
                        add = left >> 1;
                    dst[i] = curr[i] + add;
                }
                FillExtra(dst, srcN, dstN);
                curr += srcN;
                dst += dstN;
                if (op == PngUp || op == PngAvg || op == PngPaeth)
                    prev += dstN;
            }
        }

        // kind 0: 8-bit gray -> gray+alpha (1->2), kind 1: 8-bit rgb -> rgba (3->4), kind 2: 16-bit gray -> gray+alpha (2->4).
        // Value bytes are filtered as dstN-wide pixels of source expanded with zeros; the extra channel is then set to 0xFF.
        template<int kind> SIMD_INLINE __m256i ExpandSrc(const uint8_t* p);

        template<> SIMD_INLINE __m256i ExpandSrc<0>(const uint8_t* p)
        {
            return _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)p));
        }

        template<> SIMD_INLINE __m256i ExpandSrc<1>(const uint8_t* p)
        {
            __m128i x0 = _mm_loadl_epi64((const __m128i*)p);
            x0 = _mm_insert_epi32(x0, *(int*)(p + 8), 2);
            __m128i x1 = _mm_loadl_epi64((const __m128i*)(p + 12));
            x1 = _mm_insert_epi32(x1, *(int*)(p + 20), 2);
            return Combine(_mm_shuffle_epi8(x0, K8_SHUF_RGB_TO_RGBA), _mm_shuffle_epi8(x1, K8_SHUF_RGB_TO_RGBA));
        }

        template<> SIMD_INLINE __m256i ExpandSrc<2>(const uint8_t* p)
        {
            return _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p));
        }

        template<int kind> SIMD_INLINE __m256i AlphaMask();

        template<> SIMD_INLINE __m256i AlphaMask<0>()
        {
            return K16_FF00;
        }

        template<> SIMD_INLINE __m256i AlphaMask<1>()
        {
            return K32_FF000000;
        }

        template<> SIMD_INLINE __m256i AlphaMask<2>()
        {
            return K32_FFFF0000;
        }

        template<int op, int kind> void DecodeExpand(const uint8_t* curr, const uint8_t* prev, int width, uint8_t* dst)
        {
            const int srcN = kind == 0 ? 1 : (kind == 1 ? 3 : 2);
            const int dstN = kind == 0 ? 2 : 4;
            const int group = kind == 0 ? 16 : 8;
            const bool readPrev = (op == PngUp);
            int x = 0;
            __m128i sub = _mm_setzero_si128();
            for (; x + group <= width; x += group)
            {
                __m256i cv = ExpandSrc<kind>(curr + x * srcN);
                __m256i pv = readPrev ? _mm256_loadu_si256((__m256i*)(prev + x * dstN)) : K_ZERO;
                __m256i y;
                if (op == PngNone)
                    y = cv;
                else if (op == PngSub)
                {
                    __m128i lo = SubVector128<dstN>(_mm256_castsi256_si128(cv), sub);
                    __m128i hi = SubVector128<dstN>(_mm256_extracti128_si256(cv, 1), sub);
                    y = Combine(lo, hi);
                }
                else
                    y = _mm256_add_epi8(cv, pv);
                _mm256_storeu_si256((__m256i*)(dst + x * dstN), _mm256_or_si256(y, AlphaMask<kind>()));
            }
            ScalarLine(op, curr + x * srcN, readPrev ? prev + x * dstN : NULL, x, width, srcN, dstN, dst + x * dstN);
        }

        //-------------------------------------------------------------------------------------------------

        void DecodeLine0(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                int size = width * srcN, i = 0, sizeA = (int)AlignLo(size, A);
                for (; i < sizeA; i += (int)A)
                    _mm256_storeu_si256((__m256i*)(dst + i), _mm256_loadu_si256((__m256i*)(curr + i)));
                for (; i < size; ++i)
                    dst[i] = curr[i];
            }
            else if (srcN == 1 && dstN == 2)
                DecodeExpand<PngNone, 0>(curr, prev, width, dst);
            else if (srcN == 3 && dstN == 4)
                DecodeExpand<PngNone, 1>(curr, prev, width, dst);
            else if (srcN == 2 && dstN == 4)
                DecodeExpand<PngNone, 2>(curr, prev, width, dst);
            else
                ScalarLine(PngNone, curr, NULL, 0, width, srcN, dstN, dst);
        }

        void DecodeLine1(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                switch (srcN)
                {
                case 1: DecodeSubRow<1>(curr, width, dst); break;
                case 2: DecodeSubRow<2>(curr, width, dst); break;
                case 3: DecodeSubRow<3>(curr, width, dst); break;
                case 4: DecodeSubRow<4>(curr, width, dst); break;
                case 6: DecodeSubRow<6>(curr, width, dst); break;
                case 8: DecodeSubRow<8>(curr, width, dst); break;
                default:
                    for (int i = 0; i < srcN; ++i)
                        dst[i] = curr[i];
                    for (int i = srcN, n = srcN * width; i < n; ++i)
                        dst[i] = curr[i] + dst[i - dstN];
                    break;
                }
            }
            else if (srcN == 1 && dstN == 2)
                DecodeExpand<PngSub, 0>(curr, prev, width, dst);
            else if (srcN == 3 && dstN == 4)
                DecodeExpand<PngSub, 1>(curr, prev, width, dst);
            else if (srcN == 2 && dstN == 4)
                DecodeExpand<PngSub, 2>(curr, prev, width, dst);
            else
                ScalarLine(PngSub, curr, NULL, 0, width, srcN, dstN, dst);
        }

        void DecodeLine2(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                int size = width * srcN, i = 0, sizeA = (int)AlignLo(size, A);
                for (; i < sizeA; i += (int)A)
                {
                    __m256i _curr = _mm256_loadu_si256((__m256i*)(curr + i));
                    __m256i _prev = _mm256_loadu_si256((__m256i*)(prev + i));
                    _mm256_storeu_si256((__m256i*)(dst + i), _mm256_add_epi8(_curr, _prev));
                }
                for (; i < size; ++i)
                    dst[i] = curr[i] + prev[i];
            }
            else if (srcN == 1 && dstN == 2)
                DecodeExpand<PngUp, 0>(curr, prev, width, dst);
            else if (srcN == 3 && dstN == 4)
                DecodeExpand<PngUp, 1>(curr, prev, width, dst);
            else if (srcN == 2 && dstN == 4)
                DecodeExpand<PngUp, 2>(curr, prev, width, dst);
            else
                ScalarLine(PngUp, curr, prev, 0, width, srcN, dstN, dst);
        }


        void DecodeLine6(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            DecodeLine1(curr, prev, width, srcN, dstN, dst);
        }

        //-------------------------------------------------------------------------------------------------

        ImagePngLoader::ImagePngLoader(const ImageLoaderParam& param)
            : Sse41::ImagePngLoader(param)
        {
        }

        void ImagePngLoader::SetHandlers()
        {
            Sse41::ImagePngLoader::SetHandlers();
            if (_width >= A)
            {
                _decodeLine[0] = DecodeLine0;
                _decodeLine[1] = DecodeLine1;
                _decodeLine[2] = DecodeLine2;
                // Average and Paeth stay on the SSE4.1 kernels. vpblendvb is slower than
                // pblendvb for the serial per-pixel reconstruction, so the SSE4.1 object
                // code is the faster one. None, Up and Sub use the AVX2 kernels above.
                _decodeLine[3] = Sse41::DecodeLine3;
                _decodeLine[4] = Sse41::DecodeLine4;
                _decodeLine[5] = Sse41::DecodeLine5;
                _decodeLine[6] = DecodeLine6;
                if (_depth <= 8)
                {
                    size_t channels = _paletteChannels ? _paletteChannels : _outN;
                    if (channels == 1)
                    {
                        switch (_param.format)
                        {
                        case SimdPixelFormatGray8: _converter = Copy<1>; break;
                        case SimdPixelFormatBgr24: _converter = GrayToBgr; break;
                        case SimdPixelFormatRgb24: _converter = GrayToBgr; break;
                        case SimdPixelFormatBgra32: _converter = GrayToBgra; break;
                        case SimdPixelFormatRgba32: _converter = GrayToBgra; break;
                        }
                    }
                    else if (channels == 3)
                    {
                        switch (_param.format)
                        {
                        case SimdPixelFormatGray8: _converter = RgbToGray; break;
                        case SimdPixelFormatBgr24: _converter = BgrToRgb; break;
                        case SimdPixelFormatRgb24: _converter = Copy<3>; break;
                        case SimdPixelFormatBgra32: _converter = RgbToBgra; break;
                        case SimdPixelFormatRgba32: _converter = BgrToBgra; break;
                        }
                    }
                    else if (channels == 4)
                    {
                        switch (_param.format)
                        {
                        case SimdPixelFormatGray8: _converter = RgbaToGray; break;
                        case SimdPixelFormatBgr24: _converter = BgraToRgb; break;
                        case SimdPixelFormatRgb24: _converter = BgraToBgr; break;
                        case SimdPixelFormatBgra32: _converter = BgraToRgba; break;
                        case SimdPixelFormatRgba32: _converter = Copy<4>; break;
                        }
                    }
                }
            }
        }
    }
#endif
}
