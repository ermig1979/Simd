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
#include "Simd/SimdMemory.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) 
    namespace Sse41
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
        const __m128i K16_LOW4 = SIMD_MM_SETR_EPI16(255, 255, 255, 255, 0, 0, 0, 0);
        const __m128i K32_FFFF0000 = SIMD_MM_SET1_EPI32(0xFFFF0000);

        enum PngOp
        {
            PngNone = 0,
            PngSub = 1,
            PngUp = 2,
            PngAvg = 3,
            PngPaeth = 4,
            PngAvg0 = 5
        };

        template<int n> SIMD_INLINE __m128i PrefixSum(__m128i x);

        template<> SIMD_INLINE __m128i PrefixSum<1>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<2>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<3>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 3));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 6));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 12));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<4>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<6>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 6));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 12));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<8>(__m128i x)
        {
            return _mm_add_epi8(x, _mm_slli_si128(x, 8));
        }

        template<int n> SIMD_INLINE __m128i FirstN()
        {
            return _mm_srli_si128(K_INV_ZERO, 16 - n);
        }

        SIMD_INLINE __m128i Average(__m128i a, __m128i b)
        {
            return _mm_sub_epi8(_mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), K8_01));
        }

        SIMD_INLINE __m128i Paeth16(__m128i a, __m128i b, __m128i c)
        {
            __m128i pa = _mm_sub_epi16(b, c);
            __m128i pb = _mm_sub_epi16(a, c);
            __m128i pc = _mm_add_epi16(pa, pb);
            pa = _mm_abs_epi16(pa);
            pb = _mm_abs_epi16(pb);
            pc = _mm_abs_epi16(pc);
            __m128i mbc = _mm_or_si128(_mm_cmpgt_epi16(pa, pb), _mm_cmpgt_epi16(pa, pc));
            __m128i mc = _mm_cmpgt_epi16(pb, pc);
            return _mm_blendv_epi8(a, _mm_blendv_epi8(b, c, mc), mbc);
        }

        template<int n> SIMD_INLINE __m128i SubVector(__m128i curr, __m128i& carry)
        {
            enum { step = n * (16 / n) };
            __m128i x = PrefixSum<n>(_mm_add_epi8(curr, carry));
            carry = _mm_and_si128(_mm_srli_si128(x, step - n), FirstN<n>());
            return x;
        }

        template<int n, int shift, int step> struct RunAvg;

        template<int n, int step> struct RunAvg<n, step, step>
        {
            static SIMD_INLINE void Run(__m128i, __m128i, __m128i&, __m128i&)
            {
            }
        };

        template<int n, int shift, int step> struct RunAvg
        {
            static SIMD_INLINE void Run(__m128i curr, __m128i prev, __m128i& left, __m128i& dst)
            {
                __m128i pix = _mm_and_si128(_mm_add_epi8(_mm_srli_si128(curr, shift), Average(_mm_srli_si128(prev, shift), left)), FirstN<n>());
                dst = _mm_or_si128(dst, _mm_slli_si128(pix, shift));
                left = pix;
                RunAvg<n, shift + n, step>::Run(curr, prev, left, dst);
            }
        };

        template<int n> SIMD_INLINE __m128i AvgVector(__m128i curr, __m128i prev, __m128i& left)
        {
            enum { step = n * (16 / n) };
            __m128i dst = _mm_setzero_si128();
            RunAvg<n, 0, step>::Run(curr, prev, left, dst);
            return dst;
        }

        template<int n, int shift, int step> struct RunPaeth;

        template<int n, int step> struct RunPaeth<n, step, step>
        {
            static SIMD_INLINE void Run(__m128i, __m128i, __m128i&, __m128i&, __m128i&)
            {
            }
        };

        template<int n, int shift, int step> struct RunPaeth
        {
            static SIMD_INLINE void Run(__m128i curr, __m128i prev, __m128i& a, __m128i& c, __m128i& dst)
            {
                __m128i b = _mm_cvtepu8_epi16(_mm_srli_si128(prev, shift));
                __m128i d = _mm_cvtepu8_epi16(_mm_srli_si128(curr, shift));
                a = _mm_and_si128(_mm_add_epi16(d, Paeth16(a, b, c)), K16_00FF);
                __m128i pix = _mm_and_si128(_mm_packus_epi16(a, K_ZERO), FirstN<n>());
                dst = _mm_or_si128(dst, _mm_slli_si128(pix, shift));
                c = b;
                RunPaeth<n, shift + n, step>::Run(curr, prev, a, c, dst);
            }
        };

        SIMD_INLINE __m128i Paeth4(__m128i curr, __m128i prev, __m128i& a, __m128i& c)
        {
            __m128i b, d, s0, s1, s2, s3;
            b = _mm_cvtepu8_epi16(prev);
            d = _mm_cvtepu8_epi16(curr);
            a = _mm_and_si128(_mm_add_epi16(d, Paeth16(a, b, c)), K16_LOW4);
            c = _mm_and_si128(b, K16_LOW4);
            s0 = a;
            b = _mm_cvtepu8_epi16(_mm_srli_si128(prev, 4));
            d = _mm_cvtepu8_epi16(_mm_srli_si128(curr, 4));
            a = _mm_and_si128(_mm_add_epi16(d, Paeth16(a, b, c)), K16_LOW4);
            c = _mm_and_si128(b, K16_LOW4);
            s1 = a;
            b = _mm_cvtepu8_epi16(_mm_srli_si128(prev, 8));
            d = _mm_cvtepu8_epi16(_mm_srli_si128(curr, 8));
            a = _mm_and_si128(_mm_add_epi16(d, Paeth16(a, b, c)), K16_LOW4);
            c = _mm_and_si128(b, K16_LOW4);
            s2 = a;
            b = _mm_cvtepu8_epi16(_mm_srli_si128(prev, 12));
            d = _mm_cvtepu8_epi16(_mm_srli_si128(curr, 12));
            a = _mm_and_si128(_mm_add_epi16(d, Paeth16(a, b, c)), K16_LOW4);
            c = _mm_and_si128(b, K16_LOW4);
            s3 = a;
            __m128i p01 = _mm_packus_epi16(s0, s1);
            __m128i p23 = _mm_packus_epi16(s2, s3);
            __m128i a01 = _mm_unpacklo_epi32(p01, _mm_srli_si128(p01, 8));
            __m128i a23 = _mm_unpacklo_epi32(p23, _mm_srli_si128(p23, 8));
            return _mm_unpacklo_epi64(a01, a23);
        }

        SIMD_INLINE __m128i Paeth8(__m128i curr, __m128i prev, __m128i& a, __m128i& c)
        {
            __m128i b0 = _mm_cvtepu8_epi16(prev);
            __m128i d0 = _mm_cvtepu8_epi16(curr);
            __m128i s0 = _mm_and_si128(_mm_add_epi16(d0, Paeth16(a, b0, c)), K16_00FF);
            __m128i b1 = _mm_cvtepu8_epi16(_mm_srli_si128(prev, 8));
            __m128i d1 = _mm_cvtepu8_epi16(_mm_srli_si128(curr, 8));
            __m128i s1 = _mm_and_si128(_mm_add_epi16(d1, Paeth16(s0, b1, b0)), K16_00FF);
            a = s1;
            c = b1;
            return _mm_packus_epi16(s0, s1);
        }

        template<int n> SIMD_INLINE __m128i PaethVector(__m128i curr, __m128i prev, __m128i& a, __m128i& c)
        {
            enum { step = n * (16 / n) };
            __m128i dst = _mm_setzero_si128();
            RunPaeth<n, 0, step>::Run(curr, prev, a, c, dst);
            return dst;
        }

        template<> SIMD_INLINE __m128i PaethVector<4>(__m128i curr, __m128i prev, __m128i& a, __m128i& c)
        {
            return Paeth4(curr, prev, a, c);
        }

        template<> SIMD_INLINE __m128i PaethVector<8>(__m128i curr, __m128i prev, __m128i& a, __m128i& c)
        {
            return Paeth8(curr, prev, a, c);
        }

        template<int n> void DecodeSubRow(const uint8_t* curr, int width, uint8_t* dst)
        {
            enum { step = n * (16 / n) };
            int size = width * n, i = 0;
            __m128i carry = _mm_setzero_si128();
            for (; i + 16 <= size; i += step)
            {
                __m128i x = SubVector<n>(_mm_loadu_si128((__m128i*)(curr + i)), carry);
                _mm_storeu_si128((__m128i*)(dst + i), x);
            }
            if (i == 0)
            {
                for (; i < n && i < size; ++i)
                    dst[i] = curr[i];
            }
            for (; i < size; ++i)
                dst[i] = curr[i] + dst[i - n];
        }

        template<int n, bool usePrev> void DecodeAvgRow(const uint8_t* curr, const uint8_t* prev, int width, uint8_t* dst)
        {
            enum { step = n * (16 / n) };
            int size = width * n, i = 0;
            __m128i left = _mm_setzero_si128();
            for (; i + 16 <= size; i += step)
            {
                __m128i cv = _mm_loadu_si128((__m128i*)(curr + i));
                __m128i pv = usePrev ? _mm_loadu_si128((__m128i*)(prev + i)) : K_ZERO;
                _mm_storeu_si128((__m128i*)(dst + i), AvgVector<n>(cv, pv, left));
            }
            if (i == 0)
            {
                for (; i < n && i < size; ++i)
                    dst[i] = curr[i] + (usePrev ? (prev[i] >> 1) : 0);
            }
            for (; i < size; ++i)
            {
                int up = usePrev ? prev[i] : 0;
                dst[i] = curr[i] + ((up + dst[i - n]) >> 1);
            }
        }

        template<int n> void DecodePaethRow(const uint8_t* curr, const uint8_t* prev, int width, uint8_t* dst)
        {
            enum { step = n * (16 / n) };
            int size = width * n, i = 0;
            __m128i a = _mm_setzero_si128();
            __m128i c = _mm_setzero_si128();
            for (; i + 16 <= size; i += step)
            {
                __m128i cv = _mm_loadu_si128((__m128i*)(curr + i));
                __m128i pv = _mm_loadu_si128((__m128i*)(prev + i));
                _mm_storeu_si128((__m128i*)(dst + i), PaethVector<n>(cv, pv, a, c));
            }
            if (i == 0)
            {
                for (; i < n && i < size; ++i)
                    dst[i] = curr[i] + Base::Paeth(0, prev[i], 0);
            }
            for (; i < size; ++i)
                dst[i] = curr[i] + Base::Paeth(dst[i - n], prev[i], prev[i - n]);
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
        template<int kind> SIMD_INLINE __m128i ExpandSrc(const uint8_t* p);

        template<> SIMD_INLINE __m128i ExpandSrc<0>(const uint8_t* p)
        {
            return _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)p), K_ZERO);
        }

        template<> SIMD_INLINE __m128i ExpandSrc<1>(const uint8_t* p)
        {
            __m128i x = _mm_loadl_epi64((__m128i*)p);
            x = _mm_insert_epi32(x, *(int*)(p + 8), 2);
            return _mm_shuffle_epi8(x, K8_SHUF_RGB_TO_RGBA);
        }

        template<> SIMD_INLINE __m128i ExpandSrc<2>(const uint8_t* p)
        {
            return _mm_unpacklo_epi16(_mm_loadl_epi64((__m128i*)p), K_ZERO);
        }

        template<int kind> SIMD_INLINE __m128i AlphaMask();

        template<> SIMD_INLINE __m128i AlphaMask<0>()
        {
            return K16_FF00;
        }

        template<> SIMD_INLINE __m128i AlphaMask<1>()
        {
            return K32_FF000000;
        }

        template<> SIMD_INLINE __m128i AlphaMask<2>()
        {
            return K32_FFFF0000;
        }

        template<int op, int kind> void DecodeExpand(const uint8_t* curr, const uint8_t* prev, int width, uint8_t* dst)
        {
            const int srcN = kind == 0 ? 1 : (kind == 1 ? 3 : 2);
            const int dstN = kind == 0 ? 2 : 4;
            const int group = kind == 0 ? 8 : 4;
            const bool readPrev = (op == PngUp || op == PngAvg || op == PngPaeth);
            int x = 0;
            __m128i a = _mm_setzero_si128();
            __m128i c = _mm_setzero_si128();
            for (; x + group <= width; x += group)
            {
                __m128i cv = ExpandSrc<kind>(curr + x * srcN);
                __m128i pv = readPrev ? _mm_loadu_si128((__m128i*)(prev + x * dstN)) : K_ZERO;
                __m128i y;
                if (op == PngNone)
                    y = cv;
                else if (op == PngSub)
                    y = SubVector<dstN>(cv, a);
                else if (op == PngUp)
                    y = _mm_add_epi8(cv, pv);
                else if (op == PngAvg || op == PngAvg0)
                    y = AvgVector<dstN>(cv, op == PngAvg ? pv : K_ZERO, a);
                else
                    y = PaethVector<dstN>(cv, pv, a, c);
                _mm_storeu_si128((__m128i*)(dst + x * dstN), _mm_or_si128(y, AlphaMask<kind>()));
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
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_loadu_si128((__m128i*)(curr + i)));
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
                    __m128i _curr = _mm_loadu_si128((__m128i*)(curr + i));
                    __m128i _prev = _mm_loadu_si128((__m128i*)(prev + i));
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_add_epi8(_curr, _prev));
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

        void DecodeLine3(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                switch (srcN)
                {
                case 2: DecodeAvgRow<2, true>(curr, prev, width, dst); break;
                case 3: DecodeAvgRow<3, true>(curr, prev, width, dst); break;
                case 4: DecodeAvgRow<4, true>(curr, prev, width, dst); break;
                case 6: DecodeAvgRow<6, true>(curr, prev, width, dst); break;
                case 8: DecodeAvgRow<8, true>(curr, prev, width, dst); break;
                default:
                    for (int i = 0; i < srcN; ++i)
                        dst[i] = curr[i] + (prev[i] >> 1);
                    for (int i = srcN, n = srcN * width; i < n; ++i)
                        dst[i] = curr[i] + ((prev[i] + dst[i - dstN]) >> 1);
                    break;
                }
            }
            else if (srcN == 1 && dstN == 2)
                DecodeExpand<PngAvg, 0>(curr, prev, width, dst);
            else if (srcN == 3 && dstN == 4)
                DecodeExpand<PngAvg, 1>(curr, prev, width, dst);
            else if (srcN == 2 && dstN == 4)
                DecodeExpand<PngAvg, 2>(curr, prev, width, dst);
            else
                ScalarLine(PngAvg, curr, prev, 0, width, srcN, dstN, dst);
        }

        void DecodeLine4(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                switch (srcN)
                {
                case 2: DecodePaethRow<2>(curr, prev, width, dst); break;
                case 3: DecodePaethRow<3>(curr, prev, width, dst); break;
                case 4: DecodePaethRow<4>(curr, prev, width, dst); break;
                case 6: DecodePaethRow<6>(curr, prev, width, dst); break;
                case 8: DecodePaethRow<8>(curr, prev, width, dst); break;
                default:
                    for (int i = 0; i < srcN; ++i)
                        dst[i] = curr[i] + Base::Paeth(0, prev[i], 0);
                    for (int i = srcN, n = srcN * width; i < n; ++i)
                        dst[i] = curr[i] + Base::Paeth(dst[i - dstN], prev[i], prev[i - dstN]);
                    break;
                }
            }
            else if (srcN == 1 && dstN == 2)
                DecodeExpand<PngPaeth, 0>(curr, prev, width, dst);
            else if (srcN == 3 && dstN == 4)
                DecodeExpand<PngPaeth, 1>(curr, prev, width, dst);
            else if (srcN == 2 && dstN == 4)
                DecodeExpand<PngPaeth, 2>(curr, prev, width, dst);
            else
                ScalarLine(PngPaeth, curr, prev, 0, width, srcN, dstN, dst);
        }

        void DecodeLine5(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                switch (srcN)
                {
                case 2: DecodeAvgRow<2, false>(curr, prev, width, dst); break;
                case 3: DecodeAvgRow<3, false>(curr, prev, width, dst); break;
                case 4: DecodeAvgRow<4, false>(curr, prev, width, dst); break;
                case 6: DecodeAvgRow<6, false>(curr, prev, width, dst); break;
                case 8: DecodeAvgRow<8, false>(curr, prev, width, dst); break;
                default:
                    for (int i = 0; i < srcN; ++i)
                        dst[i] = curr[i];
                    for (int i = srcN, n = srcN * width; i < n; ++i)
                        dst[i] = curr[i] + (dst[i - dstN] >> 1);
                    break;
                }
            }
            else if (srcN == 1 && dstN == 2)
                DecodeExpand<PngAvg0, 0>(curr, prev, width, dst);
            else if (srcN == 3 && dstN == 4)
                DecodeExpand<PngAvg0, 1>(curr, prev, width, dst);
            else if (srcN == 2 && dstN == 4)
                DecodeExpand<PngAvg0, 2>(curr, prev, width, dst);
            else
                ScalarLine(PngAvg0, curr, NULL, 0, width, srcN, dstN, dst);
        }

        void DecodeLine6(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            DecodeLine1(curr, prev, width, srcN, dstN, dst);
        }

        //-------------------------------------------------------------------------------------------------

        ImagePngLoader::ImagePngLoader(const ImageLoaderParam& param)
            : Base::ImagePngLoader(param)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgb24;
        }

        void ImagePngLoader::SetHandlers()
        {
            Base::ImagePngLoader::SetHandlers();
            if (_width >= A)
            {
                _decodeLine[0] = DecodeLine0;
                _decodeLine[1] = DecodeLine1;
                _decodeLine[2] = DecodeLine2;
                _decodeLine[3] = DecodeLine3;
                _decodeLine[4] = DecodeLine4;
                _decodeLine[5] = DecodeLine5;
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
