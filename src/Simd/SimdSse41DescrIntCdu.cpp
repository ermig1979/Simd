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
#include "Simd/SimdExtract.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdFloat16.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template<int bits> __m128i UnpackData16(const uint8_t* src);

        template<> SIMD_INLINE __m128i UnpackData16<4>(const uint8_t* src)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C4_SHFL0), C4_MULLO), 12);
            __m128i hi = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C4_SHFL1), C4_MULLO), 12);
            return _mm_packus_epi16(lo, hi);
        }

        template<> SIMD_INLINE __m128i UnpackData16<5>(const uint8_t* src)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C5_SHFL0), C5_MULLO), 11);
            __m128i hi = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C5_SHFL1), C5_MULLO), 11);
            return _mm_packus_epi16(lo, hi);
        }

        template<> SIMD_INLINE __m128i UnpackData16<6>(const uint8_t* src)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C6_SHFL0), C6_MULLO), 10);
            __m128i hi = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C6_SHFL1), C6_MULLO), 10);
            return _mm_packus_epi16(lo, hi);
        }

        template<> SIMD_INLINE __m128i UnpackData16<7>(const uint8_t* src)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C7_SHFL0), C7_MULLO), 9);
            __m128i hi = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C7_SHFL1), C7_MULLO), 9);
            return _mm_packus_epi16(lo, hi);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void UnpackDataA(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t size16 = AlignLo(size - 1, 16);
            for (size_t i = 0; i < count; i++)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = (uint8_t*)dst + i * size;
                size_t j = 0;
                for (; j < size16; j += 16, ps += 2 * bits, pd += 16)
                    _mm_storeu_si128((__m128i*)pd, UnpackData16<bits>(ps));
                for (; j < size; j += 8, ps += bits, pd += 8)
                    _mm_storel_epi64((__m128i*)pd, UnpackData8<bits>(ps));
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void UnpackDataA8(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t size16 = AlignLo(size, 16);
            for (size_t i = 0, j; i < count; i++)
            {
                const uint8_t* ps = src[i] + 16;
                uint16_t* pd = (uint16_t*)dst + i * size;
                for (j = 0; j < size16; j += 16, ps += 16, pd += 16)
                {
                    __m128i s = _mm_loadu_si128((__m128i*)ps);
                    _mm_storeu_si128((__m128i*)pd + 0, UnpackU8<0>(s));
                    _mm_storeu_si128((__m128i*)pd + 1, UnpackU8<1>(s));
                }
                for (; j < size; j += 8, ps += 8, pd += 8)
                {
                    __m128i s = _mm_loadl_epi64((__m128i*)ps);
                    _mm_storeu_si128((__m128i*)pd, UnpackU8<0>(s));
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> SIMD_INLINE void UnpackDataBx4x16(const uint8_t* const* src, size_t offset, uint8_t* dst)
        {
            __m128i a0 = UnpackData16<bits>(src[0] + offset);
            __m128i a1 = UnpackData16<bits>(src[1] + offset);
            __m128i a2 = UnpackData16<bits>(src[2] + offset);
            __m128i a3 = UnpackData16<bits>(src[3] + offset);
            __m128i b0 = _mm_unpacklo_epi32(a0, a2);
            __m128i b1 = _mm_unpacklo_epi32(a1, a3);
            __m128i b2 = _mm_unpackhi_epi32(a0, a2);
            __m128i b3 = _mm_unpackhi_epi32(a1, a3);
            _mm_storeu_si128((__m128i*)dst + 0, _mm_unpacklo_epi32(b0, b1));
            _mm_storeu_si128((__m128i*)dst + 2, _mm_unpackhi_epi32(b0, b1));
            _mm_storeu_si128((__m128i*)dst + 4, _mm_unpacklo_epi32(b2, b3));
            _mm_storeu_si128((__m128i*)dst + 6, _mm_unpackhi_epi32(b2, b3));
        }

        template<int bits> SIMD_INLINE void UnpackDataBx4x8(const uint8_t* const* src, size_t offset, uint8_t* dst)
        {
            __m128i a0 = UnpackData8<bits>(src[0] + offset);
            __m128i a1 = UnpackData8<bits>(src[1] + offset);
            __m128i a2 = UnpackData8<bits>(src[2] + offset);
            __m128i a3 = UnpackData8<bits>(src[3] + offset);
            __m128i b0 = _mm_unpacklo_epi32(a0, a2);
            __m128i b1 = _mm_unpacklo_epi32(a1, a3);
            _mm_storeu_si128((__m128i*)dst + 0, _mm_unpacklo_epi32(b0, b1));
            _mm_storeu_si128((__m128i*)dst + 2, _mm_unpackhi_epi32(b0, b1));
        }

        template<int bits> void UnpackDataB(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t count8 = AlignLo(count, 8), size16 = AlignLo(size - 1, 16), i, j, o;
            for (i = 0; i < count8; i += 8, src += 8)
            {
                for (j = 0, o = 16; j < size16; j += 16, o += 2 * bits, dst += 8 * A)
                {
                    UnpackDataBx4x16<bits>(src + 0, o, dst + 0);
                    UnpackDataBx4x16<bits>(src + 4, o, dst + A);
                }
                for (; j < size; j += 8, o += bits, dst += 4 * A)
                {
                    UnpackDataBx4x8<bits>(src + 0, o, dst + 0);
                    UnpackDataBx4x8<bits>(src + 4, o, dst + A);
                }
            }
            if (i < count)
            {
                const uint8_t* _src[8];
                for (size_t j = 0; j < 8; i++, j++)
                    _src[j] = i < count ? *src++ : src[-1];
                for (j = 0, o = 16; j < size16; j += 16, o += 2 * bits, dst += 8 * A)
                {
                    UnpackDataBx4x16<bits>(_src + 0, o, dst + 0);
                    UnpackDataBx4x16<bits>(_src + 4, o, dst + A);
                }
                for (; j < size; j += 8, o += bits, dst += 4 * A)
                {
                    UnpackDataBx4x8<bits>(_src + 0, o, dst + 0);
                    UnpackDataBx4x8<bits>(_src + 4, o, dst + A);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void UnpackDataB8x4(const uint8_t* const* src, size_t offset, uint8_t* dst)
        {
            __m128i a0 = UnpackU8<0>(_mm_loadl_epi64((__m128i*)(src[0] + offset)));
            __m128i a1 = UnpackU8<0>(_mm_loadl_epi64((__m128i*)(src[1] + offset)));
            __m128i a2 = UnpackU8<0>(_mm_loadl_epi64((__m128i*)(src[2] + offset)));
            __m128i a3 = UnpackU8<0>(_mm_loadl_epi64((__m128i*)(src[3] + offset)));
            __m128i b0 = _mm_unpacklo_epi32(a0, a2);
            __m128i b1 = _mm_unpacklo_epi32(a1, a3);
            __m128i b2 = _mm_unpackhi_epi32(a0, a2);
            __m128i b3 = _mm_unpackhi_epi32(a1, a3);
            _mm_storeu_si128((__m128i*)dst + 0, _mm_unpacklo_epi32(b0, b1));
            _mm_storeu_si128((__m128i*)dst + 2, _mm_unpackhi_epi32(b0, b1));
            _mm_storeu_si128((__m128i*)dst + 4, _mm_unpacklo_epi32(b2, b3));
            _mm_storeu_si128((__m128i*)dst + 6, _mm_unpackhi_epi32(b2, b3));
        }

        static void UnpackDataB8(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t count8 = AlignLo(count, 8), i;
            for (i = 0, size += 16; i < count8; i += 8, src += 8)
            {
                for (size_t j = 16; j < size; j += 8, dst += 8 * A)
                {
                    UnpackDataB8x4(src + 0, j, dst + 0);
                    UnpackDataB8x4(src + 4, j, dst + A);
                }
            }
            if (i < count)
            {
                const uint8_t* _src[8];
                for (size_t j = 0; j < 8; i++, j++)
                    _src[j] = i < count ? *src++ : src[-1];
                for (size_t j = 16; j < size; j += 8, dst += 8 * A)
                {
                    UnpackDataB8x4(_src + 0, j, dst + 0);
                    UnpackDataB8x4(_src + 4, j, dst + A);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m128i Set2(const int16_t* src)
        {
            return _mm_set1_epi32(*(int32_t*)src);
        }

        SIMD_INLINE void Madd2(__m128i& ab, __m128i a, __m128i b)
        {
            ab = _mm_add_epi32(ab, _mm_madd_epi16(a, b));
        }

        template<int M> void Correlation16_2xM(size_t N, size_t K, const int16_t* ad0, const int16_t* bd, const float *an, const float *bn, size_t bnStride, float* distances, size_t stride)
        {
            __m128i ab00, ab01, ab10, ab11, ab20, ab21, ab30, ab31, ab40, ab41, ab50, ab51, a0, b0, b1;
            const int16_t* ad1 = ad0 + 1 * K;
            const int16_t* ad2 = ad0 + 2 * K;
            const int16_t* ad3 = ad0 + 3 * K;
            const int16_t* ad4 = ad0 + 4 * K;
            const int16_t* ad5 = ad0 + 5 * K;
            if (N > 4)
            {
                if (M > 0) ab00 = _mm_setzero_si128(), ab01 = _mm_setzero_si128();
                if (M > 1) ab10 = _mm_setzero_si128(), ab11 = _mm_setzero_si128();
                if (M > 2) ab20 = _mm_setzero_si128(), ab21 = _mm_setzero_si128();
                if (M > 3) ab30 = _mm_setzero_si128(), ab31 = _mm_setzero_si128();
                if (M > 4) ab40 = _mm_setzero_si128(), ab41 = _mm_setzero_si128();
                if (M > 5) ab50 = _mm_setzero_si128(), ab51 = _mm_setzero_si128();
                for (size_t k = 0; k < K; k += 2)
                {
                    b0 = _mm_loadu_si128((__m128i*)bd + 0);
                    b1 = _mm_loadu_si128((__m128i*)bd + 1);
                    if (M > 0) a0 = Set2(ad0 + k), Madd2(ab00, a0, b0), Madd2(ab01, a0, b1);
                    if (M > 1) a0 = Set2(ad1 + k), Madd2(ab10, a0, b0), Madd2(ab11, a0, b1);
                    if (M > 2) a0 = Set2(ad2 + k), Madd2(ab20, a0, b0), Madd2(ab21, a0, b1);
                    if (M > 3) a0 = Set2(ad3 + k), Madd2(ab30, a0, b0), Madd2(ab31, a0, b1);
                    if (M > 4) a0 = Set2(ad4 + k), Madd2(ab40, a0, b0), Madd2(ab41, a0, b1);
                    if (M > 5) a0 = Set2(ad5 + k), Madd2(ab50, a0, b0), Madd2(ab51, a0, b1);
                    bd += 16;
                } 
                if (N == 8)
                {
                    if (M > 0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab01, distances + 4), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab11, distances + 4), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab21, distances + 4), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab31, distances + 4), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab41, distances + 4), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab50, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab51, distances + 4), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab01, distances + 4, N - 4), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab11, distances + 4, N - 4), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab21, distances + 4, N - 4), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab31, distances + 4, N - 4), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab41, distances + 4, N - 4), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab50, distances + 0), DecodeCosineDistances1xF(an, bn + 4, bnStride, ab51, distances + 4, N - 4), an += 4, distances += stride;
                }
            }
            else
            {
                if (M > 0) ab00 = _mm_setzero_si128();
                if (M > 1) ab10 = _mm_setzero_si128();
                if (M > 2) ab20 = _mm_setzero_si128();
                if (M > 3) ab30 = _mm_setzero_si128();
                if (M > 4) ab40 = _mm_setzero_si128();
                if (M > 5) ab50 = _mm_setzero_si128();
                for (size_t k = 0; k < K; k += 2)
                {
                    b0 = _mm_loadu_si128((__m128i*)bd + 0);
                    if (M > 0) a0 = Set2(ad0 + k), Madd2(ab00, a0, b0);
                    if (M > 1) a0 = Set2(ad1 + k), Madd2(ab10, a0, b0);
                    if (M > 2) a0 = Set2(ad2 + k), Madd2(ab20, a0, b0);
                    if (M > 3) a0 = Set2(ad3 + k), Madd2(ab30, a0, b0);
                    if (M > 4) a0 = Set2(ad4 + k), Madd2(ab40, a0, b0);
                    if (M > 5) a0 = Set2(ad5 + k), Madd2(ab50, a0, b0);
                    bd += 16;
                }
                if (N == 4)
                {
                    if (M > 0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab50, distances + 0), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0, N), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0, N), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0, N), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0, N), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0, N), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab50, distances + 0, N), an += 4, distances += stride;
                }
            }
        }

        typedef void(*Correlation16_2xM_Ptr)(size_t N, size_t K, const int16_t* ad0, const int16_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride);

        SIMD_INLINE Correlation16_2xM_Ptr GetCorrelation16_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Correlation16_2xM<1>;
            case 2: return Correlation16_2xM<2>;
            case 3: return Correlation16_2xM<3>;
            case 4: return Correlation16_2xM<4>;
            case 5: return Correlation16_2xM<5>;
            case 6: return Correlation16_2xM<6>;
            }
            assert(0);
            return NULL;
        }

        void MacroCorrelation16(size_t M, size_t N, size_t K, const uint8_t* ad, const float* an, const uint8_t* bd, const float* bn, float* distances, size_t stride)
        {
            size_t M6 = AlignLoAny(M, 6);
            Correlation16_2xM_Ptr correlation_2x6 = GetCorrelation16_2xM(6);
            Correlation16_2xM_Ptr correlation_2xT = GetCorrelation16_2xM(M - M6);
            const int16_t* a = (int16_t*)ad;
            const int16_t* b = (int16_t*)bd;
            for (size_t j = 0; j < N; j += 8)
            {
                size_t dN = Simd::Min<size_t>(8, N - j);
                size_t i = 0;
                for (; i < M6; i += 6)
                    correlation_2x6(dN, K, a + i * K, b, an + i * 4, bn, N, distances + i * stride, stride);
                if(i < M)
                    correlation_2xT(dN, K, a + i * K, b, an + i * 4, bn, N, distances + i * stride, stride);
                b += K * 8;
                bn += 8;
                distances += 8;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int M> void Correlation8_2xM(size_t N, size_t K, const uint8_t* ad0, const uint8_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            __m128i ab00, ab01, ab10, ab11, ab20, ab21, ab30, ab31, ab40, ab41, a0, b0, b1;
            const uint8_t* ad1 = ad0 + 1 * K;
            const uint8_t* ad2 = ad0 + 2 * K;
            const uint8_t* ad3 = ad0 + 3 * K;
            const uint8_t* ad4 = ad0 + 4 * K;
            if (N > 4)
            {
                if (M > 0) ab00 = _mm_setzero_si128(), ab01 = _mm_setzero_si128();
                if (M > 1) ab10 = _mm_setzero_si128(), ab11 = _mm_setzero_si128();
                if (M > 2) ab20 = _mm_setzero_si128(), ab21 = _mm_setzero_si128();
                if (M > 3) ab30 = _mm_setzero_si128(), ab31 = _mm_setzero_si128();
                if (M > 4) ab40 = _mm_setzero_si128(), ab41 = _mm_setzero_si128();
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = _mm_loadu_si128((__m128i*)bd + 0);
                    b1 = _mm_loadu_si128((__m128i*)bd + 1);
                    if (M > 0) a0 = Set4(ad0 + k), Madd4<true>(ab00, a0, b0), Madd4<true>(ab01, a0, b1);
                    if (M > 1) a0 = Set4(ad1 + k), Madd4<true>(ab10, a0, b0), Madd4<true>(ab11, a0, b1);
                    if (M > 2) a0 = Set4(ad2 + k), Madd4<true>(ab20, a0, b0), Madd4<true>(ab21, a0, b1);
                    if (M > 3) a0 = Set4(ad3 + k), Madd4<true>(ab30, a0, b0), Madd4<true>(ab31, a0, b1);
                    if (M > 4) a0 = Set4(ad4 + k), Madd4<true>(ab40, a0, b0), Madd4<true>(ab41, a0, b1);
                    bd += DA;
                }
                if (N == DF)
                {
                    if (M > 0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab01, distances + F), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab11, distances + F), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab21, distances + F), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab31, distances + F), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab41, distances + F), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab01, distances + F, N - F), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab11, distances + F, N - F), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab21, distances + F, N - F), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab31, distances + F, N - F), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab41, distances + F, N - F), an += 4, distances += stride;
                }
            }
            else
            {
                if (M > 0) ab00 = _mm_setzero_si128();
                if (M > 1) ab10 = _mm_setzero_si128();
                if (M > 2) ab20 = _mm_setzero_si128();
                if (M > 3) ab30 = _mm_setzero_si128();
                if (M > 4) ab40 = _mm_setzero_si128();
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = _mm_loadu_si128((__m128i*)bd + 0);
                    if (M > 0) a0 = Set4(ad0 + k), Madd4<true>(ab00, a0, b0);
                    if (M > 1) a0 = Set4(ad1 + k), Madd4<true>(ab10, a0, b0);
                    if (M > 2) a0 = Set4(ad2 + k), Madd4<true>(ab20, a0, b0);
                    if (M > 3) a0 = Set4(ad3 + k), Madd4<true>(ab30, a0, b0);
                    if (M > 4) a0 = Set4(ad4 + k), Madd4<true>(ab40, a0, b0);
                    bd += DA;
                }
                if (N == F)
                {
                    if (M > 0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0, N), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0, N), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0, N), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0, N), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0, N), an += 4, distances += stride;
                }
            }
        }

        typedef void(*Correlation8_2xM_Ptr)(size_t N, size_t K, const uint8_t* ad0, const uint8_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride);

        SIMD_INLINE Correlation8_2xM_Ptr GetCorrelation8_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Correlation8_2xM<1>;
            case 2: return Correlation8_2xM<2>;
            case 3: return Correlation8_2xM<3>;
            case 4: return Correlation8_2xM<4>;
            case 5: return Correlation8_2xM<5>;
            }
            assert(0);
            return NULL;
        }

        void MacroCorrelation8(size_t M, size_t N, size_t K, const uint8_t* ad, const float* an, const uint8_t* bd, const float* bn, float* distances, size_t stride)
        {
            size_t M5 = AlignLoAny(M, 5);
            Correlation8_2xM_Ptr correlation_2x5 = GetCorrelation8_2xM(5);
            Correlation8_2xM_Ptr correlation_2xT = GetCorrelation8_2xM(M - M5);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min<size_t>(DF, N - j);
                size_t i = 0;
                for (; i < M5; i += 5)
                    correlation_2x5(dN, K, ad + i * K, bd, an + i * 4, bn, N, distances + i * stride, stride);
                if (i < M)
                    correlation_2xT(dN, K, ad + i * K, bd, an + i * 4, bn, N, distances + i * stride, stride);
                bd += K * DF;
                bn += DF;
                distances += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::UnpackDataPtr GetUnpackData(size_t depth, bool transpose)
        {
            switch (depth)
            {
            case 4: return transpose ? UnpackDataB<4> : UnpackDataA<4>;
            case 5: return transpose ? UnpackDataB<5> : UnpackDataA<5>;
            case 6: return transpose ? UnpackDataB<6> : UnpackDataA<6>;
            case 7: return transpose ? UnpackDataB<7> : UnpackDataA<7>;
            case 8: return transpose ? UnpackDataB8 : UnpackDataA8;
            default: return NULL;
            }
        }

        Base::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth)
        {
            //return depth == 8 ? MacroCorrelation16 : MacroCorrelation8;
            return depth == 8 ? NULL : MacroCorrelation8;
        }
    }
#endif
}
