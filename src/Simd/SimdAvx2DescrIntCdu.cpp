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
#include "Simd/SimdSynet.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template<int bits> __m128i UnpackData16(const uint8_t* src);

        template<> SIMD_INLINE __m128i UnpackData16<4>(const uint8_t* src)
        {
            __m256i s4 = _mm256_broadcastsi128_si256(Sse41::LoadLast16<4>(src));
            __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s4, C4_SHFL), C4_MULLO), 12);
            return _mm256_castsi256_si128(PackI16ToU8(s16, K_ZERO));
        }

        template<> SIMD_INLINE __m128i UnpackData16<5>(const uint8_t* src)
        {
            __m256i s5 = _mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(src));
            __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s5, C5_SHFL), C5_MULLO), 11);
            return _mm256_castsi256_si128(PackI16ToU8(s16, K_ZERO));
        }

        template<> SIMD_INLINE __m128i UnpackData16<6>(const uint8_t* src)
        {
            __m256i s6 = _mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(src));
            __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, C6_SHFL), C6_MULLO), 10);
            return _mm256_castsi256_si128(PackI16ToU8(s16, K_ZERO));
        }

        template<> SIMD_INLINE __m128i UnpackData16<7>(const uint8_t* src)
        {
            __m256i s7 = _mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(src));
            __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s7, C7_SHFL), C7_MULLO), 9);
            return _mm256_castsi256_si128(PackI16ToU8(s16, K_ZERO));
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> __m256i UnpackData32(const uint8_t* src);

        template<> SIMD_INLINE __m256i UnpackData32<4>(const uint8_t* src)
        {
            __m256i lo = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(src + 0))), C4_SHFL), C4_MULLO), 12);
            __m256i hi = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(src + 8))), C4_SHFL), C4_MULLO), 12);
            return PackI16ToU8(lo, hi);
        }

        template<> SIMD_INLINE __m256i UnpackData32<5>(const uint8_t* src)
        {
            __m256i lo = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(src + 0))), C5_SHFL), C5_MULLO), 11);
            __m256i hi = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(src + 10))), C5_SHFL), C5_MULLO), 11);
            return PackI16ToU8(lo, hi);
        }

        template<> SIMD_INLINE __m256i UnpackData32<6>(const uint8_t* src)
        {
            __m256i lo = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(src + 0))), C6_SHFL), C6_MULLO), 10);
            __m256i hi = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(src + 12))), C6_SHFL), C6_MULLO), 10);
            return PackI16ToU8(lo, hi);
        }

        template<> SIMD_INLINE __m256i UnpackData32<7>(const uint8_t* src)
        {
            __m256i lo = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(src + 0))), C7_SHFL), C7_MULLO), 9);
            __m256i hi = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(src + 14))), C7_SHFL), C7_MULLO), 9);
            return PackI16ToU8(lo, hi);
        }

        //-------------------------------------------------------------------------------------------------


        template<int bits> void UnpackDataA(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t size16 = AlignLo(size, 16), size32 = AlignLo(size - 1, 32);
            for (size_t i = 0; i < count; i++)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = (uint8_t*)dst + i * size;
                size_t j = 0;
                for (; j < size32; j += 32, ps += 4 * bits, pd += 32)
                    _mm256_storeu_si256((__m256i*)pd, UnpackData32<bits>(ps));
                for (; j < size16; j += 16, ps += 2 * bits, pd += 16)
                    _mm_storeu_si128((__m128i*)pd, UnpackData16<bits>(ps));
                for (; j < size; j += 8, ps += bits, pd += 8)
                    _mm_storel_epi64((__m128i*)pd, Sse41::UnpackData8<bits>(ps));
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> SIMD_INLINE void UnpackDataBx4x32(const uint8_t* const* src, size_t offset, uint8_t* dst)
        {
            __m256i a0 = UnpackData32<bits>(src[0] + offset);
            __m256i a1 = UnpackData32<bits>(src[1] + offset);
            __m256i a2 = UnpackData32<bits>(src[2] + offset);
            __m256i a3 = UnpackData32<bits>(src[3] + offset);
            __m256i b0 = _mm256_unpacklo_epi32(a0, a2);
            __m256i b1 = _mm256_unpacklo_epi32(a1, a3);
            __m256i b2 = _mm256_unpackhi_epi32(a0, a2);
            __m256i b3 = _mm256_unpackhi_epi32(a1, a3);
            Store<false>((__m128i*)dst + 0, (__m128i*)dst + 16, _mm256_unpacklo_epi32(b0, b1));
            Store<false>((__m128i*)dst + 4, (__m128i*)dst + 20, _mm256_unpackhi_epi32(b0, b1));
            Store<false>((__m128i*)dst + 8, (__m128i*)dst + 24, _mm256_unpacklo_epi32(b2, b3));
            Store<false>((__m128i*)dst + 12, (__m128i*)dst + 28, _mm256_unpackhi_epi32(b2, b3));
        }

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
            _mm_storeu_si128((__m128i*)dst + 4, _mm_unpackhi_epi32(b0, b1));
            _mm_storeu_si128((__m128i*)dst + 8, _mm_unpacklo_epi32(b2, b3));
            _mm_storeu_si128((__m128i*)dst + 12, _mm_unpackhi_epi32(b2, b3));
        }

        template<int bits> SIMD_INLINE void UnpackDataBx4x8(const uint8_t* const* src, size_t offset, uint8_t* dst)
        {
            __m128i a0 = Sse41::UnpackData8<bits>(src[0] + offset);
            __m128i a1 = Sse41::UnpackData8<bits>(src[1] + offset);
            __m128i a2 = Sse41::UnpackData8<bits>(src[2] + offset);
            __m128i a3 = Sse41::UnpackData8<bits>(src[3] + offset);
            __m128i b0 = _mm_unpacklo_epi32(a0, a2);
            __m128i b1 = _mm_unpacklo_epi32(a1, a3);
            _mm_storeu_si128((__m128i*)dst + 0, _mm_unpacklo_epi32(b0, b1));
            _mm_storeu_si128((__m128i*)dst + 4, _mm_unpackhi_epi32(b0, b1));
        }

        template<int bits> void UnpackDataB(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t countDF = AlignLo(count, DF), size16 = AlignLo(size, 16), size32 = AlignLo(size - 1, 32), i, j, o;
            for (i = 0; i < countDF; i += DF, src += DF)
            {
                for (j = 0, o = 16; j < size32; j += 32, o += 4 * bits, dst += 16 * A)
                {
                    UnpackDataBx4x32<bits>(src + 0, o, dst + 0 * Sse41::A);
                    UnpackDataBx4x32<bits>(src + 4, o, dst + 1 * Sse41::A);
                    UnpackDataBx4x32<bits>(src + 8, o, dst + 2 * Sse41::A);
                    UnpackDataBx4x32<bits>(src + 12, o, dst + 3 * Sse41::A);
                }
                for (; j < size16; j += 16, o += 2 * bits, dst += 16 * Sse41::A)
                {
                    UnpackDataBx4x16<bits>(src + 0, o, dst + 0 * Sse41::A);
                    UnpackDataBx4x16<bits>(src + 4, o, dst + 1 * Sse41::A);
                    UnpackDataBx4x16<bits>(src + 8, o, dst + 2 * Sse41::A);
                    UnpackDataBx4x16<bits>(src + 12, o, dst + 3 * Sse41::A);
                }
                for (; j < size; j += 8, o += bits, dst += 8 * Sse41::A)
                {
                    UnpackDataBx4x8<bits>(src + 0, o, dst + 0 * Sse41::A);
                    UnpackDataBx4x8<bits>(src + 4, o, dst + 1 * Sse41::A);
                    UnpackDataBx4x8<bits>(src + 8, o, dst + 2 * Sse41::A);
                    UnpackDataBx4x8<bits>(src + 12, o, dst + 3 * Sse41::A);
                }
            }
            if (i < count)
            {
                const uint8_t* _src[DF];
                for (size_t j = 0; j < DF; i++, j++)
                    _src[j] = i < count ? *src++ : src[-1];
                for (j = 0, o = 16; j < size32; j += 32, o += 4 * bits, dst += 16 * A)
                {
                    UnpackDataBx4x32<bits>(_src + 0, o, dst + 0 * Sse41::A);
                    UnpackDataBx4x32<bits>(_src + 4, o, dst + 1 * Sse41::A);
                    UnpackDataBx4x32<bits>(_src + 8, o, dst + 2 * Sse41::A);
                    UnpackDataBx4x32<bits>(_src + 12, o, dst + 3 * Sse41::A);
                }
                for (; j < size16; j += 16, o += 2 * bits, dst += 16 * Sse41::A)
                {
                    UnpackDataBx4x16<bits>(_src + 0, o, dst + 0 * Sse41::A);
                    UnpackDataBx4x16<bits>(_src + 4, o, dst + 1 * Sse41::A);
                    UnpackDataBx4x16<bits>(_src + 8, o, dst + 2 * Sse41::A);
                    UnpackDataBx4x16<bits>(_src + 12, o, dst + 3 * Sse41::A);
                }
                for (; j < size; j += 8, o += bits, dst += 8 * Sse41::A)
                {
                    UnpackDataBx4x8<bits>(_src + 0, o, dst + 0 * Sse41::A);
                    UnpackDataBx4x8<bits>(_src + 4, o, dst + 1 * Sse41::A);
                    UnpackDataBx4x8<bits>(_src + 8, o, dst + 2 * Sse41::A);
                    UnpackDataBx4x8<bits>(_src + 12, o, dst + 3 * Sse41::A);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int M> void Correlation8_2xM(size_t N, size_t K, const uint8_t* ad0, const uint8_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            __m256i ab00, ab01, ab10, ab11, ab20, ab21, ab30, ab31, ab40, ab41, a0, b0, b1;
            const uint8_t* ad1 = ad0 + 1 * K;
            const uint8_t* ad2 = ad0 + 2 * K;
            const uint8_t* ad3 = ad0 + 3 * K;
            const uint8_t* ad4 = ad0 + 4 * K;
            if (N > F)
            {
                if (M > 0) ab00 = _mm256_setzero_si256(), ab01 = _mm256_setzero_si256();
                if (M > 1) ab10 = _mm256_setzero_si256(), ab11 = _mm256_setzero_si256();
                if (M > 2) ab20 = _mm256_setzero_si256(), ab21 = _mm256_setzero_si256();
                if (M > 3) ab30 = _mm256_setzero_si256(), ab31 = _mm256_setzero_si256();
                if (M > 4) ab40 = _mm256_setzero_si256(), ab41 = _mm256_setzero_si256();
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = _mm256_loadu_si256((__m256i*)bd + 0);
                    b1 = _mm256_loadu_si256((__m256i*)bd + 1);
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
                if (M > 0) ab00 = _mm256_setzero_si256();
                if (M > 1) ab10 = _mm256_setzero_si256();
                if (M > 2) ab20 = _mm256_setzero_si256();
                if (M > 3) ab30 = _mm256_setzero_si256();
                if (M > 4) ab40 = _mm256_setzero_si256();
                for (size_t k = 0; k < K; k += 4)
                {
                    b0 = _mm256_loadu_si256((__m256i*)bd + 0);
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
            default: return NULL;
            }
        }

        Base::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth)
        {
            return depth == 8 ? NULL : MacroCorrelation8;
        }
    }
#endif
}
