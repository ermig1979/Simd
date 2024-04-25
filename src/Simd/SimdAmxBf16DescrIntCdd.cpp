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
#include "Simd/SimdExtract.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m512i _ab = _mm512_setzero_si512();
            size_t i = 0, size64 = AlignLo(size, 64);
            __mmask64 mask = TailMask64(8 * bits);
            for (; i < size64; i += 64, a += 8 * bits, b += 8 * bits)
            {
                __m512i _a = Load8u<bits>(a, mask);
                __m512i _b = Load8u<bits>(b, mask);
                _ab = _mm512_dpbusd_epi32(_ab, _a, _b);
            }
            if (i < size)
            {
                mask = TailMask64((size - i) / 8 * bits);
                __m512i _a = Load8u<bits>(a, mask);
                __m512i _b = Load8u<bits>(b, mask);
                _ab = _mm512_dpbusd_epi32(_ab, _a, _b);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        template<int bits> void CosineDistance(const uint8_t* a, const uint8_t* b, size_t size, float* distance)
        {
            float abSum = (float)Correlation<bits>(a + 16, b + 16, size);
            Base::DecodeCosineDistance(a, b, abSum, distance);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void MicroCosineDistancesDirect4x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size64 = AlignLo(size, 64), o = 16;
            __m512i a0, a1, a2, a3, b0;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            __m512i ab10 = _mm512_setzero_si512();
            __m512i ab11 = _mm512_setzero_si512();
            __m512i ab12 = _mm512_setzero_si512();
            __m512i ab13 = _mm512_setzero_si512();
            __m512i ab20 = _mm512_setzero_si512();
            __m512i ab21 = _mm512_setzero_si512();
            __m512i ab22 = _mm512_setzero_si512();
            __m512i ab23 = _mm512_setzero_si512();
            __m512i ab30 = _mm512_setzero_si512();
            __m512i ab31 = _mm512_setzero_si512();
            __m512i ab32 = _mm512_setzero_si512();
            __m512i ab33 = _mm512_setzero_si512();
            __mmask64 mask = TailMask64(8 * bits);
            for (; i < size64; i += 64, o += 8 * bits)
            {
                a0 = Load8u<bits>(A[0] + o, mask);
                a1 = Load8u<bits>(A[1] + o, mask);
                a2 = Load8u<bits>(A[2] + o, mask);
                a3 = Load8u<bits>(A[3] + o, mask);

                b0 = Load8u<bits>(B[0] + o, mask);
                ab00 = _mm512_dpbusd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpbusd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpbusd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpbusd_epi32(ab30, a3, b0);

                b0 = Load8u<bits>(B[1] + o, mask);
                ab01 = _mm512_dpbusd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpbusd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpbusd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpbusd_epi32(ab31, a3, b0);

                b0 = Load8u<bits>(B[2] + o, mask);
                ab02 = _mm512_dpbusd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpbusd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpbusd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpbusd_epi32(ab32, a3, b0);

                b0 = Load8u<bits>(B[3] + o, mask);
                ab03 = _mm512_dpbusd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpbusd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpbusd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpbusd_epi32(ab33, a3, b0);
            }
            if (i < size)
            {
                mask = TailMask64((size - i) / 8 * bits);
                a0 = Load8u<bits>(A[0] + o, mask);
                a1 = Load8u<bits>(A[1] + o, mask);
                a2 = Load8u<bits>(A[2] + o, mask);
                a3 = Load8u<bits>(A[3] + o, mask);

                b0 = Load8u<bits>(B[0] + o, mask);
                ab00 = _mm512_dpbusd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpbusd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpbusd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpbusd_epi32(ab30, a3, b0);

                b0 = Load8u<bits>(B[1] + o, mask);
                ab01 = _mm512_dpbusd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpbusd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpbusd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpbusd_epi32(ab31, a3, b0);

                b0 = Load8u<bits>(B[2] + o, mask);
                ab02 = _mm512_dpbusd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpbusd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpbusd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpbusd_epi32(ab32, a3, b0);

                b0 = Load8u<bits>(B[3] + o, mask);
                ab03 = _mm512_dpbusd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpbusd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpbusd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpbusd_epi32(ab33, a3, b0);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 ab2 = _mm_cvtepi32_ps(Extract4Sums(ab20, ab21, ab22, ab23));
            __m128 ab3 = _mm_cvtepi32_ps(Extract4Sums(ab30, ab31, ab32, ab33));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            Sse41::DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
            Sse41::DecodeCosineDistances1x4(A[2], B, ab2, distances + 2 * stride);
            Sse41::DecodeCosineDistances1x4(A[3], B, ab3, distances + 3 * stride);
        }

        template<int bits> void MicroCosineDistancesDirect1x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size64 = AlignLo(size, 64), o = 16;
            __m512i a0, b0;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            __mmask64 mask = TailMask64(8 * bits);
            for (; i < size64; i += 64, o += 8 * bits)
            {
                a0 = Load8u<bits>(A[0] + o, mask);

                b0 = Load8u<bits>(B[0] + o, mask);
                ab00 = _mm512_dpbusd_epi32(ab00, a0, b0);

                b0 = Load8u<bits>(B[1] + o, mask);
                ab01 = _mm512_dpbusd_epi32(ab01, a0, b0);

                b0 = Load8u<bits>(B[2] + o, mask);
                ab02 = _mm512_dpbusd_epi32(ab02, a0, b0);

                b0 = Load8u<bits>(B[3] + o, mask);
                ab03 = _mm512_dpbusd_epi32(ab03, a0, b0);
            }
            if (i < size)
            {
                mask = TailMask64((size - i) / 8 * bits);
                a0 = Load8u<bits>(A[0] + o, mask);

                b0 = Load8u<bits>(B[0] + o, mask);
                ab00 = _mm512_dpbusd_epi32(ab00, a0, b0);

                b0 = Load8u<bits>(B[1] + o, mask);
                ab01 = _mm512_dpbusd_epi32(ab01, a0, b0);

                b0 = Load8u<bits>(B[2] + o, mask);
                ab02 = _mm512_dpbusd_epi32(ab02, a0, b0);

                b0 = Load8u<bits>(B[3] + o, mask);
                ab03 = _mm512_dpbusd_epi32(ab03, a0, b0);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M4 = AlignLoAny(M, 4);
            size_t N4 = AlignLoAny(N, 4);
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistancesDirect4x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                {
                    CosineDistance<bits>(A[i + 0], B[j], size, distances + j + 0 * stride);
                    CosineDistance<bits>(A[i + 1], B[j], size, distances + j + 1 * stride);
                    CosineDistance<bits>(A[i + 2], B[j], size, distances + j + 2 * stride);
                    CosineDistance<bits>(A[i + 3], B[j], size, distances + j + 3 * stride);
                }
                distances += 4 * stride;
            }
            for (; i < M; i++)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistancesDirect1x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                    CosineDistance<bits>(A[i], B[j], size, distances + j);
                distances += 1 * stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::CosineDistancePtr GetCosineDistance(size_t depth)
        {
            switch (depth)
            {
            case 5: return CosineDistance<5>;
            case 6: return CosineDistance<6>;
            case 7: return CosineDistance<7>;
            default: assert(0); return NULL;
            }
        }

        Base::DescrInt::MacroCosineDistancesDirectPtr GetMacroCosineDistancesDirect(size_t depth)
        {
            switch (depth)
            {
            case 5: return MacroCosineDistancesDirect<5>;
            case 6: return MacroCosineDistancesDirect<6>;
            case 7: return MacroCosineDistancesDirect<7>;
            default: assert(0); return NULL;
            }
        }
    }
#endif
}
