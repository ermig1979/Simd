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
#ifdef SIMD_AVX512VNNI_ENABLE    
    namespace Avx512vnni
    {
        template<int M> void Correlation8_2xM(size_t N, size_t K, const uint8_t* ad0, const uint8_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            __m512i ab00, ab01, ab10, ab11, ab20, ab21, ab30, ab31, ab40, ab41, ab50, ab51, ab60, ab61, ab70, ab71, ab80, ab81, ab90, ab91, abA0, abA1, abB0, abB1, a0, b0, b1;
            const uint8_t* ad1 = ad0 + 1 * K;
            const uint8_t* ad2 = ad0 + 2 * K;
            const uint8_t* ad3 = ad0 + 3 * K;
            const uint8_t* ad4 = ad0 + 4 * K;
            const uint8_t* ad5 = ad0 + 5 * K;
            if (N > F)
            {
                if (M > 0x0) ab00 = _mm512_setzero_si512(), ab01 = _mm512_setzero_si512();
                if (M > 0x1) ab10 = _mm512_setzero_si512(), ab11 = _mm512_setzero_si512();
                if (M > 0x2) ab20 = _mm512_setzero_si512(), ab21 = _mm512_setzero_si512();
                if (M > 0x3) ab30 = _mm512_setzero_si512(), ab31 = _mm512_setzero_si512();
                if (M > 0x4) ab40 = _mm512_setzero_si512(), ab41 = _mm512_setzero_si512();
                if (M > 0x5) ab50 = _mm512_setzero_si512(), ab51 = _mm512_setzero_si512();
                if (M > 0x6) ab60 = _mm512_setzero_si512(), ab61 = _mm512_setzero_si512();
                if (M > 0x7) ab70 = _mm512_setzero_si512(), ab71 = _mm512_setzero_si512();
                if (M > 0x8) ab80 = _mm512_setzero_si512(), ab81 = _mm512_setzero_si512();
                if (M > 0x9) ab90 = _mm512_setzero_si512(), ab91 = _mm512_setzero_si512();
                if (M > 0xA) abA0 = _mm512_setzero_si512(), abA1 = _mm512_setzero_si512();
                if (M > 0xB) abB0 = _mm512_setzero_si512(), abB1 = _mm512_setzero_si512();
                for (size_t k0 = 0, k6 = K * 6; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)bd + 0);
                    b1 = _mm512_loadu_si512((__m512i*)bd + 1);
                    if (M > 0x0) a0 = Set4(ad0 + k0), Madd4<false>(ab00, a0, b0), Madd4<false>(ab01, a0, b1);
                    if (M > 0x1) a0 = Set4(ad1 + k0), Madd4<false>(ab10, a0, b0), Madd4<false>(ab11, a0, b1);
                    if (M > 0x2) a0 = Set4(ad2 + k0), Madd4<false>(ab20, a0, b0), Madd4<false>(ab21, a0, b1);
                    if (M > 0x3) a0 = Set4(ad3 + k0), Madd4<false>(ab30, a0, b0), Madd4<false>(ab31, a0, b1);
                    if (M > 0x4) a0 = Set4(ad4 + k0), Madd4<false>(ab40, a0, b0), Madd4<false>(ab41, a0, b1);
                    if (M > 0x5) a0 = Set4(ad5 + k0), Madd4<false>(ab50, a0, b0), Madd4<false>(ab51, a0, b1);
                    if (M > 0x6) a0 = Set4(ad0 + k6), Madd4<false>(ab60, a0, b0), Madd4<false>(ab61, a0, b1);
                    if (M > 0x7) a0 = Set4(ad1 + k6), Madd4<false>(ab70, a0, b0), Madd4<false>(ab71, a0, b1);
                    if (M > 0x8) a0 = Set4(ad2 + k6), Madd4<false>(ab80, a0, b0), Madd4<false>(ab81, a0, b1);
                    if (M > 0x9) a0 = Set4(ad3 + k6), Madd4<false>(ab90, a0, b0), Madd4<false>(ab91, a0, b1);
                    if (M > 0xA) a0 = Set4(ad4 + k6), Madd4<false>(abA0, a0, b0), Madd4<false>(abA1, a0, b1);
                    if (M > 0xB) a0 = Set4(ad5 + k6), Madd4<false>(abB0, a0, b0), Madd4<false>(abB1, a0, b1);
                    bd += DA;
                }
                __mmask16 tail = TailMask16(N - F);
                if (M > 0x0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab01, distances + F, tail), an += 4, distances += stride;
                if (M > 0x1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab11, distances + F, tail), an += 4, distances += stride;
                if (M > 0x2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab21, distances + F, tail), an += 4, distances += stride;
                if (M > 0x3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab31, distances + F, tail), an += 4, distances += stride;
                if (M > 0x4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab41, distances + F, tail), an += 4, distances += stride;
                if (M > 0x5) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab50, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab51, distances + F, tail), an += 4, distances += stride;
                if (M > 0x6) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab60, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab61, distances + F, tail), an += 4, distances += stride;
                if (M > 0x7) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab70, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab71, distances + F, tail), an += 4, distances += stride;
                if (M > 0x8) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab80, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab81, distances + F, tail), an += 4, distances += stride;
                if (M > 0x9) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab90, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, ab91, distances + F, tail), an += 4, distances += stride;
                if (M > 0xA) DecodeCosineDistances1xF(an, bn + 0, bnStride, abA0, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, abA1, distances + F, tail), an += 4, distances += stride;
                if (M > 0xB) DecodeCosineDistances1xF(an, bn + 0, bnStride, abB0, distances + 0), DecodeCosineDistances1xF(an, bn + F, bnStride, abB1, distances + F, tail), an += 4, distances += stride;
            }
            else
            {
                if (M > 0x0) ab00 = _mm512_setzero_si512();
                if (M > 0x1) ab10 = _mm512_setzero_si512();
                if (M > 0x2) ab20 = _mm512_setzero_si512();
                if (M > 0x3) ab30 = _mm512_setzero_si512();
                if (M > 0x4) ab40 = _mm512_setzero_si512();
                if (M > 0x5) ab50 = _mm512_setzero_si512();
                if (M > 0x6) ab60 = _mm512_setzero_si512();
                if (M > 0x7) ab70 = _mm512_setzero_si512();
                if (M > 0x8) ab80 = _mm512_setzero_si512();
                if (M > 0x9) ab90 = _mm512_setzero_si512();
                if (M > 0xA) abA0 = _mm512_setzero_si512();
                if (M > 0xB) abB0 = _mm512_setzero_si512();
                for (size_t k0 = 0, k6 = K * 6; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)bd + 0);
                    if (M > 0x0) a0 = Set4(ad0 + k0), Madd4<false>(ab00, a0, b0);
                    if (M > 0x1) a0 = Set4(ad1 + k0), Madd4<false>(ab10, a0, b0);
                    if (M > 0x2) a0 = Set4(ad2 + k0), Madd4<false>(ab20, a0, b0);
                    if (M > 0x3) a0 = Set4(ad3 + k0), Madd4<false>(ab30, a0, b0);
                    if (M > 0x4) a0 = Set4(ad4 + k0), Madd4<false>(ab40, a0, b0);
                    if (M > 0x5) a0 = Set4(ad5 + k0), Madd4<false>(ab50, a0, b0);
                    if (M > 0x6) a0 = Set4(ad0 + k6), Madd4<false>(ab60, a0, b0);
                    if (M > 0x7) a0 = Set4(ad1 + k6), Madd4<false>(ab70, a0, b0);
                    if (M > 0x8) a0 = Set4(ad2 + k6), Madd4<false>(ab80, a0, b0);
                    if (M > 0x9) a0 = Set4(ad3 + k6), Madd4<false>(ab90, a0, b0);
                    if (M > 0xA) a0 = Set4(ad4 + k6), Madd4<false>(abA0, a0, b0);
                    if (M > 0xB) a0 = Set4(ad5 + k6), Madd4<false>(abB0, a0, b0);
                    bd += DA;
                }
                __mmask16 tail = TailMask16(N);
                if (M > 0x0) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab00, distances + 0, tail), an += 4, distances += stride;
                if (M > 0x1) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab10, distances + 0, tail), an += 4, distances += stride;
                if (M > 0x2) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab20, distances + 0, tail), an += 4, distances += stride;
                if (M > 0x3) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab30, distances + 0, tail), an += 4, distances += stride;
                if (M > 0x4) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab40, distances + 0, tail), an += 4, distances += stride;
                if (M > 0x5) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab50, distances + 0, tail), an += 4, distances += stride;
                if (M > 0x6) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab60, distances + 0, tail), an += 4, distances += stride;
                if (M > 0x7) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab70, distances + 0, tail), an += 4, distances += stride;
                if (M > 0x8) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab80, distances + 0, tail), an += 4, distances += stride;
                if (M > 0x9) DecodeCosineDistances1xF(an, bn + 0, bnStride, ab90, distances + 0, tail), an += 4, distances += stride;
                if (M > 0xA) DecodeCosineDistances1xF(an, bn + 0, bnStride, abA0, distances + 0, tail), an += 4, distances += stride;
                if (M > 0xB) DecodeCosineDistances1xF(an, bn + 0, bnStride, abB0, distances + 0, tail), an += 4, distances += stride;
            }
        }

        typedef void(*Correlation8_2xM_Ptr)(size_t N, size_t K, const uint8_t* ad0, const uint8_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride);

        SIMD_INLINE Correlation8_2xM_Ptr GetCorrelation8_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return Correlation8_2xM<0x1>;
            case 0x2: return Correlation8_2xM<0x2>;
            case 0x3: return Correlation8_2xM<0x3>;
            case 0x4: return Correlation8_2xM<0x4>;
            case 0x5: return Correlation8_2xM<0x5>;
            case 0x6: return Correlation8_2xM<0x6>;
            case 0x7: return Correlation8_2xM<0x7>;
            case 0x8: return Correlation8_2xM<0x8>;
            case 0x9: return Correlation8_2xM<0x9>;
            case 0xA: return Correlation8_2xM<0xA>;
            case 0xB: return Correlation8_2xM<0xB>;
            case 0xC: return Correlation8_2xM<0xC>;
            }
            assert(0);
            return NULL;
        }

        void MacroCorrelation8(size_t M, size_t N, size_t K, const uint8_t* ad, const float* an, const uint8_t* bd, const float* bn, float* distances, size_t stride)
        {
            size_t M12 = AlignLoAny(M, 12);
            Correlation8_2xM_Ptr correlation_2x12 = GetCorrelation8_2xM(12);
            Correlation8_2xM_Ptr correlation_2xT = GetCorrelation8_2xM(M - M12);
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min<size_t>(DF, N - j);
                size_t i = 0;
                for (; i < M12; i += 12)
                    correlation_2x12(dN, K, ad + i * K, bd, an + i * 4, bn, N, distances + i * stride, stride);
                if (i < M)
                    correlation_2xT(dN, K, ad + i * K, bd, an + i * 4, bn, N, distances + i * stride, stride);
                bd += K * DF;
                bn += DF;
                distances += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth)
        {
            return depth == 8 ? NULL : MacroCorrelation8;
        }
    }
#endif
}
