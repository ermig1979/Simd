/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)     
    namespace Avx512bw
    {
        template<bool mask, bool nofma> SIMD_INLINE void SynetAdd8iNchwF(const uint8_t* a, const uint8_t* b, __m512 scale[3], __m512 shift[3], __m128i upper, uint8_t* c, size_t offset, __mmask16 tail = -1)
        {
            __m512 _a = Fmadd<nofma>(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(a + offset, tail)))), scale[0], shift[0]);
            __m512 _b = Fmadd<nofma>(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(b + offset, tail)))), scale[1], shift[1]);
            __m512i c32 = _mm512_cvtps_epi32(Fmadd<nofma>(_mm512_add_ps(_a, _b), scale[2], shift[2]));
            __m512i c8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(c32, K_ZERO), K_ZERO));
            Store<false, mask>(c + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(c8, 0), upper), tail);
        }

        template <bool nofma> void SynetAdd8iNchw(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 tailF = TailMask16(spatial - spatialF);
            __m128i _upper = _mm_set1_epi8(upper);
            __m512 scale[3], shift[3];
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    scale[0] = _mm512_set1_ps(aScale[c]);
                    shift[0] = _mm512_set1_ps(aShift[c]);
                    scale[1] = _mm512_set1_ps(bScale[c]);
                    shift[1] = _mm512_set1_ps(bShift[c]);
                    scale[2] = _mm512_set1_ps(cScale[c]);
                    shift[2] = _mm512_set1_ps(cShift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        SynetAdd8iNchwF<false, nofma>(aData, bData, scale, shift, _upper, cData, s);
                    if (s < spatial)
                        SynetAdd8iNchwF<true, nofma>(aData, bData, scale, shift, _upper, cData, s, tailF);
                    aData += spatial;
                    bData += spatial;
                    cData += spatial;
                }
            }
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void SynetAdd8iNhwcF(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, __m128i upper, uint8_t* c, size_t offset, __mmask16 tail = -1)
        {
            __m512 _a = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(a + offset, tail))));
            __m512 _aScale = Avx512f::Load<align, mask>(aScale + offset, tail);
            __m512 _aShift = Avx512f::Load<align, mask>(aShift + offset, tail);
            _a = Fmadd<nofma>(_a, _aScale, _aShift);
            __m512 _b = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(b + offset, tail))));
            __m512 _bScale = Avx512f::Load<align, mask>(bScale + offset, tail);
            __m512 _bShift = Avx512f::Load<align, mask>(bShift + offset, tail);
            _b = Fmadd<nofma>(_b, _bScale, _bShift);
            __m512 _cScale = Avx512f::Load<align, mask>(cScale + offset, tail);
            __m512 _cShift = Avx512f::Load<align, mask>(cShift + offset, tail);
            __m512i c32 = _mm512_cvtps_epi32(Fmadd<nofma>(_mm512_add_ps(_a, _b), _cScale, _cShift));
            __m512i c8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(c32, K_ZERO), K_ZERO));
            Store<false, mask>(c + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(c8, 0), upper), tail);
        }

        template <bool align, bool nofma> void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            if (align)
                assert(Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift));

            size_t channelsF = AlignLo(channels, F);
            __mmask16 tailF = TailMask16(channels - channelsF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        SynetAdd8iNhwcF<align, false, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c);
                    if (c < channels)
                        SynetAdd8iNhwcF<align, true, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c, tailF);
                    aData += channels;
                    bData += channels;
                    cData += channels;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            if (Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift))
                SynetAdd8iNhwc<true, nofma>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                SynetAdd8iNhwc<false, nofma>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
        }

        void SynetAdd8i(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            bool nofma = Base::FmaAvoid(compatibility);
            if (format == SimdTensorFormatNchw && spatial > HF)
            {
                if (nofma)
                    SynetAdd8iNchw<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
                else
                    SynetAdd8iNchw<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            }
            else if (format == SimdTensorFormatNhwc && channels > HF)
            {
                if (nofma)
                    SynetAdd8iNhwc<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
                else
                    SynetAdd8iNhwc<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            }
            else
                Avx2::SynetAdd8i(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, format, compatibility);
        }

        //---------------------------------------------------------------------

        static SIMD_INLINE void Save4Sums(const __m512i& sum0, const __m512i sum1, const __m512i& sum2, const __m512i& sum3, int32_t* dst)
        {
            __m512i sum02 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum0, sum2), _mm512_unpackhi_epi32(sum0, sum2));
            __m512i sum13 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum1, sum3), _mm512_unpackhi_epi32(sum1, sum3));
            __m512i sum512 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum02, sum13), _mm512_unpackhi_epi32(sum02, sum13));
            _mm_storeu_si128((__m128i*)dst, _mm_add_epi32(_mm_add_epi32(_mm512_extracti32x4_epi32(sum512, 0), _mm512_extracti32x4_epi32(sum512, 1)),
                _mm_add_epi32(_mm512_extracti32x4_epi32(sum512, 2), _mm512_extracti32x4_epi32(sum512, 3))));
        }

        template<bool overflow> static void SynetInnerProduct8i1x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i s0, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
            }
            D[0] = ExtractSum<uint32_t>(d00);
        }

        template<bool overflow> static void SynetInnerProduct8i1x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d01 = _mm512_setzero_si512();
            __m512i d02 = _mm512_setzero_si512();
            __m512i d03 = _mm512_setzero_si512();
            __m512i s0, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W1 + k));
                Madd4<overflow>(d01, s0, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W2 + k));
                Madd4<overflow>(d02, s0, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W3 + k));
                Madd4<overflow>(d03, s0, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                w0 = Load<false, true>(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                w0 = Load<false, true>(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                w0 = Load<false, true>(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
            }
            Save4Sums(d00, d01, d02, d03, D);
        }

        template<bool overflow> static void SynetInnerProduct8i2x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d10 = _mm512_setzero_si512();
            __m512i s0, s1, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                s1 = _mm512_loadu_si512((__m512i*)(S1 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                s1 = Load<false, true>(S1 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            D[0 * ldd] = ExtractSum<uint32_t>(d00);
            D[1 * ldd] = ExtractSum<uint32_t>(d10);
        }

        template<bool overflow> static void SynetInnerProduct8i2x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d01 = _mm512_setzero_si512();
            __m512i d02 = _mm512_setzero_si512();
            __m512i d03 = _mm512_setzero_si512();
            __m512i d10 = _mm512_setzero_si512();
            __m512i d11 = _mm512_setzero_si512();
            __m512i d12 = _mm512_setzero_si512();
            __m512i d13 = _mm512_setzero_si512();
            __m512i s0, s1, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                s1 = _mm512_loadu_si512((__m512i*)(S1 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W1 + k));
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W2 + k));
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W3 + k));
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                s1 = Load<false, true>(S1 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = Load<false, true>(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = Load<false, true>(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = Load<false, true>(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
            }
            Save4Sums(d00, d01, d02, d03, D + 0 * ldd);
            Save4Sums(d10, d11, d12, d13, D + 1 * ldd);
        }

        template<bool overflow> static void SynetInnerProduct8i4x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const uint8_t* S2 = S + 2 * lds;
            const uint8_t* S3 = S + 3 * lds;
            const int8_t* W0 = W + 0 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d10 = _mm512_setzero_si512();
            __m512i d20 = _mm512_setzero_si512();
            __m512i d30 = _mm512_setzero_si512();
            __m512i s0, s1, s2, s3, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                s1 = _mm512_loadu_si512((__m512i*)(S1 + k));
                s2 = _mm512_loadu_si512((__m512i*)(S2 + k));
                s3 = _mm512_loadu_si512((__m512i*)(S3 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                Madd4<overflow>(d20, s2, w0);
                Madd4<overflow>(d30, s3, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                s1 = Load<false, true>(S1 + KA, tail);
                s2 = Load<false, true>(S2 + KA, tail);
                s3 = Load<false, true>(S3 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                Madd4<overflow>(d20, s2, w0);
                Madd4<overflow>(d30, s3, w0);
            }
            D[0 * ldd] = ExtractSum<uint32_t>(d00);
            D[1 * ldd] = ExtractSum<uint32_t>(d10);
            D[2 * ldd] = ExtractSum<uint32_t>(d20);
            D[3 * ldd] = ExtractSum<uint32_t>(d30);
        }

        template<bool overflow> static void SynetInnerProduct8i4x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const uint8_t* S2 = S + 2 * lds;
            const uint8_t* S3 = S + 3 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            __m512i d00 = _mm512_setzero_si512();
            __m512i d01 = _mm512_setzero_si512();
            __m512i d02 = _mm512_setzero_si512();
            __m512i d03 = _mm512_setzero_si512();
            __m512i d10 = _mm512_setzero_si512();
            __m512i d11 = _mm512_setzero_si512();
            __m512i d12 = _mm512_setzero_si512();
            __m512i d13 = _mm512_setzero_si512();
            __m512i d20 = _mm512_setzero_si512();
            __m512i d21 = _mm512_setzero_si512();
            __m512i d22 = _mm512_setzero_si512();
            __m512i d23 = _mm512_setzero_si512();
            __m512i d30 = _mm512_setzero_si512();
            __m512i d31 = _mm512_setzero_si512();
            __m512i d32 = _mm512_setzero_si512();
            __m512i d33 = _mm512_setzero_si512();
            __m512i s0, s1, s2, s3, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm512_loadu_si512((__m512i*)(S0 + k));
                s1 = _mm512_loadu_si512((__m512i*)(S1 + k));
                s2 = _mm512_loadu_si512((__m512i*)(S2 + k));
                s3 = _mm512_loadu_si512((__m512i*)(S3 + k));
                w0 = _mm512_loadu_si512((__m512i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                Madd4<overflow>(d20, s2, w0);
                Madd4<overflow>(d30, s3, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W1 + k));
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                Madd4<overflow>(d21, s2, w0);
                Madd4<overflow>(d31, s3, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W2 + k));
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                Madd4<overflow>(d22, s2, w0);
                Madd4<overflow>(d32, s3, w0);
                w0 = _mm512_loadu_si512((__m512i*)(W3 + k));
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
                Madd4<overflow>(d23, s2, w0);
                Madd4<overflow>(d33, s3, w0);
            }
            if (KA < K)
            {
                __mmask64 tail = TailMask64(K - KA);
                s0 = Load<false, true>(S0 + KA, tail);
                s1 = Load<false, true>(S1 + KA, tail);
                s2 = Load<false, true>(S2 + KA, tail);
                s3 = Load<false, true>(S3 + KA, tail);
                w0 = Load<false, true>(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                Madd4<overflow>(d20, s2, w0);
                Madd4<overflow>(d30, s3, w0);
                w0 = Load<false, true>(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                Madd4<overflow>(d21, s2, w0);
                Madd4<overflow>(d31, s3, w0);
                w0 = Load<false, true>(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                Madd4<overflow>(d22, s2, w0);
                Madd4<overflow>(d32, s3, w0);
                w0 = Load<false, true>(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
                Madd4<overflow>(d23, s2, w0);
                Madd4<overflow>(d33, s3, w0);
            }
            Save4Sums(d00, d01, d02, d03, D + 0 * ldd);
            Save4Sums(d10, d11, d12, d13, D + 1 * ldd);
            Save4Sums(d20, d21, d22, d23, D + 2 * ldd);
            Save4Sums(d30, d31, d32, d33, D + 3 * ldd);
        }

        template<bool overflow> void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst)
        {
            size_t M2 = AlignLoAny(M, 2);
            size_t M4 = AlignLoAny(M, 4);
            size_t N4 = AlignLoAny(N, 4);
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i4x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i4x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K * 4;
                dst += N * 4;
            }
            for (; i < M2; i += 2)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i2x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i2x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K * 2;
                dst += N * 2;
            }
            for (; i < M; i += 1)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i1x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i1x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K;
                dst += N;
            }
        }

        void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::Precise(compatibility))
                SynetInnerProduct8i<false>(M, N, K, src, weight, dst);
            else
                SynetInnerProduct8i<true>(M, N, K, src, weight, dst);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
