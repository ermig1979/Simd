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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i U4_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1,-1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1);
        const __m512i U4_SHFL0 = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3,
            0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3,
            0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3,
            0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3);
        const __m512i U4_SHFL1 = SIMD_MM512_SETR_EPI8(
            0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7,
            0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7,
            0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7,
            0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7);

        const __m512i U5_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x2, -1, 0x2, 0x3, 0x4, -1, 0x5, 0x6, 0x7, -1, 0x7, 0x8, 0x9, -1);
        const __m512i U5_SHFL0 = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6);
        const __m512i U5_SHFL1 = SIMD_MM512_SETR_EPI8(
            0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9,
            0x7, 0x7, 0x7, 0x8, 0x8, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xA, 0xA, 0xB, 0xB, 0xB,
            0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9,
            0x7, 0x7, 0x7, 0x8, 0x8, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xA, 0xA, 0xB, 0xB, 0xB);

        const __m512i U6_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);
        const __m512i U6_SHFL0 = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5);
        const __m512i U6_SHFL1 = SIMD_MM512_SETR_EPI8(
            0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB,
            0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB,
            0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB,
            0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);

        const __m512i U7_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x2, 0x3, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xA, 0xB, 0xC, 0xD);
        const __m512i U7_SHFL0 = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8,
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8);
        const __m512i U7_SHFL1 = SIMD_MM512_SETR_EPI8(
            0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD,
            0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xE, 0xE, 0xF, 0xF, 0xF,
            0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD,
            0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xE, 0xE, 0xF, 0xF, 0xF);

        //-------------------------------------------------------------------------------------------------

        template<int bits> __m512i UnpackData64(const uint8_t* src, __mmask64 mask);

        template<> SIMD_INLINE __m512i UnpackData64<4>(const uint8_t* src, __mmask64 mask)
        {
            __m512i val = _mm512_permutexvar_epi32(U4_PERM, _mm512_maskz_loadu_epi8(mask, src));
            __m512i lo = _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(val, U4_SHFL0), C4_MULLO), 12);
            __m512i hi = _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(val, U4_SHFL1), C4_MULLO), 12);
            return _mm512_packus_epi16(lo, hi);
        }

        template<> SIMD_INLINE __m512i UnpackData64<5>(const uint8_t* src, __mmask64 mask)
        {
            __m512i val = _mm512_permutexvar_epi32(U5_PERM, _mm512_maskz_loadu_epi8(mask, src));
            __m512i lo = _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(val, U5_SHFL0), C5_MULLO), 11);
            __m512i hi = _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(val, U5_SHFL1), C5_MULLO), 11);
            return _mm512_packus_epi16(lo, hi);
        }

        template<> SIMD_INLINE __m512i UnpackData64<6>(const uint8_t* src, __mmask64 mask)
        {
            __m512i val = _mm512_permutexvar_epi32(U6_PERM, _mm512_maskz_loadu_epi8(mask, src));
            __m512i lo = _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(val, U6_SHFL0), C6_MULLO), 10);
            __m512i hi = _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(val, U6_SHFL1), C6_MULLO), 10);
            return _mm512_packus_epi16(lo, hi);
        }

        template<> SIMD_INLINE __m512i UnpackData64<7>(const uint8_t* src, __mmask64 mask)
        {
            __m512i val = _mm512_permutexvar_epi32(U7_PERM, _mm512_maskz_loadu_epi8(mask, src));
           __m512i lo = _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(val, U7_SHFL0), C7_MULLO), 9);
           __m512i hi = _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(val, U7_SHFL1), C7_MULLO), 9);
            return _mm512_packus_epi16(lo, hi);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void UnpackDataA(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t size64 = AlignLo(size, 64);
            __mmask64 srcBody = TailMask64(8 * bits), dstBody = __mmask64(-1), srcTail = TailMask64((size - size64) / 8 * bits), dstTail = TailMask64(size - size64);
            for (size_t i = 0; i < count; i++)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = (uint8_t*)dst + i * size;
                size_t j = 0;
                for (; j < size64; j += 64, ps += 8 * bits, pd += 64)
                    _mm512_mask_storeu_epi8(pd, dstBody, UnpackData64<bits>(ps, srcBody));
                if(j < size)
                    _mm512_mask_storeu_epi8(pd, dstTail, UnpackData64<bits>(ps, srcTail));
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits, int N> SIMD_INLINE void UnpackDataBx16xN(const uint8_t* const* src, size_t offset, uint8_t* dst)
        {
            __mmask64 mask = TailMask64(bits * N);
            __m512i a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aA, aB, aC, aD, aE, aF, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, bA, bB, bC, bD, bE, bF;

            a0 = UnpackData64<bits>(src[0x0] + offset, mask);
            a1 = UnpackData64<bits>(src[0x1] + offset, mask);
            a2 = UnpackData64<bits>(src[0x2] + offset, mask);
            a3 = UnpackData64<bits>(src[0x3] + offset, mask);
            a4 = UnpackData64<bits>(src[0x4] + offset, mask);
            a5 = UnpackData64<bits>(src[0x5] + offset, mask);
            a6 = UnpackData64<bits>(src[0x6] + offset, mask);
            a7 = UnpackData64<bits>(src[0x7] + offset, mask);
            a8 = UnpackData64<bits>(src[0x8] + offset, mask);
            a9 = UnpackData64<bits>(src[0x9] + offset, mask);
            aA = UnpackData64<bits>(src[0xA] + offset, mask);
            aB = UnpackData64<bits>(src[0xB] + offset, mask);
            aC = UnpackData64<bits>(src[0xC] + offset, mask);
            aD = UnpackData64<bits>(src[0xD] + offset, mask);
            aE = UnpackData64<bits>(src[0xE] + offset, mask);
            aF = UnpackData64<bits>(src[0xF] + offset, mask);

            b0 = _mm512_unpacklo_epi32(a0, a2);
            b1 = _mm512_unpacklo_epi32(a1, a3);
            b2 = _mm512_unpackhi_epi32(a0, a2);
            b3 = _mm512_unpackhi_epi32(a1, a3);
            b4 = _mm512_unpacklo_epi32(a4, a6);
            b5 = _mm512_unpacklo_epi32(a5, a7);
            b6 = _mm512_unpackhi_epi32(a4, a6);
            b7 = _mm512_unpackhi_epi32(a5, a7);
            b8 = _mm512_unpacklo_epi32(a8, aA);
            b9 = _mm512_unpacklo_epi32(a9, aB);
            bA = _mm512_unpackhi_epi32(a8, aA);
            bB = _mm512_unpackhi_epi32(a9, aB);
            bC = _mm512_unpacklo_epi32(aC, aE);
            bD = _mm512_unpacklo_epi32(aD, aF);
            bE = _mm512_unpackhi_epi32(aC, aE);
            bF = _mm512_unpackhi_epi32(aD, aF);

            a0 = _mm512_unpacklo_epi32(b0, b1);
            a1 = _mm512_unpackhi_epi32(b0, b1);
            a2 = _mm512_unpacklo_epi32(b2, b3);
            a3 = _mm512_unpackhi_epi32(b2, b3);
            a4 = _mm512_unpacklo_epi32(b4, b5);
            a5 = _mm512_unpackhi_epi32(b4, b5);
            a6 = _mm512_unpacklo_epi32(b6, b7);
            a7 = _mm512_unpackhi_epi32(b6, b7);
            a8 = _mm512_unpacklo_epi32(b8, b9);
            a9 = _mm512_unpackhi_epi32(b8, b9);
            aA = _mm512_unpacklo_epi32(bA, bB);
            aB = _mm512_unpackhi_epi32(bA, bB);
            aC = _mm512_unpacklo_epi32(bC, bD);
            aD = _mm512_unpackhi_epi32(bC, bD);
            aE = _mm512_unpacklo_epi32(bE, bF);
            aF = _mm512_unpackhi_epi32(bE, bF);

            b0 = _mm512_shuffle_i32x4(a0, a4, 0x44);
            b1 = _mm512_shuffle_i32x4(a1, a5, 0x44);
            b2 = _mm512_shuffle_i32x4(a2, a6, 0x44);
            b3 = _mm512_shuffle_i32x4(a3, a7, 0x44);
            b4 = _mm512_shuffle_i32x4(a0, a4, 0xEE);
            b5 = _mm512_shuffle_i32x4(a1, a5, 0xEE);
            b6 = _mm512_shuffle_i32x4(a2, a6, 0xEE);
            b7 = _mm512_shuffle_i32x4(a3, a7, 0xEE);
            b8 = _mm512_shuffle_i32x4(a8, aC, 0x44);
            b9 = _mm512_shuffle_i32x4(a9, aD, 0x44);
            bA = _mm512_shuffle_i32x4(aA, aE, 0x44);
            bB = _mm512_shuffle_i32x4(aB, aF, 0x44);
            bC = _mm512_shuffle_i32x4(a8, aC, 0xEE);
            bD = _mm512_shuffle_i32x4(a9, aD, 0xEE);
            bE = _mm512_shuffle_i32x4(aA, aE, 0xEE);
            bF = _mm512_shuffle_i32x4(aB, aF, 0xEE);

            a0 = _mm512_shuffle_i32x4(b0, b8, 0x88);
            a1 = _mm512_shuffle_i32x4(b1, b9, 0x88);
            a2 = _mm512_shuffle_i32x4(b2, bA, 0x88);
            a3 = _mm512_shuffle_i32x4(b3, bB, 0x88);
            a4 = _mm512_shuffle_i32x4(b0, b8, 0xDD);
            a5 = _mm512_shuffle_i32x4(b1, b9, 0xDD);
            a6 = _mm512_shuffle_i32x4(b2, bA, 0xDD);
            a7 = _mm512_shuffle_i32x4(b3, bB, 0xDD);
            a8 = _mm512_shuffle_i32x4(b4, bC, 0x88);
            a9 = _mm512_shuffle_i32x4(b5, bD, 0x88);
            aA = _mm512_shuffle_i32x4(b6, bE, 0x88);
            aB = _mm512_shuffle_i32x4(b7, bF, 0x88);
            aC = _mm512_shuffle_i32x4(b4, bC, 0xDD);
            aD = _mm512_shuffle_i32x4(b5, bD, 0xDD);
            aE = _mm512_shuffle_i32x4(b6, bE, 0xDD);
            aF = _mm512_shuffle_i32x4(b7, bF, 0xDD);

            if (N > 0) _mm512_storeu_si512(dst + 0x0 * DA, a0);
            if (N > 0) _mm512_storeu_si512(dst + 0x1 * DA, a1);
            if (N > 1) _mm512_storeu_si512(dst + 0x2 * DA, a2);
            if (N > 1) _mm512_storeu_si512(dst + 0x3 * DA, a3);
            if (N > 2) _mm512_storeu_si512(dst + 0x4 * DA, a4);
            if (N > 2) _mm512_storeu_si512(dst + 0x5 * DA, a5);
            if (N > 3) _mm512_storeu_si512(dst + 0x6 * DA, a6);
            if (N > 3) _mm512_storeu_si512(dst + 0x7 * DA, a7);
            if (N > 4) _mm512_storeu_si512(dst + 0x8 * DA, a8);
            if (N > 4) _mm512_storeu_si512(dst + 0x9 * DA, a9);
            if (N > 5) _mm512_storeu_si512(dst + 0xA * DA, aA);
            if (N > 5) _mm512_storeu_si512(dst + 0xB * DA, aB);
            if (N > 6) _mm512_storeu_si512(dst + 0xC * DA, aC);
            if (N > 6) _mm512_storeu_si512(dst + 0xD * DA, aD);
            if (N > 7) _mm512_storeu_si512(dst + 0xE * DA, aE);
            if (N > 7) _mm512_storeu_si512(dst + 0xF * DA, aF);
        }

        typedef void (*UnpackDataBx16xN_Ptr)(const uint8_t* const* src, size_t offset, uint8_t* dst);

        template<int bits> UnpackDataBx16xN_Ptr GetUnpackDataBx16xN(size_t tail)
        {
            switch (tail / 8)
            {
            case 0: return NULL;
            case 1: return UnpackDataBx16xN<bits, 1>;
            case 2: return UnpackDataBx16xN<bits, 2>;
            case 3: return UnpackDataBx16xN<bits, 3>;
            case 4: return UnpackDataBx16xN<bits, 4>;
            case 5: return UnpackDataBx16xN<bits, 5>;
            case 6: return UnpackDataBx16xN<bits, 6>;
            case 7: return UnpackDataBx16xN<bits, 7>;
            case 8: return UnpackDataBx16xN<bits, 8>;
            default:
                assert(0);  return NULL;
            }
        }

        template<int bits> void UnpackDataB(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t countDF = AlignLo(count, DF), size64 = AlignLo(size, 64), tail = size - size64, i, j, o;
            UnpackDataBx16xN_Ptr unpackDataMain = GetUnpackDataBx16xN<bits>(64);
            UnpackDataBx16xN_Ptr unpackDataTail = GetUnpackDataBx16xN<bits>(tail);
            for (i = 0; i < countDF; i += DF, src += DF)
            {
                for (j = 0, o = 16; j < size64; j += 64, o += 8 * bits, dst += 32 * A)
                {
                    unpackDataMain(src + 0, o, dst + 0);
                    unpackDataMain(src + F, o, dst + A);
                }
                if (j < size)
                {
                    unpackDataTail(src + 0, o, dst + 0);
                    unpackDataTail(src + F, o, dst + A);
                    dst += tail * DF;
                }
            }
            if (i < count)
            {
                size_t tail = count - countDF;
                const uint8_t* _src[DF];
                for (size_t j = 0; j < DF; i++, j++)
                    _src[j] = i < count ? *src++ : src[-1];
                for (j = 0, o = 16; j < size64; j += 64, o += 8 * bits, dst += 32 * A)
                {
                    unpackDataMain(_src + 0, o, dst + 0);
                    if(tail > F)
                        unpackDataMain(_src + F, o, dst + A);
                }
                if (j < size)
                {
                    unpackDataTail(_src + 0, o, dst + 0);
                    if (tail > F)
                        unpackDataTail(_src + F, o, dst + A);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

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
                    if (M > 0x0) a0 = Set4(ad0 + k0), Madd4<true>(ab00, a0, b0), Madd4<true>(ab01, a0, b1);
                    if (M > 0x1) a0 = Set4(ad1 + k0), Madd4<true>(ab10, a0, b0), Madd4<true>(ab11, a0, b1);
                    if (M > 0x2) a0 = Set4(ad2 + k0), Madd4<true>(ab20, a0, b0), Madd4<true>(ab21, a0, b1);
                    if (M > 0x3) a0 = Set4(ad3 + k0), Madd4<true>(ab30, a0, b0), Madd4<true>(ab31, a0, b1);
                    if (M > 0x4) a0 = Set4(ad4 + k0), Madd4<true>(ab40, a0, b0), Madd4<true>(ab41, a0, b1);
                    if (M > 0x5) a0 = Set4(ad5 + k0), Madd4<true>(ab50, a0, b0), Madd4<true>(ab51, a0, b1);
                    if (M > 0x6) a0 = Set4(ad0 + k6), Madd4<true>(ab60, a0, b0), Madd4<true>(ab61, a0, b1);
                    if (M > 0x7) a0 = Set4(ad1 + k6), Madd4<true>(ab70, a0, b0), Madd4<true>(ab71, a0, b1);
                    if (M > 0x8) a0 = Set4(ad2 + k6), Madd4<true>(ab80, a0, b0), Madd4<true>(ab81, a0, b1);
                    if (M > 0x9) a0 = Set4(ad3 + k6), Madd4<true>(ab90, a0, b0), Madd4<true>(ab91, a0, b1);
                    if (M > 0xA) a0 = Set4(ad4 + k6), Madd4<true>(abA0, a0, b0), Madd4<true>(abA1, a0, b1);
                    if (M > 0xB) a0 = Set4(ad5 + k6), Madd4<true>(abB0, a0, b0), Madd4<true>(abB1, a0, b1);
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
                    if (M > 0x0) a0 = Set4(ad0 + k0), Madd4<true>(ab00, a0, b0);
                    if (M > 0x1) a0 = Set4(ad1 + k0), Madd4<true>(ab10, a0, b0);
                    if (M > 0x2) a0 = Set4(ad2 + k0), Madd4<true>(ab20, a0, b0);
                    if (M > 0x3) a0 = Set4(ad3 + k0), Madd4<true>(ab30, a0, b0);
                    if (M > 0x4) a0 = Set4(ad4 + k0), Madd4<true>(ab40, a0, b0);
                    if (M > 0x5) a0 = Set4(ad5 + k0), Madd4<true>(ab50, a0, b0);
                    if (M > 0x6) a0 = Set4(ad0 + k6), Madd4<true>(ab60, a0, b0);
                    if (M > 0x7) a0 = Set4(ad1 + k6), Madd4<true>(ab70, a0, b0);
                    if (M > 0x8) a0 = Set4(ad2 + k6), Madd4<true>(ab80, a0, b0);
                    if (M > 0x9) a0 = Set4(ad3 + k6), Madd4<true>(ab90, a0, b0);
                    if (M > 0xA) a0 = Set4(ad4 + k6), Madd4<true>(abA0, a0, b0);
                    if (M > 0xB) a0 = Set4(ad5 + k6), Madd4<true>(abB0, a0, b0);
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
