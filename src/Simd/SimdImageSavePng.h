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
#ifndef __SimdImageSavePng_h__
#define __SimdImageSavePng_h__

#include "Simd/SimdImageSave.h"
#include "Simd/SimdLoad.h"

#define SIMD_PNG_ZLIB_BIT_REV_TABLE

namespace Simd
{
    namespace Base
    {
        extern const uint16_t ZlibLenC[30];
        extern const uint8_t  ZlibLenEb[29];
        extern const uint16_t ZlibDistC[31];
        extern const uint8_t  ZlibDistEb[30];

#if defined(SIMD_PNG_ZLIB_BIT_REV_TABLE)
        const int ZlibBitRevShift = 9;
        const int ZlibBitRevSize = 1 << ZlibBitRevShift;
        extern int ZlibBitRevTable[ZlibBitRevSize];
        SIMD_INLINE int ZlibBitRev(int bits, int count)
        {
            assert(bits < ZlibBitRevSize&& count <= ZlibBitRevShift);
            return ZlibBitRevTable[bits] >> (ZlibBitRevShift - count);
        }
#else
        SIMD_INLINE int ZlibBitRev(int bits, int count)
        {
            int rev = 0;
            for (size_t b = 0; b < count; b++)
            {
                rev = (rev << 1) | (bits & 1);
                bits >>= 1;
            }
            return rev;
        }
#endif

        SIMD_INLINE uint32_t ZlibHash(const uint8_t* data)
        {
            uint32_t hash = data[0] + (data[1] << 8) + (data[2] << 16);
            hash ^= hash << 3;
            hash += hash >> 5;
            hash ^= hash << 4;
            hash += hash >> 17;
            hash ^= hash << 25;
            hash += hash >> 6;
            return hash;
        }

        SIMD_INLINE void ZlibHuffA(int bits, int count, OutputMemoryStream& stream)
        {
            stream.WriteBits(ZlibBitRev(bits, count), count);
        }

        SIMD_INLINE void ZlibHuff1(int bits, OutputMemoryStream& stream)
        {
            ZlibHuffA(0x30 + bits, 8, stream);
        }

        SIMD_INLINE void ZlibHuff2(int bits, OutputMemoryStream& stream)
        {
            ZlibHuffA(0x190 + bits - 144, 9, stream);
        }

        SIMD_INLINE void ZlibHuff3(int bits, OutputMemoryStream& stream)
        {
            ZlibHuffA(0 + bits - 256, 7, stream);
        }

        SIMD_INLINE void ZlibHuff4(int bits, OutputMemoryStream& stream)
        {
            ZlibHuffA(0xc0 + bits - 280, 8, stream);
        }

        SIMD_INLINE void ZlibHuff(int bits, OutputMemoryStream& stream)
        {
            if (bits <= 143)
                ZlibHuff1(bits, stream);
            else if (bits <= 255)
                ZlibHuff2(bits, stream);
            else if (bits <= 279)
                ZlibHuff3(bits, stream);
            else
                ZlibHuff4(bits, stream);
        }

        SIMD_INLINE void ZlibHuffB(int bits, OutputMemoryStream& stream)
        {
            if (bits <= 143)
                ZlibHuff1(bits, stream);
            else
                ZlibHuff2(bits, stream);
        }

        SIMD_INLINE int ZlibCount(const uint8_t* a, const uint8_t* b, int limit)
        {
            limit = Min(limit, 258);
            int i = 0;
#if defined(SIMD_X64_ENABLE) || defined(SIMD_ARM64_ENABLE)
            int limit8 = limit & (~7);
            for (; i < limit8; i += 8)
                if (*(uint64_t*)(a + i) != *(uint64_t*)(b + i))
                    break;
#else
            int limit4 = limit & (~3);
            for (; i < limit4; i += 4)
                if (*(uint32_t*)(a + i) != *(uint32_t*)(b + i))
                    break;
#endif
            for (; i < limit; i += 1)
                if (a[i] != b[i])
                    break;
            return i;
        }

        SIMD_INLINE uint8_t Paeth(int a, int b, int c)
        {
            int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
            if (pa <= pb && pa <= pc)
                return uint8_t(a);
            if (pb <= pc)
                return uint8_t(b);
            return uint8_t(c);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE int ZlibCount(const uint8_t* a, const uint8_t* b, int limit)
        {
            limit = Min(limit, 258);
            int i = 0;
            int limit16 = limit & (~15);
            for (; i < limit16; i += 16)
                if (_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((__m128i*)(a + i)), _mm_loadu_si128((__m128i*)(b + i)))) != 0xFFFF)
                    break;
#if defined(SIMD_X64_ENABLE)
            int limit8 = limit & (~7);
            for (; i < limit8; i += 8)
                if (*(uint64_t*)(a + i) != *(uint64_t*)(b + i))
                    break;
#else
            int limit4 = limit & (~3);
            for (; i < limit4; i += 4)
                if (*(uint32_t*)(a + i) != *(uint32_t*)(b + i))
                    break;
#endif
            for (; i < limit; i += 1)
                if (a[i] != b[i])
                    break;
            return i;
        }
    }
#endif// SIMD_SSE41_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE int ZlibCount(const uint8_t* a, const uint8_t* b, int limit)
        {
            limit = Min(limit, 258);
            int i = 0;
            for (; i < limit; i += 32)
            {
                __m256i _a = _mm256_loadu_si256((__m256i*)(a + i));
                __m256i _b = _mm256_loadu_si256((__m256i*)(b + i));
                uint32_t mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_a, _b));
                if (mask != 0xFFFFFFFF)
                {
                    i += _tzcnt_u32(~mask);
                    break;
                }
            }
            return Min(i, limit);
        }
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE int ZlibCount(const uint8_t* a, const uint8_t* b, int limit)
        {
            limit = Min(limit, 258);
            int i = 0;
            for (; i < limit; i += 64)
            {
                __m512i _a = _mm512_loadu_si512(a + i);
                __m512i _b = _mm512_loadu_si512(b + i);
                uint64_t mask = _mm512_cmp_epi8_mask(_a, _b, _MM_CMPINT_NE);
                if (mask != 0)
                {
                    i += (int)FirstNotZero64(mask);
                    break;
                }
            }
            return Min(i, limit);
        }
    }
#endif// SIMD_AVX512BW_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif// SIMD_NEON_ENABLE
}

#endif//__SimdImageSavePng_h__
