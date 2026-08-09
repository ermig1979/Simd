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
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase64.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        const uint8_t BASE64_FROM_DIG_SHUFFLE[16] = {
            62, 0xFF, 0xFF, 0xFF, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0xFF };

        // Per 16-byte lane shuffle patterns from the SSE4.1 Base64Decode pack.
        const uint8_t BASE64_DECODE_SHUFFLE_LO[16] = {
            0x1, 0x3, 0x2, 0x5, 0x7, 0x6, 0x9, 0xB, 0xA, 0xD, 0xF, 0xE, 0xFF, 0xFF, 0xFF, 0xFF };
        const uint8_t BASE64_DECODE_SHUFFLE_HI[16] = {
            0x1, 0x0, 0x2, 0x5, 0x4, 0x6, 0x9, 0x8, 0xA, 0xD, 0xC, 0xE, 0xFF, 0xFF, 0xFF, 0xFF };

        SIMD_INLINE bool InitBase64DecodeShuffleIndex(uint8_t shuffleLo[SIMD_SVE2_VECTOR_SIZE_MAX],
            uint8_t shuffleHi[SIMD_SVE2_VECTOR_SIZE_MAX], uint8_t compact[SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX && (A % 16) == 0);
            for (size_t i = 0; i < A; ++i)
            {
                size_t lane = i / 16, j = i % 16;
                uint8_t lo = BASE64_DECODE_SHUFFLE_LO[j];
                uint8_t hi = BASE64_DECODE_SHUFFLE_HI[j];
                shuffleLo[i] = lo == 0xFF ? 0xFF : (uint8_t)(lane * 16 + lo);
                shuffleHi[i] = hi == 0xFF ? 0xFF : (uint8_t)(lane * 16 + hi);
            }
            size_t dstSize = A * 3 / 4;
            for (size_t i = 0; i < dstSize; ++i)
                compact[i] = (uint8_t)((i / 12) * 16 + (i % 12));
            for (size_t i = dstSize; i < A; ++i)
                compact[i] = 0xFF;
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t BASE64_DECODE_SHUFFLE_LO_INDEX[SIMD_SVE2_VECTOR_SIZE_MAX];
        SIMD_ALIGNED(SIMD_ALIGN) uint8_t BASE64_DECODE_SHUFFLE_HI_INDEX[SIMD_SVE2_VECTOR_SIZE_MAX];
        SIMD_ALIGNED(SIMD_ALIGN) uint8_t BASE64_DECODE_COMPACT_INDEX[SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool BASE64_DECODE_INDEX_INITED = InitBase64DecodeShuffleIndex(
            BASE64_DECODE_SHUFFLE_LO_INDEX, BASE64_DECODE_SHUFFLE_HI_INDEX, BASE64_DECODE_COMPACT_INDEX);

        SIMD_INLINE svuint8_t FromBase64(svuint8_t src, svbool_t mask, svuint8_t fromDig)
        {
            svuint8_t zero = svdup_n_u8(0);
            svbool_t letMask = svcmpgt_n_u8(mask, src, '9');
            svbool_t lowMask = svcmpgt_n_u8(mask, src, 'Z');
            svuint8_t lowValue = svsel_u8(lowMask, svsub_n_u8_x(mask, src, 'a' - 26), zero);
            svuint8_t uppValue = svsel_u8(svand_b_z(mask, letMask, svnot_b_z(mask, lowMask)),
                svsub_n_u8_x(mask, src, 'A'), zero);
            svuint8_t digValue = svsel_u8(svnot_b_z(mask, letMask),
                svtbl_u8(fromDig, svsub_n_u8_x(mask, src, '+')), zero);
            svuint8_t dst = svorr_u8_x(mask, svorr_u8_x(mask, uppValue, lowValue), digValue);
            return svsel_u8(svcmpeq_n_u8(mask, src, '_'), svdup_n_u8(63), dst);
        }

        SIMD_INLINE svuint8_t Base64DecodePack(svuint8_t from,
            svuint16_t mulLoK, svuint16_t mulHiK, svuint8_t shuffleLo, svuint8_t shuffleHi)
        {
            const svbool_t mask16 = svptrue_b16();
            svuint16_t words = svreinterpret_u16_u8(from);
            svuint16_t mulLo = svmul_u16_x(mask16, svand_n_u16_x(mask16, words, 0x003F), mulLoK);
            svuint16_t mulHi = svmulh_u16_x(mask16, svand_n_u16_x(mask16, words, 0x3F00), mulHiK);
            const svbool_t mask8 = svptrue_b8();
            return svorr_u8_x(mask8,
                svtbl_u8(svreinterpret_u8_u16(mulHi), shuffleHi),
                svtbl_u8(svreinterpret_u8_u16(mulLo), shuffleLo));
        }

        template<bool narrow> SIMD_INLINE void Base64Decode(const uint8_t* src, uint8_t* dst, svbool_t mask,
            svuint8_t fromDig, svuint16_t mulLoK, svuint16_t mulHiK,
            svuint8_t shuffleLo, svuint8_t shuffleHi, svuint8_t compact)
        {
            svuint8_t packed = Base64DecodePack(FromBase64(svld1_u8(mask, src), mask, fromDig),
                mulLoK, mulHiK, shuffleLo, shuffleHi);
            if (narrow)
                packed = svtbl_u8(packed, compact);
            // Full-vector store: only the first 3/4 lanes are valid; the next overlapping
            // store (or scalar tail) overwrites the garbage, matching SSE4.1/AVX.
            svst1_u8(mask, dst, packed);
        }

        void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize)
        {
            assert(srcSize % 4 == 0 && srcSize >= 4);

            size_t A = svlen(svuint8_t()), dstStep = A * 3 / 4;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX && (A % 16) == 0);

            size_t srcSizeA = srcSize >= A ? AlignLoAny(srcSize - (A - 1), A) : 0;

            const svbool_t body = svptrue_b8();
            const svuint8_t fromDig = svld1rq_u8(body, BASE64_FROM_DIG_SHUFFLE);
            const svuint16_t mulLoK = svreinterpret_u16_u32(svdup_n_u32(0x00400400));
            const svuint16_t mulHiK = svreinterpret_u16_u32(svdup_n_u32(0x01001000));
            const svuint8_t shuffleLo = svld1_u8(body, BASE64_DECODE_SHUFFLE_LO_INDEX);
            const svuint8_t shuffleHi = svld1_u8(body, BASE64_DECODE_SHUFFLE_HI_INDEX);
            const svuint8_t compact = svld1_u8(body, BASE64_DECODE_COMPACT_INDEX);

            if (A == 16)
            {
                size_t srcSize64 = AlignLo(srcSizeA, 64);
                for (const uint8_t* body64 = src + srcSize64; src < body64; src += 64, dst += 48)
                {
                    Base64Decode<false>(src + 0, dst + 0, body, fromDig, mulLoK, mulHiK, shuffleLo, shuffleHi, compact);
                    Base64Decode<false>(src + 16, dst + 12, body, fromDig, mulLoK, mulHiK, shuffleLo, shuffleHi, compact);
                    Base64Decode<false>(src + 32, dst + 24, body, fromDig, mulLoK, mulHiK, shuffleLo, shuffleHi, compact);
                    Base64Decode<false>(src + 48, dst + 36, body, fromDig, mulLoK, mulHiK, shuffleLo, shuffleHi, compact);
                }
                for (const uint8_t* bodyA = src + srcSizeA - srcSize64; src < bodyA; src += A, dst += dstStep)
                    Base64Decode<false>(src, dst, body, fromDig, mulLoK, mulHiK, shuffleLo, shuffleHi, compact);
            }
            else
            {
                for (const uint8_t* bodyA = src + srcSizeA; src < bodyA; src += A, dst += dstStep)
                    Base64Decode<true>(src, dst, body, fromDig, mulLoK, mulHiK, shuffleLo, shuffleHi, compact);
            }
            for (const uint8_t* tail = src + srcSize - srcSizeA - 4; src < tail; src += 4, dst += 3)
                Base::Base64Decode3(src, dst);
            *dstSize = srcSize / 4 * 3 + Base::Base64DecodeTail(src, dst) - 3;
        }

        //---------------------------------------------------------------------------------------------

        SIMD_INLINE svuint8_t ToBase64(const svuint8_t& src, const svbool_t& mask)
        {
            svbool_t upper = svnot_b_z(mask, svcmpgt_n_u8(mask, src, 25));
            svbool_t lower = svand_b_z(mask, svcmpgt_n_u8(mask, src, 25), svnot_b_z(mask, svcmpgt_n_u8(mask, src, 51)));
            svbool_t digit = svand_b_z(mask, svcmpgt_n_u8(mask, src, 51), svnot_b_z(mask, svcmpgt_n_u8(mask, src, 61)));

            svuint8_t dst = svdup_n_u8('/');
            dst = svsel_u8(svcmpeq_n_u8(mask, src, 62), svdup_n_u8('+'), dst);
            dst = svsel_u8(digit, svsub_n_u8_x(mask, src, 4), dst);
            dst = svsel_u8(lower, svadd_n_u8_x(mask, src, 'a' - 26), dst);
            dst = svsel_u8(upper, svadd_n_u8_x(mask, src, 'A'), dst);
            return dst;
        }

        SIMD_INLINE void Base64Encode(const uint8_t* src, uint8_t* dst, const svbool_t& mask)
        {
            svuint8x3_t _src = svld3_u8(mask, src);
            svuint8_t src0 = svget3(_src, 0);
            svuint8_t src1 = svget3(_src, 1);
            svuint8_t src2 = svget3(_src, 2);

            svuint8_t dst0 = svlsr_n_u8_x(mask, svand_n_u8_x(mask, src0, 0xFC), 2);
            svuint8_t dst1 = svorr_u8_x(mask, svlsl_n_u8_x(mask, svand_n_u8_x(mask, src0, 0x03), 4), svlsr_n_u8_x(mask, svand_n_u8_x(mask, src1, 0xF0), 4));
            svuint8_t dst2 = svorr_u8_x(mask, svlsl_n_u8_x(mask, svand_n_u8_x(mask, src1, 0x0F), 2), svlsr_n_u8_x(mask, svand_n_u8_x(mask, src2, 0xC0), 6));
            svuint8_t dst3 = svand_n_u8_x(mask, src2, 0x3F);

            dst0 = ToBase64(dst0, mask);
            dst1 = ToBase64(dst1, mask);
            dst2 = ToBase64(dst2, mask);
            dst3 = ToBase64(dst3, mask);

            svst4_u8(mask, dst, svcreate4_u8(dst0, dst1, dst2, dst3));
        }

        void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst)
        {
            size_t size3 = AlignLoAny(size, 3);
            size_t A = svlen(svuint8_t()), srcStep = A * 3, dstStep = A * 4;
            size_t sizeA = size >= srcStep - 1 ? AlignLoAny(size - (srcStep - 1), srcStep) : 0;
            const svbool_t body = svptrue_b8();
            for (const uint8_t* bodyA = src + sizeA; src < bodyA; src += srcStep, dst += dstStep)
                Base64Encode(src, dst, body);
            for (const uint8_t* body3 = src + size3 - sizeA; src < body3; src += 3, dst += 4)
                Base::Base64Encode3(src, dst);
            if (size - size3)
                Base::Base64EncodeTail(src, size - size3, dst);
        }
    }
#endif
}
