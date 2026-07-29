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
        SIMD_INLINE bool InitBase64DecodeCompactIndex(uint8_t index[SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            size_t dstSize = A * 3 / 4;
            for (size_t i = 0; i < dstSize; ++i)
                index[i] = (uint8_t)(i + i / 3);
            for (size_t i = dstSize; i < A; ++i)
                index[i] = 0xFF;
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t BASE64_DECODE_COMPACT_INDEX[SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool BASE64_DECODE_COMPACT_INDEX_INITED = InitBase64DecodeCompactIndex(BASE64_DECODE_COMPACT_INDEX);

        const uint8_t BASE64_FROM_DIG_SHUFFLE[16] = {
            62, 0xFF, 0xFF, 0xFF, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0xFF };

        SIMD_INLINE svuint8_t FromBase64(const svuint8_t& src, const svbool_t& mask, const svuint8_t& fromDig)
        {
            const svuint8_t zero = svdup_n_u8(0);
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

        SIMD_INLINE void Base64Decode(const uint8_t* src, uint8_t* dst, const svbool_t& load,
            const svbool_t& store, const svbool_t& mask32, const svuint8_t& fromDig, const svuint8_t& compact)
        {
            svuint8_t from = FromBase64(svld1_u8(load, src), load, fromDig);

            svuint32_t v = svreinterpret_u32_u8(from);
            svuint32_t s0 = svand_n_u32_x(mask32, v, 0xFF);
            svuint32_t s1 = svand_n_u32_x(mask32, svlsr_n_u32_x(mask32, v, 8), 0xFF);
            svuint32_t s2 = svand_n_u32_x(mask32, svlsr_n_u32_x(mask32, v, 16), 0xFF);
            svuint32_t s3 = svlsr_n_u32_x(mask32, v, 24);

            svuint32_t d0 = svorr_u32_x(mask32, svlsl_n_u32_x(mask32, s0, 2), svlsr_n_u32_x(mask32, s1, 4));
            svuint32_t d1 = svorr_u32_x(mask32, svlsl_n_u32_x(mask32, svand_n_u32_x(mask32, s1, 0x0F), 4),
                svlsr_n_u32_x(mask32, s2, 2));
            svuint32_t d2 = svorr_u32_x(mask32, svlsl_n_u32_x(mask32, svand_n_u32_x(mask32, s2, 0x03), 6), s3);

            svuint32_t out = svorr_u32_x(mask32, d0, svlsl_n_u32_x(mask32, d1, 8));
            out = svorr_u32_x(mask32, out, svlsl_n_u32_x(mask32, d2, 16));

            svst1_u8(store, dst, svtbl_u8(svreinterpret_u8_u32(out), compact));
        }

        void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize)
        {
            assert(srcSize % 4 == 0 && srcSize >= 4);

            size_t A = svlen(svuint8_t()), dstStep = A * 3 / 4;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX && (A % 4) == 0);

            size_t srcSize4 = srcSize - 4;
            size_t srcSizeA = AlignLo(srcSize4, A);

            const svbool_t body = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            const svbool_t store = svwhilelt_b8((size_t)0, dstStep);
            const svuint8_t fromDig = svld1rq_u8(body, BASE64_FROM_DIG_SHUFFLE);
            const svuint8_t compact = svld1_u8(body, BASE64_DECODE_COMPACT_INDEX);

            for (const uint8_t* bodyA = src + srcSizeA; src < bodyA; src += A, dst += dstStep)
                Base64Decode(src, dst, body, store, body32, fromDig, compact);
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
