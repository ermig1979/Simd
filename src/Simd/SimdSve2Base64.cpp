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
        SIMD_INLINE svuint8_t FromBase64(const svuint8_t& src, const svbool_t& mask)
        {
            const svuint8_t zero = svdup_n_u8(0);
            svuint8_t dst = zero;

            svbool_t upper = svand_b_z(mask, svcmpgt_n_u8(mask, src, 'A' - 1), svnot_b_z(mask, svcmpgt_n_u8(mask, src, 'Z')));
            svbool_t lower = svand_b_z(mask, svcmpgt_n_u8(mask, src, 'a' - 1), svnot_b_z(mask, svcmpgt_n_u8(mask, src, 'z')));
            svbool_t digit = svand_b_z(mask, svcmpgt_n_u8(mask, src, '0' - 1), svnot_b_z(mask, svcmpgt_n_u8(mask, src, '9')));
            svbool_t slash = svorr_b_z(mask, svcmpeq_n_u8(mask, src, '/'), svcmpeq_n_u8(mask, src, '_'));

            dst = svsel_u8(upper, svsub_n_u8_x(mask, src, 'A'), dst);
            dst = svsel_u8(lower, svsub_n_u8_x(mask, src, 'a' - 26), dst);
            dst = svsel_u8(digit, svadd_n_u8_x(mask, src, 52 - '0'), dst);
            dst = svsel_u8(svcmpeq_n_u8(mask, src, '+'), svdup_n_u8(62), dst);
            dst = svsel_u8(slash, svdup_n_u8(63), dst);
            return dst;
        }

        SIMD_INLINE void Base64Decode(const uint8_t* src, uint8_t* dst, const svbool_t& mask)
        {
            svuint8x4_t _src = svld4_u8(mask, src);
            svuint8_t src0 = FromBase64(svget4(_src, 0), mask);
            svuint8_t src1 = FromBase64(svget4(_src, 1), mask);
            svuint8_t src2 = FromBase64(svget4(_src, 2), mask);
            svuint8_t src3 = FromBase64(svget4(_src, 3), mask);

            svuint8_t dst0 = svorr_u8_x(mask, svlsl_n_u8_x(mask, src0, 2), svlsr_n_u8_x(mask, svand_n_u8_x(mask, src1, 0x30), 4));
            svuint8_t dst1 = svorr_u8_x(mask, svlsl_n_u8_x(mask, svand_n_u8_x(mask, src1, 0x0F), 4), svlsr_n_u8_x(mask, svand_n_u8_x(mask, src2, 0x3C), 2));
            svuint8_t dst2 = svorr_u8_x(mask, svlsl_n_u8_x(mask, svand_n_u8_x(mask, src2, 0x03), 6), src3);

            svst3_u8(mask, dst, svcreate3_u8(dst0, dst1, dst2));
        }

        void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize)
        {
            assert(srcSize % 4 == 0 && srcSize >= 4);

            size_t A = svlen(svuint8_t()), srcStep = A * 4, dstStep = A * 3;
            size_t srcSizeA = srcSize > srcStep ? AlignLoAny(srcSize - srcStep, srcStep) : 0;
            const svbool_t body = svptrue_b8();
            for (const uint8_t* bodyA = src + srcSizeA; src < bodyA; src += srcStep, dst += dstStep)
                Base64Decode(src, dst, body);
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

