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
    }
#endif
}

