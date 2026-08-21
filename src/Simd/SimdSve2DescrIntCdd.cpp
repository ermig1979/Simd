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
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        const uint8_t C5_TBL[16] = { 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4 };
        const uint16_t C5_SHR[8] = { 8, 5, 10, 7, 4, 9, 6, 11 };
        const uint8_t C6_TBL[16] = { 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5 };
        const uint16_t C6_SHR[8] = { 8, 6, 4, 2, 8, 6, 4, 2 };
        const uint8_t C7_TBL[16] = { 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6 };
        const uint16_t C7_SHR[8] = { 8, 7, 6, 5, 4, 3, 2, 1 };

        SIMD_INLINE svuint8_t UnpackTbl(const uint8_t* tbl16, uint8_t bits)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t base = svld1rq_u8(all, tbl16);
            svuint8_t index = svindex_u8(0, 1);
            svuint8_t group = svlsr_n_u8_x(all, index, 4);
            return svadd_u8_x(all, base, svmul_n_u8_x(all, group, bits));
        }

        SIMD_INLINE svuint16_t UnpackShr(const uint16_t* shr8)
        {
            return svld1rq_u16(svptrue_b16(), shr8);
        }

        template<int bits> SIMD_INLINE svuint8_t UnpackTbl();
        template<int bits> SIMD_INLINE svuint16_t UnpackShr();

        template<> SIMD_INLINE svuint8_t UnpackTbl<5>() { return UnpackTbl(C5_TBL, 5); }
        template<> SIMD_INLINE svuint16_t UnpackShr<5>() { return UnpackShr(C5_SHR); }
        template<> SIMD_INLINE svuint8_t UnpackTbl<6>() { return UnpackTbl(C6_TBL, 6); }
        template<> SIMD_INLINE svuint16_t UnpackShr<6>() { return UnpackShr(C6_SHR); }
        template<> SIMD_INLINE svuint8_t UnpackTbl<7>() { return UnpackTbl(C7_TBL, 7); }
        template<> SIMD_INLINE svuint16_t UnpackShr<7>() { return UnpackShr(C7_SHR); }

        SIMD_INLINE svuint16_t UnpackTo16(const uint8_t* src, size_t packed, const svuint8_t& tbl, const svuint16_t& shr, uint16_t mask)
        {
            svuint8_t raw = svld1_u8(svwhilelt_b8((size_t)0, packed), src);
            svuint16_t wide = svreinterpret_u16_u8(svtbl_u8(raw, tbl));
            const svbool_t all = svptrue_b16();
            return svand_n_u16_x(all, svlsr_u16_x(all, wide, shr), mask);
        }

        template<int bits> SIMD_INLINE svuint8_t UnpackTo8(const uint8_t* src, size_t packed, const svuint8_t& tbl, const svuint16_t& shr)
        {
            const size_t packedHalf = svcnth() * bits / 8;
            const uint16_t mask = (uint16_t)((1 << bits) - 1);
            size_t packedLo = packed < packedHalf ? packed : packedHalf;
            size_t packedHi = packed - packedLo;
            svuint16_t lo = UnpackTo16(src, packedLo, tbl, shr, mask);
            svuint16_t hi = UnpackTo16(src + packedLo, packedHi, tbl, shr, mask);
            return svuzp1_u8(svreinterpret_u8_u16(lo), svreinterpret_u8_u16(hi));
        }

        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0 && size >= 8);
            const size_t valuesPerVec = svcntb();
            const size_t packedPerVec = valuesPerVec * bits / 8;
            const svuint8_t tbl = UnpackTbl<bits>();
            const svuint16_t shr = UnpackShr<bits>();
            svuint32_t sums = svdup_n_u32(0);
            size_t i = 0, packed = 0;
            for (; i + valuesPerVec <= size; i += valuesPerVec, packed += packedPerVec)
            {
                svuint8_t a8 = UnpackTo8<bits>(a + packed, packedPerVec, tbl, shr);
                svuint8_t b8 = UnpackTo8<bits>(b + packed, packedPerVec, tbl, shr);
                sums = svdot_u32(sums, a8, b8);
            }
            if (i < size)
            {
                size_t packedTail = (size - i) * bits / 8;
                svuint8_t a8 = UnpackTo8<bits>(a + packed, packedTail, tbl, shr);
                svuint8_t b8 = UnpackTo8<bits>(b + packed, packedTail, tbl, shr);
                sums = svdot_u32(sums, a8, b8);
            }
            return (int32_t)svaddv_u32(svptrue_b32(), sums);
        }

        template<> int32_t Correlation<4>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0 && size >= 8);
            const size_t byteSize = size / 2;
            const svuint8_t zero = svdup_n_u8(0);
            svuint32_t sums = svdup_n_u32(0);
            for (size_t i = 0; i < byteSize; i += svcntb())
            {
                svbool_t mask = svwhilelt_b8(i, byteSize);
                svuint8_t _a = svsel_u8(mask, svld1_u8(mask, a + i), zero);
                svuint8_t _b = svsel_u8(mask, svld1_u8(mask, b + i), zero);
                sums = svdot_u32(sums, svand_n_u8_z(mask, _a, 0x0F), svand_n_u8_z(mask, _b, 0x0F));
                sums = svdot_u32(sums, svlsr_n_u8_z(mask, _a, 4), svlsr_n_u8_z(mask, _b, 4));
            }
            return (int32_t)svaddv_u32(svptrue_b32(), sums);
        }

        template<> int32_t Correlation<8>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0 && size >= 8);
            const svuint8_t zero = svdup_n_u8(0);
            svuint32_t sums = svdup_n_u32(0);
            for (size_t i = 0; i < size; i += svcntb())
            {
                svbool_t mask = svwhilelt_b8(i, size);
                svuint8_t _a = svsel_u8(mask, svld1_u8(mask, a + i), zero);
                svuint8_t _b = svsel_u8(mask, svld1_u8(mask, b + i), zero);
                sums = svdot_u32(sums, _a, _b);
            }
            return (int32_t)svaddv_u32(svptrue_b32(), sums);
        }

        template<int bits> void CosineDistance(const uint8_t* a, const uint8_t* b, size_t size, float* distance)
        {
            float abSum = (float)Correlation<bits>(a + 16, b + 16, size);
            Base::DecodeCosineDistance(a, b, abSum, distance);
        }

        template<int bits, int M> SIMD_INLINE void MicroCosineDistancesDirectMx4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            const size_t valuesPerVec = svcntb();
            const size_t packedPerVec = valuesPerVec * bits / 8;
            const svuint8_t tbl = UnpackTbl<bits>();
            const svuint16_t shr = UnpackShr<bits>();
            svuint32_t ab00 = svdup_n_u32(0), ab01 = ab00, ab02 = ab00, ab03 = ab00;
            svuint32_t ab10 = ab00, ab11 = ab00, ab12 = ab00, ab13 = ab00;
            svuint32_t ab20 = ab00, ab21 = ab00, ab22 = ab00, ab23 = ab00;
            svuint32_t ab30 = ab00, ab31 = ab00, ab32 = ab00, ab33 = ab00;
            size_t i = 0, o = 16;
            for (; i + valuesPerVec <= size; i += valuesPerVec, o += packedPerVec)
            {
                svuint8_t b0 = UnpackTo8<bits>(B[0] + o, packedPerVec, tbl, shr);
                svuint8_t b1 = UnpackTo8<bits>(B[1] + o, packedPerVec, tbl, shr);
                svuint8_t b2 = UnpackTo8<bits>(B[2] + o, packedPerVec, tbl, shr);
                svuint8_t b3 = UnpackTo8<bits>(B[3] + o, packedPerVec, tbl, shr);
                if (M > 0)
                {
                    svuint8_t a0 = UnpackTo8<bits>(A[0] + o, packedPerVec, tbl, shr);
                    ab00 = svdot_u32(ab00, a0, b0);
                    ab01 = svdot_u32(ab01, a0, b1);
                    ab02 = svdot_u32(ab02, a0, b2);
                    ab03 = svdot_u32(ab03, a0, b3);
                }
                if (M > 1)
                {
                    svuint8_t a1 = UnpackTo8<bits>(A[1] + o, packedPerVec, tbl, shr);
                    ab10 = svdot_u32(ab10, a1, b0);
                    ab11 = svdot_u32(ab11, a1, b1);
                    ab12 = svdot_u32(ab12, a1, b2);
                    ab13 = svdot_u32(ab13, a1, b3);
                }
                if (M > 2)
                {
                    svuint8_t a2 = UnpackTo8<bits>(A[2] + o, packedPerVec, tbl, shr);
                    ab20 = svdot_u32(ab20, a2, b0);
                    ab21 = svdot_u32(ab21, a2, b1);
                    ab22 = svdot_u32(ab22, a2, b2);
                    ab23 = svdot_u32(ab23, a2, b3);
                }
                if (M > 3)
                {
                    svuint8_t a3 = UnpackTo8<bits>(A[3] + o, packedPerVec, tbl, shr);
                    ab30 = svdot_u32(ab30, a3, b0);
                    ab31 = svdot_u32(ab31, a3, b1);
                    ab32 = svdot_u32(ab32, a3, b2);
                    ab33 = svdot_u32(ab33, a3, b3);
                }
            }
            if (i < size)
            {
                size_t packedTail = (size - i) * bits / 8;
                svuint8_t b0 = UnpackTo8<bits>(B[0] + o, packedTail, tbl, shr);
                svuint8_t b1 = UnpackTo8<bits>(B[1] + o, packedTail, tbl, shr);
                svuint8_t b2 = UnpackTo8<bits>(B[2] + o, packedTail, tbl, shr);
                svuint8_t b3 = UnpackTo8<bits>(B[3] + o, packedTail, tbl, shr);
                if (M > 0)
                {
                    svuint8_t a0 = UnpackTo8<bits>(A[0] + o, packedTail, tbl, shr);
                    ab00 = svdot_u32(ab00, a0, b0);
                    ab01 = svdot_u32(ab01, a0, b1);
                    ab02 = svdot_u32(ab02, a0, b2);
                    ab03 = svdot_u32(ab03, a0, b3);
                }
                if (M > 1)
                {
                    svuint8_t a1 = UnpackTo8<bits>(A[1] + o, packedTail, tbl, shr);
                    ab10 = svdot_u32(ab10, a1, b0);
                    ab11 = svdot_u32(ab11, a1, b1);
                    ab12 = svdot_u32(ab12, a1, b2);
                    ab13 = svdot_u32(ab13, a1, b3);
                }
                if (M > 2)
                {
                    svuint8_t a2 = UnpackTo8<bits>(A[2] + o, packedTail, tbl, shr);
                    ab20 = svdot_u32(ab20, a2, b0);
                    ab21 = svdot_u32(ab21, a2, b1);
                    ab22 = svdot_u32(ab22, a2, b2);
                    ab23 = svdot_u32(ab23, a2, b3);
                }
                if (M > 3)
                {
                    svuint8_t a3 = UnpackTo8<bits>(A[3] + o, packedTail, tbl, shr);
                    ab30 = svdot_u32(ab30, a3, b0);
                    ab31 = svdot_u32(ab31, a3, b1);
                    ab32 = svdot_u32(ab32, a3, b2);
                    ab33 = svdot_u32(ab33, a3, b3);
                }
            }
            if (M > 0)
            {
                Base::DecodeCosineDistance(A[0], B[0], (float)svaddv_u32(svptrue_b32(), ab00), distances + 0 * stride + 0);
                Base::DecodeCosineDistance(A[0], B[1], (float)svaddv_u32(svptrue_b32(), ab01), distances + 0 * stride + 1);
                Base::DecodeCosineDistance(A[0], B[2], (float)svaddv_u32(svptrue_b32(), ab02), distances + 0 * stride + 2);
                Base::DecodeCosineDistance(A[0], B[3], (float)svaddv_u32(svptrue_b32(), ab03), distances + 0 * stride + 3);
            }
            if (M > 1)
            {
                Base::DecodeCosineDistance(A[1], B[0], (float)svaddv_u32(svptrue_b32(), ab10), distances + 1 * stride + 0);
                Base::DecodeCosineDistance(A[1], B[1], (float)svaddv_u32(svptrue_b32(), ab11), distances + 1 * stride + 1);
                Base::DecodeCosineDistance(A[1], B[2], (float)svaddv_u32(svptrue_b32(), ab12), distances + 1 * stride + 2);
                Base::DecodeCosineDistance(A[1], B[3], (float)svaddv_u32(svptrue_b32(), ab13), distances + 1 * stride + 3);
            }
            if (M > 2)
            {
                Base::DecodeCosineDistance(A[2], B[0], (float)svaddv_u32(svptrue_b32(), ab20), distances + 2 * stride + 0);
                Base::DecodeCosineDistance(A[2], B[1], (float)svaddv_u32(svptrue_b32(), ab21), distances + 2 * stride + 1);
                Base::DecodeCosineDistance(A[2], B[2], (float)svaddv_u32(svptrue_b32(), ab22), distances + 2 * stride + 2);
                Base::DecodeCosineDistance(A[2], B[3], (float)svaddv_u32(svptrue_b32(), ab23), distances + 2 * stride + 3);
            }
            if (M > 3)
            {
                Base::DecodeCosineDistance(A[3], B[0], (float)svaddv_u32(svptrue_b32(), ab30), distances + 3 * stride + 0);
                Base::DecodeCosineDistance(A[3], B[1], (float)svaddv_u32(svptrue_b32(), ab31), distances + 3 * stride + 1);
                Base::DecodeCosineDistance(A[3], B[2], (float)svaddv_u32(svptrue_b32(), ab32), distances + 3 * stride + 2);
                Base::DecodeCosineDistance(A[3], B[3], (float)svaddv_u32(svptrue_b32(), ab33), distances + 3 * stride + 3);
            }
        }

        template<int bits> struct CorrelationsMx1
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* B, size_t size, uint32_t* ab)
            {
                assert(size % 8 == 0 && size >= 8);
                const size_t valuesPerVec = svcntb();
                const size_t packedPerVec = valuesPerVec * bits / 8;
                const svuint8_t tbl = UnpackTbl<bits>();
                const svuint16_t shr = UnpackShr<bits>();
                svuint32_t sums0 = svdup_n_u32(0), sums1 = sums0, sums2 = sums0, sums3 = sums0;
                const uint8_t* b = B + 16;
                size_t i = 0, packed = 0;
                for (; i + valuesPerVec <= size; i += valuesPerVec, packed += packedPerVec)
                {
                    svuint8_t _b = UnpackTo8<bits>(b + packed, packedPerVec, tbl, shr);
                    if (M > 0)
                        sums0 = svdot_u32(sums0, UnpackTo8<bits>(A[0] + 16 + packed, packedPerVec, tbl, shr), _b);
                    if (M > 1)
                        sums1 = svdot_u32(sums1, UnpackTo8<bits>(A[1] + 16 + packed, packedPerVec, tbl, shr), _b);
                    if (M > 2)
                        sums2 = svdot_u32(sums2, UnpackTo8<bits>(A[2] + 16 + packed, packedPerVec, tbl, shr), _b);
                    if (M > 3)
                        sums3 = svdot_u32(sums3, UnpackTo8<bits>(A[3] + 16 + packed, packedPerVec, tbl, shr), _b);
                }
                if (i < size)
                {
                    size_t packedTail = (size - i) * bits / 8;
                    svuint8_t _b = UnpackTo8<bits>(b + packed, packedTail, tbl, shr);
                    if (M > 0)
                        sums0 = svdot_u32(sums0, UnpackTo8<bits>(A[0] + 16 + packed, packedTail, tbl, shr), _b);
                    if (M > 1)
                        sums1 = svdot_u32(sums1, UnpackTo8<bits>(A[1] + 16 + packed, packedTail, tbl, shr), _b);
                    if (M > 2)
                        sums2 = svdot_u32(sums2, UnpackTo8<bits>(A[2] + 16 + packed, packedTail, tbl, shr), _b);
                    if (M > 3)
                        sums3 = svdot_u32(sums3, UnpackTo8<bits>(A[3] + 16 + packed, packedTail, tbl, shr), _b);
                }
                if (M > 0) ab[0] = (uint32_t)svaddv_u32(svptrue_b32(), sums0);
                if (M > 1) ab[1] = (uint32_t)svaddv_u32(svptrue_b32(), sums1);
                if (M > 2) ab[2] = (uint32_t)svaddv_u32(svptrue_b32(), sums2);
                if (M > 3) ab[3] = (uint32_t)svaddv_u32(svptrue_b32(), sums3);
            }
        };

        template<> struct CorrelationsMx1<4>
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* B, size_t size, uint32_t* ab)
            {
                assert(size % 8 == 0 && size >= 8);
                const size_t byteSize = size / 2;
                const svuint8_t zero = svdup_n_u8(0);
                svuint32_t sums0 = svdup_n_u32(0), sums1 = sums0, sums2 = sums0, sums3 = sums0;
                const uint8_t* b = B + 16;
                for (size_t i = 0; i < byteSize; i += svcntb())
                {
                    svbool_t mask = svwhilelt_b8(i, byteSize);
                    svuint8_t _b = svsel_u8(mask, svld1_u8(mask, b + i), zero);
                    svuint8_t b0 = svand_n_u8_z(mask, _b, 0x0F);
                    svuint8_t b1 = svlsr_n_u8_z(mask, _b, 4);
                    if (M > 0)
                    {
                        svuint8_t _a = svsel_u8(mask, svld1_u8(mask, A[0] + 16 + i), zero);
                        sums0 = svdot_u32(sums0, svand_n_u8_z(mask, _a, 0x0F), b0);
                        sums0 = svdot_u32(sums0, svlsr_n_u8_z(mask, _a, 4), b1);
                    }
                    if (M > 1)
                    {
                        svuint8_t _a = svsel_u8(mask, svld1_u8(mask, A[1] + 16 + i), zero);
                        sums1 = svdot_u32(sums1, svand_n_u8_z(mask, _a, 0x0F), b0);
                        sums1 = svdot_u32(sums1, svlsr_n_u8_z(mask, _a, 4), b1);
                    }
                    if (M > 2)
                    {
                        svuint8_t _a = svsel_u8(mask, svld1_u8(mask, A[2] + 16 + i), zero);
                        sums2 = svdot_u32(sums2, svand_n_u8_z(mask, _a, 0x0F), b0);
                        sums2 = svdot_u32(sums2, svlsr_n_u8_z(mask, _a, 4), b1);
                    }
                    if (M > 3)
                    {
                        svuint8_t _a = svsel_u8(mask, svld1_u8(mask, A[3] + 16 + i), zero);
                        sums3 = svdot_u32(sums3, svand_n_u8_z(mask, _a, 0x0F), b0);
                        sums3 = svdot_u32(sums3, svlsr_n_u8_z(mask, _a, 4), b1);
                    }
                }
                if (M > 0) ab[0] = (uint32_t)svaddv_u32(svptrue_b32(), sums0);
                if (M > 1) ab[1] = (uint32_t)svaddv_u32(svptrue_b32(), sums1);
                if (M > 2) ab[2] = (uint32_t)svaddv_u32(svptrue_b32(), sums2);
                if (M > 3) ab[3] = (uint32_t)svaddv_u32(svptrue_b32(), sums3);
            }
        };

        template<> struct CorrelationsMx1<8>
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* B, size_t size, uint32_t* ab)
            {
                assert(size % 8 == 0 && size >= 8);
                const svuint8_t zero = svdup_n_u8(0);
                svuint32_t sums0 = svdup_n_u32(0), sums1 = sums0, sums2 = sums0, sums3 = sums0;
                const uint8_t* b = B + 16;
                for (size_t i = 0; i < size; i += svcntb())
                {
                    svbool_t mask = svwhilelt_b8(i, size);
                    svuint8_t _b = svsel_u8(mask, svld1_u8(mask, b + i), zero);
                    if (M > 0)
                    {
                        svuint8_t _a = svsel_u8(mask, svld1_u8(mask, A[0] + 16 + i), zero);
                        sums0 = svdot_u32(sums0, _a, _b);
                    }
                    if (M > 1)
                    {
                        svuint8_t _a = svsel_u8(mask, svld1_u8(mask, A[1] + 16 + i), zero);
                        sums1 = svdot_u32(sums1, _a, _b);
                    }
                    if (M > 2)
                    {
                        svuint8_t _a = svsel_u8(mask, svld1_u8(mask, A[2] + 16 + i), zero);
                        sums2 = svdot_u32(sums2, _a, _b);
                    }
                    if (M > 3)
                    {
                        svuint8_t _a = svsel_u8(mask, svld1_u8(mask, A[3] + 16 + i), zero);
                        sums3 = svdot_u32(sums3, _a, _b);
                    }
                }
                if (M > 0) ab[0] = (uint32_t)svaddv_u32(svptrue_b32(), sums0);
                if (M > 1) ab[1] = (uint32_t)svaddv_u32(svptrue_b32(), sums1);
                if (M > 2) ab[2] = (uint32_t)svaddv_u32(svptrue_b32(), sums2);
                if (M > 3) ab[3] = (uint32_t)svaddv_u32(svptrue_b32(), sums3);
            }
        };

        template<int bits, int M> SIMD_INLINE void MicroCosineDistancesDirectMx1(const uint8_t* const* A, const uint8_t* B, size_t size, float* distances, size_t stride)
        {
            uint32_t ab[4];
            CorrelationsMx1<bits>::template Run<M>(A, B, size, ab);
            if (M > 0) Base::DecodeCosineDistance(A[0], B, (float)ab[0], distances + 0 * stride);
            if (M > 1) Base::DecodeCosineDistance(A[1], B, (float)ab[1], distances + 1 * stride);
            if (M > 2) Base::DecodeCosineDistance(A[2], B, (float)ab[2], distances + 2 * stride);
            if (M > 3) Base::DecodeCosineDistance(A[3], B, (float)ab[3], distances + 3 * stride);
        }

        template<int bits> void MacroCosineDistancesDirectMx1(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M4 = AlignLoAny(M, 4), i = 0;
            for (; i < M4; i += 4)
            {
                for (size_t j = 0; j < N; ++j)
                    MicroCosineDistancesDirectMx1<bits, 4>(A + i, B[j], size, distances + j, stride);
                distances += 4 * stride;
            }
            if (i < M)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    switch (M - i)
                    {
                    case 1: MicroCosineDistancesDirectMx1<bits, 1>(A + i, B[j], size, distances + j, stride); break;
                    case 2: MicroCosineDistancesDirectMx1<bits, 2>(A + i, B[j], size, distances + j, stride); break;
                    case 3: MicroCosineDistancesDirectMx1<bits, 3>(A + i, B[j], size, distances + j, stride); break;
                    }
                }
            }
        }

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M4 = AlignLoAny(M, 4), N4 = AlignLo(N, 4), i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistancesDirectMx4<bits, 4>(A + i, B + j, size, distances + j, stride);
                for (; j < N; ++j)
                    MicroCosineDistancesDirectMx1<bits, 4>(A + i, B[j], size, distances + j, stride);
                distances += 4 * stride;
            }
            if (i < M)
            {
                size_t m = M - i;
                size_t j = 0;
                for (; j < N4; j += 4)
                {
                    if (m == 1)
                        MicroCosineDistancesDirectMx4<bits, 1>(A + i, B + j, size, distances + j, stride);
                    else if (m == 2)
                        MicroCosineDistancesDirectMx4<bits, 2>(A + i, B + j, size, distances + j, stride);
                    else
                        MicroCosineDistancesDirectMx4<bits, 3>(A + i, B + j, size, distances + j, stride);
                }
                for (; j < N; ++j)
                {
                    if (m == 1)
                        MicroCosineDistancesDirectMx1<bits, 1>(A + i, B[j], size, distances + j, stride);
                    else if (m == 2)
                        MicroCosineDistancesDirectMx1<bits, 2>(A + i, B[j], size, distances + j, stride);
                    else
                        MicroCosineDistancesDirectMx1<bits, 3>(A + i, B[j], size, distances + j, stride);
                }
            }
        }

        Base::DescrInt::CosineDistancePtr GetCosineDistance(size_t depth)
        {
            switch (depth)
            {
            case 4: return CosineDistance<4>;
            case 5: return CosineDistance<5>;
            case 6: return CosineDistance<6>;
            case 7: return CosineDistance<7>;
            case 8: return CosineDistance<8>;
            default: return Neon::GetCosineDistance(depth);
            }
        }

        Base::DescrInt::MacroCosineDistancesDirectPtr GetMacroCosineDistancesDirect(size_t depth)
        {
            switch (depth)
            {
            case 4: return MacroCosineDistancesDirectMx1<4>;
            case 5: return MacroCosineDistancesDirect<5>;
            case 6: return MacroCosineDistancesDirect<6>;
            case 7: return MacroCosineDistancesDirect<7>;
            case 8: return MacroCosineDistancesDirectMx1<8>;
            default: return NULL;
            }
        }
    }
#endif
}
