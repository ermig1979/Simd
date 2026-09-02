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
            const size_t width = svcntb();
            const size_t main = byteSize & ~(width - 1);
            const svbool_t all = svptrue_b8();
            svuint32_t sums = svdup_n_u32(0);
            size_t i = 0;
            for (; i < main; i += width)
            {
                svuint8_t _a = svld1_u8(all, a + i);
                svuint8_t _b = svld1_u8(all, b + i);
                sums = svdot_u32(sums, svand_n_u8_x(all, _a, 0x0F), svand_n_u8_x(all, _b, 0x0F));
                sums = svdot_u32(sums, svlsr_n_u8_x(all, _a, 4), svlsr_n_u8_x(all, _b, 4));
            }
            if (i < byteSize)
            {
                svbool_t mask = svwhilelt_b8(i, byteSize);
                svuint8_t _a = svld1_u8(mask, a + i);
                svuint8_t _b = svld1_u8(mask, b + i);
                sums = svdot_u32(sums, svand_n_u8_z(mask, _a, 0x0F), svand_n_u8_z(mask, _b, 0x0F));
                sums = svdot_u32(sums, svlsr_n_u8_z(mask, _a, 4), svlsr_n_u8_z(mask, _b, 4));
            }
            return (int32_t)svaddv_u32(svptrue_b32(), sums);
        }

        template<> int32_t Correlation<8>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0 && size >= 8);
            const size_t width = svcntb();
            const size_t main = size & ~(width - 1);
            const svbool_t all = svptrue_b8();
            svuint32_t sums = svdup_n_u32(0);
            size_t i = 0;
            for (; i < main; i += width)
                sums = svdot_u32(sums, svld1_u8(all, a + i), svld1_u8(all, b + i));
            if (i < size)
            {
                svbool_t mask = svwhilelt_b8(i, size);
                sums = svdot_u32(sums, svld1_u8(mask, a + i), svld1_u8(mask, b + i));
            }
            return (int32_t)svaddv_u32(svptrue_b32(), sums);
        }

        template<int bits> void CosineDistance(const uint8_t* a, const uint8_t* b, size_t size, float* distance)
        {
            float abSum = (float)Correlation<bits>(a + 16, b + 16, size);
            Base::DecodeCosineDistance(a, b, abSum, distance);
        }

        template<int bits> struct DirectMx4
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
                if (M > 0) DecodeCosineDistances1x4(A[0], B, ab00, ab01, ab02, ab03, distances + 0 * stride);
                if (M > 1) DecodeCosineDistances1x4(A[1], B, ab10, ab11, ab12, ab13, distances + 1 * stride);
                if (M > 2) DecodeCosineDistances1x4(A[2], B, ab20, ab21, ab22, ab23, distances + 2 * stride);
                if (M > 3) DecodeCosineDistances1x4(A[3], B, ab30, ab31, ab32, ab33, distances + 3 * stride);
            }
        };

        template<> struct DirectMx4<4>
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
            {
                const size_t width = svcntb();
                const size_t byteSize = size / 2;
                const size_t main = byteSize & ~(width - 1);
                const svbool_t all = svptrue_b8();
                svuint32_t ab00 = svdup_n_u32(0), ab01 = ab00, ab02 = ab00, ab03 = ab00;
                svuint32_t ab10 = ab00, ab11 = ab00, ab12 = ab00, ab13 = ab00;
                svuint32_t ab20 = ab00, ab21 = ab00, ab22 = ab00, ab23 = ab00;
                svuint32_t ab30 = ab00, ab31 = ab00, ab32 = ab00, ab33 = ab00;
                const uint8_t* a0 = A[0] + 16;
                const uint8_t* a1 = M > 1 ? A[1] + 16 : a0;
                const uint8_t* a2 = M > 2 ? A[2] + 16 : a0;
                const uint8_t* a3 = M > 3 ? A[3] + 16 : a0;
                const uint8_t* b0 = B[0] + 16;
                const uint8_t* b1 = B[1] + 16;
                const uint8_t* b2 = B[2] + 16;
                const uint8_t* b3 = B[3] + 16;
                size_t i = 0;
                for (; i < main; i += width)
                {
                    svuint8_t B0 = svld1_u8(all, b0 + i);
                    svuint8_t B1 = svld1_u8(all, b1 + i);
                    svuint8_t B2 = svld1_u8(all, b2 + i);
                    svuint8_t B3 = svld1_u8(all, b3 + i);
                    svuint8_t B0l = svand_n_u8_x(all, B0, 0x0F), B0h = svlsr_n_u8_x(all, B0, 4);
                    svuint8_t B1l = svand_n_u8_x(all, B1, 0x0F), B1h = svlsr_n_u8_x(all, B1, 4);
                    svuint8_t B2l = svand_n_u8_x(all, B2, 0x0F), B2h = svlsr_n_u8_x(all, B2, 4);
                    svuint8_t B3l = svand_n_u8_x(all, B3, 0x0F), B3h = svlsr_n_u8_x(all, B3, 4);
                    if (M > 0)
                    {
                        svuint8_t A0 = svld1_u8(all, a0 + i);
                        svuint8_t A0l = svand_n_u8_x(all, A0, 0x0F), A0h = svlsr_n_u8_x(all, A0, 4);
                        ab00 = svdot_u32(svdot_u32(ab00, A0l, B0l), A0h, B0h);
                        ab01 = svdot_u32(svdot_u32(ab01, A0l, B1l), A0h, B1h);
                        ab02 = svdot_u32(svdot_u32(ab02, A0l, B2l), A0h, B2h);
                        ab03 = svdot_u32(svdot_u32(ab03, A0l, B3l), A0h, B3h);
                    }
                    if (M > 1)
                    {
                        svuint8_t A1 = svld1_u8(all, a1 + i);
                        svuint8_t A1l = svand_n_u8_x(all, A1, 0x0F), A1h = svlsr_n_u8_x(all, A1, 4);
                        ab10 = svdot_u32(svdot_u32(ab10, A1l, B0l), A1h, B0h);
                        ab11 = svdot_u32(svdot_u32(ab11, A1l, B1l), A1h, B1h);
                        ab12 = svdot_u32(svdot_u32(ab12, A1l, B2l), A1h, B2h);
                        ab13 = svdot_u32(svdot_u32(ab13, A1l, B3l), A1h, B3h);
                    }
                    if (M > 2)
                    {
                        svuint8_t A2 = svld1_u8(all, a2 + i);
                        svuint8_t A2l = svand_n_u8_x(all, A2, 0x0F), A2h = svlsr_n_u8_x(all, A2, 4);
                        ab20 = svdot_u32(svdot_u32(ab20, A2l, B0l), A2h, B0h);
                        ab21 = svdot_u32(svdot_u32(ab21, A2l, B1l), A2h, B1h);
                        ab22 = svdot_u32(svdot_u32(ab22, A2l, B2l), A2h, B2h);
                        ab23 = svdot_u32(svdot_u32(ab23, A2l, B3l), A2h, B3h);
                    }
                    if (M > 3)
                    {
                        svuint8_t A3 = svld1_u8(all, a3 + i);
                        svuint8_t A3l = svand_n_u8_x(all, A3, 0x0F), A3h = svlsr_n_u8_x(all, A3, 4);
                        ab30 = svdot_u32(svdot_u32(ab30, A3l, B0l), A3h, B0h);
                        ab31 = svdot_u32(svdot_u32(ab31, A3l, B1l), A3h, B1h);
                        ab32 = svdot_u32(svdot_u32(ab32, A3l, B2l), A3h, B2h);
                        ab33 = svdot_u32(svdot_u32(ab33, A3l, B3l), A3h, B3h);
                    }
                }
                if (i < byteSize)
                {
                    svbool_t mask = svwhilelt_b8(i, byteSize);
                    svuint8_t B0 = svld1_u8(mask, b0 + i);
                    svuint8_t B1 = svld1_u8(mask, b1 + i);
                    svuint8_t B2 = svld1_u8(mask, b2 + i);
                    svuint8_t B3 = svld1_u8(mask, b3 + i);
                    svuint8_t B0l = svand_n_u8_z(mask, B0, 0x0F), B0h = svlsr_n_u8_z(mask, B0, 4);
                    svuint8_t B1l = svand_n_u8_z(mask, B1, 0x0F), B1h = svlsr_n_u8_z(mask, B1, 4);
                    svuint8_t B2l = svand_n_u8_z(mask, B2, 0x0F), B2h = svlsr_n_u8_z(mask, B2, 4);
                    svuint8_t B3l = svand_n_u8_z(mask, B3, 0x0F), B3h = svlsr_n_u8_z(mask, B3, 4);
                    if (M > 0)
                    {
                        svuint8_t A0 = svld1_u8(mask, a0 + i);
                        svuint8_t A0l = svand_n_u8_z(mask, A0, 0x0F), A0h = svlsr_n_u8_z(mask, A0, 4);
                        ab00 = svdot_u32(svdot_u32(ab00, A0l, B0l), A0h, B0h);
                        ab01 = svdot_u32(svdot_u32(ab01, A0l, B1l), A0h, B1h);
                        ab02 = svdot_u32(svdot_u32(ab02, A0l, B2l), A0h, B2h);
                        ab03 = svdot_u32(svdot_u32(ab03, A0l, B3l), A0h, B3h);
                    }
                    if (M > 1)
                    {
                        svuint8_t A1 = svld1_u8(mask, a1 + i);
                        svuint8_t A1l = svand_n_u8_z(mask, A1, 0x0F), A1h = svlsr_n_u8_z(mask, A1, 4);
                        ab10 = svdot_u32(svdot_u32(ab10, A1l, B0l), A1h, B0h);
                        ab11 = svdot_u32(svdot_u32(ab11, A1l, B1l), A1h, B1h);
                        ab12 = svdot_u32(svdot_u32(ab12, A1l, B2l), A1h, B2h);
                        ab13 = svdot_u32(svdot_u32(ab13, A1l, B3l), A1h, B3h);
                    }
                    if (M > 2)
                    {
                        svuint8_t A2 = svld1_u8(mask, a2 + i);
                        svuint8_t A2l = svand_n_u8_z(mask, A2, 0x0F), A2h = svlsr_n_u8_z(mask, A2, 4);
                        ab20 = svdot_u32(svdot_u32(ab20, A2l, B0l), A2h, B0h);
                        ab21 = svdot_u32(svdot_u32(ab21, A2l, B1l), A2h, B1h);
                        ab22 = svdot_u32(svdot_u32(ab22, A2l, B2l), A2h, B2h);
                        ab23 = svdot_u32(svdot_u32(ab23, A2l, B3l), A2h, B3h);
                    }
                    if (M > 3)
                    {
                        svuint8_t A3 = svld1_u8(mask, a3 + i);
                        svuint8_t A3l = svand_n_u8_z(mask, A3, 0x0F), A3h = svlsr_n_u8_z(mask, A3, 4);
                        ab30 = svdot_u32(svdot_u32(ab30, A3l, B0l), A3h, B0h);
                        ab31 = svdot_u32(svdot_u32(ab31, A3l, B1l), A3h, B1h);
                        ab32 = svdot_u32(svdot_u32(ab32, A3l, B2l), A3h, B2h);
                        ab33 = svdot_u32(svdot_u32(ab33, A3l, B3l), A3h, B3h);
                    }
                }
                if (M > 0) DecodeCosineDistances1x4(A[0], B, ab00, ab01, ab02, ab03, distances + 0 * stride);
                if (M > 1) DecodeCosineDistances1x4(A[1], B, ab10, ab11, ab12, ab13, distances + 1 * stride);
                if (M > 2) DecodeCosineDistances1x4(A[2], B, ab20, ab21, ab22, ab23, distances + 2 * stride);
                if (M > 3) DecodeCosineDistances1x4(A[3], B, ab30, ab31, ab32, ab33, distances + 3 * stride);
            }
        };

        template<> struct DirectMx4<8>
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
            {
                const size_t width = svcntb();
                const size_t main = size & ~(width - 1);
                const svbool_t all = svptrue_b8();
                svuint32_t ab00 = svdup_n_u32(0), ab01 = ab00, ab02 = ab00, ab03 = ab00;
                svuint32_t ab10 = ab00, ab11 = ab00, ab12 = ab00, ab13 = ab00;
                svuint32_t ab20 = ab00, ab21 = ab00, ab22 = ab00, ab23 = ab00;
                svuint32_t ab30 = ab00, ab31 = ab00, ab32 = ab00, ab33 = ab00;
                const uint8_t* a0 = A[0] + 16;
                const uint8_t* a1 = M > 1 ? A[1] + 16 : a0;
                const uint8_t* a2 = M > 2 ? A[2] + 16 : a0;
                const uint8_t* a3 = M > 3 ? A[3] + 16 : a0;
                const uint8_t* b0 = B[0] + 16;
                const uint8_t* b1 = B[1] + 16;
                const uint8_t* b2 = B[2] + 16;
                const uint8_t* b3 = B[3] + 16;
                size_t i = 0;
                for (; i < main; i += width)
                {
                    svuint8_t B0 = svld1_u8(all, b0 + i);
                    svuint8_t B1 = svld1_u8(all, b1 + i);
                    svuint8_t B2 = svld1_u8(all, b2 + i);
                    svuint8_t B3 = svld1_u8(all, b3 + i);
                    if (M > 0)
                    {
                        svuint8_t A0 = svld1_u8(all, a0 + i);
                        ab00 = svdot_u32(ab00, A0, B0);
                        ab01 = svdot_u32(ab01, A0, B1);
                        ab02 = svdot_u32(ab02, A0, B2);
                        ab03 = svdot_u32(ab03, A0, B3);
                    }
                    if (M > 1)
                    {
                        svuint8_t A1 = svld1_u8(all, a1 + i);
                        ab10 = svdot_u32(ab10, A1, B0);
                        ab11 = svdot_u32(ab11, A1, B1);
                        ab12 = svdot_u32(ab12, A1, B2);
                        ab13 = svdot_u32(ab13, A1, B3);
                    }
                    if (M > 2)
                    {
                        svuint8_t A2 = svld1_u8(all, a2 + i);
                        ab20 = svdot_u32(ab20, A2, B0);
                        ab21 = svdot_u32(ab21, A2, B1);
                        ab22 = svdot_u32(ab22, A2, B2);
                        ab23 = svdot_u32(ab23, A2, B3);
                    }
                    if (M > 3)
                    {
                        svuint8_t A3 = svld1_u8(all, a3 + i);
                        ab30 = svdot_u32(ab30, A3, B0);
                        ab31 = svdot_u32(ab31, A3, B1);
                        ab32 = svdot_u32(ab32, A3, B2);
                        ab33 = svdot_u32(ab33, A3, B3);
                    }
                }
                if (i < size)
                {
                    svbool_t mask = svwhilelt_b8(i, size);
                    svuint8_t B0 = svld1_u8(mask, b0 + i);
                    svuint8_t B1 = svld1_u8(mask, b1 + i);
                    svuint8_t B2 = svld1_u8(mask, b2 + i);
                    svuint8_t B3 = svld1_u8(mask, b3 + i);
                    if (M > 0)
                    {
                        svuint8_t A0 = svld1_u8(mask, a0 + i);
                        ab00 = svdot_u32(ab00, A0, B0);
                        ab01 = svdot_u32(ab01, A0, B1);
                        ab02 = svdot_u32(ab02, A0, B2);
                        ab03 = svdot_u32(ab03, A0, B3);
                    }
                    if (M > 1)
                    {
                        svuint8_t A1 = svld1_u8(mask, a1 + i);
                        ab10 = svdot_u32(ab10, A1, B0);
                        ab11 = svdot_u32(ab11, A1, B1);
                        ab12 = svdot_u32(ab12, A1, B2);
                        ab13 = svdot_u32(ab13, A1, B3);
                    }
                    if (M > 2)
                    {
                        svuint8_t A2 = svld1_u8(mask, a2 + i);
                        ab20 = svdot_u32(ab20, A2, B0);
                        ab21 = svdot_u32(ab21, A2, B1);
                        ab22 = svdot_u32(ab22, A2, B2);
                        ab23 = svdot_u32(ab23, A2, B3);
                    }
                    if (M > 3)
                    {
                        svuint8_t A3 = svld1_u8(mask, a3 + i);
                        ab30 = svdot_u32(ab30, A3, B0);
                        ab31 = svdot_u32(ab31, A3, B1);
                        ab32 = svdot_u32(ab32, A3, B2);
                        ab33 = svdot_u32(ab33, A3, B3);
                    }
                }
                if (M > 0) DecodeCosineDistances1x4(A[0], B, ab00, ab01, ab02, ab03, distances + 0 * stride);
                if (M > 1) DecodeCosineDistances1x4(A[1], B, ab10, ab11, ab12, ab13, distances + 1 * stride);
                if (M > 2) DecodeCosineDistances1x4(A[2], B, ab20, ab21, ab22, ab23, distances + 2 * stride);
                if (M > 3) DecodeCosineDistances1x4(A[3], B, ab30, ab31, ab32, ab33, distances + 3 * stride);
            }
        };

        template<int bits> struct DirectMx1
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* B, size_t size, uint32_t* ab)
            {
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
                    if (M > 0) sums0 = svdot_u32(sums0, UnpackTo8<bits>(A[0] + 16 + packed, packedPerVec, tbl, shr), _b);
                    if (M > 1) sums1 = svdot_u32(sums1, UnpackTo8<bits>(A[1] + 16 + packed, packedPerVec, tbl, shr), _b);
                    if (M > 2) sums2 = svdot_u32(sums2, UnpackTo8<bits>(A[2] + 16 + packed, packedPerVec, tbl, shr), _b);
                    if (M > 3) sums3 = svdot_u32(sums3, UnpackTo8<bits>(A[3] + 16 + packed, packedPerVec, tbl, shr), _b);
                }
                if (i < size)
                {
                    size_t packedTail = (size - i) * bits / 8;
                    svuint8_t _b = UnpackTo8<bits>(b + packed, packedTail, tbl, shr);
                    if (M > 0) sums0 = svdot_u32(sums0, UnpackTo8<bits>(A[0] + 16 + packed, packedTail, tbl, shr), _b);
                    if (M > 1) sums1 = svdot_u32(sums1, UnpackTo8<bits>(A[1] + 16 + packed, packedTail, tbl, shr), _b);
                    if (M > 2) sums2 = svdot_u32(sums2, UnpackTo8<bits>(A[2] + 16 + packed, packedTail, tbl, shr), _b);
                    if (M > 3) sums3 = svdot_u32(sums3, UnpackTo8<bits>(A[3] + 16 + packed, packedTail, tbl, shr), _b);
                }
                if (M > 0) ab[0] = (uint32_t)svaddv_u32(svptrue_b32(), sums0);
                if (M > 1) ab[1] = (uint32_t)svaddv_u32(svptrue_b32(), sums1);
                if (M > 2) ab[2] = (uint32_t)svaddv_u32(svptrue_b32(), sums2);
                if (M > 3) ab[3] = (uint32_t)svaddv_u32(svptrue_b32(), sums3);
            }
        };

        template<> struct DirectMx1<4>
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* B, size_t size, uint32_t* ab)
            {
                const size_t width = svcntb();
                const size_t byteSize = size / 2;
                const size_t main = byteSize & ~(width - 1);
                const svbool_t all = svptrue_b8();
                svuint32_t sums0 = svdup_n_u32(0), sums1 = sums0, sums2 = sums0, sums3 = sums0;
                const uint8_t* b = B + 16;
                size_t i = 0;
                for (; i < main; i += width)
                {
                    svuint8_t _b = svld1_u8(all, b + i);
                    svuint8_t b0 = svand_n_u8_x(all, _b, 0x0F);
                    svuint8_t b1 = svlsr_n_u8_x(all, _b, 4);
                    if (M > 0)
                    {
                        svuint8_t _a = svld1_u8(all, A[0] + 16 + i);
                        sums0 = svdot_u32(svdot_u32(sums0, svand_n_u8_x(all, _a, 0x0F), b0), svlsr_n_u8_x(all, _a, 4), b1);
                    }
                    if (M > 1)
                    {
                        svuint8_t _a = svld1_u8(all, A[1] + 16 + i);
                        sums1 = svdot_u32(svdot_u32(sums1, svand_n_u8_x(all, _a, 0x0F), b0), svlsr_n_u8_x(all, _a, 4), b1);
                    }
                    if (M > 2)
                    {
                        svuint8_t _a = svld1_u8(all, A[2] + 16 + i);
                        sums2 = svdot_u32(svdot_u32(sums2, svand_n_u8_x(all, _a, 0x0F), b0), svlsr_n_u8_x(all, _a, 4), b1);
                    }
                    if (M > 3)
                    {
                        svuint8_t _a = svld1_u8(all, A[3] + 16 + i);
                        sums3 = svdot_u32(svdot_u32(sums3, svand_n_u8_x(all, _a, 0x0F), b0), svlsr_n_u8_x(all, _a, 4), b1);
                    }
                }
                if (i < byteSize)
                {
                    svbool_t mask = svwhilelt_b8(i, byteSize);
                    svuint8_t _b = svld1_u8(mask, b + i);
                    svuint8_t b0 = svand_n_u8_z(mask, _b, 0x0F);
                    svuint8_t b1 = svlsr_n_u8_z(mask, _b, 4);
                    if (M > 0)
                    {
                        svuint8_t _a = svld1_u8(mask, A[0] + 16 + i);
                        sums0 = svdot_u32(svdot_u32(sums0, svand_n_u8_z(mask, _a, 0x0F), b0), svlsr_n_u8_z(mask, _a, 4), b1);
                    }
                    if (M > 1)
                    {
                        svuint8_t _a = svld1_u8(mask, A[1] + 16 + i);
                        sums1 = svdot_u32(svdot_u32(sums1, svand_n_u8_z(mask, _a, 0x0F), b0), svlsr_n_u8_z(mask, _a, 4), b1);
                    }
                    if (M > 2)
                    {
                        svuint8_t _a = svld1_u8(mask, A[2] + 16 + i);
                        sums2 = svdot_u32(svdot_u32(sums2, svand_n_u8_z(mask, _a, 0x0F), b0), svlsr_n_u8_z(mask, _a, 4), b1);
                    }
                    if (M > 3)
                    {
                        svuint8_t _a = svld1_u8(mask, A[3] + 16 + i);
                        sums3 = svdot_u32(svdot_u32(sums3, svand_n_u8_z(mask, _a, 0x0F), b0), svlsr_n_u8_z(mask, _a, 4), b1);
                    }
                }
                if (M > 0) ab[0] = (uint32_t)svaddv_u32(svptrue_b32(), sums0);
                if (M > 1) ab[1] = (uint32_t)svaddv_u32(svptrue_b32(), sums1);
                if (M > 2) ab[2] = (uint32_t)svaddv_u32(svptrue_b32(), sums2);
                if (M > 3) ab[3] = (uint32_t)svaddv_u32(svptrue_b32(), sums3);
            }
        };

        template<> struct DirectMx1<8>
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* B, size_t size, uint32_t* ab)
            {
                const size_t width = svcntb();
                const size_t main = size & ~(width - 1);
                const svbool_t all = svptrue_b8();
                svuint32_t sums0 = svdup_n_u32(0), sums1 = sums0, sums2 = sums0, sums3 = sums0;
                const uint8_t* b = B + 16;
                size_t i = 0;
                for (; i < main; i += width)
                {
                    svuint8_t _b = svld1_u8(all, b + i);
                    if (M > 0) sums0 = svdot_u32(sums0, svld1_u8(all, A[0] + 16 + i), _b);
                    if (M > 1) sums1 = svdot_u32(sums1, svld1_u8(all, A[1] + 16 + i), _b);
                    if (M > 2) sums2 = svdot_u32(sums2, svld1_u8(all, A[2] + 16 + i), _b);
                    if (M > 3) sums3 = svdot_u32(sums3, svld1_u8(all, A[3] + 16 + i), _b);
                }
                if (i < size)
                {
                    svbool_t mask = svwhilelt_b8(i, size);
                    svuint8_t _b = svld1_u8(mask, b + i);
                    if (M > 0) sums0 = svdot_u32(sums0, svld1_u8(mask, A[0] + 16 + i), _b);
                    if (M > 1) sums1 = svdot_u32(sums1, svld1_u8(mask, A[1] + 16 + i), _b);
                    if (M > 2) sums2 = svdot_u32(sums2, svld1_u8(mask, A[2] + 16 + i), _b);
                    if (M > 3) sums3 = svdot_u32(sums3, svld1_u8(mask, A[3] + 16 + i), _b);
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
            DirectMx1<bits>::template Run<M>(A, B, size, ab);
            if (M > 0) Base::DecodeCosineDistance(A[0], B, (float)ab[0], distances + 0 * stride);
            if (M > 1) Base::DecodeCosineDistance(A[1], B, (float)ab[1], distances + 1 * stride);
            if (M > 2) Base::DecodeCosineDistance(A[2], B, (float)ab[2], distances + 2 * stride);
            if (M > 3) Base::DecodeCosineDistance(A[3], B, (float)ab[3], distances + 3 * stride);
        }

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M4 = AlignLoAny(M, 4), N4 = AlignLo(N, 4), i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    DirectMx4<bits>::template Run<4>(A + i, B + j, size, distances + j, stride);
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
                        DirectMx4<bits>::template Run<1>(A + i, B + j, size, distances + j, stride);
                    else if (m == 2)
                        DirectMx4<bits>::template Run<2>(A + i, B + j, size, distances + j, stride);
                    else
                        DirectMx4<bits>::template Run<3>(A + i, B + j, size, distances + j, stride);
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
#ifdef SIMD_NEON_ENABLE
            default: return Neon::GetCosineDistance(depth);
#else
            default: return Base::GetCosineDistance(depth);
#endif
            }
        }

        Base::DescrInt::MacroCosineDistancesDirectPtr GetMacroCosineDistancesDirect(size_t depth)
        {
            switch (depth)
            {
            case 4: return MacroCosineDistancesDirect<4>;
            case 5: return MacroCosineDistancesDirect<5>;
            case 6: return MacroCosineDistancesDirect<6>;
            case 7: return MacroCosineDistancesDirect<7>;
            case 8: return MacroCosineDistancesDirect<8>;
            default: return NULL;
            }
        }
    }
#endif
}
