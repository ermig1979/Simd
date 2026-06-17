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
        template<int bits> SIMD_INLINE uint64_t LoadBlock(const uint8_t* src)
        {
            uint64_t value = 0;
            for (size_t i = 0; i < bits; ++i)
                value |= uint64_t(src[i]) << (8 * i);
            return value;
        }

        template<int bits> SIMD_INLINE uint64_t CorrelationBlock(uint64_t a, uint64_t b)
        {
            const uint64_t mask = (uint64_t(1) << bits) - 1;
            uint64_t sum = 0;
            for (size_t i = 0; i < 8; ++i)
                sum += ((a >> (i * bits)) & mask) * ((b >> (i * bits)) & mask);
            return sum;
        }

        template<int bits> SIMD_INLINE svuint64_t CorrelationBlock(const svuint64_t& a, const svuint64_t& b, const svbool_t& mask)
        {
            const uint64_t valueMask = (uint64_t(1) << bits) - 1;
            svuint64_t sum = svdup_n_u64(0);
#define SIMD_SVE2_DESCR_INT_CORRELATE(shift) \
            { \
                svuint64_t _a = svand_n_u64_x(mask, svlsr_n_u64_x(mask, a, shift), valueMask); \
                svuint64_t _b = svand_n_u64_x(mask, svlsr_n_u64_x(mask, b, shift), valueMask); \
                sum = svmla_u64_m(mask, sum, _a, _b); \
            }
            SIMD_SVE2_DESCR_INT_CORRELATE(0 * bits);
            SIMD_SVE2_DESCR_INT_CORRELATE(1 * bits);
            SIMD_SVE2_DESCR_INT_CORRELATE(2 * bits);
            SIMD_SVE2_DESCR_INT_CORRELATE(3 * bits);
            SIMD_SVE2_DESCR_INT_CORRELATE(4 * bits);
            SIMD_SVE2_DESCR_INT_CORRELATE(5 * bits);
            SIMD_SVE2_DESCR_INT_CORRELATE(6 * bits);
            SIMD_SVE2_DESCR_INT_CORRELATE(7 * bits);
#undef SIMD_SVE2_DESCR_INT_CORRELATE
            return sum;
        }

        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0 && size >= 8);
            size_t blocks = size / 8, vectorBlocks = blocks - 1, i = 0;
            svuint64_t sums = svdup_n_u64(0);
            for (; i < vectorBlocks; i += svcntd())
            {
                svbool_t mask = svwhilelt_b64(i, vectorBlocks);
                svuint64_t offsets = svindex_u64(uint64_t(i * bits), bits);
                svuint64_t _a = svld1_gather_u64offset_u64(mask, (const uint64_t*)a, offsets);
                svuint64_t _b = svld1_gather_u64offset_u64(mask, (const uint64_t*)b, offsets);
                sums = svadd_u64_m(mask, sums, CorrelationBlock<bits>(_a, _b, mask));
            }
            uint64_t sum = svaddv_u64(svptrue_b64(), sums);
            for (size_t tail = vectorBlocks; tail < blocks; ++tail)
                sum += CorrelationBlock<bits>(LoadBlock<bits>(a + tail * bits), LoadBlock<bits>(b + tail * bits));
            return (int32_t)sum;
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

        template<int bits> struct CorrelationsMx1
        {
            template<int M> static SIMD_INLINE void Run(const uint8_t* const* A, const uint8_t* B, size_t size, uint32_t* ab)
            {
                assert(size % 8 == 0 && size >= 8);
                size_t blocks = size / 8, vectorBlocks = blocks - 1, i = 0;
                svuint64_t sums0 = svdup_n_u64(0), sums1 = sums0, sums2 = sums0, sums3 = sums0;
                const uint8_t* b = B + 16;
                for (; i < vectorBlocks; i += svcntd())
                {
                    svbool_t mask = svwhilelt_b64(i, vectorBlocks);
                    svuint64_t offsets = svindex_u64(uint64_t(i * bits), bits);
                    svuint64_t _b = svld1_gather_u64offset_u64(mask, (const uint64_t*)b, offsets);
                    if (M > 0)
                    {
                        svuint64_t _a = svld1_gather_u64offset_u64(mask, (const uint64_t*)(A[0] + 16), offsets);
                        sums0 = svadd_u64_m(mask, sums0, CorrelationBlock<bits>(_a, _b, mask));
                    }
                    if (M > 1)
                    {
                        svuint64_t _a = svld1_gather_u64offset_u64(mask, (const uint64_t*)(A[1] + 16), offsets);
                        sums1 = svadd_u64_m(mask, sums1, CorrelationBlock<bits>(_a, _b, mask));
                    }
                    if (M > 2)
                    {
                        svuint64_t _a = svld1_gather_u64offset_u64(mask, (const uint64_t*)(A[2] + 16), offsets);
                        sums2 = svadd_u64_m(mask, sums2, CorrelationBlock<bits>(_a, _b, mask));
                    }
                    if (M > 3)
                    {
                        svuint64_t _a = svld1_gather_u64offset_u64(mask, (const uint64_t*)(A[3] + 16), offsets);
                        sums3 = svadd_u64_m(mask, sums3, CorrelationBlock<bits>(_a, _b, mask));
                    }
                }
                uint64_t ab0 = svaddv_u64(svptrue_b64(), sums0), ab1 = svaddv_u64(svptrue_b64(), sums1);
                uint64_t ab2 = svaddv_u64(svptrue_b64(), sums2), ab3 = svaddv_u64(svptrue_b64(), sums3);
                for (size_t tail = vectorBlocks; tail < blocks; ++tail)
                {
                    uint64_t _b = LoadBlock<bits>(b + tail * bits);
                    if (M > 0)
                        ab0 += CorrelationBlock<bits>(LoadBlock<bits>(A[0] + 16 + tail * bits), _b);
                    if (M > 1)
                        ab1 += CorrelationBlock<bits>(LoadBlock<bits>(A[1] + 16 + tail * bits), _b);
                    if (M > 2)
                        ab2 += CorrelationBlock<bits>(LoadBlock<bits>(A[2] + 16 + tail * bits), _b);
                    if (M > 3)
                        ab3 += CorrelationBlock<bits>(LoadBlock<bits>(A[3] + 16 + tail * bits), _b);
                }
                if (M > 0) ab[0] = (uint32_t)ab0;
                if (M > 1) ab[1] = (uint32_t)ab1;
                if (M > 2) ab[2] = (uint32_t)ab2;
                if (M > 3) ab[3] = (uint32_t)ab3;
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

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
