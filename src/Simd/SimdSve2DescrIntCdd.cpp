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

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            for (size_t i = 0; i < M; ++i)
            {
                const uint8_t* a = A[i];
                for (size_t j = 0; j < N; ++j)
                    CosineDistance<bits>(a, B[j], size, distances + i * stride + j);
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
