/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_MSA_ENABLE  
    namespace Msa
    {
        template <SimdOperationBinary8uType type> SIMD_INLINE v16u8 OperationBinary8u(const v16u8 & a, const v16u8 & b);

        template <> SIMD_INLINE v16u8 OperationBinary8u<SimdOperationBinary8uAverage>(const v16u8 & a, const v16u8 & b)
        {
            return __msa_aver_u_b(a, b);
        }

        template <> SIMD_INLINE v16u8 OperationBinary8u<SimdOperationBinary8uAnd>(const v16u8 & a, const v16u8 & b)
        {
            return __msa_and_v(a, b);
        }

        template <> SIMD_INLINE v16u8 OperationBinary8u<SimdOperationBinary8uOr>(const v16u8 & a, const v16u8 & b)
        {
            return __msa_or_v(a, b);
        }

        template <> SIMD_INLINE v16u8 OperationBinary8u<SimdOperationBinary8uMaximum>(const v16u8 & a, const v16u8 & b)
        {
            return __msa_max_u_b(a, b);
        }

        template <> SIMD_INLINE v16u8 OperationBinary8u<SimdOperationBinary8uMinimum>(const v16u8 & a, const v16u8 & b)
        {
            return __msa_min_u_b(a, b);
        }

        template <> SIMD_INLINE v16u8 OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(const v16u8 & a, const v16u8 & b)
        {
            return __msa_subs_u_b(a, b);
        }

        template <> SIMD_INLINE v16u8 OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(const v16u8 & a, const v16u8 & b)
        {
            return __msa_adds_u_b(a, b);
        }

        template <bool align, SimdOperationBinary8uType type> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(width*channelCount >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

            size_t size = channelCount*width;
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t offset = 0; offset < alignedSize; offset += A)
                {
                    const v16u8 a_ = Load<align>(a + offset);
                    const v16u8 b_ = Load<align>(b + offset);
                    Store<align>(dst + offset, OperationBinary8u<type>(a_, b_));
                }
                if (alignedSize != size)
                {
                    const v16u8 a_ = Load<false>(a + size - A);
                    const v16u8 b_ = Load<false>(b + size - A);
                    Store<false>(dst + size - A, OperationBinary8u<type>(a_, b_));
                }
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        template <bool align> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            switch (type)
            {
            case SimdOperationBinary8uAverage:
                return OperationBinary8u<align, SimdOperationBinary8uAverage>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uAnd:
                return OperationBinary8u<align, SimdOperationBinary8uAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uOr:
                return OperationBinary8u<align, SimdOperationBinary8uOr>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMaximum:
                return OperationBinary8u<align, SimdOperationBinary8uMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMinimum:
                return OperationBinary8u<align, SimdOperationBinary8uMinimum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedSubtraction:
                return OperationBinary8u<align, SimdOperationBinary8uSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedAddition:
                return OperationBinary8u<align, SimdOperationBinary8uSaturatedAddition>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            default:
                assert(0);
            }
        }

        void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
                OperationBinary8u<true>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
            else
                OperationBinary8u<false>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
        }

        template <SimdOperationBinary16iType type> SIMD_INLINE v8i16 OperationBinary16i(const v8i16 & a, const v8i16 & b);

        template <> SIMD_INLINE v8i16 OperationBinary16i<SimdOperationBinary16iAddition>(const v8i16 & a, const v8i16 & b)
        {
            return __msa_addv_h(a, b);
        }

        template <> SIMD_INLINE v8i16 OperationBinary16i<SimdOperationBinary16iSubtraction>(const v8i16 & a, const v8i16 & b)
        {
            return __msa_subv_h(a, b);
        }

        template <bool align, SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(width * sizeof(int16_t) >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

            size_t size = width * sizeof(int16_t);
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t offset = 0; offset < alignedSize; offset += A)
                {
                    const v8i16 a_ = (v8i16)Load<align>(a + offset);
                    const v8i16 b_ = (v8i16)Load<align>(b + offset);
                    Store<align>(dst + offset, (v16u8)OperationBinary16i<type>(a_, b_));
                }
                if (alignedSize != size)
                {
                    const v8i16 a_ = (v8i16)Load<false>(a + size - A);
                    const v8i16 b_ = (v8i16)Load<false>(b + size - A);
                    Store<false>(dst + size - A, (v16u8)OperationBinary16i<type>(a_, b_));
                }
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        template <bool align> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
        {
            switch (type)
            {
            case SimdOperationBinary16iAddition:
                return OperationBinary16i<align, SimdOperationBinary16iAddition>(a, aStride, b, bStride, width, height, dst, dstStride);
            case SimdOperationBinary16iSubtraction:
                return OperationBinary16i<align, SimdOperationBinary16iSubtraction>(a, aStride, b, bStride, width, height, dst, dstStride);
            default:
                assert(0);
            }
        }

        void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
                OperationBinary16i<true>(a, aStride, b, bStride, width, height, dst, dstStride, type);
            else
                OperationBinary16i<false>(a, aStride, b, bStride, width, height, dst, dstStride, type);
        }
    }
#endif// SIMD_MSA_ENABLE
}
