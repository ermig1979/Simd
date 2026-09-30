/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_HVX_ENABLE
    namespace Hvx
    {
        namespace
        {
            template<class T> struct Buffer
            {
                Buffer(size_t rowSize, size_t histogramSize)
                {
                    _p = Allocate(sizeof(T) * rowSize + 4 * sizeof(uint32_t) * histogramSize);
                    v = (T*)_p;
                    h[0] = (uint32_t*)(v + rowSize);
                    h[1] = h[0] + histogramSize;
                    h[2] = h[1] + histogramSize;
                    h[3] = h[2] + histogramSize;
                    memset(h[0], 0, 4 * sizeof(uint32_t) * histogramSize);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                T* v;
                uint32_t* h[4];
            private:
                void* _p;
            };
        }

        SIMD_INLINE void SumHistograms(uint32_t* src, size_t start, uint32_t* dst)
        {
            uint32_t* src0 = src + start;
            uint32_t* src1 = src0 + start + HISTOGRAM_SIZE;
            uint32_t* src2 = src1 + start + HISTOGRAM_SIZE;
            uint32_t* src3 = src2 + start + HISTOGRAM_SIZE;
            for (size_t i = 0; i < HISTOGRAM_SIZE; i += A / sizeof(uint32_t))
            {
                HVX_Vector s0 = Load<false>((const uint8_t*)(src0 + i));
                HVX_Vector s1 = Load<false>((const uint8_t*)(src1 + i));
                HVX_Vector s2 = Load<false>((const uint8_t*)(src2 + i));
                HVX_Vector s3 = Load<false>((const uint8_t*)(src3 + i));
                HVX_Vector sum = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(s0, s1), Q6_Vw_vadd_VwVw(s2, s3));
                Store<false>((uint8_t*)(dst + i), sum);
            }
        }

        //-----------------------------------------------------------------------

        template <SimdCompareType compareType, bool srcAlign>
        SIMD_INLINE void ConditionalSrc(const uint8_t* src, const uint8_t* mask, const HVX_Vector& value, ptrdiff_t offset, uint16_t* dst)
        {
            const HVX_Vector _src = Load<srcAlign>(src + offset);
            const HVX_VectorPred cmp = Compare8u<compareType>(Load<srcAlign>(mask + offset), value);
            const HVX_Vector one = Q6_V_vsplat_R(0x01010101);
            const HVX_Vector _mask = Q6_V_vmux_QVV(cmp, one, Q6_V_vzero());
            const HVX_Vector K16_0004 = Q6_V_vsplat_R(0x00040004);
            HVX_VectorPair srcU16 = Q6_Wuh_vunpack_Vub(_src);
            HVX_Vector srcLo = Q6_Vh_vadd_VhVh(Q6_V_lo_W(srcU16), K16_0004);
            HVX_Vector srcHi = Q6_Vh_vadd_VhVh(Q6_V_hi_W(srcU16), K16_0004);
            HVX_VectorPair maskU16 = Q6_Wuh_vunpack_Vub(_mask);
            HVX_Vector resLo = Q6_Vh_vmpyi_VhVh(srcLo, Q6_V_lo_W(maskU16));
            HVX_Vector resHi = Q6_Vh_vmpyi_VhVh(srcHi, Q6_V_hi_W(maskU16));
            Store<false>((uint8_t*)(dst + offset), resLo);
            Store<false>((uint8_t*)(dst + offset + A / 2), resHi);
        }

        template<SimdCompareType compareType, bool align> void HistogramConditional(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, uint32_t* histogram)
        {
            Buffer<uint16_t> buffer(AlignHi(width, A), HISTOGRAM_SIZE + 4);
            size_t widthAligned4 = Simd::AlignLo(width, 4);
            size_t widthAlignedA = Simd::AlignLo(width, A);
            HVX_Vector _value = Q6_V_vsplat_R(
                (uint32_t)value | ((uint32_t)value << 8) | ((uint32_t)value << 16) | ((uint32_t)value << 24));
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthAlignedA; col += A)
                    ConditionalSrc<compareType, align>(src, mask, _value, col, buffer.v);
                if (width != widthAlignedA)
                    ConditionalSrc<compareType, false>(src, mask, _value, width - A, buffer.v);

                for (col = 0; col < widthAligned4; col += 4)
                {
                    ++buffer.h[0][buffer.v[col + 0]];
                    ++buffer.h[1][buffer.v[col + 1]];
                    ++buffer.h[2][buffer.v[col + 2]];
                    ++buffer.h[3][buffer.v[col + 3]];
                }
                for (; col < width; ++col)
                    ++buffer.h[0][buffer.v[col]];

                src += srcStride;
                mask += maskStride;
            }

            SumHistograms(buffer.h[0], 4, histogram);
        }

        template <SimdCompareType compareType>
        void HistogramConditional(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, uint32_t* histogram)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                return HistogramConditional<compareType, true>(src, srcStride, width, height, mask, maskStride, value, histogram);
            else
                return HistogramConditional<compareType, false>(src, srcStride, width, height, mask, maskStride, value, histogram);
        }

        void HistogramConditional(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t* histogram)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return HistogramConditional<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareNotEqual:
                return HistogramConditional<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareGreater:
                return HistogramConditional<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareGreaterOrEqual:
                return HistogramConditional<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareLesser:
                return HistogramConditional<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareLesserOrEqual:
                return HistogramConditional<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            default:
                assert(0);
            }
        }
    }
#endif
}
