/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_HVX_ENABLE
    namespace Hvx
    {
        template <SimdOperationBinary8uType type> SIMD_INLINE HVX_Vector OperationBinary8u(const HVX_Vector& a, const HVX_Vector& b);

        template <> SIMD_INLINE HVX_Vector OperationBinary8u<SimdOperationBinary8uAverage>(const HVX_Vector& a, const HVX_Vector& b)
        {
            return Q6_Vub_vavg_VubVub_rnd(a, b);
        }

        template <> SIMD_INLINE HVX_Vector OperationBinary8u<SimdOperationBinary8uAnd>(const HVX_Vector& a, const HVX_Vector& b)
        {
            return Q6_V_vand_VV(a, b);
        }

        template <> SIMD_INLINE HVX_Vector OperationBinary8u<SimdOperationBinary8uOr>(const HVX_Vector& a, const HVX_Vector& b)
        {
            return Q6_V_vor_VV(a, b);
        }

        template <> SIMD_INLINE HVX_Vector OperationBinary8u<SimdOperationBinary8uMaximum>(const HVX_Vector& a, const HVX_Vector& b)
        {
            return Q6_Vub_vmax_VubVub(a, b);
        }

        template <> SIMD_INLINE HVX_Vector OperationBinary8u<SimdOperationBinary8uMinimum>(const HVX_Vector& a, const HVX_Vector& b)
        {
            return Q6_Vub_vmin_VubVub(a, b);
        }

        template <> SIMD_INLINE HVX_Vector OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(const HVX_Vector& a, const HVX_Vector& b)
        {
            return Q6_Vub_vsub_VubVub_sat(a, b);
        }

        template <> SIMD_INLINE HVX_Vector OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(const HVX_Vector& a, const HVX_Vector& b)
        {
            return Q6_Vub_vadd_VubVub_sat(a, b);
        }

        template <bool align, SimdOperationBinary8uType type> void OperationBinary8u(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            size_t size = channelCount * width;
            assert(size >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedSize = AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t offset = 0; offset < alignedSize; offset += A)
                {
                    const HVX_Vector a_ = Load<align>(a + offset);
                    const HVX_Vector b_ = Load<align>(b + offset);
                    Store<align>(dst + offset, OperationBinary8u<type>(a_, b_));
                }
                if (alignedSize != size)
                {
                    const HVX_Vector a_ = Load<false>(a + size - A);
                    const HVX_Vector b_ = Load<false>(b + size - A);
                    Store<false>(dst + size - A, OperationBinary8u<type>(a_, b_));
                }
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        template <bool align> void OperationBinary8u(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride, SimdOperationBinary8uType type)
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

        void OperationBinary8u(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
                OperationBinary8u<true>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
            else
                OperationBinary8u<false>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
        }
    }
#endif
}
