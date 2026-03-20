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
        template <bool align> SIMD_INLINE HVX_Vector AbsGradientSaturatedSum(const uint8_t* src, size_t stride)
        {
            const HVX_Vector dx = Q6_Vub_vabsdiff_VubVub(Load<false>(src + 1), Load<false>(src - 1));
            const HVX_Vector dy = Q6_Vub_vabsdiff_VubVub(Load<align>(src + stride), Load<align>(src - stride));
            return Q6_Vub_vadd_VubVub_sat(dx, dy);
        }

        template <bool align> void AbsGradientSaturatedSum(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t* dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            memset(dst, 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 2; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    Store<align>(dst + col, AbsGradientSaturatedSum<align>(src + col, srcStride));
                if (width != alignedWidth)
                    Store<false>(dst + width - A, AbsGradientSaturatedSum<false>(src + width - A, srcStride));

                dst[0] = 0;
                dst[width - 1] = 0;

                src += srcStride;
                dst += dstStride;
            }
            memset(dst, 0, width);
        }

        void AbsGradientSaturatedSum(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AbsGradientSaturatedSum<true>(src, srcStride, width, height, dst, dstStride);
            else
                AbsGradientSaturatedSum<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif
}
