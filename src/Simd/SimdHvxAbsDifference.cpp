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
        template <bool align> void AbsDifference(
            const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, uint8_t* c, size_t cStride,
            size_t width, size_t height)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(c) && Aligned(cStride));

            size_t bodyWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const HVX_Vector a_ = Load<align>(a + col);
                    const HVX_Vector b_ = Load<align>(b + col);
                    Store<align>(c + col, Q6_Vub_vabsdiff_VubVub(a_, b_));
                }
                if (width - bodyWidth)
                {
                    const HVX_Vector a_ = Load<false>(a + width - A);
                    const HVX_Vector b_ = Load<false>(b + width - A);
                    Store<false>(c + width - A, Q6_Vub_vabsdiff_VubVub(a_, b_));
                }
                a += aStride;
                b += bStride;
                c += cStride;
            }
        }

        void AbsDifference(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, uint8_t* c, size_t cStride,
            size_t width, size_t height)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(c) && Aligned(cStride))
                AbsDifference<true>(a, aStride, b, bStride, c, cStride, width, height);
            else
                AbsDifference<false>(a, aStride, b, bStride, c, cStride, width, height);
        }
    }
#endif
}
