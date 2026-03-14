/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_HVX_ENABLE
    namespace Hvx
    {
        template <bool align> void FillBgra(uint8_t* dst, size_t stride, size_t width, size_t height,
            uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
        {
            size_t size = width * 4;
            size_t alignedSize = AlignLo(size, A);
            uint32_t pixel = blue | (green << 8) | (red << 16) | (alpha << 24);
            HVX_Vector _pixel = Q6_V_vsplat_R(pixel);

            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < alignedSize; offset += A)
                    Store<align>(dst + offset, _pixel);
                if (offset < size)
                    Store<false>(dst + size - A, _pixel);
                dst += stride;
            }
        }

        void FillBgra(uint8_t* dst, size_t stride, size_t width, size_t height,
            uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
        {
            if (Aligned(dst) && Aligned(stride))
                FillBgra<true>(dst, stride, width, height, blue, green, red, alpha);
            else
                FillBgra<false>(dst, stride, width, height, blue, green, red, alpha);
        }

        template <bool align> void FillPixel(uint8_t* dst, size_t stride, size_t width, size_t height,
            const uint8_t* pixel, size_t pixelSize)
        {
            if (pixelSize == 3)
            {
                Base::Fill(dst, stride, width, height, 3, 0);
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                    {
                        dst[col * 3 + 0] = pixel[0];
                        dst[col * 3 + 1] = pixel[1];
                        dst[col * 3 + 2] = pixel[2];
                    }
                    dst += stride;
                }
                return;
            }

            uint32_t val32;
            switch (pixelSize)
            {
            case 1:
                Base::Fill(dst, stride, width, height, 1, pixel[0]);
                return;
            case 2:
                val32 = *(uint16_t*)pixel;
                val32 = val32 | (val32 << 16);
                break;
            case 4:
                val32 = *(uint32_t*)pixel;
                break;
            default:
                assert(0);
                return;
            }

            HVX_Vector _pixel = Q6_V_vsplat_R(val32);
            size_t size = width * pixelSize;
            size_t alignedSize = AlignLo(size, A);

            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < alignedSize; offset += A)
                    Store<align>(dst + offset, _pixel);
                if (offset < size)
                    Store<false>(dst + size - A, _pixel);
                dst += stride;
            }
        }

        void FillPixel(uint8_t* dst, size_t stride, size_t width, size_t height,
            const uint8_t* pixel, size_t pixelSize)
        {
            if (Aligned(dst) && Aligned(stride))
                FillPixel<true>(dst, stride, width, height, pixel, pixelSize);
            else
                FillPixel<false>(dst, stride, width, height, pixel, pixelSize);
        }
    }
#endif
}
