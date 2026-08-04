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
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE  
    namespace Neon
    {
        template <bool align> SIMD_INLINE void CreateMask(const uint16x8_t & vertical, const uint8_t * horizontal, uint8_t * dst)
        {
            uint8x16x2_t _horizontal = vzipq_u8(Load<align>(horizontal), K8_00);
            _horizontal.val[0] = (uint8x16_t)DivideI16By255(vmulq_u16(vertical, (uint16x8_t)_horizontal.val[0]));
            _horizontal.val[1] = (uint8x16_t)DivideI16By255(vmulq_u16(vertical, (uint16x8_t)_horizontal.val[1]));
            Store<align>(dst, vuzpq_u8(_horizontal.val[0], _horizontal.val[1]).val[0]);
        }

        template <bool align> void CreateMask(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(horizontal) && Aligned(dst) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                uint16x8_t _vertical = vmovq_n_u16(vertical[row]);
                for (size_t col = 0; col < alignedWidth; col += A)
                    CreateMask<align>(_vertical, horizontal + col, dst + col);
                if (alignedWidth != width)
                    CreateMask<false>(_vertical, horizontal + width - A, dst + width - A);
                dst += stride;
            }
        }

        void CreateMask(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            if (Aligned(horizontal) && Aligned(dst) && Aligned(stride))
                CreateMask<true>(vertical, horizontal, dst, stride, width, height);
            else
                CreateMask<false>(vertical, horizontal, dst, stride, width, height);
        }
    }
#endif
}
