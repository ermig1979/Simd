/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t size8 = Simd::AlignLo(size, 8);
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size8; i += 8)
            {
                uint32x4_t d0 = Float32ToBFloat16(Load<align>(src + i + 0));
                uint32x4_t d1 = Float32ToBFloat16(Load<align>(src + i + 4));
                Store<align>(dst + i, PackU32(d0, d1));
            }
            for (; i < size4; i += 4)
            {
                uint32x4_t d0 = Float32ToBFloat16(Load<align>(src + i + 0));
                Store<align>(dst + i, vmovn_u32(d0));
            }
            for (; i < size; ++i)
                dst[i] = Base::Float32ToBFloat16(src[i]);
        }

        void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst)
        {
            if (Aligned(src) && Aligned(dst))
                Float32ToBFloat16<true>(src, size, dst);
            else
                Float32ToBFloat16<false>(src, size, dst);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t size8 = Simd::AlignLo(size, 8);
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size8; i += 8)
            {
                uint16x8_t s = Load<align>(src + i);
                Store<align>(dst + i + 0, BFloat16ToFloat32(UnpackU16<0>(s)));
                Store<align>(dst + i + 4, BFloat16ToFloat32(UnpackU16<1>(s)));
            }
            for (; i < size4; i += 4)
            {
                uint16x4_t s = LoadHalf<align>(src + i);
                Store<align>(dst + i, BFloat16ToFloat32(vmovl_u16(s)));
            }
            for (; i < size; ++i)
                dst[i] = Base::BFloat16ToFloat32(src[i]);
        }

        void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                BFloat16ToFloat32<true>(src, size, dst);
            else
                BFloat16ToFloat32<false>(src, size, dst);
        }
    }
#endif
}
