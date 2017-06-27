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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> SIMD_INLINE uint32x4_t Float32ToUint32(const float * src, const float32x4_t & lower, const float32x4_t & upper, const float32x4_t & boost)
        {
            return vcvtq_u32_f32(vmulq_f32(vsubq_f32(vminq_f32(vmaxq_f32(Load<align>(src), lower), upper), lower), boost));
        }

        template <bool align> SIMD_INLINE void Float32ToUint8(const float * src, const float32x4_t & lower, const float32x4_t & upper, const float32x4_t & boost, uint8_t * dst)
        {
            uint32x4_t d0 = Float32ToUint32<align>(src + F * 0, lower, upper, boost);
            uint32x4_t d1 = Float32ToUint32<align>(src + F * 1, lower, upper, boost);
            uint32x4_t d2 = Float32ToUint32<align>(src + F * 2, lower, upper, boost);
            uint32x4_t d3 = Float32ToUint32<align>(src + F * 3, lower, upper, boost);
            Store<align>(dst, PackU16(PackU32(d0, d1), PackU32(d2, d3)));
        }

        template <bool align> void Float32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst)
        {
            assert(size >= A);
            if (align)
                assert(Aligned(src) && Aligned(dst));

            float32x4_t _lower = vdupq_n_f32(lower[0]);
            float32x4_t _upper = vdupq_n_f32(upper[0]);
            float32x4_t boost = vdupq_n_f32(255.0f / (upper[0] - lower[0]));

            size_t alignedSize = AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                Float32ToUint8<align>(src + i, _lower, _upper, boost, dst + i);
            if (alignedSize != size)
                Float32ToUint8<false>(src + size - A, _lower, _upper, boost, dst + size - A);
        }

        void Float32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Float32ToUint8<true>(src, size, lower, upper, dst);
            else
                Float32ToUint8<false>(src, size, lower, upper, dst);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
