/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE uint32x4_t Encode32f(float32x4_t src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t value = (uint32x4_t)Round(vmulq_f32(vsubq_f32(src, min), scale));
            sum = vaddq_u32(value, sum);
            sqsum = vmlaq_u32(sqsum, value, value);
            return value;
        }

        template<bool align> SIMD_INLINE uint32x4_t Encode32f(const float* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            return Encode32f(Load<align>(src), scale, min, sum, sqsum);
        }

        static void Encode32f8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeA = AlignLo(size, A), i = 0;
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _min = vdupq_n_f32(min);
            uint32x4_t _sum = K32_00000000;
            uint32x4_t _sqsum = K32_00000000;
            if (Aligned(src))
            {
                for (; i < sizeA; i += A)
                {
                    uint32x4_t d0 = Encode32f<true>(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d1 = Encode32f<true>(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d2 = Encode32f<true>(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d3 = Encode32f<true>(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                    Store<false>(dst + i, PackU16(PackU32(d0, d1), PackU32(d2, d3)));
                }
            }
            else
            {
                for (; i < sizeA; i += A)
                {
                    uint32x4_t d0 = Encode32f<false>(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d1 = Encode32f<false>(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d2 = Encode32f<false>(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d3 = Encode32f<false>(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                    Store<false>(dst + i, PackU16(PackU32(d0, d1), PackU32(d2, d3)));
                }
            }
            for (; i < size; i += F)
            {
                uint32x4_t d0 = Encode32f<false>(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                uint32x4_t d1 = Encode32f<false>(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                Store<false>(dst + i, Half<0>(PackU16(PackU32(d0, d1), K16_0000)));
            }
            sum = ExtractSum32u(_sum);
            sqsum = ExtractSum32u(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::Encode32fPtr GetEncode32f(size_t depth)
        {
            switch (depth)
            {
            //case 4: return Encode32f4;
            //case 5: return Encode32f5;
            //case 6: return Encode32f6;
            //case 7: return Encode32f7;
            case 8: return Encode32f8;
            default: assert(0); return NULL;
            }
        }

        Base::DescrInt::Encode16fPtr GetEncode16f(size_t depth)
        {
            switch (depth)
            {
            //case 4: return Encode16f4;
            //case 5: return Encode16f5;
            //case 6: return Encode16f6;
            //case 7: return Encode16f7;
            //case 8: return Encode16f8;
            default: assert(0); return NULL;
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
