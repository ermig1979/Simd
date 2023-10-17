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
#include "Simd/SimdExtract.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE uint32x4_t Encode32f(float32x4_t src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t value = RoundPositive(vmulq_f32(vsubq_f32(src, min), scale));
            sum = vaddq_u32(value, sum);
            sqsum = vmlaq_u32(sqsum, value, value);
            return value;
        }

        template<bool align> SIMD_INLINE uint32x4_t Encode32f(const float* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            return Encode32f(Load<align>(src), scale, min, sum, sqsum);
        }

        template<bool align> static SIMD_INLINE uint16x4_t Encode32f4(const float* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t i0 = Encode32f<align>(src + 0, scale, min, sum, sqsum);
            uint32x4_t i4 = Encode32f<align>(src + 4, scale, min, sum, sqsum);
            return vmovn_u32(vshrq_n_u32((uint32x4_t)vmulq_u16(PackU32(i0, i4), E4_MULLO), 12));
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode32f4x8(const float* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint16x4_t s0 = Encode32f4<align>(src + 0, scale, min, sum, sqsum);
            return vmovn_u16(vcombine_u16(s0, s0));
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode32f4x16(const float* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint16x4_t s0 = Encode32f4<align>(src + 0, scale, min, sum, sqsum);
            uint16x4_t s8 = Encode32f4<align>(src + 8, scale, min, sum, sqsum);
            return vmovn_u16(vcombine_u16(s0, s8));
        }

        static void Encode32f4(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, size16 = AlignLo(size, 16);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _min = vdupq_n_f32(min);
            uint32x4_t _sum = K32_00000000;
            uint32x4_t _sqsum = K32_00000000;
            if (Aligned(src))
            {
                for (; i < size16; i += 16, src += 16, dst += 8)
                    Store<false>(dst, Encode32f4x16<true>(src, _scale, _min, _sum, _sqsum));
            }
            else
            {
                for (; i < size16; i += 16, src += 16, dst += 8)
                    Store<false>(dst, Encode32f4x16<false>(src, _scale, _min, _sum, _sqsum));
            }
            for (; i < size; i += 8, src += 8, dst += 4)
            {
                uint8x8_t d0 = Encode32f4x8<false>(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = vget_lane_u32((uint32x2_t)d0, 0);
            }
            sum = ExtractSum32u(_sum);
            sqsum = ExtractSum32u(_sqsum);
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode32f5(const float* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t i0 = Encode32f<align>(src + 0, scale, min, sum, sqsum);
            uint32x4_t i4 = Encode32f<align>(src + 4, scale, min, sum, sqsum);
            uint16x8_t s0 = vmulq_u16(PackU32(i0, i4), E5_MULLO);
            return vorr_u8(vorr_u8(vtbl2_u8((const uint8x8x2_t&)s0, E5_SHFL0), vtbl2_u8((const uint8x8x2_t&)s0, E5_SHFL1)), vtbl2_u8((const uint8x8x2_t&)s0, E5_SHFL2));
        }

        static void Encode32f5(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _min = vdupq_n_f32(min);
            uint32x4_t _sum = K32_00000000;
            uint32x4_t _sqsum = K32_00000000;
            if (Aligned(src))
            {
                for (; i < main; i += 8, src += 8, dst += 5)
                    Store<false>(dst, Encode32f5<true>(src, _scale, _min, _sum, _sqsum));
            }
            else
            {
                for (; i < main; i += 8, src += 8, dst += 5)
                    Store<false>(dst, Encode32f5<false>(src, _scale, _min, _sum, _sqsum));
            }
            for (; i < size; i += 8, src += 8, dst += 5)
            {
                uint8x8_t d0 = Encode32f5<false>(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = vget_lane_u32((uint32x2_t)d0, 0);
                *(uint8_t*)(dst + 4) = vget_lane_u8(d0, 4);
            }
            sum = ExtractSum32u(_sum);
            sqsum = ExtractSum32u(_sqsum);
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode32f6(const float* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t i0 = Encode32f<align>(src + 0, scale, min, sum, sqsum);
            uint32x4_t i4 = Encode32f<align>(src + 4, scale, min, sum, sqsum);
            uint16x8_t s0 = vmulq_u16(PackU32(i0, i4), E6_MULLO);
            return vorr_u8(vtbl2_u8((const uint8x8x2_t&)s0, E6_SHFL0), vtbl2_u8((const uint8x8x2_t&)s0, E6_SHFL1));
        }

        static void Encode32f6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _min = vdupq_n_f32(min);
            uint32x4_t _sum = K32_00000000;
            uint32x4_t _sqsum = K32_00000000;
            if (Aligned(src))
            {
                for (; i < main; i += 8, src += 8, dst += 6)
                    Store<false>(dst, Encode32f6<true>(src, _scale, _min, _sum, _sqsum));
            }
            else
            {
                for (; i < main; i += 8, src += 8, dst += 6)
                    Store<false>(dst, Encode32f6<false>(src, _scale, _min, _sum, _sqsum));
            }
            for (; i < size; i += 8, src += 8, dst += 6)
            {
                uint8x8_t d0 = Encode32f6<false>(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = vget_lane_u32((uint32x2_t)d0, 0);
                *(uint16_t*)(dst + 4) = vget_lane_u16((uint16x4_t)d0, 2);
            }
            sum = ExtractSum32u(_sum);
            sqsum = ExtractSum32u(_sqsum);
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode32f7(const float* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t i0 = Encode32f<align>(src + 0, scale, min, sum, sqsum);
            uint32x4_t i4 = Encode32f<align>(src + 4, scale, min, sum, sqsum);
            uint16x8_t s0 = vmulq_u16(PackU32(i0, i4), E7_MULLO);
            return vorr_u8(vtbl2_u8((const uint8x8x2_t&)s0, E7_SHFL0), vtbl2_u8((const uint8x8x2_t&)s0, E7_SHFL1));
        }

        static void Encode32f7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _min = vdupq_n_f32(min);
            uint32x4_t _sum = K32_00000000;
            uint32x4_t _sqsum = K32_00000000;
            if (Aligned(src))
            {
                for (; i < main; i += 8, src += 8, dst += 7)
                    Store<false>(dst, Encode32f7<true>(src, _scale, _min, _sum, _sqsum));
            }
            else
            {
                for (; i < main; i += 8, src += 8, dst += 7)
                    Store<false>(dst, Encode32f7<false>(src, _scale, _min, _sum, _sqsum));
            }
            for (; i < size; i += 8, src += 8, dst += 7)
            {
                uint8x8_t d0 = Encode32f7<false>(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = vget_lane_u32((uint32x2_t)d0, 0);
                *(uint16_t*)(dst + 4) = vget_lane_u16((uint16x4_t)d0, 2);
                *(uint8_t*)(dst + 6) = vget_lane_u8(d0, 6);
            }
            sum = ExtractSum32u(_sum);
            sqsum = ExtractSum32u(_sqsum);
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

        template<bool align> SIMD_INLINE uint32x4_t Encode16f(const uint16_t* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            return Encode32f(vcvt_f32_f16((float16x4_t)LoadHalf<align>(src)), scale, min, sum, sqsum);
        }

        template<bool align> static SIMD_INLINE uint16x4_t Encode16f4(const uint16_t* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t i0 = Encode16f<align>(src + 0, scale, min, sum, sqsum);
            uint32x4_t i4 = Encode16f<align>(src + 4, scale, min, sum, sqsum);
            return vmovn_u32(vshrq_n_u32((uint32x4_t)vmulq_u16(PackU32(i0, i4), E4_MULLO), 12));
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode16f4x8(const uint16_t* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint16x4_t s0 = Encode16f4<align>(src + 0, scale, min, sum, sqsum);
            return vmovn_u16(vcombine_u16(s0, s0));
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode16f4x16(const uint16_t* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint16x4_t s0 = Encode16f4<align>(src + 0, scale, min, sum, sqsum);
            uint16x4_t s8 = Encode16f4<align>(src + 8, scale, min, sum, sqsum);
            return vmovn_u16(vcombine_u16(s0, s8));
        }

        static void Encode16f4(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, size16 = AlignLo(size, 16);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _min = vdupq_n_f32(min);
            uint32x4_t _sum = K32_00000000;
            uint32x4_t _sqsum = K32_00000000;
            if (Aligned(src))
            {
                for (; i < size16; i += 16, src += 16, dst += 8)
                    Store<false>(dst, Encode16f4x16<true>(src, _scale, _min, _sum, _sqsum));
            }
            else
            {
                for (; i < size16; i += 16, src += 16, dst += 8)
                    Store<false>(dst, Encode16f4x16<false>(src, _scale, _min, _sum, _sqsum));
            }
            for (; i < size; i += 8, src += 8, dst += 4)
            {
                uint8x8_t d0 = Encode16f4x8<false>(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = vget_lane_u32((uint32x2_t)d0, 0);
            }
            sum = ExtractSum32u(_sum);
            sqsum = ExtractSum32u(_sqsum);
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode16f5(const uint16_t* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t i0 = Encode16f<align>(src + 0, scale, min, sum, sqsum);
            uint32x4_t i4 = Encode16f<align>(src + 4, scale, min, sum, sqsum);
            uint16x8_t s0 = vmulq_u16(PackU32(i0, i4), E5_MULLO);
            return vorr_u8(vorr_u8(vtbl2_u8((const uint8x8x2_t&)s0, E5_SHFL0), vtbl2_u8((const uint8x8x2_t&)s0, E5_SHFL1)), vtbl2_u8((const uint8x8x2_t&)s0, E5_SHFL2));
        }

        static void Encode16f5(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _min = vdupq_n_f32(min);
            uint32x4_t _sum = K32_00000000;
            uint32x4_t _sqsum = K32_00000000;
            if (Aligned(src))
            {
                for (; i < main; i += 8, src += 8, dst += 5)
                    Store<false>(dst, Encode16f5<true>(src, _scale, _min, _sum, _sqsum));
            }
            else
            {
                for (; i < main; i += 8, src += 8, dst += 5)
                    Store<false>(dst, Encode16f5<false>(src, _scale, _min, _sum, _sqsum));
            }
            for (; i < size; i += 8, src += 8, dst += 5)
            {
                uint8x8_t d0 = Encode16f5<false>(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = vget_lane_u32((uint32x2_t)d0, 0);
                *(uint8_t*)(dst + 4) = vget_lane_u8(d0, 4);
            }
            sum = ExtractSum32u(_sum);
            sqsum = ExtractSum32u(_sqsum);
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode16f6(const uint16_t* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t i0 = Encode16f<align>(src + 0, scale, min, sum, sqsum);
            uint32x4_t i4 = Encode16f<align>(src + 4, scale, min, sum, sqsum);
            uint16x8_t s0 = vmulq_u16(PackU32(i0, i4), E6_MULLO);
            return vorr_u8(vtbl2_u8((const uint8x8x2_t&)s0, E6_SHFL0), vtbl2_u8((const uint8x8x2_t&)s0, E6_SHFL1));
        }

        static void Encode16f6(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _min = vdupq_n_f32(min);
            uint32x4_t _sum = K32_00000000;
            uint32x4_t _sqsum = K32_00000000;
            if (Aligned(src))
            {
                for (; i < main; i += 8, src += 8, dst += 6)
                    Store<false>(dst, Encode16f6<true>(src, _scale, _min, _sum, _sqsum));
            }
            else
            {
                for (; i < main; i += 8, src += 8, dst += 6)
                    Store<false>(dst, Encode16f6<false>(src, _scale, _min, _sum, _sqsum));
            }
            for (; i < size; i += 8, src += 8, dst += 6)
            {
                uint8x8_t d0 = Encode16f6<false>(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = vget_lane_u32((uint32x2_t)d0, 0);
                *(uint16_t*)(dst + 4) = vget_lane_u16((uint16x4_t)d0, 2);
            }
            sum = ExtractSum32u(_sum);
            sqsum = ExtractSum32u(_sqsum);
        }

        template<bool align> static SIMD_INLINE uint8x8_t Encode16f7(const uint16_t* src, float32x4_t scale, float32x4_t min, uint32x4_t& sum, uint32x4_t& sqsum)
        {
            uint32x4_t i0 = Encode16f<align>(src + 0, scale, min, sum, sqsum);
            uint32x4_t i4 = Encode16f<align>(src + 4, scale, min, sum, sqsum);
            uint16x8_t s0 = vmulq_u16(PackU32(i0, i4), E7_MULLO);
            return vorr_u8(vtbl2_u8((const uint8x8x2_t&)s0, E7_SHFL0), vtbl2_u8((const uint8x8x2_t&)s0, E7_SHFL1));
        }

        static void Encode16f7(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _min = vdupq_n_f32(min);
            uint32x4_t _sum = K32_00000000;
            uint32x4_t _sqsum = K32_00000000;
            if (Aligned(src))
            {
                for (; i < main; i += 8, src += 8, dst += 7)
                    Store<false>(dst, Encode16f7<true>(src, _scale, _min, _sum, _sqsum));
            }
            else
            {
                for (; i < main; i += 8, src += 8, dst += 7)
                    Store<false>(dst, Encode16f7<false>(src, _scale, _min, _sum, _sqsum));
            }
            for (; i < size; i += 8, src += 8, dst += 7)
            {
                uint8x8_t d0 = Encode16f7<false>(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = vget_lane_u32((uint32x2_t)d0, 0);
                *(uint16_t*)(dst + 4) = vget_lane_u16((uint16x4_t)d0, 2);
                *(uint8_t*)(dst + 6) = vget_lane_u8(d0, 6);
            }
            sum = ExtractSum32u(_sum);
            sqsum = ExtractSum32u(_sqsum);
        }

        static void Encode16f8(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
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
                    uint32x4_t d0 = Encode16f<true>(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d1 = Encode16f<true>(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d2 = Encode16f<true>(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d3 = Encode16f<true>(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                    Store<false>(dst + i, PackU16(PackU32(d0, d1), PackU32(d2, d3)));
                }
            }
            else
            {
                for (; i < sizeA; i += A)
                {
                    uint32x4_t d0 = Encode16f<false>(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d1 = Encode16f<false>(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d2 = Encode16f<false>(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                    uint32x4_t d3 = Encode16f<false>(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                    Store<false>(dst + i, PackU16(PackU32(d0, d1), PackU32(d2, d3)));
                }
}
            for (; i < size; i += F)
            {
                uint32x4_t d0 = Encode16f<false>(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                uint32x4_t d1 = Encode16f<false>(src + i + 1 * F, _scale, _min, _sum, _sqsum);
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
            case 4: return Encode32f4;
            case 5: return Encode32f5;
            case 6: return Encode32f6;
            case 7: return Encode32f7;
            case 8: return Encode32f8;
            default: return Base::GetEncode32f(depth);
            }
        }

        Base::DescrInt::Encode16fPtr GetEncode16f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Encode16f4;
            case 5: return Encode16f5;
            case 6: return Encode16f6;
            case 7: return Encode16f7;
            case 8: return Encode16f8;
            default: return Base::GetEncode16f(depth);
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
