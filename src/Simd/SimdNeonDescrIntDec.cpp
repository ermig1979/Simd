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
#include "Simd/SimdShuffle.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        static void Decode32f4(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            if (Aligned(dst))
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C4_SHL0), C4_AND);
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s0, C4_SHL1), C4_AND);
                    float32x4_t d0 = vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0));
                    float32x4_t d1 = vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1));
                    Store<true>(dst + 0, d0);
                    Store<true>(dst + 4, d1);
                    src += 4;
                    dst += 8;
                }
            }
            else
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C4_SHL0), C4_AND);
                    Store<false>(dst + 0, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0)));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s0, C4_SHL1), C4_AND);
                    Store<false>(dst + 4, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1)));
                    src += 4;
                    dst += 8;
                }
            }
        }

        static void Decode32f5(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            if (Aligned(dst))
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C5_SHL0), C5_AND);
                    Store<true>(dst + 0, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0)));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 1));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C5_SHL1), C5_AND);
                    Store<true>(dst + 4, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1)));
                    src += 5;
                    dst += 8;
                }
            }
            else
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C5_SHL0), C5_AND);
                    Store<false>(dst + 0, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0)));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 1));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C5_SHL1), C5_AND);
                    Store<false>(dst + 4, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1)));
                    src += 5;
                    dst += 8;
                }
            }
        }        

        static void Decode32f6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            if (Aligned(dst))
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C6_SHL0), C6_AND);
                    Store<true>(dst + 0, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0)));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 2));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C6_SHL1), C6_AND);
                    Store<true>(dst + 4, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1)));
                    src += 6;
                    dst += 8;
                }
            }
            else
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C6_SHL0), C6_AND);
                    Store<false>(dst + 0, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0)));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 2));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C6_SHL1), C6_AND);
                    Store<false>(dst + 4, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1)));
                    src += 6;
                    dst += 8;
                }
            }
        }

        static void Decode32f7(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            if (Aligned(dst))
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C7_SHL0), C7_AND);
                    Store<true>(dst + 0, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0)));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 3));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C7_SHL1), C7_AND);
                    Store<true>(dst + 4, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1)));
                    src += 7;
                    dst += 8;
                }
            }
            else
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C7_SHL0), C7_AND);
                    Store<false>(dst + 0, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0)));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 3));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C7_SHL1), C7_AND);
                    Store<false>(dst + 4, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1)));
                    src += 7;
                    dst += 8;
                }
            }
        }

        static void Decode32f8(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            size_t i = 0;
            if (Aligned(src) && Aligned(dst))
            {
                for (; i < size; i += 8)
                {
                    uint16x8_t u16 = vmovl_u8(LoadHalf<true>(src));
                    Store<true>(dst + 0, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(UnpackU16<0>(u16))));
                    Store<true>(dst + 4, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(UnpackU16<1>(u16))));
                    src += 8;
                    dst += 8;
                }
            }
            else
            {
                for (; i < size; i += 8)
                {
                    uint16x8_t u16 = vmovl_u8(LoadHalf<false>(src));
                    Store<false>(dst + 0, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(UnpackU16<0>(u16))));
                    Store<false>(dst + 4, vmlaq_f32(_shift, _scale, vcvtq_f32_u32(UnpackU16<1>(u16))));
                    src += 8;
                    dst += 8;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void Decode16f4(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            if (Aligned(dst))
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C4_SHL0), C4_AND);
                    Store<true>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0))));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s0, C4_SHL1), C4_AND);
                    Store<true>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1))));
                    src += 4;
                    dst += 8;
                }
            }
            else
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C4_SHL0), C4_AND);
                    Store<false>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0))));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s0, C4_SHL1), C4_AND);
                    Store<false>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1))));
                    src += 4;
                    dst += 8;
                }
            }
        }

        static void Decode16f5(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            if (Aligned(dst))
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C5_SHL0), C5_AND);
                    Store<true>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0))));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 1));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C5_SHL1), C5_AND);
                    Store<true>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1))));
                    src += 5;
                    dst += 8;
                }
            }
            else
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C5_SHL0), C5_AND);
                    Store<false>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0))));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 1));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C5_SHL1), C5_AND);
                    Store<false>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1))));
                    src += 5;
                    dst += 8;
                }
            }
        }

        static void Decode16f6(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            if (Aligned(dst))
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C6_SHL0), C6_AND);
                    Store<true>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0))));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 2));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C6_SHL1), C6_AND);
                    Store<true>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1))));
                    src += 6;
                    dst += 8;
                }
            }
            else
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C6_SHL0), C6_AND);
                    Store<false>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0))));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 2));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C6_SHL1), C6_AND);
                    Store<false>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1))));
                    src += 6;
                    dst += 8;
                }
            }
        }

        static void Decode16f7(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            if (Aligned(dst))
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C7_SHL0), C7_AND);
                    Store<true>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0))));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 3));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C7_SHL1), C7_AND);
                    Store<true>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1))));
                    src += 7;
                    dst += 8;
                }
            }
            else
            {
                for (size_t i = 0; i < size; i += 8)
                {
                    uint32x4_t s0 = vdupq_n_u32(*(uint32_t*)(src + 0));
                    uint32x4_t u0 = vandq_u32(vshlq_u32(s0, C7_SHL0), C7_AND);
                    Store<false>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u0))));
                    uint32x4_t s1 = vdupq_n_u32(*(uint32_t*)(src + 3));
                    uint32x4_t u1 = vandq_u32(vshlq_u32(s1, C7_SHL1), C7_AND);
                    Store<false>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(u1))));
                    src += 7;
                    dst += 8;
                }
            }
        }

        static void Decode16f8(const uint8_t* src, float scale, float shift, size_t size, uint16_t * dst)
        {
            assert(size % 8 == 0);
            float32x4_t _scale = vdupq_n_f32(scale);
            float32x4_t _shift = vdupq_n_f32(shift);
            size_t i = 0;
            if (Aligned(src) && Aligned(dst))
            {
                for (; i < size; i += 8)
                {
                    uint16x8_t u16 = vmovl_u8(LoadHalf<true>(src));
                    Store<true>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(UnpackU16<0>(u16)))));
                    Store<true>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(UnpackU16<1>(u16)))));
                    src += 8;
                    dst += 8;
                }
            }
            else
            {
                for (; i < size; i += 8)
                {
                    uint16x8_t u16 = vmovl_u8(LoadHalf<false>(src));
                    Store<false>(dst + 0, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(UnpackU16<0>(u16)))));
                    Store<false>(dst + 4, (uint16x4_t)vcvt_f16_f32(vmlaq_f32(_shift, _scale, vcvtq_f32_u32(UnpackU16<1>(u16)))));
                    src += 8;
                    dst += 8;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::Decode32fPtr GetDecode32f(size_t depth)
        {
            switch (depth)
            {
            //case 4: return Decode32f4;
            case 5: return Decode32f5;
            case 6: return Decode32f6;
            case 7: return Decode32f7;
            //case 8: return Decode32f8;
            default: return Base::GetDecode32f(depth);
            }
        }

        Base::DescrInt::Decode16fPtr GetDecode16f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Decode16f4;
            case 5: return Decode16f5;
            case 6: return Decode16f6;
            case 7: return Decode16f7;
            case 8: return Decode16f8;
            default: return Base::GetDecode16f(depth);
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
