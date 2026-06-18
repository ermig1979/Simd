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
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint32_t Encode32f(const svfloat32_t& src, const svbool_t& mask, const svfloat32_t& scale,
            const svfloat32_t& min, svuint32_t& sum, svuint32_t& sqsum)
        {
            svfloat32_t value = svmul_f32_x(mask, svsub_f32_x(mask, src, min), scale);
            svuint32_t encoded = svmin_u32_x(mask, svcvt_u32_f32_x(mask, svadd_n_f32_x(mask, value, 0.5f)), svdup_n_u32(0xFF));
            sum = svadd_u32_m(mask, sum, encoded);
            sqsum = svmla_u32_m(mask, sqsum, encoded, encoded);
            return encoded;
        }

        SIMD_INLINE svuint32_t Encode32f(const float* src, const svbool_t& mask, const svfloat32_t& scale,
            const svfloat32_t& min, svuint32_t& sum, svuint32_t& sqsum)
        {
            return Encode32f(svld1_f32(mask, src), mask, scale, min, sum, sqsum);
        }

        SIMD_INLINE void Encode32f8(const float* src, const svbool_t& mask, const svfloat32_t& scale,
            const svfloat32_t& min, svuint32_t& sum, svuint32_t& sqsum, uint8_t* dst)
        {
            svst1b_u32(mask, dst, Encode32f(src, mask, scale, min, sum, sqsum));
        }

        SIMD_INLINE void Encode32f8(const float* src, const svfloat32_t& scale, const svfloat32_t& min,
            svuint32_t& sum, svuint32_t& sqsum, uint32_t* dst)
        {
            size_t done = 0;
            do
            {
                svbool_t mask = svwhilelt_b32(done, size_t(8));
                svst1_u32(mask, dst + done, Encode32f(src + done, mask, scale, min, sum, sqsum));
                done += svcntw();
            } while (done < 8);
        }

        static void Encode32f4(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            uint32_t buf[8];
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _min = svdup_n_f32(min);
            svuint32_t _sum = svdup_n_u32(0);
            svuint32_t _sqsum = svdup_n_u32(0);
            for (size_t i = 0; i < size; i += 8, src += 8, dst += 4)
            {
                Encode32f8(src, _scale, _min, _sum, _sqsum, buf);
                dst[0] = uint8_t(buf[0] | (buf[1] << 4));
                dst[1] = uint8_t(buf[2] | (buf[3] << 4));
                dst[2] = uint8_t(buf[4] | (buf[5] << 4));
                dst[3] = uint8_t(buf[6] | (buf[7] << 4));
            }
            sum = (int32_t)svaddv_u32(svptrue_b32(), _sum);
            sqsum = (int32_t)svaddv_u32(svptrue_b32(), _sqsum);
        }

        static void Encode32f5(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            uint32_t buf[8];
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _min = svdup_n_f32(min);
            svuint32_t _sum = svdup_n_u32(0);
            svuint32_t _sqsum = svdup_n_u32(0);
            for (size_t i = 0; i < size; i += 8, src += 8, dst += 5)
            {
                Encode32f8(src, _scale, _min, _sum, _sqsum, buf);
                dst[0] = uint8_t(buf[0] | (buf[1] << 5));
                dst[1] = uint8_t((buf[1] >> 3) | (buf[2] << 2) | (buf[3] << 7));
                dst[2] = uint8_t((buf[3] >> 1) | (buf[4] << 4));
                dst[3] = uint8_t((buf[4] >> 4) | (buf[5] << 1) | (buf[6] << 6));
                dst[4] = uint8_t((buf[6] >> 2) | (buf[7] << 3));
            }
            sum = (int32_t)svaddv_u32(svptrue_b32(), _sum);
            sqsum = (int32_t)svaddv_u32(svptrue_b32(), _sqsum);
        }

        static void Encode32f6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            uint32_t buf[8];
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _min = svdup_n_f32(min);
            svuint32_t _sum = svdup_n_u32(0);
            svuint32_t _sqsum = svdup_n_u32(0);
            for (size_t i = 0; i < size; i += 8, src += 8, dst += 6)
            {
                Encode32f8(src, _scale, _min, _sum, _sqsum, buf);
                dst[0] = uint8_t(buf[0] | (buf[1] << 6));
                dst[1] = uint8_t((buf[1] >> 2) | (buf[2] << 4));
                dst[2] = uint8_t((buf[2] >> 4) | (buf[3] << 2));
                dst[3] = uint8_t(buf[4] | (buf[5] << 6));
                dst[4] = uint8_t((buf[5] >> 2) | (buf[6] << 4));
                dst[5] = uint8_t((buf[6] >> 4) | (buf[7] << 2));
            }
            sum = (int32_t)svaddv_u32(svptrue_b32(), _sum);
            sqsum = (int32_t)svaddv_u32(svptrue_b32(), _sqsum);
        }

        static void Encode32f7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            uint32_t buf[8];
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _min = svdup_n_f32(min);
            svuint32_t _sum = svdup_n_u32(0);
            svuint32_t _sqsum = svdup_n_u32(0);
            for (size_t i = 0; i < size; i += 8, src += 8, dst += 7)
            {
                Encode32f8(src, _scale, _min, _sum, _sqsum, buf);
                dst[0] = uint8_t(buf[0] | (buf[1] << 7));
                dst[1] = uint8_t((buf[1] >> 1) | (buf[2] << 6));
                dst[2] = uint8_t((buf[2] >> 2) | (buf[3] << 5));
                dst[3] = uint8_t((buf[3] >> 3) | (buf[4] << 4));
                dst[4] = uint8_t((buf[4] >> 4) | (buf[5] << 3));
                dst[5] = uint8_t((buf[5] >> 5) | (buf[6] << 2));
                dst[6] = uint8_t((buf[6] >> 6) | (buf[7] << 1));
            }
            sum = (int32_t)svaddv_u32(svptrue_b32(), _sum);
            sqsum = (int32_t)svaddv_u32(svptrue_b32(), _sqsum);
        }

        static void Encode32f8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _min = svdup_n_f32(min);
            svuint32_t _sum = svdup_n_u32(0);
            svuint32_t _sqsum = svdup_n_u32(0);
            for (size_t i = 0; i < size; i += svcntw())
                Encode32f8(src + i, svwhilelt_b32(i, size), _scale, _min, _sum, _sqsum, dst + i);
            sum = (int32_t)svaddv_u32(svptrue_b32(), _sum);
            sqsum = (int32_t)svaddv_u32(svptrue_b32(), _sqsum);
        }

#ifdef SIMD_NEON_FP16_ENABLE
        SIMD_INLINE void Encode16f(const uint16_t* src, const svbool_t& mask16, const svbool_t& mask32,
            const svfloat32_t& scale, const svfloat32_t& min, svuint32_t& sum, svuint32_t& sqsum, svuint32_t& even, svuint32_t& odd)
        {
            svfloat16_t half = svreinterpret_f16_u16(svld1_u16(mask16, src));
            even = Encode32f(svcvt_f32_f16_x(mask32, half), mask32, scale, min, sum, sqsum);
            odd = Encode32f(svcvtlt_f32_f16_x(mask32, half), mask32, scale, min, sum, sqsum);
        }

        static void Encode16f4(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t step = svcntw() * 2, i = 0;
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _min = svdup_n_f32(min);
            svuint32_t _sum = svdup_n_u32(0);
            svuint32_t _sqsum = svdup_n_u32(0);
            svuint32_t even, odd;
            for (; i < size; i += step, src += step, dst += step / 2)
            {
                size_t tail = Simd::Min(step, size - i);
                svbool_t mask16 = svwhilelt_b16(size_t(0), tail);
                svbool_t mask32 = svwhilelt_b32(size_t(0), tail / 2);
                Encode16f(src, mask16, mask32, _scale, _min, _sum, _sqsum, even, odd);
                svst1b_u32(mask32, dst, svorr_u32_x(mask32, even, svlsl_n_u32_x(mask32, odd, 4)));
            }
            sum = (int32_t)svaddv_u32(svptrue_b32(), _sum);
            sqsum = (int32_t)svaddv_u32(svptrue_b32(), _sqsum);
        }

        SIMD_INLINE svuint8_t PackU32ToU8(const svuint32_t& src)
        {
            return svqxtnb_u16(svqxtnb_u32(src));
        }

        static void Encode16f8(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t step = svcntw() * 2, i = 0;
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _min = svdup_n_f32(min);
            svuint32_t _sum = svdup_n_u32(0);
            svuint32_t _sqsum = svdup_n_u32(0);
            svuint32_t even, odd;
            for (; i < size; i += step, src += step, dst += step)
            {
                size_t tail = Simd::Min(step, size - i);
                svbool_t mask16 = svwhilelt_b16(size_t(0), tail);
                svbool_t mask32 = svwhilelt_b32(size_t(0), tail / 2);
                svbool_t mask8 = svwhilelt_b8(size_t(0), tail / 2);
                Encode16f(src, mask16, mask32, _scale, _min, _sum, _sqsum, even, odd);
                svst2_u8(mask8, dst, svcreate2_u8(PackU32ToU8(even), PackU32ToU8(odd)));
            }
            sum = (int32_t)svaddv_u32(svptrue_b32(), _sum);
            sqsum = (int32_t)svaddv_u32(svptrue_b32(), _sqsum);
        }
#endif

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
#ifdef SIMD_NEON_FP16_ENABLE
            switch (depth)
            {
            case 4: return Encode16f4;
#ifdef SIMD_NEON_ENABLE
            case 5: return Neon::GetEncode16f(depth);
            case 6: return Neon::GetEncode16f(depth);
            case 7: return Neon::GetEncode16f(depth);
#endif
            case 8: return Encode16f8;
            default: return Base::GetEncode16f(depth);
            }
#else
#ifdef SIMD_NEON_ENABLE
            return Neon::GetEncode16f(depth);
#else
            return Base::GetEncode16f(depth);
#endif
#endif
        }
    }
#endif// SIMD_SVE2_ENABLE
}
