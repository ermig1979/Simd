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

#include "Simd/SimdDescrInt.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svfloat32_t Decode32f(uint32_t src, const svbool_t& mask, const svuint32_t& shifts,
            uint32_t valueMask, const svfloat32_t& scale, const svfloat32_t& shift)
        {
            svuint32_t value = svand_n_u32_x(mask, svlsr_u32_x(mask, svdup_n_u32(src), shifts), valueMask);
            return svmla_f32_x(mask, shift, scale, svcvt_f32_u32_x(mask, value));
        }

        SIMD_INLINE void Store16f(const svbool_t& mask32, const svbool_t& mask16, uint16_t* dst, const svfloat32_t& value)
        {
            svuint16_t half = svreinterpret_u16_f16(svcvt_f16_f32_x(mask32, value));
            svst1_u16(mask16, dst, svuzp1_u16(half, half));
        }

        template<int bits> static void Decode32fN(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            const uint32_t valueMask = (1 << bits) - 1;
            const int hiOffset = bits == 4 ? 0 : bits - 4;
            const int hiShift = bits == 4 ? 16 : 4 * bits - hiOffset * 8;
            svbool_t mask = svwhilelt_b32(size_t(0), size_t(4));
            svuint32_t loShifts = svindex_u32(0, bits);
            svuint32_t hiShifts = svindex_u32(hiShift, bits);
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _shift = svdup_n_f32(shift);
            for (size_t i = 0; i < size; i += 8, src += bits, dst += 8)
            {
                svst1_f32(mask, dst + 0, Decode32f(*(uint32_t*)(src + 0), mask, loShifts, valueMask, _scale, _shift));
                svst1_f32(mask, dst + 4, Decode32f(*(uint32_t*)(src + hiOffset), mask, hiShifts, valueMask, _scale, _shift));
            }
        }

        template<int bits> static void Decode16fN(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            const uint32_t valueMask = (1 << bits) - 1;
            const int hiOffset = bits == 4 ? 0 : bits - 4;
            const int hiShift = bits == 4 ? 16 : 4 * bits - hiOffset * 8;
            svbool_t mask32 = svwhilelt_b32(size_t(0), size_t(4));
            svbool_t mask16 = svwhilelt_b16(size_t(0), size_t(4));
            svuint32_t loShifts = svindex_u32(0, bits);
            svuint32_t hiShifts = svindex_u32(hiShift, bits);
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _shift = svdup_n_f32(shift);
            for (size_t i = 0; i < size; i += 8, src += bits, dst += 8)
            {
                Store16f(mask32, mask16, dst + 0, Decode32f(*(uint32_t*)(src + 0), mask32, loShifts, valueMask, _scale, _shift));
                Store16f(mask32, mask16, dst + 4, Decode32f(*(uint32_t*)(src + hiOffset), mask32, hiShifts, valueMask, _scale, _shift));
            }
        }

        static void Decode32f8(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _shift = svdup_n_f32(shift);
            for (size_t i = 0; i < size; i += svcntw())
            {
                svbool_t mask = svwhilelt_b32(i, size);
                svuint32_t value = svld1ub_u32(mask, src + i);
                svst1_f32(mask, dst + i, svmla_f32_x(mask, _shift, _scale, svcvt_f32_u32_x(mask, value)));
            }
        }

        static void Decode16f8(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            svbool_t mask32 = svwhilelt_b32(size_t(0), size_t(4));
            svbool_t mask16 = svwhilelt_b16(size_t(0), size_t(4));
            svfloat32_t _scale = svdup_n_f32(scale);
            svfloat32_t _shift = svdup_n_f32(shift);
            for (size_t i = 0; i < size; i += 8, src += 8, dst += 8)
            {
                svuint32_t lo = svld1ub_u32(mask32, src + 0);
                svuint32_t hi = svld1ub_u32(mask32, src + 4);
                Store16f(mask32, mask16, dst + 0, svmla_f32_x(mask32, _shift, _scale, svcvt_f32_u32_x(mask32, lo)));
                Store16f(mask32, mask16, dst + 4, svmla_f32_x(mask32, _shift, _scale, svcvt_f32_u32_x(mask32, hi)));
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::Decode32fPtr GetDecode32f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Decode32fN<4>;
            case 5: return Decode32fN<5>;
            case 6: return Decode32fN<6>;
            case 7: return Decode32fN<7>;
            case 8: return Decode32f8;
            default: return Base::GetDecode32f(depth);
            }
        }

        Base::DescrInt::Decode16fPtr GetDecode16f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Decode16fN<4>;
            case 5: return Decode16fN<5>;
            case 6: return Decode16fN<6>;
            case 7: return Decode16fN<7>;
            case 8: return Decode16f8;
            default: return Base::GetDecode16f(depth);
            }
        }
    }
#endif// SIMD_SVE2_ENABLE
}
