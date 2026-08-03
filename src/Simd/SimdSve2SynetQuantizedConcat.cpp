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

#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svint32_t QuantizeLinear(const svfloat32_t& value, const svfloat32_t& norm, const svint32_t& zero, const svbool_t& mask)
        {
            svfloat32_t scaled = svmul_f32_x(mask, value, norm);
            svfloat32_t round = svsel_f32(svcmpgt_n_f32(mask, scaled, 0.0f), svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svadd_s32_x(mask, svcvt_s32_f32_x(mask, svadd_f32_x(mask, scaled, round)), zero);
        }

        SIMD_INLINE void DequantizeQuantizeLinear(const uint8_t* src, const svint32_t& bias, const svfloat32_t& norm, const svfloat32_t& scale, const svint32_t& zero, uint8_t* dst, const svbool_t& mask)
        {
            svint32_t value = svadd_s32_x(mask, svreinterpret_s32_u32(svld1ub_u32(mask, src)), bias);
            svint32_t i32 = QuantizeLinear(svmul_f32_x(mask, svcvt_f32_s32_x(mask, value), norm), scale, zero, mask);
            i32 = svmin_n_s32_x(mask, svmax_n_s32_x(mask, i32, 0), 255);
            svst1b_u32(mask, dst, svreinterpret_u32_s32(i32));
        }

        SIMD_INLINE void DequantizeQuantizeLinear(const uint8_t* src, size_t size, const svint32_t& bias, const svfloat32_t& norm, const svfloat32_t& scale, const svint32_t& zero, uint8_t* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            size_t i = 0;
            for (; i + QF <= size; i += QF)
            {
                DequantizeQuantizeLinear(src + i + 0 * F, bias, norm, scale, zero, dst + i + 0 * F, body);
                DequantizeQuantizeLinear(src + i + 1 * F, bias, norm, scale, zero, dst + i + 1 * F, body);
                DequantizeQuantizeLinear(src + i + 2 * F, bias, norm, scale, zero, dst + i + 2 * F, body);
                DequantizeQuantizeLinear(src + i + 3 * F, bias, norm, scale, zero, dst + i + 3 * F, body);
            }
            for (; i < size; i += F)
                DequantizeQuantizeLinear(src + i, bias, norm, scale, zero, dst + i, svwhilelt_b32(i, size));
        }

        static void SynetQuantizedConcatLayerForward1(const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst)
        {
            size_t size0 = size[0];
            svint32_t _bias0 = svdup_n_s32(bias[0]), _zero = svdup_n_s32(zero);
            svfloat32_t _norm0 = svdup_n_f32(norm[0]), _scale = svdup_n_f32(scale[0]);
            const uint8_t* ps0 = src[0];
            for (size_t o = 0; o < num; ++o)
            {
                DequantizeQuantizeLinear(ps0, size0, _bias0, _norm0, _scale, _zero, dst);
                ps0 += size0;
                dst += size0;
            }
        }

        static void SynetQuantizedConcatLayerForward2(const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst)
        {
            size_t size0 = size[0], size1 = size[1];
            svint32_t _bias0 = svdup_n_s32(bias[0]), _bias1 = svdup_n_s32(bias[1]), _zero = svdup_n_s32(zero);
            svfloat32_t _norm0 = svdup_n_f32(norm[0]), _norm1 = svdup_n_f32(norm[1]), _scale = svdup_n_f32(scale[0]);
            const uint8_t* ps0 = src[0], * ps1 = src[1];
            for (size_t o = 0; o < num; ++o)
            {
                DequantizeQuantizeLinear(ps0, size0, _bias0, _norm0, _scale, _zero, dst);
                ps0 += size0;
                dst += size0;
                DequantizeQuantizeLinear(ps1, size1, _bias1, _norm1, _scale, _zero, dst);
                ps1 += size1;
                dst += size1;
            }
        }

        static void SynetQuantizedConcatLayerForward3(const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst)
        {
            size_t size0 = size[0], size1 = size[1], size2 = size[2];
            svint32_t _bias0 = svdup_n_s32(bias[0]), _bias1 = svdup_n_s32(bias[1]), _bias2 = svdup_n_s32(bias[2]), _zero = svdup_n_s32(zero);
            svfloat32_t _norm0 = svdup_n_f32(norm[0]), _norm1 = svdup_n_f32(norm[1]), _norm2 = svdup_n_f32(norm[2]), _scale = svdup_n_f32(scale[0]);
            const uint8_t* ps0 = src[0], * ps1 = src[1], * ps2 = src[2];
            for (size_t o = 0; o < num; ++o)
            {
                DequantizeQuantizeLinear(ps0, size0, _bias0, _norm0, _scale, _zero, dst);
                ps0 += size0;
                dst += size0;
                DequantizeQuantizeLinear(ps1, size1, _bias1, _norm1, _scale, _zero, dst);
                ps1 += size1;
                dst += size1;
                DequantizeQuantizeLinear(ps2, size2, _bias2, _norm2, _scale, _zero, dst);
                ps2 += size2;
                dst += size2;
            }
        }

        static void SynetQuantizedConcatLayerForwardN(size_t count, const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst)
        {
            svfloat32_t _scale = svdup_n_f32(scale[0]);
            svint32_t _zero = svdup_n_s32(zero);
            for (size_t o = 0; o < num; ++o)
            {
                for (size_t s = 0; s < count; ++s)
                {
                    size_t _size = size[s];
                    const uint8_t* _src = src[s] + o * _size;
                    DequantizeQuantizeLinear(_src, _size, svdup_n_s32(bias[s]), svdup_n_f32(norm[s]), _scale, _zero, dst);
                    dst += _size;
                }
            }
        }

        void SynetQuantizedConcatLayerForward(size_t count, const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst)
        {
            switch (count)
            {
            case 1: SynetQuantizedConcatLayerForward1(src, num, size, bias, norm, scale, zero, dst); break;
            case 2: SynetQuantizedConcatLayerForward2(src, num, size, bias, norm, scale, zero, dst); break;
            case 3: SynetQuantizedConcatLayerForward3(src, num, size, bias, norm, scale, zero, dst); break;
            default: SynetQuantizedConcatLayerForwardN(count, src, num, size, bias, norm, scale, zero, dst); break;
            }
        }
    }
#endif
}
