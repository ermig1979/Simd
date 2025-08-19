/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        static void SynetQuantizedConcatLayerForward1(const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst)
        {
            __m256i bias0 = _mm256_set1_epi32(bias[0]), _zero = _mm256_set1_epi32(zero);
            __m256 norm0 = _mm256_set1_ps(norm[0]), _scale = _mm256_set1_ps(scale[0]);
            size_t size01 = size[0], size08 = size01 & (~7), size032 = size01 & (~31);
            const uint8_t* ps0 = src[0];
            for (size_t o = 0, i = 0; o < num; ++o)
            {
                for (i = 0; i < size032; i += 32)
                    DequantizeQuantizeLinear32(ps0 + i, bias0, norm0, _scale, _zero, dst + i);
                for (; i < size08; i += 8)
                    DequantizeQuantizeLinear8(ps0 + i, bias0, norm0, _scale, _zero, dst + i);
                for (; i < size01; i += 1)
                    DequantizeQuantizeLinear1(ps0 + i, bias0, norm0, _scale, _zero, dst + i);
                ps0 += size01;
                dst += size01;
            }
        }

        static void SynetQuantizedConcatLayerForward2(const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst)
        {
            __m256i bias0 = _mm256_set1_epi32(bias[0]), bias1 = _mm256_set1_epi32(bias[1]), _zero = _mm256_set1_epi32(zero);
            __m256 norm0 = _mm256_set1_ps(norm[0]), norm1 = _mm256_set1_ps(norm[1]), _scale = _mm256_set1_ps(scale[0]);
            size_t size01 = size[0], size08 = size01 & (~7), size032 = size01 & (~31);
            size_t size11 = size[1], size18 = size11 & (~7), size132 = size11 & (~31);
            const uint8_t* ps0 = src[0], * ps1 = src[1];
            for (size_t o = 0, i = 0; o < num; ++o)
            {
                for (i = 0; i < size032; i += 32)
                    DequantizeQuantizeLinear32(ps0 + i, bias0, norm0, _scale, _zero, dst + i);
                for (; i < size08; i += 8)
                    DequantizeQuantizeLinear8(ps0 + i, bias0, norm0, _scale, _zero, dst + i);
                for (; i < size01; i += 1)
                    DequantizeQuantizeLinear1(ps0 + i, bias0, norm0, _scale, _zero, dst + i);
                ps0 += size01;
                dst += size01;
                for (i = 0; i < size132; i += 32)
                    DequantizeQuantizeLinear32(ps1 + i, bias1, norm1, _scale, _zero, dst + i);
                for (; i < size18; i += 8)
                    DequantizeQuantizeLinear8(ps1 + i, bias1, norm1, _scale, _zero, dst + i);
                for (; i < size11; i += 1)
                    DequantizeQuantizeLinear1(ps1 + i, bias1, norm1, _scale, _zero, dst + i);
                ps1 += size11;
                dst += size11;
            }
        }

        static void SynetQuantizedConcatLayerForward3(const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst)
        {
            __m256i bias0 = _mm256_set1_epi32(bias[0]), bias1 = _mm256_set1_epi32(bias[1]), bias2 = _mm256_set1_epi32(bias[2]), _zero = _mm256_set1_epi32(zero);
            __m256 norm0 = _mm256_set1_ps(norm[0]), norm1 = _mm256_set1_ps(norm[1]), norm2 = _mm256_set1_ps(norm[2]), _scale = _mm256_set1_ps(scale[0]);
            size_t size01 = size[0], size08 = size01 & (~7), size032 = size01 & (~31);
            size_t size11 = size[1], size18 = size11 & (~7), size132 = size11 & (~31);
            size_t size21 = size[2], size28 = size21 & (~7), size232 = size21 & (~31);
            const uint8_t* ps0 = src[0], * ps1 = src[1], * ps2 = src[2];
            for (size_t o = 0, i = 0; o < num; ++o)
            {
                for (i = 0; i < size032; i += 32)
                    DequantizeQuantizeLinear32(ps0 + i, bias0, norm0, _scale, _zero, dst + i);
                for (; i < size08; i += 8)
                    DequantizeQuantizeLinear8(ps0 + i, bias0, norm0, _scale, _zero, dst + i);
                for (; i < size01; i += 1)
                    DequantizeQuantizeLinear1(ps0 + i, bias0, norm0, _scale, _zero, dst + i);
                ps0 += size01;
                dst += size01;
                for (i = 0; i < size132; i += 32)
                    DequantizeQuantizeLinear32(ps1 + i, bias1, norm1, _scale, _zero, dst + i);
                for (; i < size18; i += 8)
                    DequantizeQuantizeLinear8(ps1 + i, bias1, norm1, _scale, _zero, dst + i);
                for (; i < size11; i += 1)
                    DequantizeQuantizeLinear1(ps1 + i, bias1, norm1, _scale, _zero, dst + i);
                ps1 += size11;
                dst += size11;
                for (i = 0; i < size232; i += 32)
                    DequantizeQuantizeLinear32(ps2 + i, bias2, norm2, _scale, _zero, dst + i);
                for (; i < size28; i += 8)
                    DequantizeQuantizeLinear8(ps2 + i, bias2, norm2, _scale, _zero, dst + i);
                for (; i < size21; i += 1)
                    DequantizeQuantizeLinear1(ps2 + i, bias2, norm2, _scale, _zero, dst + i);
                ps2 += size21;
                dst += size21;

            }
        }

        static void SynetQuantizedConcatLayerForwardN(size_t count, const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float *scale, int32_t zero, uint8_t* dst)
        {
            __m256i _bias, _zero = _mm256_set1_epi32(zero);
            __m256 _norm, _scale = _mm256_set1_ps(scale[0]);
            for (size_t o = 0; o < num; ++o)
            {
                for (size_t s = 0; s < count; ++s)
                {
                    size_t size1 = size[s], size8 = size1 & (~7), size32 = size1 & (~31), i = 0;
                    const uint8_t* ps = src[s] + o * size1;
                    _bias = _mm256_set1_epi32(bias[s]);
                    _norm = _mm256_set1_ps(norm[s]);
                    for (; i < size32; i += 32)
                        DequantizeQuantizeLinear32(ps + i, _bias, _norm, _scale, _zero, dst + i);
                    for (; i < size8; i += 8)
                        DequantizeQuantizeLinear8(ps + i, _bias, _norm, _scale, _zero, dst + i);
                    for (; i < size1; i += 1)
                        DequantizeQuantizeLinear1(ps + i, _bias, _norm, _scale, _zero, dst + i);
                    dst += size1;
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
