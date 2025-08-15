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
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        void SynetQuantizedShuffleLayerForwardNchw0(const uint8_t* src0, int bias0, float norm0, size_t srcC0, const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cd = 0, spatial4  = AlignLo(spatial, 4), spatial16 = AlignLo(spatial, 16), s;
            __m128i _bias0 = _mm_set1_epi32(bias0), _bias1 = _mm_set1_epi32(bias1), _zero = _mm_set1_epi32(zero);
            __m128 _norm0 = _mm_set1_ps(norm0), _norm1 = _mm_set1_ps(norm1), _scale = _mm_set1_ps(scale);
            for (size_t cs = 0; cs < srcC0; cs += 2, cd += 1)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                src0 += spatial;
                dst1 += spatial;
            }
            for (size_t cs = 0; cs < srcC1; cs += 2, cd += 1)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                src1 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                src1 += spatial;
                dst1 += spatial;
            }        
        }

        void SynetQuantizedShuffleLayerForwardNhwc0(const uint8_t* src0, int bias0, float norm0, size_t srcC0, const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2;
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t cd = 0;
                for (size_t cs = 0; cs < srcC0; cs += 2, cd += 1)
                {
                    dst0[cd] = Base::DequantizeQuantizeLinear(src0[cs + 0], bias0, norm0, scale, zero, 0, 255);
                    dst1[cd] = Base::DequantizeQuantizeLinear(src0[cs + 1], bias0, norm0, scale, zero, 0, 255);
                }
                for (size_t cs = 0; cs < srcC1; cs += 2, cd += 1)
                {
                    dst0[cd] = Base::DequantizeQuantizeLinear(src1[cs + 0], bias1, norm1, scale, zero, 0, 255);
                    dst1[cd] = Base::DequantizeQuantizeLinear(src1[cs + 1], bias1, norm1, scale, zero, 0, 255);
                }
                src0 += srcC0;
                src1 += srcC1;
                dst0 += dstC;
                dst1 += dstC;
            }

        }

        void SynetQuantizedShuffleLayerForwardNchw1(const uint8_t* src0, int bias0, float norm0, size_t srcC0, const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2, cs = 0, spatial4 = AlignLo(spatial, 4), spatial16 = AlignLo(spatial, 16), s;
            __m128i _bias0 = _mm_set1_epi32(bias0), _bias1 = _mm_set1_epi32(bias1), _zero = _mm_set1_epi32(zero);
            __m128 _norm0 = _mm_set1_ps(norm0), _norm1 = _mm_set1_ps(norm1), _scale = _mm_set1_ps(scale);
            for (size_t cd = 0; cd < srcC0; cs += 1, cd += 2)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst0 + s);
                src0 += spatial;
                dst0 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst0 + s);
                src1 += spatial;
                dst0 += spatial;
            }
            for (size_t cd = 0; cd < srcC1; cs += 1, cd += 2)
            {
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src0 + s, _bias0, _norm0, _scale, _zero, dst1 + s);
                src0 += spatial;
                dst1 += spatial;
                for (s = 0; s < spatial16; s += 16)
                    DequantizeQuantizeLinear16(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial4; s += 4)
                    DequantizeQuantizeLinear4(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                for (; s < spatial; s += 1)
                    DequantizeQuantizeLinear1(src1 + s, _bias1, _norm1, _scale, _zero, dst1 + s);
                src1 += spatial;
                dst1 += spatial;
            }
        }

        void SynetQuantizedShuffleLayerForwardNhwc1(const uint8_t* src0, int bias0, float norm0, size_t srcC0, const uint8_t* src1, int bias1, float norm1, size_t srcC1, size_t spatial, uint8_t* dst0, uint8_t* dst1, float scale, int zero)
        {
            size_t dstC = (srcC0 + srcC1) / 2;
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t cs = 0;
                for (size_t cd = 0; cd < srcC0; cd += 2, cs += 1)
                {
                    dst0[cd + 0] = Base::DequantizeQuantizeLinear(src0[cs], bias0, norm0, scale, zero, 0, 255);
                    dst0[cd + 1] = Base::DequantizeQuantizeLinear(src1[cs], bias1, norm1, scale, zero, 0, 255);
                }
                for (size_t cd = 0; cd < srcC1; cd += 2, cs += 1)
                {
                    dst1[cd + 0] = Base::DequantizeQuantizeLinear(src0[cs], bias0, norm0, scale, zero, 0, 255);
                    dst1[cd + 1] = Base::DequantizeQuantizeLinear(src1[cs], bias1, norm1, scale, zero, 0, 255);
                }
                src0 += dstC;
                src1 += dstC;
                dst0 += srcC0;
                dst1 += srcC1;
            }
        }

        void SynetQuantizedShuffleLayerForward(const uint8_t* src0, int bias0, const float* norm0, size_t srcC0, const uint8_t* src1, int bias1, const float* norm1, size_t srcC1,
            size_t spatial, uint8_t* dst0, uint8_t* dst1, const float* scale, int zero, SimdTensorFormatType format, int shuffleType)
        {
            size_t dstC = (srcC0 + srcC1) / 2;
            switch (shuffleType)
            {
            case 0:
                if (format == SimdTensorFormatNhwc)
                    SynetQuantizedShuffleLayerForwardNhwc0(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                else
                    SynetQuantizedShuffleLayerForwardNchw0(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                break;
            case 1:
                if (format == SimdTensorFormatNhwc)
                    SynetQuantizedShuffleLayerForwardNhwc1(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                else
                    SynetQuantizedShuffleLayerForwardNchw1(src0, bias0, *norm0, srcC0, src1, bias1, *norm1, srcC1, spatial, dst0, dst1, *scale, zero);
                break;
            }
        }
    }
#endif
}
