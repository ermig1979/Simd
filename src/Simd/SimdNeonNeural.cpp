/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool inversion> uint8x16_t Invert(const uint8x16_t & value);

        template <> uint8x16_t Invert<true>(const uint8x16_t & value)
        {
            return vsubq_u8(K8_FF, value);
        }

        template <> uint8x16_t Invert<false>(const uint8x16_t & value)
        {
            return value;
        }

        template <bool align> void Convert(const uint16x8_t & src, const float32x4_t &_1_255, float * dst)
        {
            Store<align>(dst + 0, vmulq_f32(ToFloat<0>(src), _1_255));
            Store<align>(dst + F, vmulq_f32(ToFloat<1>(src), _1_255));
        }

        template <bool inversion, bool align> void Convert(const uint8_t * src, const float32x4_t &_1_255, float * dst)
        {
            uint8x16_t _src = Invert<inversion>(Load<align>(src));
            Convert<align>(UnpackU8<0>(_src), _1_255, dst + 0);
            Convert<align>(UnpackU8<1>(_src), _1_255, dst + DF);
        }

        template <bool inversion, bool align> void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride) && Aligned(dst) && Aligned(width));

            size_t alignedWidth = AlignLo(width, A);
            float32x4_t _1_255 = vdupq_n_f32(1.0f / 255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    Convert<inversion, align>(src + col, _1_255, dst + col);
                if (width != alignedWidth)
                    Convert<inversion, false>(src + width - A, _1_255, dst + width - A);
                src += stride;
                dst += width;
            }
        }

        template <bool inversion> void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst)
        {
            if (Aligned(src) && Aligned(stride) && Aligned(dst) && Aligned(width))
                NeuralConvert<inversion, true>(src, stride, width, height, dst);
            else
                NeuralConvert<inversion, false>(src, stride, width, height, dst);
        }

        void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst, int inversion)
        {
            if (inversion)
                NeuralConvert<true>(src, stride, width, height, dst);
            else
                NeuralConvert<false>(src, stride, width, height, dst);
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t offset, float32x4_t & sum)
        {
            float32x4_t _a = Load<align>(a + offset);
            float32x4_t _b = Load<align>(b + offset);
            sum = vaddq_f32(sum, vmulq_f32(_a, _b));
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, DF);
            size_t i = 0;
            if(partialAlignedSize)
            {
                float32x4_t sums[2] = {vdupq_n_f32(0), vdupq_n_f32(0)};
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += DF)
                    {
                        NeuralProductSum<align>(a, b, i + 0, sums[0]);
                        NeuralProductSum<align>(a, b, i + F, sums[1]);
                    }
                    sums[0] = vaddq_f32(sums[0], sums[1]);
                }
                for(; i < partialAlignedSize; i += F)
                    NeuralProductSum<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for(; i < size; ++i)
                *sum += a[i]*b[i];
        }

        void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(Aligned(a) && Aligned(b))
                NeuralProductSum<true>(a, b, size, sum);
            else
                NeuralProductSum<false>(a, b, size, sum);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            float32x4_t _slope = vdupq_n_f32(*slope);
            float32x4_t _0 = vdupq_n_f32(-0.0f);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            float32x4_t _a = vdupq_n_f32(0.5417f);
            float32x4_t _b = vdupq_n_f32(0.1460f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                float32x4_t _src = Load<align>(src + i);
                float32x4_t x = vabsq_f32(vmulq_f32(_src, _slope));
                float32x4_t x2 = vmulq_f32(x, x);
                float32x4_t x4 = vmulq_f32(x2, x2);
                float32x4_t series = vaddq_f32(vaddq_f32(_1, x), vaddq_f32(vmulq_f32(x2, _a), vmulq_f32(x4, _b)));
                uint32x4_t mask = vcgtq_f32(_src, _0);
                float32x4_t exp = vbslq_f32(mask, Reciprocal<1>(series), series);
                float32x4_t sigmoid = Reciprocal<1>(vaddq_f32(_1, exp));
                Store<align>(dst + i, sigmoid);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughSigmoid(src[i] * slope[0]);
        }

        void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, const float32x4_t & k, const float32x4_t & o, const float32x4_t & m, float * dst)
        {
            float32x4_t _src = Load<align>(src);
            float32x4_t e1 = vmaxq_f32(m, vsubq_f32(o, vmulq_f32(_src, k)));
            float32x4_t e2 = vmulq_f32(e1, e1);
            float32x4_t e4 = vmulq_f32(e2, e2);
            float32x4_t e8 = vmulq_f32(e4, e4);
            float32x4_t e16 = vmulq_f32(e8, e8);
            float32x4_t e32 = vmulq_f32(e16, e16);
            float32x4_t e64 = vmulq_f32(e32, e32);
            float32x4_t sigmoid = Reciprocal<1>(vaddq_f32(o, vmulq_f32(e64, e64)));
            Store<align>(dst, sigmoid);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            float32x4_t _k = vdupq_n_f32((*slope)*0.0078125f);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            float32x4_t _05 = vdupq_n_f32(0.5f);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                NeuralRoughSigmoid2<align>(src + i + 0 * F, _k, _1, _05, dst + i + 0 * F);
                NeuralRoughSigmoid2<align>(src + i + 1 * F, _k, _1, _05, dst + i + 1 * F);
                NeuralRoughSigmoid2<align>(src + i + 2 * F, _k, _1, _05, dst + i + 2 * F);
                NeuralRoughSigmoid2<align>(src + i + 3 * F, _k, _1, _05, dst + i + 3 * F);
            }
            for (; i < partialAlignedSize; i += F)
                NeuralRoughSigmoid2<align>(src + i, _k, _1, _05, dst + i);
            for (; i < size; ++i)
                dst[i] = Base::RoughSigmoid2(src[i] * slope[0]);
        }

        void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid2<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid2<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void NeuralRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            float32x4_t _slope = vdupq_n_f32(*slope);
            float32x4_t _0 = vdupq_n_f32(-0.0f);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            float32x4_t _a = vdupq_n_f32(0.5658f);
            float32x4_t _b = vdupq_n_f32(0.1430f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                float32x4_t _src = Load<align>(src + i);
                float32x4_t x = vabsq_f32(vmulq_f32(_src, _slope));
                float32x4_t x2 = vmulq_f32(x, x);
                float32x4_t x4 = vmulq_f32(x2, x2);
                float32x4_t pe = vaddq_f32(vaddq_f32(_1, x), vaddq_f32(vmulq_f32(x2, _a), vmulq_f32(x4, _b)));
                float32x4_t ne = Reciprocal<1>(pe);
                float32x4_t absTanh = vmulq_f32(vsubq_f32(pe, ne), Reciprocal<1>(vaddq_f32(pe, ne)));
                float32x4_t tanh = (float32x4_t)veorq_u32((uint32x4_t)absTanh, vandq_u32((uint32x4_t)_0, vcgtq_f32(_0, _src)));
                Store<align>(dst + i, tanh);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughTanh(src[i] * slope[0]);
        }

        void NeuralRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughTanh<true>(src, size, slope, dst);
            else
                NeuralRoughTanh<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void NeuralRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float s = slope[0];
            assert(s >= 0.0f && s <= 1.0f);
            size_t alignedSize = Simd::AlignLo(size, F);
            size_t i = 0;
            if (s == 0)
            {
                float32x4_t _0 = vdupq_n_f32(0.0f);
                for (; i < alignedSize; i += F)
                {
                    float32x4_t _src = Load<align>(src + i);
                    float32x4_t relu = vmaxq_f32(_0, _src);
                    Store<align>(dst + i, relu);
                }
                for (; i < size; ++i)
                    dst[i] = Simd::Max(0.0f, src[i]);
            }
            else
            {
                float32x4_t _s = vdupq_n_f32(s);
                for (; i < alignedSize; i += F)
                {
                    float32x4_t _src = Load<align>(src + i);
                    float32x4_t relu = vmaxq_f32(vmulq_f32(_src, _s), _src);
                    Store<align>(dst + i, relu);
                }
                for (; i < size; ++i)
                    dst[i] = Simd::Max(src[i] * s, src[i]);
            }
        }

        void NeuralRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRelu<true>(src, size, slope, dst);
            else
                NeuralRelu<false>(src, size, slope, dst);
        }
    }
#endif// SIMD_NEON_ENABLE
}
