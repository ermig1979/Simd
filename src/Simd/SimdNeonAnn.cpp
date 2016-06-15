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
            Store<align>(dst + 4, vmulq_f32(ToFloat<1>(src), _1_255));
        }

        template <bool inversion, bool align> void Convert(const uint8_t * src, const float32x4_t &_1_255, float * dst)
        {
            uint8x16_t _src = Invert<inversion>(Load<align>(src));
            Convert<align>(UnpackU8<0>(_src), _1_255, dst + 0);
            Convert<align>(UnpackU8<1>(_src), _1_255, dst + 8);
        }

        template <bool inversion, bool align> void AnnConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst)
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

        template <bool inversion> void AnnConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst)
        {
            if (Aligned(src) && Aligned(stride) && Aligned(dst) && Aligned(width))
                AnnConvert<inversion, true>(src, stride, width, height, dst);
            else
                AnnConvert<inversion, false>(src, stride, width, height, dst);
        }

        void AnnConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst, int inversion)
        {
            if (inversion)
                AnnConvert<true>(src, stride, width, height, dst);
            else
                AnnConvert<false>(src, stride, width, height, dst);
        }

        template <bool align> SIMD_INLINE void AnnProductSum(const float * a, const float * b, size_t offset, float32x4_t & sum)
        {
            float32x4_t _a = Load<align>(a + offset);
            float32x4_t _b = Load<align>(b + offset);
            sum = vaddq_f32(sum, vmulq_f32(_a, _b));
        }

        template <bool align> SIMD_INLINE void AnnProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, 4);
            size_t fullAlignedSize = AlignLo(size, 8);
            size_t i = 0;
            if(partialAlignedSize)
            {
                float32x4_t sums[2] = {vdupq_n_f32(0), vdupq_n_f32(0)};
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += 8)
                    {
                        AnnProductSum<align>(a, b, i + 0, sums[0]);
                        AnnProductSum<align>(a, b, i + 4, sums[1]);
                    }
                    sums[0] = vaddq_f32(sums[0], sums[1]);
                }
                for(; i < partialAlignedSize; i += 4)
					AnnProductSum<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for(; i < size; ++i)
                *sum += a[i]*b[i];
        }

        void AnnProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(Aligned(a) && Aligned(b))
				AnnProductSum<true>(a, b, size, sum);
            else
				AnnProductSum<false>(a, b, size, sum);
        }

        template <bool align> SIMD_INLINE void AnnRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            float32x4_t _slope = vdupq_n_f32(*slope);
            float32x4_t _0 = vdupq_n_f32(-0.0f);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            float32x4_t _0555 = vdupq_n_f32(0.555f);
            float32x4_t _0143 = vdupq_n_f32(0.143f);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                float32x4_t _src = Load<align>(src + i);
                float32x4_t x = vabsq_f32(vmulq_f32(_src, _slope));
                float32x4_t x2 = vmulq_f32(x, x);
                float32x4_t x4 = vmulq_f32(x2, x2);
                float32x4_t series = vaddq_f32(vaddq_f32(_1, x), vaddq_f32(vmulq_f32(x2, _0555), vmulq_f32(x4, _0143)));
                uint32x4_t mask = vcgtq_f32(_src, _0);
                float32x4_t exp = vbslq_f32(mask, Reciprocal<1>(series), series);
                float32x4_t sigmoid = Reciprocal<1>(vaddq_f32(_1, exp));
                Store<align>(dst + i, sigmoid);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughSigmoid(src[i] * slope[0]);
        }

        void AnnRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                AnnRoughSigmoid<true>(src, size, slope, dst);
            else
                AnnRoughSigmoid<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void AnnRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            float32x4_t _slope = vdupq_n_f32(*slope);
            float32x4_t _0 = vdupq_n_f32(-0.0f);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            float32x4_t _0559 = vdupq_n_f32(0.559f);
            float32x4_t _0148 = vdupq_n_f32(0.148f);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                float32x4_t _src = Load<align>(src + i);
                float32x4_t x = vabsq_f32(vmulq_f32(_src, _slope));
                float32x4_t x2 = vmulq_f32(x, x);
                float32x4_t x4 = vmulq_f32(x2, x2);
                float32x4_t pe = vaddq_f32(vaddq_f32(_1, x), vaddq_f32(vmulq_f32(x2, _0559), vmulq_f32(x4, _0148)));
                float32x4_t ne = Reciprocal<1>(pe);
                float32x4_t absTanh = vmulq_f32(vsubq_f32(pe, ne), Reciprocal<1>(vaddq_f32(pe, ne)));
                float32x4_t tanh = (float32x4_t)veorq_u32((uint32x4_t)absTanh, vandq_u32((uint32x4_t)_0, vcgtq_f32(_0, _src)));
                Store<align>(dst + i, tanh);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughTanh(src[i] * slope[0]);
        }

        void AnnRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                AnnRoughTanh<true>(src, size, slope, dst);
            else
                AnnRoughTanh<false>(src, size, slope, dst);
        }
    }
#endif// SIMD_NEON_ENABLE
}
