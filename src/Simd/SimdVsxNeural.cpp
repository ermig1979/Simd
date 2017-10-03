/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        template <bool inversion> v128_u8 Invert(v128_u8 value);

        template <> v128_u8 Invert<true>(v128_u8 value)
        {
            return vec_sub(K8_FF, value);
        }

        template <> v128_u8 Invert<false>(v128_u8 value)
        {
            return value;
        }

        const v128_u8 K8_PERM_0 = SIMD_VEC_SETR_EPI8(0x10, 0x10, 0x10, 0x00, 0x10, 0x10, 0x10, 0x01, 0x10, 0x10, 0x10, 0x02, 0x10, 0x10, 0x10, 0x03);
        const v128_u8 K8_PERM_1 = SIMD_VEC_SETR_EPI8(0x10, 0x10, 0x10, 0x04, 0x10, 0x10, 0x10, 0x05, 0x10, 0x10, 0x10, 0x06, 0x10, 0x10, 0x10, 0x07);
        const v128_u8 K8_PERM_2 = SIMD_VEC_SETR_EPI8(0x10, 0x10, 0x10, 0x08, 0x10, 0x10, 0x10, 0x09, 0x10, 0x10, 0x10, 0x0A, 0x10, 0x10, 0x10, 0x0B);
        const v128_u8 K8_PERM_3 = SIMD_VEC_SETR_EPI8(0x10, 0x10, 0x10, 0x0C, 0x10, 0x10, 0x10, 0x0D, 0x10, 0x10, 0x10, 0x0E, 0x10, 0x10, 0x10, 0x0F);

        template <bool inversion, bool align, bool first> void Convert(const uint8_t * src, const v128_f32 &_1_255, Storer<align> & dst)
        {
            v128_u8 _src = Invert<inversion>(Load<align>(src));
            Store<align, first>(dst, vec_mul(vec_ctf((v128_u32)vec_perm(_src, K8_00, K8_PERM_0), 0), _1_255));
            Store<align, false>(dst, vec_mul(vec_ctf((v128_u32)vec_perm(_src, K8_00, K8_PERM_1), 0), _1_255));
            Store<align, false>(dst, vec_mul(vec_ctf((v128_u32)vec_perm(_src, K8_00, K8_PERM_2), 0), _1_255));
            Store<align, false>(dst, vec_mul(vec_ctf((v128_u32)vec_perm(_src, K8_00, K8_PERM_3), 0), _1_255));
        }

        template <bool inversion, bool align> void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, A);
            v128_f32 _1_255 = SIMD_VEC_SET1_PS(1.0f / 255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _dst(dst);
                Convert<inversion, align, true>(src, _1_255, _dst);
                for (size_t col = A; col < alignedWidth; col += A)
                    Convert<inversion, align, false>(src + col, _1_255, _dst);
                Flush(_dst);
                if (width != alignedWidth)
                {
                    Storer<false> _dst(dst + width - A);
                    Convert<inversion, false, true>(src + width - A, _1_255, _dst);
                    Flush(_dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool inversion> void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                NeuralConvert<inversion, true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralConvert<inversion, false>(src, srcStride, width, height, dst, dstStride);
        }

        void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride, int inversion)
        {
            if (inversion)
                NeuralConvert<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralConvert<false>(src, srcStride, width, height, dst, dstStride);
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t offset, v128_f32 & sum)
        {
            v128_f32 _a = Load<align>(a + offset);
            v128_f32 _b = Load<align>(b + offset);
            sum = vec_add(sum, vec_mul(_a, _b));
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, 4);
            size_t fullAlignedSize = AlignLo(size, 16);
            size_t i = 0;
            if (partialAlignedSize)
            {
                v128_f32 sums[4] = { K_0_0f, K_0_0f, K_0_0f, K_0_0f };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += 16)
                    {
                        NeuralProductSum<align>(a, b, i, sums[0]);
                        NeuralProductSum<align>(a, b, i + 4, sums[1]);
                        NeuralProductSum<align>(a, b, i + 8, sums[2]);
                        NeuralProductSum<align>(a, b, i + 12, sums[3]);
                    }
                    sums[0] = vec_add(vec_add(sums[0], sums[1]), vec_add(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += 4)
                    NeuralProductSum<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for (; i < size; ++i)
                *sum += a[i] * b[i];
        }

        void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                NeuralProductSum<true>(a, b, size, sum);
            else
                NeuralProductSum<false>(a, b, size, sum);
        }

        template <bool align, bool first> SIMD_INLINE void NeuralRoughSigmoid(const float * src, const v128_f32 & slope,
            const v128_f32 & _0, const v128_f32 & _1, const v128_f32 & _a, const v128_f32 & _b, Storer<align> & dst)
        {
            v128_f32 _src = Load<align>(src);
            v128_f32 x = vec_abs(vec_mul(_src, slope));
            v128_f32 x2 = vec_mul(x, x);
            v128_f32 x4 = vec_mul(x2, x2);
            v128_f32 series = vec_add(vec_add(_1, x), vec_add(vec_mul(x2, _a), vec_mul(x4, _b)));
            v128_f32 exp = vec_sel(series, vec_div(_1, series), vec_cmpgt(_src, _0));
            v128_f32 sigmoid = vec_div(_1, vec_add(_1, exp));
            Store<align, first>(dst, sigmoid);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t alignedSize = Simd::AlignLo(size, 4);
            const v128_f32 _slope = SetF32(*slope);
            const v128_f32 _0 = SetF32(0.0f);
            const v128_f32 _1 = SetF32(1.0f);
            const v128_f32 _a = SetF32(0.5417f);
            const v128_f32 _b = SetF32(0.1460f);

            Storer<align> _dst(dst);
            NeuralRoughSigmoid<align, true>(src, _slope, _0, _1, _a, _b, _dst);
            for (size_t i = 4; i < alignedSize; i += 4)
                NeuralRoughSigmoid<align, false>(src + i, _slope, _0, _1, _a, _b, _dst);
            Flush(_dst);

            for (size_t i = alignedSize; i < size; ++i)
                dst[i] = Base::RoughSigmoid(src[i] * slope[0]);
        }

        void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid<false>(src, size, slope, dst);
        }
    }
#endif// SIMD_VSX_ENABLE
}
