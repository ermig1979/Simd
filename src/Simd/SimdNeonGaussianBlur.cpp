/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdLoadBlock.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdGaussianBlur.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE float32x4_t LoadAs32f(const uint8_t* src)
        {
            return vcvtq_f32_u32(UnpackU16<0>(vmovl_u8((uint8x8_t)vdup_n_u32(*(uint32_t*)src))));
        }

        SIMD_INLINE void BlurColsAny(const uint8_t* src, size_t size, size_t channels, const float* weight, size_t kernel, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = vmlaq_f32(sum, vdupq_n_f32(weight[k]), LoadAs32f(src + i + k * channels));
                Store<false>(dst + i, sum);
            }
            for (; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * float(src[i + k * channels]);
                dst[i] = sum;
            }
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const float32x4_t& f0)
        {
            uint32x4_t i0 = vcvtq_u32_f32(f0);
            ((uint32_t*)dst)[0] = vget_lane_u32((uint32x2_t)vqmovn_u16(vcombine_u16(vqmovn_u32(i0), vcreate_u16(0))), 0);
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const float32x4_t& f0, const float32x4_t& f1, const float32x4_t& f2, const float32x4_t& f3)
        {
            uint32x4_t i0 = vcvtq_u32_f32(f0);
            uint32x4_t i1 = vcvtq_u32_f32(f1);
            uint32x4_t i2 = vcvtq_u32_f32(f2);
            uint32x4_t i3 = vcvtq_u32_f32(f3);
            Store<false>(dst, PackU16(PackU32(i0, i1), PackU32(i2, i3)));
        }

        SIMD_INLINE void BlurRowsAny(const float* src, size_t size, size_t stride, const float* weight, size_t kernel, uint8_t* dst)
        {
            size_t sizeA = AlignLo(size, A);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                {
                    float32x4_t w = vdupq_n_f32(weight[k]);
                    const float* ps = src + i + k * stride;
                    sum0 = vmlaq_f32(sum0, w, Load<false>(ps + 0 * F));
                    sum1 = vmlaq_f32(sum1, w, Load<false>(ps + 1 * F));
                    sum2 = vmlaq_f32(sum2, w, Load<false>(ps + 2 * F));
                    sum3 = vmlaq_f32(sum3, w, Load<false>(ps + 3 * F));
                }
                StoreAs8u(dst + i, sum0, sum1, sum2, sum3);
            }
            for (; i < sizeF; i += F)
            {
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = vmlaq_f32(sum, vdupq_n_f32(weight[k]), Load<false>(src + i + k * stride));
                StoreAs8u(dst + i, sum);
            }
            for (; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * src[i + k * stride];
                dst[i] = Simd::Round(sum);
            }
        }

        template<int channels> void BlurImageAny(const BlurParam& p, const Base::AlgDefault& a, const uint8_t* src, size_t srcStride, uint8_t* cols, float* rows, uint8_t* dst, size_t dstStride)
        {
            Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
            BlurColsAny(cols, a.size, p.channels, a.weight.data, a.kernel, rows + a.half * a.stride);
            for (size_t row = 0; row < a.half; ++row)
                memcpy(rows + row * a.stride, rows + a.half * a.stride, a.size * sizeof(float));
            for (size_t row = 1; row < a.nose; ++row)
            {
                Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurColsAny(cols, a.size, p.channels, a.weight.data, a.kernel, rows + (a.half + row) * a.stride);
            }
            for (size_t row = a.nose; row <= a.half; ++row)
                memcpy(rows + (a.half + row) * a.stride, rows + (a.half + a.nose - 1) * a.stride, a.size * sizeof(float));
            BlurRowsAny(rows, a.size, a.stride, a.weight.data, a.kernel, dst), dst += dstStride;

            for (size_t row = 1, b = row % a.kernel + 2 * a.half, w = a.kernel - row % a.kernel; row < a.body; ++row, ++b, --w)
            {
                if (b >= a.kernel)
                    b -= a.kernel;
                if (w == 0)
                    w += a.kernel;
                Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurColsAny(cols, a.size, p.channels, a.weight.data, a.kernel, rows + b * a.stride);
                BlurRowsAny(rows, a.size, a.stride, a.weight.data + w, a.kernel, dst), dst += dstStride;
            }

            size_t last = (a.body + 2 * a.half - 1) % a.kernel;
            for (size_t row = a.body, b = row % a.kernel + 2 * a.half, w = a.kernel - row % a.kernel; row < p.height; ++row, ++b, --w)
            {
                if (b >= a.kernel)
                    b -= a.kernel;
                if (w == 0)
                    w += a.kernel;
                memcpy(rows + b * a.stride, rows + last * a.stride, a.size * sizeof(float));
                BlurRowsAny(rows, a.size, a.stride, a.weight.data + w, a.kernel, dst), dst += dstStride;
            }
        }

        //---------------------------------------------------------------------

        template<int kernel> SIMD_INLINE void BlurCols(const uint8_t* src, size_t size, size_t channels, const float* weight, float* dst)
        {
            float32x4_t w[kernel];
            for (size_t k = 0; k < kernel; ++k)
                w[k] = vdupq_n_f32(weight[k]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = vmlaq_f32(sum, w[k], LoadAs32f(src + i + k * channels));
                Store<false>(dst + i, sum);
            }
            for (; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * float(src[i + k * channels]);
                dst[i] = sum;
            }
        }

        template<> SIMD_INLINE void BlurCols<3>(const uint8_t* src, size_t size, size_t channels, const float* weight, float* dst)
        {
            float32x4_t w0 = vdupq_n_f32(weight[0]);
            float32x4_t w1 = vdupq_n_f32(weight[1]);
            float32x4_t w2 = vdupq_n_f32(weight[2]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                float32x4_t sum = vmulq_f32(w0, LoadAs32f(src + i + 0 * channels));
                sum = vmlaq_f32(sum, w1, LoadAs32f(src + i + 1 * channels));
                sum = vmlaq_f32(sum, w2, LoadAs32f(src + i + 2 * channels));
                Store<false>(dst + i, sum);
            }
            for (; i < size; ++i)
            {
                dst[i] =
                    weight[0] * float(src[i + 0 * channels]) +
                    weight[1] * float(src[i + 1 * channels]) +
                    weight[2] * float(src[i + 2 * channels]);
            }
        }

        template<> SIMD_INLINE void BlurCols<5>(const uint8_t* src, size_t size, size_t channels, const float* weight, float* dst)
        {
            float32x4_t w0 = vdupq_n_f32(weight[0]);
            float32x4_t w1 = vdupq_n_f32(weight[1]);
            float32x4_t w2 = vdupq_n_f32(weight[2]);
            float32x4_t w3 = vdupq_n_f32(weight[3]);
            float32x4_t w4 = vdupq_n_f32(weight[4]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                float32x4_t sum0 = vmulq_f32(w0, LoadAs32f(src + i + 0 * channels));
                float32x4_t sum1 = vmulq_f32(w1, LoadAs32f(src + i + 1 * channels));
                sum0 = vmlaq_f32(sum0, w2, LoadAs32f(src + i + 2 * channels));
                sum1 = vmlaq_f32(sum1, w3, LoadAs32f(src + i + 3 * channels));
                sum0 = vmlaq_f32(sum0, w4, LoadAs32f(src + i + 4 * channels));
                Store<false>(dst + i, vaddq_f32(sum0, sum1));
            }
            for (; i < size; ++i)
            {
                dst[i] =
                    weight[0] * float(src[i + 0 * channels]) +
                    weight[1] * float(src[i + 1 * channels]) +
                    weight[2] * float(src[i + 2 * channels]) +
                    weight[3] * float(src[i + 3 * channels]) +
                    weight[4] * float(src[i + 4 * channels]);
            }
        }

        template<int kernel> SIMD_INLINE void BlurRows(const float* src, size_t size, size_t stride, const float* weight, uint8_t* dst)
        {
            float32x4_t w[kernel];
            for (size_t k = 0; k < kernel; ++k)
                w[k] = vdupq_n_f32(weight[k]);
            size_t sizeA = AlignLo(size, A);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                float32x4_t sum2 = vdupq_n_f32(0.0f);
                float32x4_t sum3 = vdupq_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                {
                    const float* ps = src + i + k * stride;
                    sum0 = vmlaq_f32(sum0, w[k], Load<false>(ps + 0 * F));
                    sum1 = vmlaq_f32(sum1, w[k], Load<false>(ps + 1 * F));
                    sum2 = vmlaq_f32(sum2, w[k], Load<false>(ps + 2 * F));
                    sum3 = vmlaq_f32(sum3, w[k], Load<false>(ps + 3 * F));
                }
                StoreAs8u(dst + i, sum0, sum1, sum2, sum3);
            }
            for (; i < sizeF; i += F)
            {
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = vmlaq_f32(sum, w[k], Load<false>(src + i + k * stride));
                StoreAs8u(dst + i, sum);
            }
            for (; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * src[i + k * stride];
                dst[i] = Simd::Round(sum);
            }
        }

        template<int channels, int kernel> void BlurImage(const BlurParam& p, const Base::AlgDefault& a, const uint8_t* src, size_t srcStride, uint8_t* cols, float* rows, uint8_t* dst, size_t dstStride)
        {
            Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
            BlurCols<kernel>(cols, a.size, p.channels, a.weight.data, rows + a.half * a.stride);
            for (size_t row = 0; row < a.half; ++row)
                memcpy(rows + row * a.stride, rows + a.half * a.stride, a.size * sizeof(float));
            for (size_t row = 1; row < a.nose; ++row)
            {
                Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurCols<kernel>(cols, a.size, p.channels, a.weight.data, rows + (a.half + row) * a.stride);
            }
            for (size_t row = a.nose; row <= a.half; ++row)
                memcpy(rows + (a.half + row) * a.stride, rows + (a.half + a.nose - 1) * a.stride, a.size * sizeof(float));
            BlurRows<kernel>(rows, a.size, a.stride, a.weight.data, dst), dst += dstStride;

            for (size_t row = 1, b = row % a.kernel + 2 * a.half, w = kernel - row % kernel; row < a.body; ++row, ++b, --w)
            {
                if (b >= kernel)
                    b -= kernel;
                if (w == 0)
                    w += kernel;
                Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurCols<kernel>(cols, a.size, p.channels, a.weight.data, rows + b * a.stride);
                BlurRows<kernel>(rows, a.size, a.stride, a.weight.data + w, dst), dst += dstStride;
            }

            size_t last = (a.body + 2 * a.half - 1) % kernel;
            for (size_t row = a.body, b = row % kernel + 2 * a.half, w = kernel - row % kernel; row < p.height; ++row, ++b, --w)
            {
                if (b >= kernel)
                    b -= kernel;
                if (w == 0)
                    w += kernel;
                memcpy(rows + b * a.stride, rows + last * a.stride, a.size * sizeof(float));
                BlurRows<kernel>(rows, a.size, a.stride, a.weight.data + w, dst), dst += dstStride;
            }
        }

        //---------------------------------------------------------------------

        template<int channels> Base::BlurDefaultPtr GetBlurDefaultPtr(const BlurParam& p, const Base::AlgDefault& a)
        {
            switch (a.kernel)
            {
            case 3: return BlurImage<channels, 3>;
            case 5: return BlurImage<channels, 5>;
            case 7: return BlurImage<channels, 7>;
            case 9: return BlurImage<channels, 9>;
            default: return BlurImageAny<channels>;
            }
        }

        //---------------------------------------------------------------------

        GaussianBlurDefault::GaussianBlurDefault(const BlurParam& param)
            : Base::GaussianBlurDefault(param)
        {
            if (_param.width >= F && _alg.kernel > 5)
            {
                switch (_param.channels)
                {
                case 1: _blur = GetBlurDefaultPtr<1>(_param, _alg); break;
                case 2: _blur = GetBlurDefaultPtr<2>(_param, _alg); break;
                case 3: _blur = GetBlurDefaultPtr<3>(_param, _alg); break;
                case 4: _blur = GetBlurDefaultPtr<4>(_param, _alg); break;
                }
            }
        }

        //---------------------------------------------------------------------

        void* GaussianBlurInit(size_t width, size_t height, size_t channels, const float* sigma, const float* epsilon)
        {
            BlurParam param(width, height, channels, sigma, epsilon, A);
            if (!param.Valid())
                return NULL;
            return new GaussianBlurDefault(param);
        }

        //---------------------------------------------------------------------

        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t) * 3 * width);
                    src0 = (uint16_t*)_p;
                    src1 = src0 + width;
                    src2 = src1 + width;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * src0;
                uint16_t * src1;
                uint16_t * src2;
            private:
                void * _p;
            };
        }

        template<bool align> SIMD_INLINE void BlurCol(uint8x16_t a[3], uint16_t * b)
        {
            Store<align>(b + 0, BinomialSum16(UnpackU8<0>(a[0]), UnpackU8<0>(a[1]), UnpackU8<0>(a[2])));
            Store<align>(b + HA, BinomialSum16(UnpackU8<1>(a[0]), UnpackU8<1>(a[1]), UnpackU8<1>(a[2])));
        }

        template<bool align> SIMD_INLINE uint16x8_t BlurRow16(const Buffer & buffer, size_t offset)
        {
            return DivideBy16(BinomialSum16(
                Load<align>(buffer.src0 + offset),
                Load<align>(buffer.src1 + offset),
                Load<align>(buffer.src2 + offset)));
        }

        template<bool align> SIMD_INLINE uint8x16_t BlurRow(const Buffer & buffer, size_t offset)
        {
            return PackU16(BlurRow16<align>(buffer, offset), BlurRow16<align>(buffer, offset + HA));
        }

        template <bool align, size_t step> void GaussianBlur3x3(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(step*(width - 1) >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(step*width) && Aligned(dst) && Aligned(dstStride));

            uint8x16_t a[3];

            size_t size = step*width;
            size_t bodySize = Simd::AlignHi(size, A) - A;

            Buffer buffer(Simd::AlignHi(size, A));

            LoadNose3<align, step>(src + 0, a);
            BlurCol<true>(a, buffer.src0 + 0);
            for (size_t col = A; col < bodySize; col += A)
            {
                LoadBody3<align, step>(src + col, a);
                BlurCol<true>(a, buffer.src0 + col);
            }
            LoadTail3<align, step>(src + size - A, a);
            BlurCol<align>(a, buffer.src0 + size - A);

            memcpy(buffer.src1, buffer.src0, sizeof(uint16_t)*size);

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                const uint8_t *src2 = src + srcStride*(row + 1);
                if (row >= height - 2)
                    src2 = src + srcStride*(height - 1);

                LoadNose3<align, step>(src2 + 0, a);
                BlurCol<true>(a, buffer.src2 + 0);
                for (size_t col = A; col < bodySize; col += A)
                {
                    LoadBody3<align, step>(src2 + col, a);
                    BlurCol<true>(a, buffer.src2 + col);
                }
                LoadTail3<align, step>(src2 + size - A, a);
                BlurCol<align>(a, buffer.src2 + size - A);

                for (size_t col = 0; col < bodySize; col += A)
                    Store<align>(dst + col, BlurRow<true>(buffer, col));
                Store<align>(dst + size - A, BlurRow<align>(buffer, size - A));

                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src0, buffer.src1);
            }
        }

        template <bool align> void GaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: GaussianBlur3x3<align, 1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: GaussianBlur3x3<align, 2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: GaussianBlur3x3<align, 3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: GaussianBlur3x3<align, 4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        void GaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(channelCount*width) && Aligned(dst) && Aligned(dstStride))
                GaussianBlur3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride);
            else
                GaussianBlur3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
