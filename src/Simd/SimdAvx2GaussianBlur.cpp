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
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256 LoadAs32f(const uint8_t* src)
        {
            return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src))));
        }

        SIMD_INLINE void BlurColsAny(const uint8_t* src, size_t size, size_t channels, const float* weight, size_t kernel, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m256 sum = _mm256_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm256_fmadd_ps(_mm256_set1_ps(weight[k]), LoadAs32f(src + i + k * channels), sum);
                _mm256_storeu_ps(dst + i, sum);
            }
            for (; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * float(src[i + k * channels]);
                dst[i] = sum;
            }
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const __m256& f0)
        {
            __m256i i0 = _mm256_cvtps_epi32(f0);
            ((int64_t*)dst)[0] = Extract64i<0>(PackI16ToU8(PackI32ToI16(i0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const __m256& f0, const __m256& f1, const __m256& f2, const __m256& f3)
        {
            __m256i i0 = _mm256_cvtps_epi32(f0);
            __m256i i1 = _mm256_cvtps_epi32(f1);
            __m256i i2 = _mm256_cvtps_epi32(f2);
            __m256i i3 = _mm256_cvtps_epi32(f3);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(i0, i1), PackI32ToI16(i2, i3)));
        }

        SIMD_INLINE void BlurRowsAny(const float* src, size_t size, size_t stride, const float* weight, size_t kernel, uint8_t* dst)
        {
            size_t sizeA = AlignLo(size, A);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                {
                    __m256 w = _mm256_set1_ps(weight[k]);
                    const float* ps = src + i + k * stride;
                    sum0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(ps + 0 * F), sum0);
                    sum1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(ps + 1 * F), sum1);
                    sum2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(ps + 2 * F), sum2);
                    sum3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(ps + 3 * F), sum3);
                }
                StoreAs8u(dst + i, sum0, sum1, sum2, sum3);
            }
            for (; i < sizeF; i += F)
            {
                __m256 sum = _mm256_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm256_fmadd_ps(_mm256_set1_ps(weight[k]), _mm256_loadu_ps(src + i + k * stride), sum);
                StoreAs8u(dst + i, sum);
            }
            for (; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * src[i + k * stride];
                dst[i] = Round(sum);
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
            __m256 w[kernel];
            for (size_t k = 0; k < kernel; ++k)
                w[k] = _mm256_set1_ps(weight[k]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m256 sum = _mm256_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm256_fmadd_ps(w[k], LoadAs32f(src + i + k * channels), sum);
                _mm256_storeu_ps(dst + i, sum);
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
            __m256 w0 = _mm256_set1_ps(weight[0]);
            __m256 w1 = _mm256_set1_ps(weight[1]);
            __m256 w2 = _mm256_set1_ps(weight[2]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m256 sum = _mm256_mul_ps(w0, LoadAs32f(src + i + 0 * channels));
                sum = _mm256_fmadd_ps(w1, LoadAs32f(src + i + 1 * channels), sum);
                sum = _mm256_fmadd_ps(w2, LoadAs32f(src + i + 2 * channels), sum);
                _mm256_storeu_ps(dst + i, sum);
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
            __m256 w0 = _mm256_set1_ps(weight[0]);
            __m256 w1 = _mm256_set1_ps(weight[1]);
            __m256 w2 = _mm256_set1_ps(weight[2]);
            __m256 w3 = _mm256_set1_ps(weight[3]);
            __m256 w4 = _mm256_set1_ps(weight[4]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m256 sum0 = _mm256_mul_ps(w0, LoadAs32f(src + i + 0 * channels));
                __m256 sum1 = _mm256_mul_ps(w1, LoadAs32f(src + i + 1 * channels));
                sum0 = _mm256_fmadd_ps(w2, LoadAs32f(src + i + 2 * channels), sum0);
                sum1 = _mm256_fmadd_ps(w3, LoadAs32f(src + i + 3 * channels), sum1);
                sum0 = _mm256_fmadd_ps(w4, LoadAs32f(src + i + 4 * channels), sum0);
                _mm256_storeu_ps(dst + i, _mm256_add_ps(sum0, sum1));
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
            __m256 w[kernel];
            for (size_t k = 0; k < kernel; ++k)
                w[k] = _mm256_set1_ps(weight[k]);
            size_t sizeA = AlignLo(size, A);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                {
                    const float* ps = src + i + k * stride;
                    sum0 = _mm256_fmadd_ps(w[k], _mm256_loadu_ps(ps + 0 * F), sum0);
                    sum1 = _mm256_fmadd_ps(w[k], _mm256_loadu_ps(ps + 1 * F), sum1);
                    sum2 = _mm256_fmadd_ps(w[k], _mm256_loadu_ps(ps + 2 * F), sum2);
                    sum3 = _mm256_fmadd_ps(w[k], _mm256_loadu_ps(ps + 3 * F), sum3);
                }
                StoreAs8u(dst + i, sum0, sum1, sum2, sum3);
            }
            for (; i < sizeF; i += F)
            {
                __m256 sum = _mm256_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm256_fmadd_ps(w[k], _mm256_loadu_ps(src + i + k * stride), sum);
                StoreAs8u(dst + i, sum);
            }
            for (; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * src[i + k * stride];
                dst[i] = Round(sum);
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
            : Sse41::GaussianBlurDefault(param)
        {
            if (_param.width >= F)
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

        SIMD_INLINE __m256i DivideBy16(__m256i value)
        {
            return _mm256_srli_epi16(_mm256_add_epi16(value, K16_0008), 4);
        }

        const __m256i K8_01_02 = SIMD_MM256_SET2_EPI8(0x01, 0x02);

        template<int part> SIMD_INLINE __m256i BinomialSumUnpackedU8(__m256i a[3])
        {
            return _mm256_add_epi16(_mm256_maddubs_epi16(UnpackU8<part>(a[0], a[1]), K8_01_02), UnpackU8<part>(a[2]));
        }

        template<bool align> SIMD_INLINE void BlurCol(__m256i a[3], uint16_t * b)
        {
            Store<align>((__m256i*)b + 0, BinomialSumUnpackedU8<0>(a));
            Store<align>((__m256i*)b + 1, BinomialSumUnpackedU8<1>(a));
        }

        template<bool align> SIMD_INLINE __m256i BlurRow16(const Buffer & buffer, size_t offset)
        {
            return DivideBy16(BinomialSum16(
                Load<align>((__m256i*)(buffer.src0 + offset)),
                Load<align>((__m256i*)(buffer.src1 + offset)),
                Load<align>((__m256i*)(buffer.src2 + offset))));
        }

        template<bool align> SIMD_INLINE __m256i BlurRow(const Buffer & buffer, size_t offset)
        {
            return _mm256_packus_epi16(BlurRow16<align>(buffer, offset), BlurRow16<align>(buffer, offset + HA));
        }

        template <bool align, size_t step> void GaussianBlur3x3(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(step*(width - 1) >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(step*width) && Aligned(dst) && Aligned(dstStride));

            __m256i a[3];

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
            BlurCol<true>(a, buffer.src0 + bodySize);

            memcpy(buffer.src1, buffer.src0, sizeof(uint16_t)*(bodySize + A));

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
                BlurCol<true>(a, buffer.src2 + bodySize);

                for (size_t col = 0; col < bodySize; col += A)
                    Store<align>((__m256i*)(dst + col), BlurRow<true>(buffer, col));
                Store<align>((__m256i*)(dst + size - A), BlurRow<true>(buffer, bodySize));

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
#endif// SIMD_AVX2_ENABLE
}
