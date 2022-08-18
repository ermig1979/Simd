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

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE __m128 LoadAs32f(const uint8_t * src)
        {
            return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)src)));
        }

        SIMD_INLINE void BlurColsAny(const uint8_t* src, size_t size, size_t channels, const float* weight, size_t kernel, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(weight[k]), LoadAs32f(src + i + k * channels)), sum);
                _mm_storeu_ps(dst + i, sum);
            }
            for (; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * float(src[i + k * channels]);
                dst[i] = sum;
            }
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const __m128 & f0)
        {
            __m128i i0 = _mm_cvtps_epi32(f0);
            ((int32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(i0, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const __m128& f0, const __m128& f1, const __m128& f2, const __m128& f3)
        {
            __m128i i0 = _mm_cvtps_epi32(f0);
            __m128i i1 = _mm_cvtps_epi32(f1);
            __m128i i2 = _mm_cvtps_epi32(f2);
            __m128i i3 = _mm_cvtps_epi32(f3);
            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(i0, i1), _mm_packs_epi32(i2, i3)));
        }

        SIMD_INLINE void BlurRowsAny(const float* src, size_t size, size_t stride, const float* weight, size_t kernel, uint8_t* dst)
        {
            size_t sizeA = AlignLo(size, A);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                __m128 sum0 = _mm_setzero_ps();
                __m128 sum1 = _mm_setzero_ps();
                __m128 sum2 = _mm_setzero_ps();
                __m128 sum3 = _mm_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                {
                    __m128 w = _mm_set1_ps(weight[k]);
                    const float* ps = src + i + k * stride;
                    sum0 = _mm_add_ps(_mm_mul_ps(w, _mm_loadu_ps(ps + 0 * F)), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(w, _mm_loadu_ps(ps + 1 * F)), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(w, _mm_loadu_ps(ps + 2 * F)), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(w, _mm_loadu_ps(ps + 3 * F)), sum3);
                }
                StoreAs8u(dst + i, sum0, sum1, sum2, sum3);
            }
            for (; i < sizeF; i += F)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(weight[k]), _mm_loadu_ps(src + i + k * stride)), sum);
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
            __m128 w[kernel];
            for (size_t k = 0; k < kernel; ++k)
                w[k] = _mm_set1_ps(weight[k]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm_add_ps(_mm_mul_ps(w[k], LoadAs32f(src + i + k * channels)), sum);
                _mm_storeu_ps(dst + i, sum);
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
            __m128 w0 = _mm_set1_ps(weight[0]);
            __m128 w1 = _mm_set1_ps(weight[1]);
            __m128 w2 = _mm_set1_ps(weight[2]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m128 sum = _mm_mul_ps(w0, LoadAs32f(src + i + 0 * channels));
                sum = _mm_add_ps(_mm_mul_ps(w1, LoadAs32f(src + i + 1 * channels)), sum);
                sum = _mm_add_ps(_mm_mul_ps(w2, LoadAs32f(src + i + 2 * channels)), sum);
                _mm_storeu_ps(dst + i, sum);
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
            __m128 w0 = _mm_set1_ps(weight[0]);
            __m128 w1 = _mm_set1_ps(weight[1]);
            __m128 w2 = _mm_set1_ps(weight[2]);
            __m128 w3 = _mm_set1_ps(weight[3]);
            __m128 w4 = _mm_set1_ps(weight[4]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m128 sum0 = _mm_mul_ps(w0, LoadAs32f(src + i + 0 * channels));
                __m128 sum1 = _mm_mul_ps(w1, LoadAs32f(src + i + 1 * channels));
                sum0 = _mm_add_ps(_mm_mul_ps(w2, LoadAs32f(src + i + 2 * channels)), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(w3, LoadAs32f(src + i + 3 * channels)), sum1);
                sum0 = _mm_add_ps(_mm_mul_ps(w4, LoadAs32f(src + i + 4 * channels)), sum0);
                _mm_storeu_ps(dst + i, _mm_add_ps(sum0, sum1));
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
            __m128 w[kernel];
            for (size_t k = 0; k < kernel; ++k)
                w[k] = _mm_set1_ps(weight[k]);
            size_t sizeA = AlignLo(size, A);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                __m128 sum0 = _mm_setzero_ps();
                __m128 sum1 = _mm_setzero_ps();
                __m128 sum2 = _mm_setzero_ps();
                __m128 sum3 = _mm_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                {
                    const float* ps = src + i + k * stride;
                    sum0 = _mm_add_ps(_mm_mul_ps(w[k], _mm_loadu_ps(ps + 0 * F)), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(w[k], _mm_loadu_ps(ps + 1 * F)), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(w[k], _mm_loadu_ps(ps + 2 * F)), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(w[k], _mm_loadu_ps(ps + 3 * F)), sum3);
                }
                StoreAs8u(dst + i, sum0, sum1, sum2, sum3);
            }
            for (; i < sizeF; i += F)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm_add_ps(_mm_mul_ps(w[k], _mm_loadu_ps(src + i + k * stride)), sum);
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
            : Base::GaussianBlurDefault(param)
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
    }
#endif// SIMD_SSE41_ENABLE
}
