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
#include "Simd/SimdMemory.h"
#include "Simd/SimdGaussianBlur.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE    
    namespace Sve2
    {
        SIMD_INLINE svfloat32_t LoadAs32f(const uint8_t* src, const svbool_t& mask)
        {
            return svcvt_f32_u32_x(mask, svld1ub_u32(mask, src));
        }

        SIMD_INLINE void BlurColsAny(const uint8_t* src, size_t size, size_t channels, const float* weight, size_t kernel, float* dst)
        {
            size_t F = svcntw(), sizeF = AlignLoAny(size, F), i = 0;
            const svbool_t body = svptrue_b32();
            for (; i < sizeF; i += F)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = svmla_f32_m(body, sum, svdup_n_f32(weight[k]), LoadAs32f(src + i + k * channels, body));
                svst1_f32(body, dst + i, sum);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = svmla_f32_m(tail, sum, svdup_n_f32(weight[k]), LoadAs32f(src + i + k * channels, tail));
                svst1_f32(tail, dst + i, sum);
            }
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const svfloat32_t& src, const svbool_t& mask)
        {
            svfloat32_t value = svmin_n_f32_x(mask, svadd_n_f32_x(mask, src, 0.5f), 255.0f);
            svst1b_u32(mask, dst, svcvt_u32_f32_x(mask, value));
        }

        SIMD_INLINE void BlurRowsAny(const float* src, size_t size, size_t stride, const float* weight, size_t kernel, uint8_t* dst)
        {
            size_t F = svcntw(), A = 4 * F, sizeA = AlignLoAny(size, A), sizeF = AlignLoAny(size, F), i = 0;
            const svbool_t body = svptrue_b32();
            for (; i < sizeA; i += A)
            {
                svfloat32_t sum0 = svdup_n_f32(0.0f);
                svfloat32_t sum1 = svdup_n_f32(0.0f);
                svfloat32_t sum2 = svdup_n_f32(0.0f);
                svfloat32_t sum3 = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                {
                    svfloat32_t w = svdup_n_f32(weight[k]);
                    const float* ps = src + i + k * stride;
                    sum0 = svmla_f32_m(body, sum0, w, svld1_f32(body, ps + 0 * F));
                    sum1 = svmla_f32_m(body, sum1, w, svld1_f32(body, ps + 1 * F));
                    sum2 = svmla_f32_m(body, sum2, w, svld1_f32(body, ps + 2 * F));
                    sum3 = svmla_f32_m(body, sum3, w, svld1_f32(body, ps + 3 * F));
                }
                StoreAs8u(dst + i + 0 * F, sum0, body);
                StoreAs8u(dst + i + 1 * F, sum1, body);
                StoreAs8u(dst + i + 2 * F, sum2, body);
                StoreAs8u(dst + i + 3 * F, sum3, body);
            }
            for (; i < sizeF; i += F)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = svmla_f32_m(body, sum, svdup_n_f32(weight[k]), svld1_f32(body, src + i + k * stride));
                StoreAs8u(dst + i, sum, body);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = svmla_f32_m(tail, sum, svdup_n_f32(weight[k]), svld1_f32(tail, src + i + k * stride));
                StoreAs8u(dst + i, sum, tail);
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
            size_t F = svcntw(), sizeF = AlignLoAny(size, F), i = 0;
            const svbool_t body = svptrue_b32();
            for (; i < sizeF; i += F)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = svmla_f32_m(body, sum, svdup_n_f32(weight[k]), LoadAs32f(src + i + k * channels, body));
                svst1_f32(body, dst + i, sum);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = svmla_f32_m(tail, sum, svdup_n_f32(weight[k]), LoadAs32f(src + i + k * channels, tail));
                svst1_f32(tail, dst + i, sum);
            }
        }

        template<> SIMD_INLINE void BlurCols<3>(const uint8_t* src, size_t size, size_t channels, const float* weight, float* dst)
        {
            svfloat32_t w0 = svdup_n_f32(weight[0]);
            svfloat32_t w1 = svdup_n_f32(weight[1]);
            svfloat32_t w2 = svdup_n_f32(weight[2]);
            size_t F = svcntw(), sizeF = AlignLoAny(size, F), i = 0;
            const svbool_t body = svptrue_b32();
            for (; i < sizeF; i += F)
            {
                svfloat32_t sum = svmul_f32_x(body, w0, LoadAs32f(src + i + 0 * channels, body));
                sum = svmla_f32_m(body, sum, w1, LoadAs32f(src + i + 1 * channels, body));
                sum = svmla_f32_m(body, sum, w2, LoadAs32f(src + i + 2 * channels, body));
                svst1_f32(body, dst + i, sum);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svfloat32_t sum = svmul_f32_x(tail, w0, LoadAs32f(src + i + 0 * channels, tail));
                sum = svmla_f32_m(tail, sum, w1, LoadAs32f(src + i + 1 * channels, tail));
                sum = svmla_f32_m(tail, sum, w2, LoadAs32f(src + i + 2 * channels, tail));
                svst1_f32(tail, dst + i, sum);
            }
        }

        template<> SIMD_INLINE void BlurCols<5>(const uint8_t* src, size_t size, size_t channels, const float* weight, float* dst)
        {
            svfloat32_t w0 = svdup_n_f32(weight[0]);
            svfloat32_t w1 = svdup_n_f32(weight[1]);
            svfloat32_t w2 = svdup_n_f32(weight[2]);
            svfloat32_t w3 = svdup_n_f32(weight[3]);
            svfloat32_t w4 = svdup_n_f32(weight[4]);
            size_t F = svcntw(), sizeF = AlignLoAny(size, F), i = 0;
            const svbool_t body = svptrue_b32();
            for (; i < sizeF; i += F)
            {
                svfloat32_t sum0 = svmul_f32_x(body, w0, LoadAs32f(src + i + 0 * channels, body));
                svfloat32_t sum1 = svmul_f32_x(body, w1, LoadAs32f(src + i + 1 * channels, body));
                sum0 = svmla_f32_m(body, sum0, w2, LoadAs32f(src + i + 2 * channels, body));
                sum1 = svmla_f32_m(body, sum1, w3, LoadAs32f(src + i + 3 * channels, body));
                sum0 = svmla_f32_m(body, sum0, w4, LoadAs32f(src + i + 4 * channels, body));
                svst1_f32(body, dst + i, svadd_f32_x(body, sum0, sum1));
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svfloat32_t sum0 = svmul_f32_x(tail, w0, LoadAs32f(src + i + 0 * channels, tail));
                svfloat32_t sum1 = svmul_f32_x(tail, w1, LoadAs32f(src + i + 1 * channels, tail));
                sum0 = svmla_f32_m(tail, sum0, w2, LoadAs32f(src + i + 2 * channels, tail));
                sum1 = svmla_f32_m(tail, sum1, w3, LoadAs32f(src + i + 3 * channels, tail));
                sum0 = svmla_f32_m(tail, sum0, w4, LoadAs32f(src + i + 4 * channels, tail));
                svst1_f32(tail, dst + i, svadd_f32_x(tail, sum0, sum1));
            }
        }

        template<int kernel> SIMD_INLINE void BlurRows(const float* src, size_t size, size_t stride, const float* weight, uint8_t* dst)
        {
            size_t F = svcntw(), A = 4 * F, sizeA = AlignLoAny(size, A), sizeF = AlignLoAny(size, F), i = 0;
            const svbool_t body = svptrue_b32();
            for (; i < sizeA; i += A)
            {
                svfloat32_t sum0 = svdup_n_f32(0.0f);
                svfloat32_t sum1 = svdup_n_f32(0.0f);
                svfloat32_t sum2 = svdup_n_f32(0.0f);
                svfloat32_t sum3 = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                {
                    svfloat32_t w = svdup_n_f32(weight[k]);
                    const float* ps = src + i + k * stride;
                    sum0 = svmla_f32_m(body, sum0, w, svld1_f32(body, ps + 0 * F));
                    sum1 = svmla_f32_m(body, sum1, w, svld1_f32(body, ps + 1 * F));
                    sum2 = svmla_f32_m(body, sum2, w, svld1_f32(body, ps + 2 * F));
                    sum3 = svmla_f32_m(body, sum3, w, svld1_f32(body, ps + 3 * F));
                }
                StoreAs8u(dst + i + 0 * F, sum0, body);
                StoreAs8u(dst + i + 1 * F, sum1, body);
                StoreAs8u(dst + i + 2 * F, sum2, body);
                StoreAs8u(dst + i + 3 * F, sum3, body);
            }
            for (; i < sizeF; i += F)
            {
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = svmla_f32_m(body, sum, svdup_n_f32(weight[k]), svld1_f32(body, src + i + k * stride));
                StoreAs8u(dst + i, sum, body);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b32(i, size);
                svfloat32_t sum = svdup_n_f32(0.0f);
                for (size_t k = 0; k < kernel; ++k)
                    sum = svmla_f32_m(tail, sum, svdup_n_f32(weight[k]), svld1_f32(tail, src + i + k * stride));
                StoreAs8u(dst + i, sum, tail);
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
#ifdef SIMD_NEON_ENABLE
            : Neon::GaussianBlurDefault(param)
#else
            : Base::GaussianBlurDefault(param)
#endif
        {
            switch (_param.channels)
            {
            case 1: _blur = GetBlurDefaultPtr<1>(_param, _alg); break;
            case 2: _blur = GetBlurDefaultPtr<2>(_param, _alg); break;
            case 3: _blur = GetBlurDefaultPtr<3>(_param, _alg); break;
            case 4: _blur = GetBlurDefaultPtr<4>(_param, _alg); break;
            }
        }

        //---------------------------------------------------------------------

        void* GaussianBlurInit(size_t width, size_t height, size_t channels, const float* sigma, const float* epsilon)
        {
            BlurParam param(width, height, channels, sigma, epsilon, SIMD_ALIGN);
            if (!param.Valid())
                return NULL;
            return new GaussianBlurDefault(param);
        }
    }
#endif// SIMD_SVE2_ENABLE
}
