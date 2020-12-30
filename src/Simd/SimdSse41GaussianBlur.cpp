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
#include "Simd/SimdGaussianBlur.h"
#include "Simd/SimdCopyPixel.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template<int channels> SIMD_INLINE void PadCols(const uint8_t* src, size_t half, size_t size, uint8_t* dst)
        {
            for (size_t x = 0; x < half; x += 1, dst += channels)
                Base::CopyPixel<channels>(src, dst);
            memcpy(dst, src, size), dst += size, src += size - channels;
            for (size_t x = 0; x < half; x += 1, dst += channels)
                Base::CopyPixel<channels>(src, dst);
        }

        template<int channels> SIMD_INLINE void BlurCols(const uint8_t* src, size_t size, const float* weight, size_t kernel, float* dst)
        {
            for (size_t i = 0; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * float(src[i + k * channels]);
                dst[i] = sum;
            }
        }

        SIMD_INLINE void BlurRows(const float* src, size_t size, size_t stride, const float* weight, size_t kernel, uint8_t* dst)
        {
            for (size_t i = 0; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * src[i + k * stride];
                dst[i] = int(sum);
            }
        }

        template<int channels> void BlurImageAny(const BlurParam& p, const Base::AlgDefault& a, const uint8_t* src, size_t srcStride, uint8_t* cols, float* rows, uint8_t* dst, size_t dstStride)
        {
            PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
            BlurCols<channels>(cols, a.size, a.weight.data, a.kernel, rows + a.half * a.stride);
            for (size_t row = 0; row < a.half; ++row)
                memcpy(rows + row * a.stride, rows + a.half * a.stride, a.size * sizeof(float));
            for (size_t row = 1; row < a.nose; ++row)
            {
                PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurCols<channels>(cols, a.size, a.weight.data, a.kernel, rows + (a.half + row) * a.stride);
            }
            for (size_t row = a.nose; row <= a.half; ++row)
                memcpy(rows + (a.half + row) * a.stride, rows + (a.half + a.nose + 1) * a.stride, a.size * sizeof(float));
            BlurRows(rows, a.size, a.stride, a.weight.data, a.kernel, dst), dst += dstStride;

            for (size_t row = 1, b = row % a.kernel + 2 * a.half, w = a.kernel - row % a.kernel; row < a.body; ++row, ++b, --w)
            {
                if (b >= a.kernel)
                    b -= a.kernel;
                if (w == 0)
                    w += a.kernel;
                PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurCols<channels>(cols, a.size, a.weight.data, a.kernel, rows + b * a.stride);
                BlurRows(rows, a.size, a.stride, a.weight.data + w, a.kernel, dst), dst += dstStride;
            }

            size_t last = (a.body + 2 * a.half - 1) % a.kernel;
            for (size_t row = a.body, b = row % a.kernel + 2 * a.half, w = a.kernel - row % a.kernel; row < p.height; ++row, ++b, --w)
            {
                if (b >= a.kernel)
                    b -= a.kernel;
                if (w == 0)
                    w += a.kernel;
                memcpy(rows + b * a.stride, rows + last * a.stride, a.size * sizeof(float));
                BlurRows(rows, a.size, a.stride, a.weight.data + w, a.kernel, dst), dst += dstStride;
            }
        }

        template<int channels> Base::BlurDefaultPtr GetBlurDefaultPtr(const BlurParam& p, const Base::AlgDefault& a)
        {
            return BlurImageAny<channels>;
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

        void* GaussianBlurInit(size_t width, size_t height, size_t channels, const float* radius)
        {
            BlurParam param(width, height, channels, radius, F);
            if (!param.Valid())
                return NULL;
            return new GaussianBlurDefault(param);
        }
    }
#endif// SIMD_SSE41_ENABLE
}
