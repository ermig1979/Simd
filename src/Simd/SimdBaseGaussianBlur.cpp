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
#include "Simd/SimdDefs.h"
#include "Simd/SimdGaussianBlur.h"

namespace Simd
{
    BlurParam::BlurParam(size_t w, size_t h, size_t c, const float* r, size_t a)
        : width(w)
        , height(h)
        , channels(c)
        , radius(*r)
        , align(a)
    {
    }

    namespace Base
    {
        GaussianBlur::GaussianBlur(const BlurParam& param)
            : _param(param)
        {
            _half = (int)::floor(::sqrt(::log(1000.0f)) * _param.radius);
            _kernel = 2 * _half + 1;
            _weight.Resize(2 * _kernel);
            _weight[_half] = 1.0f;
            for (size_t i = 0; i < _half; ++i)
            {
                _weight[_half + 1 + i] = ::exp(-Simd::Square(float(1 + i) / _param.radius));
                _weight[_half - 1 - i] = _weight[_half + 1 + i];
            }
            float sum = 0;
            for (size_t i = 0; i < _kernel; ++i)
                sum += _weight[i];
            for (size_t i = 0; i < _kernel; ++i)
            {
                _weight[i] /= sum;
                _weight[_kernel + i] = _weight[i];
            }
            _size = _param.width * _param.channels;
            _stride = AlignHi(_size, _param.align / sizeof(float));
            _edge = AlignHi(_half * _param.channels, _param.align);
            _start = _edge - _half * _param.channels;
            _buf.Resize(_size + 2 * _edge, true);
            _rows.Resize(_kernel * _stride);
        }

        SIMD_INLINE void PadRow(const uint8_t* src, size_t half, size_t channels, size_t size, uint8_t * dst)
        {
            for (size_t x = 0; x < half; x += 1, dst += channels)
                for (size_t c = 0; c < channels; ++c)
                    dst[c] = src[c];
            memcpy(dst, src, size), dst += size, src += size - channels;
            for (size_t x = 0; x < half; x += 1, dst += channels)
                for (size_t c = 0; c < channels; ++c)
                    dst[c] = src[c];
        }

        SIMD_INLINE void BlurRow(const uint8_t* src, size_t size, size_t channels, const float * weight, size_t kernel, float* dst)
        {
            for (size_t i = 0; i < size; ++i)
            {
                float sum = 0;
                for (size_t k = 0; k < kernel; ++k)
                    sum += weight[k] * float(src[i + k * channels]);
                dst[i] = sum;
            }
        }

        void GaussianBlur::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            if (_param.height == 0 || _param.width == 0)
                return;
            PadRow(src, _half, _param.channels, _size, _buf.data + _start);
            BlurRow(_buf.data + _start, _size, _param.channels, _weight.data, _kernel, _rows.data + _half * _stride);
            for (size_t row = 0; row < _half; ++row)
                memcpy(_rows.data + row * _stride, _rows.data + _half * _stride, _size * sizeof(float));
        }

        //---------------------------------------------------------------------

        void* GaussianBlurInit(size_t width, size_t height, size_t channels, const float* radius)
        {
            BlurParam param(width, height, channels, radius, sizeof(void*));

            return new GaussianBlur(param);
        }

        //---------------------------------------------------------------------

        SIMD_INLINE int DivideBy16(int value)
        {
            return (value + 8) >> 4;
        }

        SIMD_INLINE int GaussianBlur(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, size_t x0, size_t x1, size_t x2)
        {
            return DivideBy16(s0[x0] + 2 * s0[x1] + s0[x2] + (s1[x0] + 2 * s1[x1] + s1[x2]) * 2 + s2[x0] + 2 * s2[x1] + s2[x2]);
        }

        void GaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            const uint8_t *src0, *src1, *src2;

            size_t size = channelCount*width;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                size_t col = 0;
                for (; col < channelCount; col++)
                    dst[col] = GaussianBlur(src0, src1, src2, col, col, col + channelCount);

                for (; col < size - channelCount; ++col)
                    dst[col] = GaussianBlur(src0, src1, src2, col - channelCount, col, col + channelCount);

                for (; col < size; col++)
                    dst[col] = GaussianBlur(src0, src1, src2, col - channelCount, col, col);

                dst += dstStride;
            }
        }
    }
}
