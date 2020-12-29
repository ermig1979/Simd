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
#include "Simd/SimdBase.h"
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

    bool BlurParam::Valid() const
    {
        return 
            height > 0 && width > 0 && 
            channels > 0 && channels <= 4 &&
            radius > 0.0f && align >= sizeof(float);
    }

    //---------------------------------------------------------------------

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
            _cols.Resize(_size + 2 * _edge, true);
            _rows.Resize(_kernel * _stride);
            _nose = Simd::Min(_half + 1, _param.height);
            _body = Simd::Max<ptrdiff_t>(_param.height - _half - 1, 0);
        }

        SIMD_INLINE void PadCols(const uint8_t* src, size_t half, size_t channels, size_t size, uint8_t * dst)
        {
            for (size_t x = 0; x < half; x += 1, dst += channels)
                for (size_t c = 0; c < channels; ++c)
                    dst[c] = src[c];
            memcpy(dst, src, size), dst += size, src += size - channels;
            for (size_t x = 0; x < half; x += 1, dst += channels)
                for (size_t c = 0; c < channels; ++c)
                    dst[c] = src[c];
        }

        SIMD_INLINE void BlurCols(const uint8_t* src, size_t size, size_t channels, const float * weight, size_t kernel, float* dst)
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

        void GaussianBlur::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            if (_half == 0)
            {
                if (src != dst)
                    Copy(src, srcStride, _param.width, _param.height, _param.channels, dst, dstStride);
                return;
            }

            PadCols(src, _half, _param.channels, _size, _cols.data + _start);
            BlurCols(_cols.data + _start, _size, _param.channels, _weight.data, _kernel, _rows.data + _half * _stride);
            for (size_t row = 0; row < _half; ++row)
                memcpy(_rows.data + row * _stride, _rows.data + _half * _stride, _size * sizeof(float));
            for (size_t row = 1; row < _nose; ++row)
            {
                PadCols(src + row * srcStride, _half, _param.channels, _size, _cols.data + _start);
                BlurCols(_cols.data + _start, _size, _param.channels, _weight.data, _kernel, _rows.data + (_half + row) * _stride);
            }
            for (size_t row = _nose; row <= _half; ++row)
                memcpy(_rows.data + (_half + row) * _stride, _rows.data + (_half + _nose + 1) * _stride, _size * sizeof(float));
            BlurRows(_rows.data, _size, _stride, _weight.data, _kernel, dst);

            for (size_t row = 1, rows = row % _kernel + 2*_half, weight = _kernel - row % _kernel; row < _body; ++row, ++rows, --weight)
            {
                if (rows >= _kernel)
                    rows -= _kernel;
                if (weight == 0)
                    weight += _kernel;
                PadCols(src + (row + _half)* srcStride, _half, _param.channels, _size, _cols.data + _start);
                BlurCols(_cols.data + _start, _size, _param.channels, _weight.data, _kernel, _rows.data + rows * _stride);
                BlurRows(_rows.data, _size, _stride, _weight.data + weight, _kernel, dst + row * dstStride);
            }

            size_t last = (_body + 2 * _half - 1) % _kernel;
            for (size_t row = _body, rows = row % _kernel + 2 * _half, weight = _kernel - row % _kernel; row < _param.height; ++row, ++rows, --weight)
            {
                if (rows >= _kernel)
                    rows -= _kernel;
                if (weight == 0)
                    weight += _kernel;
                memcpy(_rows.data + rows * _stride, _rows.data + last * _stride, _size * sizeof(float));
                BlurRows(_rows.data, _size, _stride, _weight.data + weight, _kernel, dst + row * dstStride);
            }
        }

        //---------------------------------------------------------------------

        void* GaussianBlurInit(size_t width, size_t height, size_t channels, const float* radius)
        {
            BlurParam param(width, height, channels, radius, sizeof(void*));
            if (!param.Valid())
                return NULL;
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
