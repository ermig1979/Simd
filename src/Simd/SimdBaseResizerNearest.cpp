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
#include "Simd/SimdResizer.h"
#include "Simd/SimdCopyPixel.h"

namespace Simd
{
    namespace Base
    {
        ResizerNearest::ResizerNearest(const ResParam& param)
            : Resizer(param)
            , _pixelSize(0)
        {
        }

        void ResizerNearest::EstimateIndex(size_t srcSize, size_t dstSize, size_t channelSize, size_t channels, int32_t* indices)
        {
            if (_param.method == SimdResizeMethodNearest)
            {
                float scale = (float)srcSize / dstSize;
                for (size_t i = 0; i < dstSize; ++i)
                {
                    float alpha = (i + 0.5f) * scale;
                    int index = RestrictRange((int)::floor(alpha), 0, (int)srcSize - 1);
                    for (size_t c = 0; c < channels; c++)
                    {
                        size_t offset = i * channels + c;
                        indices[offset] = (int32_t)((channels * index + c) * channelSize);
                    }
                }
            }
            else if (_param.method == SimdResizeMethodNearestPytorch)
            {
                for (size_t i = 0; i < dstSize; ++i)
                {
                    int index = RestrictRange((int)(i * srcSize / dstSize), 0, (int)srcSize - 1);
                    for (size_t c = 0; c < channels; c++)
                    {
                        size_t offset = i * channels + c;
                        indices[offset] = (int32_t)((channels * index + c) * channelSize);
                    }
                }
            }
            else
                assert(0);
        }

        void ResizerNearest::EstimateParams()
        {
            if (_pixelSize)
                return;
            _pixelSize = _param.PixelSize();
            _iy.Resize(_param.dstH, false, _param.align);
            EstimateIndex(_param.srcH, _param.dstH, 1, 1, _iy.data);
            _ix.Resize(_param.dstW, false, _param.align);
            EstimateIndex(_param.srcW, _param.dstW, _pixelSize, 1, _ix.data);
        }

        void ResizerNearest::Resize(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            for (size_t dy = 0; dy < _param.dstH; dy++)
            {
                const uint8_t* srcRow = src + _iy[dy] * srcStride;
                for (size_t dx = 0, offset = 0; dx < _param.dstW; dx++, offset += _pixelSize)
                    memcpy(dst + offset, srcRow + _ix[dx], _pixelSize);
                dst += dstStride;
            }
        }

        template<size_t N> void ResizerNearest::Resize(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            for (size_t dy = 0; dy < _param.dstH; dy++)
            {
                const uint8_t * srcRow = src + _iy[dy] * srcStride;
                for (size_t dx = 0, offset = 0; dx < _param.dstW; dx++, offset += N)
                    CopyPixel<N>(srcRow + _ix[dx], dst + offset);
                dst += dstStride;
            }
        }

        void ResizerNearest::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            EstimateParams();
            switch (_pixelSize)
            {
            case 1: Resize<1>(src, srcStride, dst, dstStride); break;
            case 2: Resize<2>(src, srcStride, dst, dstStride); break;
            case 3: Resize<3>(src, srcStride, dst, dstStride); break;
            case 4: Resize<4>(src, srcStride, dst, dstStride); break;
            case 6: Resize<6>(src, srcStride, dst, dstStride); break;
            case 8: Resize<8>(src, srcStride, dst, dstStride); break;
            case 12: Resize<12>(src, srcStride, dst, dstStride); break;
            default:
                Resize(src, srcStride, dst, dstStride);
            }
        }
    }
}

