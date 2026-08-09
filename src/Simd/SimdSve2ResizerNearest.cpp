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
#include "Simd/SimdResizer.h"
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdParallel.hpp"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        ResizerNearest::ResizerNearest(const ResParam& param)
            : Base::ResizerNearest(param)
            , _blocks(0)
            , _tails(0)
        {
        }

        size_t ResizerNearest::BlockCountMax(size_t align)
        {
            return (size_t)::ceil(float(Simd::Max(_param.srcW, _param.dstW) * _param.PixelSize()) / (align - _param.PixelSize()));
        }

        void ResizerNearest::EstimateParams()
        {
            if (_blocks)
                return;
            Base::ResizerNearest::EstimateParams();
            const size_t A = svcntb();
            const size_t pixelSize = _param.PixelSize();
            if (pixelSize * _param.dstW < A || pixelSize * _param.srcW < A)
                return;
            if (pixelSize < 4 && _param.srcW < 4 * _param.dstW)
                _blocks = BlockCountMax(A);
            float scale = (float)_param.srcW / _param.dstW;
            if (_blocks)
            {
                _tails = 0;
                _ix8x1.Resize(_blocks);
                _tail8x1.Resize((size_t)::ceil(A * scale / pixelSize));
                size_t dstRowSize = _param.dstW * pixelSize;
                int block = 0;
                _ix8x1[0].src = 0;
                _ix8x1[0].dst = 0;
                for (int dstIndex = 0; dstIndex < (int)_param.dstW; ++dstIndex)
                {
                    int srcIndex = _ix[dstIndex] / (int)pixelSize;
                    int dst = dstIndex * (int)pixelSize - _ix8x1[block].dst;
                    int src = srcIndex * (int)pixelSize - _ix8x1[block].src;
                    if (src >= int(A - pixelSize) || dst >= int(A - pixelSize))
                    {
                        block++;
                        _ix8x1[block].src = srcIndex * (int)pixelSize;
                        _ix8x1[block].dst = dstIndex * (int)pixelSize;
                        if (_ix8x1[block].dst > int(dstRowSize - A))
                        {
                            _tail8x1[_tails] = dstRowSize - _ix8x1[block].dst;
                            _tails++;
                        }
                        dst = 0;
                        src = srcIndex * (int)pixelSize - _ix8x1[block].src;
                    }
                    for (size_t i = 0; i < pixelSize; ++i)
                        _ix8x1[block].shuffle[dst + i] = uint8_t(src + i);
                }
                _blocks = block + 1;
            }
        }

        void ResizerNearest::Shuffle8x1(const uint8_t* src, size_t srcStride, size_t dyBeg, size_t dyEnd, uint8_t* dst, size_t dstStride)
        {
            size_t body = _blocks - _tails;
            const svbool_t full = svptrue_b8();
            for (size_t dy = dyBeg; dy < dyEnd; dy++)
            {
                const uint8_t* srcRow = src + _iy[dy] * srcStride;
                size_t i = 0, t = 0;
                for (; i < body; ++i)
                {
                    const IndexShuffle8x1& index = _ix8x1[i];
                    svuint8_t _src = svld1_u8(full, srcRow + index.src);
                    svuint8_t _shuffle = svld1_u8(full, index.shuffle);
                    svst1_u8(full, dst + index.dst, svtbl_u8(_src, _shuffle));
                }
                for (; i < _blocks; ++i, ++t)
                {
                    const IndexShuffle8x1& index = _ix8x1[i];
                    svuint8_t _src = svld1_u8(full, srcRow + index.src);
                    svuint8_t _shuffle = svld1_u8(full, index.shuffle);
                    svst1_u8(svwhilelt_b8((size_t)0, _tail8x1[t]), dst + index.dst, svtbl_u8(_src, _shuffle));
                }
                dst += dstStride;
            }
        }

        void ResizerNearest::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            EstimateParams();
            if (_blocks)
            {
                Simd::Parallel(0, _param.dstH, [&](size_t thread, size_t dstBeg, size_t dstEnd)
                {
                    this->Shuffle8x1(src, srcStride, dstBeg, dstEnd, dst + dstBeg * dstStride, dstStride);
                }, _threads, 1);
            }
            else
                Base::ResizerNearest::Run(src, srcStride, dst, dstStride);
        }
    }
#endif
}

