/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE 
    namespace Avx512bw
    {
        ResizerNearest::ResizerNearest(const ResParam& param)
            : Avx512f::ResizerNearest(param)
        {
        }

        void ResizerNearest::EstimateParams()
        {
            if (_blocks)
                return;
            Base::ResizerNearest::EstimateParams();
            const size_t pixelSize = _param.PixelSize();
            if (pixelSize * _param.dstW < A || pixelSize * _param.srcW < A)
                return;
            if (pixelSize < 4 && _param.srcW < 4 * _param.dstW)
                _blocks = BlockCountMax(A);
            float scale = (float)_param.srcW / _param.dstW;
            if (_blocks)
            {
                _ix32x2.Resize(_blocks);
                int block = 0, tail = 0;
                _ix32x2[0].src = 0;
                _ix32x2[0].dst = 0;
                for (int dstIndex = 0; dstIndex < (int)_param.dstW; ++dstIndex)
                {
                    int srcIndex = _ix[dstIndex] / (int)pixelSize;
                    int dst = dstIndex * (int)pixelSize - _ix32x2[block].dst;
                    int src = srcIndex * (int)pixelSize - _ix32x2[block].src;
                    if (src >= A - pixelSize || dst >= A - pixelSize)
                    {
                        block++;
                        _ix32x2[block].src = srcIndex * (int)pixelSize;
                        _ix32x2[block].dst = dstIndex * (int)pixelSize;
                        dst = 0;
                        src = srcIndex * (int)pixelSize - _ix32x2[block].src;
                    }
                    for (size_t i = 0; i < pixelSize; i += 2)
                        _ix32x2[block].shuffle[(dst + i) / 2] = uint16_t((src + i) / 2);
                    tail = (dst + (int)pixelSize) / 2;
                }
                _tail32x2 = TailMask32(tail);
                _blocks = block + 1;
            }
        }

        void ResizerNearest::Shuffle32x2(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t blocks = _blocks - 1;
            for (size_t dy = 0; dy < _param.dstH; dy++)
            {
                const uint8_t* srcRow = src + _iy[dy] * srcStride;
                for (size_t i = 0; i < blocks; ++i)
                {
                    const IndexShuffle32x2& index = _ix32x2[i];
                    __m512i _src = _mm512_loadu_si512((__m512i*)(srcRow + index.src));
                    __m512i _shuffle = _mm512_loadu_si512((__m512i*) & index.shuffle);
                    _mm512_storeu_si512((__m512i*)(dst + index.dst), _mm512_permutexvar_epi16(_shuffle, _src));
                }
                {
                    const IndexShuffle32x2& index = _ix32x2[blocks];
                    __m512i _src = _mm512_loadu_si512((__m512i*)(srcRow + index.src));
                    __m512i _shuffle = _mm512_loadu_si512((__m512i*)&index.shuffle);
                    _mm512_mask_storeu_epi16(dst + index.dst, _tail32x2, _mm512_permutexvar_epi16(_shuffle, _src));
                }
                dst += dstStride;
            }
        }

        void ResizerNearest::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_param.dstW >= A);
            EstimateParams();
            if (_blocks)
                Shuffle32x2(src, srcStride, dst, dstStride);
            else
                Avx512f::ResizerNearest::Run(src, srcStride, dst, dstStride);
        }

        bool ResizerNearest::Preferable(const ResParam& param)
        {
            const size_t pixelSize = param.PixelSize();
            return (pixelSize & 1) == 0 && pixelSize < 8 && param.srcW < 8 * param.dstW;
        }
    }
#endif //SIMD_AVX512BW_ENABLE 
}

