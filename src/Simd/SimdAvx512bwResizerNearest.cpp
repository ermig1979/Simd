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
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdParallel.hpp"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE 
    namespace Avx512bw
    {
        ResizerNearest::ResizerNearest(const ResParam& param)
            : Avx2::ResizerNearest(param)
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
                _tails = 0;
                _ix32x2.Resize(_blocks);
                _tail32x2.Resize((size_t)::ceil(A * scale / pixelSize));
                size_t dstRowSize = _param.dstW * pixelSize;
                int block = 0;
                _ix32x2[0].src = 0;
                _ix32x2[0].dst = 0;
                for (int dstIndex = 0; dstIndex < (int)_param.dstW; ++dstIndex)
                {
                    int srcIndex = _ix[dstIndex] / (int)pixelSize;
                    int dst = dstIndex * (int)pixelSize - _ix32x2[block].dst;
                    int src = srcIndex * (int)pixelSize - _ix32x2[block].src;
                    if (src >= int(A - pixelSize) || dst >= int(A - pixelSize))
                    {
                        block++;
                        _ix32x2[block].src = srcIndex * (int)pixelSize;
                        _ix32x2[block].dst = dstIndex * (int)pixelSize;
                        if (_ix32x2[block].dst > int(dstRowSize - A))
                        {
                            _tail32x2[_tails] = TailMask32((dstRowSize - _ix32x2[block].dst) / 2);
                            _tails++;
                        }
                        dst = 0;
                        src = srcIndex * (int)pixelSize - _ix32x2[block].src;
                    }
                    for (size_t i = 0; i < pixelSize; i += 2)
                        _ix32x2[block].shuffle[(dst + i) / 2] = uint16_t((src + i) / 2);
                }
                _blocks = block + 1;
            }
        }

        SIMD_INLINE void Gather4(const int32_t* src, const int32_t* idx, int32_t* dst, __mmask16 mask = -1)
        {
            __m512i _idx = _mm512_maskz_loadu_epi32(mask, idx);
            __m512i val = _mm512_i32gather_epi32(_idx, src, 1);
            _mm512_mask_storeu_epi32(dst, mask, val);
        }

        void ResizerNearest::Gather4(const uint8_t* src, size_t srcStride, size_t dyBeg, size_t dyEnd, uint8_t* dst, size_t dstStride)
        {
            size_t body = AlignLo(_param.dstW, F);
            __mmask16 tail = TailMask16(_param.dstW - body);
            for (size_t dy = dyBeg; dy < dyEnd; dy++)
            {
                const int32_t* srcRow = (int32_t*)(src + _iy[dy] * srcStride);
                size_t dx = 0;
                for (; dx < body; dx += F)
                    Avx512bw::Gather4(srcRow, _ix.data + dx, (int32_t*)dst + dx);
                if (tail)
                    Avx512bw::Gather4(srcRow, _ix.data + dx, (int32_t*)dst + dx, tail);
                dst += dstStride;
            }
        }

        SIMD_INLINE void Gather8(const int64_t* src, const int32_t* idx, int64_t* dst)
        {
            __m256i _idx = _mm256_loadu_si256((__m256i*)idx);
            __m512i val = _mm512_i32gather_epi64(_idx, src, 1);
            _mm512_storeu_si512((__m512i*)dst, val);
        }

        void ResizerNearest::Gather8(const uint8_t* src, size_t srcStride, size_t dyBeg, size_t dyEnd, uint8_t* dst, size_t dstStride)
        {
            size_t body = AlignLo(_param.dstW, 8);
            size_t tail = _param.dstW - 8;
            for (size_t dy = dyBeg; dy < dyEnd; dy++)
            {
                const int64_t* srcRow = (int64_t*)(src + _iy[dy] * srcStride);
                for (size_t dx = 0; dx < body; dx += 8)
                    Avx512bw::Gather8(srcRow, _ix.data + dx, (int64_t*)dst + dx);
                Avx512bw::Gather8(srcRow, _ix.data + tail, (int64_t*)dst + tail);
                dst += dstStride;
            }
        }

        void ResizerNearest::Shuffle32x2(const uint8_t* src, size_t srcStride, size_t dyBeg, size_t dyEnd, uint8_t* dst, size_t dstStride)
        {
            size_t body = _blocks - _tails;
            for (size_t dy = dyBeg; dy < dyEnd; dy++)
            {
                const uint8_t* srcRow = src + _iy[dy] * srcStride;
                size_t i = 0, t = 0;
                for (; i < body; ++i)
                {
                    const IndexShuffle32x2& index = _ix32x2[i];
                    __m512i _src = _mm512_loadu_si512((__m512i*)(srcRow + index.src));
                    __m512i _shuffle = _mm512_loadu_si512((__m512i*) & index.shuffle);
                    _mm512_storeu_si512((__m512i*)(dst + index.dst), _mm512_permutexvar_epi16(_shuffle, _src));
                }
                for (; i < _blocks; ++i, t++)
                {
                    const IndexShuffle32x2& index = _ix32x2[i];
                    __m512i _src = _mm512_loadu_si512((__m512i*)(srcRow + index.src));
                    __m512i _shuffle = _mm512_loadu_si512((__m512i*)&index.shuffle);
                    _mm512_mask_storeu_epi16(dst + index.dst, _tail32x2[t], _mm512_permutexvar_epi16(_shuffle, _src));
                }
                dst += dstStride;
            }
        }

        void ResizerNearest::FillConst(const uint8_t* src, size_t srcStride, size_t dyBeg, size_t dyEnd, uint8_t* dst, size_t dstStride)
        {
            size_t size = _param.PixelSize();
            for (size_t dy = dyBeg; dy < dyEnd; dy++)
            {
                for (size_t dx = 0; dx < _param.dstW; dx += 1)
                    memcpy(dst + dx * size, src, size);
                dst += dstStride;
            }
        }

        void ResizerNearest::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            EstimateParams();
            if (_param.srcH == 1 && _param.srcW == 1)
            {
                Simd::Parallel(0, _param.dstH, [&](size_t thread, size_t dstBeg, size_t dstEnd)
                {
                    this->FillConst(src, srcStride, dstBeg, dstEnd, dst + dstBeg * dstStride, dstStride);
                }, _threads, 1);
            }
            else if (_blocks)
            {
                Simd::Parallel(0, _param.dstH, [&](size_t thread, size_t dstBeg, size_t dstEnd)
                {
                    this->Shuffle32x2(src, srcStride, dstBeg, dstEnd, dst + dstBeg * dstStride, dstStride);
                }, _threads, 1);
            }
            else
            {
                Avx2::ResizerNearest::EstimateParams();
                if (_pixelSize == 4)
                {
                    Simd::Parallel(0, _param.dstH, [&](size_t thread, size_t dstBeg, size_t dstEnd)
                    {
                        this->Gather4(src, srcStride, dstBeg, dstEnd, dst + dstBeg * dstStride, dstStride);
                    }, _threads, 1);
                }
                else if (_pixelSize == 8)
                {
                    Simd::Parallel(0, _param.dstH, [&](size_t thread, size_t dstBeg, size_t dstEnd)
                    {
                        this->Gather8(src, srcStride, dstBeg, dstEnd, dst + dstBeg * dstStride, dstStride);
                    }, _threads, 1);
                }
                else
                    Avx2::ResizerNearest::Run(src, srcStride, dst, dstStride);
            }
        }

        bool ResizerNearest::Preferable(const ResParam& param)
        {
            const size_t pixelSize = param.PixelSize();
            return 
                (pixelSize == 4 || (pixelSize == 8 && param.dstW >= F)) ||
                ((pixelSize & 1) == 0 && pixelSize < 8 && param.srcW < 8 * param.dstW) ||
                (pixelSize >= A && param.srcH == 1 && param.srcW == 1);
        }
    }
#endif //SIMD_AVX512BW_ENABLE 
}

