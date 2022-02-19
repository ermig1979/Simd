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
#ifdef SIMD_AVX2_ENABLE 
    namespace Avx2
    {
        ResizerNearest::ResizerNearest(const ResParam& param)
            : Sse41::ResizerNearest(param)
        {
        }

        void ResizerNearest::EstimateParams()
        {
            if (_pixelSize)
                return;
            size_t pixelSize = _param.PixelSize();
            if (pixelSize == 4 || pixelSize == 8 || 
                (pixelSize == 3 && _param.dstW <= _param.srcW) ||
                (pixelSize == 2 && _param.dstW * 2 <= _param.srcW))
                Base::ResizerNearest::EstimateParams();
        }

        const __m256i K8_SHUFFLE_UVXX_TO_UV = SIMD_MM256_SETR_EPI8(
            0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
            0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m256i K32_PERMUTE_UVXX_TO_UV = SIMD_MM256_SETR_EPI32(0x0, 0x1, 0x4, 0x5, -1, -1, -1, -1);

        SIMD_INLINE void Gather8x2(const uint8_t* src, const int32_t* idx, uint8_t* dst)
        {
            __m256i _idx = _mm256_loadu_si256((__m256i*)idx);
            __m256i uvxx = _mm256_i32gather_epi32((int32_t*)src, _idx, 1);
            __m256i uv = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(uvxx, K8_SHUFFLE_UVXX_TO_UV), K32_PERMUTE_UVXX_TO_UV);
            _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(uv));
        }

        void ResizerNearest::Gather2(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t body = AlignLo(_param.dstW, 8);
            size_t tail = _param.dstW - 8;
            for (size_t dy = 0; dy < _param.dstH; dy++)
            {
                const uint8_t* srcRow = src + _iy[dy] * srcStride;
                for (size_t dx = 0, offs = 0; dx < body; dx += 8, offs += 16)
                    Avx2::Gather8x2(srcRow, _ix.data + dx, dst + offs);
                Avx2::Gather8x2(srcRow, _ix.data + tail, dst + tail * 2);
                dst += dstStride;
            }
        }

        SIMD_INLINE __m256i Gather8x3(const uint8_t* src, const int32_t* idx)
        {
            __m256i _idx = _mm256_loadu_si256((__m256i*)idx);
            __m256i bgrx = _mm256_i32gather_epi32((int32_t*)src, _idx, 1);
            return _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bgrx, K8_SHUFFLE_BGRA_TO_BGR), K32_PERMUTE_BGRA_TO_BGR);
        }

        void ResizerNearest::Gather3(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t body = AlignLo(_param.dstW, 8);
            size_t tail = _param.dstW - 8;
            for (size_t dy = 0; dy < _param.dstH; dy++)
            {
                const uint8_t* srcRow = src + _iy[dy] * srcStride;
                for (size_t dx = 0, offs = 0; dx < body; dx +=8, offs += 24)
                    _mm256_storeu_si256((__m256i*)(dst + offs), Gather8x3(srcRow, _ix.data + dx));
                Store24<false>(dst + tail * 3, Gather8x3(srcRow, _ix.data + tail));
                dst += dstStride;
            }
        }

        SIMD_INLINE void Gather8x4(const int32_t * src, const int32_t* idx, int32_t* dst)
        {
            __m256i _idx = _mm256_loadu_si256((__m256i*)idx);
            __m256i val = _mm256_i32gather_epi32(src, _idx, 1);
            _mm256_storeu_si256((__m256i*)dst, val);
        }

        void ResizerNearest::Gather4(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t body = AlignLo(_param.dstW, 8);
            size_t tail = _param.dstW - 8;
            for (size_t dy = 0; dy < _param.dstH; dy++)
            {
                const int32_t* srcRow = (int32_t*)(src + _iy[dy] * srcStride);
                for (size_t dx = 0; dx < body; dx += 8)
                    Avx2::Gather8x4(srcRow, _ix.data + dx, (int32_t*)dst + dx);
                Avx2::Gather8x4(srcRow, _ix.data + tail, (int32_t*)dst + tail);
                dst += dstStride;
            }
        }

        SIMD_INLINE void Gather4x8(const int64_t* src, const int32_t* idx, int64_t* dst)
        {
            __m128i _idx = _mm_loadu_si128((__m128i*)idx);
            __m256i val = _mm256_i32gather_epi64((long long*)src, _idx, 1);
            _mm256_storeu_si256((__m256i*)dst, val);
        }

        void ResizerNearest::Gather8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t body = AlignLo(_param.dstW, 4);
            size_t tail = _param.dstW - 4;
            for (size_t dy = 0; dy < _param.dstH; dy++)
            {
                const int64_t* srcRow = (int64_t*)(src + _iy[dy] * srcStride);
                for (size_t dx = 0; dx < body; dx += 4)
                    Avx2::Gather4x8(srcRow, _ix.data + dx, (int64_t*)dst + dx);
                Avx2::Gather4x8(srcRow, _ix.data + tail, (int64_t*)dst + tail);
                dst += dstStride;
            }
        }

        void ResizerNearest::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            EstimateParams();
            if (_pixelSize == 2 && _param.dstW * 2 <= _param.srcW)
                Gather2(src, srcStride, dst, dstStride);
            else if (_pixelSize == 3 && _param.dstW <= _param.srcW)
                Gather3(src, srcStride, dst, dstStride);
            else if (_pixelSize == 4)
                Gather4(src, srcStride, dst, dstStride);
            else if (_pixelSize == 8)
                Gather8(src, srcStride, dst, dstStride);
            else 
                Sse41::ResizerNearest::Run(src, srcStride, dst, dstStride);
        }
    }
#endif //SIMD_AVX2_ENABLE 
}

