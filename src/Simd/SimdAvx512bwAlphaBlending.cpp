/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i AlphaBlendingI16(const __m512i &  src, const __m512i &  dst, const __m512i &  alpha)
        {
            return DivideI16By255(_mm512_add_epi16(_mm512_mullo_epi16(src, alpha), _mm512_mullo_epi16(dst, _mm512_sub_epi16(K16_00FF, alpha))));
        }

        template <bool align, bool mask> SIMD_INLINE void AlphaBlending(const uint8_t * src, uint8_t * dst, const __m512i & alpha, __mmask64 m)
        {
            __m512i _src = Load<align, mask>(src, m);
            __m512i _dst = Load<align, mask>(dst, m);
            __m512i lo = AlphaBlendingI16(UnpackU8<0>(_src), UnpackU8<0>(_dst), UnpackU8<0>(alpha));
            __m512i hi = AlphaBlendingI16(UnpackU8<1>(_src), UnpackU8<1>(_dst), UnpackU8<1>(alpha));
            Store<align, mask>(dst, _mm512_packus_epi16(lo, hi), m);
        }

        template <bool align, bool mask, size_t channelCount> struct AlphaBlender
        {
            void operator()(const uint8_t * src, uint8_t * dst, const uint8_t * alpha, __mmask64 m[channelCount + 1]);
        };

        template <bool align, bool mask> struct AlphaBlender<align, mask, 1>
        {
            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, const uint8_t * alpha, __mmask64 m[2])
            {
                __m512i _alpha = Load<align, mask>(alpha, m[0]);
                AlphaBlending<align, mask>(src, dst, _alpha, m[1]);
            }
        };

        template <bool align, bool mask> struct AlphaBlender<align, mask, 2>
        {
            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, const uint8_t * alpha, __mmask64 m[3])
            {
                __m512i _alpha = Load<align, mask>(alpha, m[0]);
                _alpha = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _alpha);
                AlphaBlending<align, mask>(src + 0, dst + 0, UnpackU8<0>(_alpha, _alpha), m[1]);
                AlphaBlending<align, mask>(src + A, dst + A, UnpackU8<1>(_alpha, _alpha), m[2]);
            }
        };

        template <bool align, bool mask> struct AlphaBlender<align, mask, 3>
        {
            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, const uint8_t * alpha, __mmask64 m[4])
            {
                __m512i _alpha = Load<align, mask>(alpha, m[0]);
                AlphaBlending<align, mask>(src + 0 * A, dst + 0 * A, GrayToBgr<0>(_alpha), m[1]);
                AlphaBlending<align, mask>(src + 1 * A, dst + 1 * A, GrayToBgr<1>(_alpha), m[2]);
                AlphaBlending<align, mask>(src + 2 * A, dst + 2 * A, GrayToBgr<2>(_alpha), m[3]);
            }
        };

        template <bool align, bool mask> struct AlphaBlender<align, mask, 4>
        {
            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, const uint8_t * alpha, __mmask64 m[5])
            {
                __m512i _alpha = Load<align, mask>(alpha, m[0]);
                _alpha = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _alpha);
                __m512i lo = UnpackU8<0>(_alpha, _alpha);
                AlphaBlending<align, mask>(src + 0 * A, dst + 0 * A, UnpackU8<0>(lo, lo), m[1]);
                AlphaBlending<align, mask>(src + 1 * A, dst + 1 * A, UnpackU8<1>(lo, lo), m[2]);
                __m512i hi = UnpackU8<1>(_alpha, _alpha);
                AlphaBlending<align, mask>(src + 2 * A, dst + 2 * A, UnpackU8<0>(hi, hi), m[3]);
                AlphaBlending<align, mask>(src + 3 * A, dst + 3 * A, UnpackU8<1>(hi, hi), m[4]);
            }
        };

        template <bool align, size_t channelCount> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[channelCount + 1];
            tailMasks[0] = TailMask64(width - alignedWidth);
            for (size_t channel = 0; channel < channelCount; ++channel)
                tailMasks[channel + 1] = TailMask64((width - alignedWidth)*channelCount - A * channel);
            size_t step = channelCount * A;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < alignedWidth; col += A, offset += step)
                    AlphaBlender<align, false, channelCount>()(src + offset, dst + offset, alpha + col, tailMasks);
                if (col < width)
                    AlphaBlender<align, true, channelCount>()(src + offset, dst + offset, alpha + col, tailMasks);
                src += srcStride;
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        template <bool align> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(alpha) && Aligned(alphaStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            switch (channelCount)
            {
            case 1: AlphaBlending<align, 1>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 2: AlphaBlending<align, 2>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 3: AlphaBlending<align, 3>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 4: AlphaBlending<align, 4>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(alpha) && Aligned(alphaStride) && Aligned(dst) && Aligned(dstStride))
                AlphaBlending<true>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
            else
                AlphaBlending<false>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
        }

        template <bool align, bool mask> SIMD_INLINE void AlphaFilling(uint8_t * dst, __m512i channelLo, __m512i channelHi, __m512i alpha, __mmask64 m)
        {
            __m512i _dst = Load<align, mask>(dst, m);
            __m512i lo = AlphaBlendingI16(channelLo, UnpackU8<0>(_dst), UnpackU8<0>(alpha));
            __m512i hi = AlphaBlendingI16(channelHi, UnpackU8<1>(_dst), UnpackU8<1>(alpha));
            Store<align, mask>(dst, _mm512_packus_epi16(lo, hi), m);
        }

        template <bool align, bool mask, size_t channelCount> struct AlphaFiller
        {
            void operator()(uint8_t * dst, const __m512i * channel, const uint8_t * alpha, __mmask64 m[channelCount + 1]);
        };

        template <bool align, bool mask> struct AlphaFiller<align, mask, 1>
        {
            SIMD_INLINE void operator()(uint8_t * dst, const __m512i * channel, const uint8_t * alpha, __mmask64 m[2])
            {
                __m512i _alpha = Load<align, mask>(alpha, m[0]);
                AlphaFilling<align, mask>(dst, channel[0], channel[0], _alpha, m[1]);
            }
        };

        template <bool align, bool mask> struct AlphaFiller<align, mask, 2>
        {
            SIMD_INLINE void operator()(uint8_t * dst, const __m512i * channel, const uint8_t * alpha, __mmask64 m[3])
            {
                __m512i _alpha = Load<align, mask>(alpha, m[0]);
                _alpha = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _alpha);
                AlphaFilling<align, mask>(dst + 0 * A, channel[0], channel[0], UnpackU8<0>(_alpha, _alpha), m[1]);
                AlphaFilling<align, mask>(dst + 1 * A, channel[0], channel[0], UnpackU8<1>(_alpha, _alpha), m[2]);
            }
        };

        template <bool align, bool mask> struct AlphaFiller<align, mask, 3>
        {
            SIMD_INLINE void operator()(uint8_t * dst, const __m512i * channel, const uint8_t * alpha, __mmask64 m[4])
            {
                __m512i _alpha = Load<align, mask>(alpha, m[0]);
                AlphaFilling<align, mask>(dst + 0 * A, channel[0], channel[1], GrayToBgr<0>(_alpha), m[1]);
                AlphaFilling<align, mask>(dst + 1 * A, channel[2], channel[0], GrayToBgr<1>(_alpha), m[2]);
                AlphaFilling<align, mask>(dst + 2 * A, channel[1], channel[2], GrayToBgr<2>(_alpha), m[3]);
            }
        };

        template <bool align, bool mask> struct AlphaFiller<align, mask, 4>
        {
            SIMD_INLINE void operator()(uint8_t * dst, const __m512i * channel, const uint8_t * alpha, __mmask64 m[5])
            {
                __m512i _alpha = Load<align, mask>(alpha, m[0]);
                _alpha = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _alpha);
                __m512i lo = UnpackU8<0>(_alpha, _alpha);
                AlphaFilling<align, mask>(dst + 0 * A, channel[0], channel[0], UnpackU8<0>(lo, lo), m[1]);
                AlphaFilling<align, mask>(dst + 1 * A, channel[0], channel[0], UnpackU8<1>(lo, lo), m[2]);
                __m512i hi = UnpackU8<1>(_alpha, _alpha);
                AlphaFilling<align, mask>(dst + 2 * A, channel[0], channel[0], UnpackU8<0>(hi, hi), m[3]);
                AlphaFilling<align, mask>(dst + 3 * A, channel[0], channel[0], UnpackU8<1>(hi, hi), m[4]);
            }
        };

        template <bool align, size_t channelCount> void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const __m512i * channel, const uint8_t * alpha, size_t alphaStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[channelCount + 1];
            tailMasks[0] = TailMask64(width - alignedWidth);
            for (size_t c = 0; c < channelCount; ++c)
                tailMasks[c + 1] = TailMask64((width - alignedWidth)*channelCount - A * c);
            size_t step = channelCount * A;
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < alignedWidth; col += A, offset += step)
                    AlphaFiller<align, false, channelCount>()(dst + offset, channel, alpha + col, tailMasks);
                if (col < width)
                    AlphaFiller<align, true, channelCount>()(dst + offset, channel, alpha + col, tailMasks);
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        template <bool align> void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride)
        {
            if (align)
            {
                assert(Aligned(dst) && Aligned(dstStride));
                assert(Aligned(alpha) && Aligned(alphaStride));
            }

            __m512i _channel[3];
            switch (channelCount)
            {
            case 1:
                _channel[0] = UnpackU8<0>(_mm512_set1_epi8(*(uint8_t*)channel));
                AlphaFilling<align, 1>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            case 2:
                _channel[0] = UnpackU8<0>(_mm512_set1_epi16(*(uint16_t*)channel));
                AlphaFilling<align, 2>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            case 3:
            {
                uint64_t _0120 = uint64_t(channel[0]) | (uint64_t(channel[1]) << 16) | (uint64_t(channel[2]) << 32) | (uint64_t(channel[0]) << 48);
                uint64_t _1201 = uint64_t(channel[1]) | (uint64_t(channel[2]) << 16) | (uint64_t(channel[0]) << 32) | (uint64_t(channel[1]) << 48);
                uint64_t _2012 = uint64_t(channel[2]) | (uint64_t(channel[0]) << 16) | (uint64_t(channel[1]) << 32) | (uint64_t(channel[2]) << 48);
                _channel[0] = _mm512_setr_epi64(_0120, _1201, _1201, _2012, _2012, _0120, _0120, _1201);
                _channel[1] = _mm512_setr_epi64(_2012, _0120, _0120, _1201, _1201, _2012, _2012, _0120);
                _channel[2] = _mm512_setr_epi64(_1201, _2012, _2012, _0120, _0120, _1201, _1201, _2012);
                AlphaFilling<align, 3>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            }
            case 4:
                _channel[0] = UnpackU8<0>(_mm512_set1_epi32(*(uint32_t*)channel));
                AlphaFilling<align, 4>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            default:
                assert(0);
            }
        }

        void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride)
        {
            if (Aligned(dst) && Aligned(dstStride) && Aligned(alpha) && Aligned(alphaStride))
                AlphaFilling<true>(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
            else
                AlphaFilling<false>(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
