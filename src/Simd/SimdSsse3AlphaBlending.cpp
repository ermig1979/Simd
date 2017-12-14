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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SSSE3_ENABLE    
    namespace Ssse3
    {
        template <bool align, size_t channelCount> struct AlphaBlender
        {
            void operator()(const __m128i * src, __m128i * dst, __m128i alpha);
        };

        template <bool align> struct AlphaBlender<align, 1>
        {
            SIMD_INLINE void operator()(const __m128i * src, __m128i * dst, __m128i alpha)
            {
                Sse2::AlphaBlending<align>(src, dst, alpha);
            }
        };

        template <bool align> struct AlphaBlender<align, 2>
        {
            SIMD_INLINE void operator()(const __m128i * src, __m128i * dst, __m128i alpha)
            {
                Sse2::AlphaBlending<align>(src + 0, dst + 0, _mm_unpacklo_epi8(alpha, alpha));
                Sse2::AlphaBlending<align>(src + 1, dst + 1, _mm_unpackhi_epi8(alpha, alpha));
            }
        };

        template <bool align> struct AlphaBlender<align, 3>
        {
            SIMD_INLINE void operator()(const __m128i * src, __m128i * dst, __m128i alpha)
            {
                Sse2::AlphaBlending<align>(src + 0, dst + 0, _mm_shuffle_epi8(alpha, K8_SHUFFLE_GRAY_TO_BGR0));
                Sse2::AlphaBlending<align>(src + 1, dst + 1, _mm_shuffle_epi8(alpha, K8_SHUFFLE_GRAY_TO_BGR1));
                Sse2::AlphaBlending<align>(src + 2, dst + 2, _mm_shuffle_epi8(alpha, K8_SHUFFLE_GRAY_TO_BGR2));
            }
        };

        template <bool align> struct AlphaBlender<align, 4>
        {
            SIMD_INLINE void operator()(const __m128i * src, __m128i * dst, __m128i alpha)
            {
                __m128i lo = _mm_unpacklo_epi8(alpha, alpha);
                Sse2::AlphaBlending<align>(src + 0, dst + 0, _mm_unpacklo_epi8(lo, lo));
                Sse2::AlphaBlending<align>(src + 1, dst + 1, _mm_unpackhi_epi8(lo, lo));
                __m128i hi = _mm_unpackhi_epi8(alpha, alpha);
                Sse2::AlphaBlending<align>(src + 2, dst + 2, _mm_unpacklo_epi8(hi, hi));
                Sse2::AlphaBlending<align>(src + 3, dst + 3, _mm_unpackhi_epi8(hi, hi));
            }
        };

        template <bool align, size_t channelCount> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
            size_t step = channelCount * A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += step)
                {
                    __m128i _alpha = Load<align>((__m128i*)(alpha + col));
                    AlphaBlender<align, channelCount>()((__m128i*)(src + offset), (__m128i*)(dst + offset), _alpha);
                }
                if (alignedWidth != width)
                {
                    __m128i _alpha = _mm_and_si128(Load<false>((__m128i*)(alpha + width - A)), tailMask);
                    AlphaBlender<false, channelCount>()((__m128i*)(src + (width - A)*channelCount), (__m128i*)(dst + (width - A)*channelCount), _alpha);
                }
                src += srcStride;
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        template <bool align> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            assert(width >= A);
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


        template <bool align, size_t channelCount> struct AlphaFiller
        {
            void operator() (__m128i * dst, const __m128i * channel, __m128i alpha);
        };

        template <bool align> struct AlphaFiller<align, 1>
        {
            SIMD_INLINE void operator()(__m128i * dst, const __m128i * channel, __m128i alpha)
            {
                Sse2::AlphaFilling<align>(dst, channel[0], channel[0], alpha);
            }
        };

        template <bool align> struct AlphaFiller<align, 2>
        {
            SIMD_INLINE void operator()(__m128i * dst, const __m128i * channel, __m128i alpha)
            {
                Sse2::AlphaFilling<align>(dst + 0, channel[0], channel[0], UnpackU8<0>(alpha, alpha));
                Sse2::AlphaFilling<align>(dst + 1, channel[0], channel[0], UnpackU8<1>(alpha, alpha));
            }
        };

        template <bool align> struct AlphaFiller<align, 3>
        {
            SIMD_INLINE void operator()(__m128i * dst, const __m128i * channel, __m128i alpha)
            {
                Sse2::AlphaFilling<align>(dst + 0, channel[0], channel[1], _mm_shuffle_epi8(alpha, K8_SHUFFLE_GRAY_TO_BGR0));
                Sse2::AlphaFilling<align>(dst + 1, channel[2], channel[0], _mm_shuffle_epi8(alpha, K8_SHUFFLE_GRAY_TO_BGR1));
                Sse2::AlphaFilling<align>(dst + 2, channel[1], channel[2], _mm_shuffle_epi8(alpha, K8_SHUFFLE_GRAY_TO_BGR2));
            }
        };

        template <bool align> struct AlphaFiller<align, 4>
        {
            SIMD_INLINE void operator()(__m128i * dst, const __m128i * channel, __m128i alpha)
            {
                __m128i lo = UnpackU8<0>(alpha, alpha);
                Sse2::AlphaFilling<align>(dst + 0, channel[0], channel[0], UnpackU8<0>(lo, lo));
                Sse2::AlphaFilling<align>(dst + 1, channel[0], channel[0], UnpackU8<1>(lo, lo));
                __m128i hi = UnpackU8<1>(alpha, alpha);
                Sse2::AlphaFilling<align>(dst + 2, channel[0], channel[0], UnpackU8<0>(hi, hi));
                Sse2::AlphaFilling<align>(dst + 3, channel[0], channel[0], UnpackU8<1>(hi, hi));
            }
        };

        template <bool align, size_t channelCount> void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const __m128i * channel, const uint8_t * alpha, size_t alphaStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
            size_t step = channelCount * A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += step)
                {
                    __m128i _alpha = Load<align>((__m128i*)(alpha + col));
                    AlphaFiller<align, channelCount>()((__m128i*)(dst + offset), channel, _alpha);
                }
                if (alignedWidth != width)
                {
                    __m128i _alpha = _mm_and_si128(Load<false>((__m128i*)(alpha + width - A)), tailMask);
                    AlphaFiller<false, channelCount>()((__m128i*)(dst + (width - A)*channelCount), channel, _alpha);
                }
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        template <bool align> void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(dst) && Aligned(dstStride));
                assert(Aligned(alpha) && Aligned(alphaStride));
            }

            __m128i _channel[3];
            switch (channelCount)
            {
            case 1:
                _channel[0] = UnpackU8<0>(_mm_set1_epi8(*(uint8_t*)channel));
                AlphaFilling<align, 1>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            case 2:
                _channel[0] = UnpackU8<0>(_mm_set1_epi16(*(uint16_t*)channel));
                AlphaFilling<align, 2>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            case 3:
                _channel[0] = _mm_setr_epi16(channel[0], channel[1], channel[2], channel[0], channel[1], channel[2], channel[0], channel[1]);
                _channel[1] = _mm_setr_epi16(channel[2], channel[0], channel[1], channel[2], channel[0], channel[1], channel[2], channel[0]);
                _channel[2] = _mm_setr_epi16(channel[1], channel[2], channel[0], channel[1], channel[2], channel[0], channel[1], channel[2]);
                AlphaFilling<align, 3>(dst, dstStride, width, height, _channel, alpha, alphaStride);
                break;
            case 4:
                _channel[0] = UnpackU8<0>(_mm_set1_epi32(*(uint32_t*)channel));
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
#endif// SIMD_SSSE3_ENABLE
}
