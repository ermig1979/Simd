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
#include "Simd/SimdAlphaBlending.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template <bool align, size_t channelCount> struct AlphaBlender
        {
            void operator()(const __m128i * src, __m128i * dst, __m128i alpha);
        };

        template <bool align> struct AlphaBlender<align, 1>
        {
            SIMD_INLINE void operator()(const __m128i * src, __m128i * dst, __m128i alpha)
            {
                AlphaBlending<align>(src, dst, alpha);
            }
        };

        template <bool align> struct AlphaBlender<align, 2>
        {
            SIMD_INLINE void operator()(const __m128i * src, __m128i * dst, __m128i alpha)
            {
                AlphaBlending<align>(src + 0, dst + 0, _mm_unpacklo_epi8(alpha, alpha));
                AlphaBlending<align>(src + 1, dst + 1, _mm_unpackhi_epi8(alpha, alpha));
            }
        };

        template <bool align> struct AlphaBlender<align, 4>
        {
            SIMD_INLINE void operator()(const __m128i * src, __m128i * dst, __m128i alpha)
            {
                __m128i lo = _mm_unpacklo_epi8(alpha, alpha);
                AlphaBlending<align>(src + 0, dst + 0, _mm_unpacklo_epi8(lo, lo));
                AlphaBlending<align>(src + 1, dst + 1, _mm_unpackhi_epi8(lo, lo));
                __m128i hi = _mm_unpackhi_epi8(alpha, alpha);
                AlphaBlending<align>(src + 2, dst + 2, _mm_unpacklo_epi8(hi, hi));
                AlphaBlending<align>(src + 3, dst + 3, _mm_unpackhi_epi8(hi, hi));
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
            case 4: AlphaBlending<align, 4>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            if (channelCount == 3)
                Base::AlphaBlending(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
            else
            {
                if (Aligned(src) && Aligned(srcStride) && Aligned(alpha) && Aligned(alphaStride) && Aligned(dst) && Aligned(dstStride))
                    AlphaBlending<true>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
                else
                    AlphaBlending<false>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
            }
        }

        //---------------------------------------------------------------------

        template <bool align> void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, 
            size_t channelCount, uint8_t alpha, uint8_t* dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }
            size_t size = width * channelCount;
            size_t sizeA = AlignLo(size, A);
            __m128i _alpha = _mm_set1_epi8(alpha);
            __m128i tail = _mm_and_si128(ShiftLeft(K_INV_ZERO, A - size + sizeA), _alpha);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t offs = 0; offs < sizeA; offs += A)
                    AlphaBlending<align>((__m128i*)(src + offs), (__m128i*)(dst + offs), _alpha);
                if (sizeA != size)
                    AlphaBlending<false>((__m128i*)(src + size - A), (__m128i*)(dst + size - A), tail);
                src += srcStride;
                dst += dstStride;
            }            
        }

        void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            uint8_t alpha, uint8_t* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AlphaBlendingUniform<true>(src, srcStride, width, height, channelCount, alpha, dst, dstStride);
            else
                AlphaBlendingUniform<false>(src, srcStride, width, height, channelCount, alpha, dst, dstStride);
        }

        //---------------------------------------------------------------------

        template <bool align, size_t channelCount> struct AlphaFiller
        {
            void operator() (__m128i * dst, __m128i channel, __m128i alpha);
        };

        template <bool align> struct AlphaFiller<align, 1>
        {
            SIMD_INLINE void operator()(__m128i * dst, __m128i channel, __m128i alpha)
            {
                AlphaFilling<align>(dst, channel, channel, alpha);
            }
        };

        template <bool align> struct AlphaFiller<align, 2>
        {
            SIMD_INLINE void operator()(__m128i * dst, __m128i channel, __m128i alpha)
            {
                AlphaFilling<align>(dst + 0, channel, channel, UnpackU8<0>(alpha, alpha));
                AlphaFilling<align>(dst + 1, channel, channel, UnpackU8<1>(alpha, alpha));
            }
        };

        template <bool align> struct AlphaFiller<align, 4>
        {
            SIMD_INLINE void operator()(__m128i * dst, __m128i channel, __m128i alpha)
            {
                __m128i lo = UnpackU8<0>(alpha, alpha);
                AlphaFilling<align>(dst + 0, channel, channel, UnpackU8<0>(lo, lo));
                AlphaFilling<align>(dst + 1, channel, channel, UnpackU8<1>(lo, lo));
                __m128i hi = UnpackU8<1>(alpha, alpha);
                AlphaFilling<align>(dst + 2, channel, channel, UnpackU8<0>(hi, hi));
                AlphaFilling<align>(dst + 3, channel, channel, UnpackU8<1>(hi, hi));
            }
        };

        template <bool align, size_t channelCount> void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const __m128i & channel, const uint8_t * alpha, size_t alphaStride)
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

            switch (channelCount)
            {
            case 1:
                AlphaFilling<align, 1>(dst, dstStride, width, height, UnpackU8<0>(_mm_set1_epi8(*(uint8_t*)channel)), alpha, alphaStride);
                break;
            case 2:
                AlphaFilling<align, 2>(dst, dstStride, width, height, UnpackU8<0>(_mm_set1_epi16(*(uint16_t*)channel)), alpha, alphaStride);
                break;
            case 4:
                AlphaFilling<align, 4>(dst, dstStride, width, height, UnpackU8<0>(_mm_set1_epi32(*(uint32_t*)channel)), alpha, alphaStride);
                break;
            default:
                assert(0);
            }
        }

        void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride)
        {
            if (channelCount == 3)
                Base::AlphaFilling(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
            else
            {
                if (Aligned(dst) && Aligned(dstStride) && Aligned(alpha) && Aligned(alphaStride))
                    AlphaFilling<true>(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
                else
                    AlphaFilling<false>(dst, dstStride, width, height, channel, channelCount, alpha, alphaStride);
            }
        }

        //---------------------------------------------------------------------

        SIMD_INLINE void AlphaPremultiply(const uint8_t* src, uint8_t* dst)
        {
            __m128i bgra = _mm_loadu_si128((__m128i*)src);
            __m128i a000 = _mm_and_si128(_mm_srli_si128(bgra, 3), K32_000000FF);
            __m128i a0a0 = _mm_or_si128(a000, _mm_slli_si128(a000, 2));
            __m128i b0r0 = _mm_and_si128(bgra, K16_00FF);
            __m128i g0f0 = _mm_or_si128(_mm_and_si128(_mm_srli_si128(bgra, 1), K32_000000FF), K32_00FF0000);
            __m128i B0R0 = AlphaPremultiply16i(b0r0, a0a0);
            __m128i G0A0 = AlphaPremultiply16i(g0f0, a0a0);
            _mm_storeu_si128((__m128i*)dst, _mm_or_si128(B0R0, _mm_slli_si128(G0A0, 1)));
        }

        void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t i = 0;
                for (; i < sizeA; i += A)
                    AlphaPremultiply(src + i, dst + i);
                for (;i < size; i += 4)
                    Base::AlphaPremultiply(src + i, dst + i);
                src += srcStride;
                dst += dstStride;
            }
        }
    }
#endif// SIMD_SSE2_ENABLE
}
