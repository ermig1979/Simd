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
#include "Simd/SimdAlphaBlending.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <int part> SIMD_INLINE uint8x8_t AlphaBlending(const uint8x16_t & src, const uint8x16_t & dst,
            const uint8x16_t & alpha, const uint8x16_t & ff_alpha)
        {
            uint16x8_t value = vaddq_u16(
                vmull_u8(Half<part>(src), Half<part>(alpha)),
                vmull_u8(Half<part>(dst), Half<part>(ff_alpha)));
            return vshrn_n_u16(vaddq_u16(vaddq_u16(value, K16_0001), vshrq_n_u16(value, 8)), 8);
        }

        template <bool align> SIMD_INLINE void AlphaBlending(const uint8_t * src, uint8_t * dst, const uint8x16_t & alpha)
        {
            uint8x16_t _src = Load<align>(src);
            uint8x16_t _dst = Load<align>(dst);
            uint8x16_t ff_alpha = vsubq_u8(K8_FF, alpha);
            uint8x8_t lo = AlphaBlending<0>(_src, _dst, alpha, ff_alpha);
            uint8x8_t hi = AlphaBlending<1>(_src, _dst, alpha, ff_alpha);
            Store<align>(dst, vcombine_u8(lo, hi));
        }

        template <bool align, size_t channelCount> struct AlphaBlender
        {
            void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha);
        };

        template <bool align> struct AlphaBlender<align, 1>
        {
            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha)
            {
                AlphaBlending<align>(src, dst, alpha);
            }
        };

        template <bool align> struct AlphaBlender<align, 2>
        {
            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha)
            {
                uint8x16x2_t _alpha = vzipq_u8(alpha, alpha);
                AlphaBlending<align>(src + 0, dst + 0, _alpha.val[0]);
                AlphaBlending<align>(src + A, dst + A, _alpha.val[1]);
            }
        };

        template <bool align> struct AlphaBlender<align, 3>
        {
            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha)
            {
                uint8x16x3_t _alpha;
                _alpha.val[0] = alpha;
                _alpha.val[1] = alpha;
                _alpha.val[2] = alpha;
                Store3<align>((uint8_t*)&_alpha, _alpha);
                AlphaBlending<align>(src + 0 * A, dst + 0 * A, _alpha.val[0]);
                AlphaBlending<align>(src + 1 * A, dst + 1 * A, _alpha.val[1]);
                AlphaBlending<align>(src + 2 * A, dst + 2 * A, _alpha.val[2]);
            }
        };

        template <bool align> struct AlphaBlender<align, 4>
        {
            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha)
            {
                uint8x16x2_t _alpha = vzipq_u8(alpha, alpha);
                AlphaBlender<align, 2>()(src + A * 0, dst + A * 0, _alpha.val[0]);
                AlphaBlender<align, 2>()(src + A * 2, dst + A * 2, _alpha.val[1]);
            }
        };

        template <bool align, size_t channelCount> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t step = channelCount*A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += step)
                {
                    uint8x16_t _alpha = Load<align>(alpha + col);
                    AlphaBlender<align, channelCount>()(src + offset, dst + offset, _alpha);
                }
                if (alignedWidth != width)
                {
                    uint8x16_t _alpha = vandq_u8(Load<false>(alpha + width - A), tailMask);
                    AlphaBlender<false, channelCount>()(src + (width - A)*channelCount, dst + (width - A)*channelCount, _alpha);
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

        //-----------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AlphaBlending2x(const uint8_t* src0, uint8x16_t alpha0, const uint8_t* src1, uint8x16_t alpha1, uint8_t* dst)
        {
            uint8x16_t _src = Load<align>(src0);
            uint8x16_t _dst = Load<align>(dst);
            uint8x16_t ff_alpha = vsubq_u8(K8_FF, alpha0);
            uint8x8_t lo = AlphaBlending<0>(_src, _dst, alpha0, ff_alpha);
            uint8x8_t hi = AlphaBlending<1>(_src, _dst, alpha0, ff_alpha);
            _src = Load<align>(src1);
            _dst = vcombine_u8(lo, hi);
            ff_alpha = vsubq_u8(K8_FF, alpha1);
            lo = AlphaBlending<0>(_src, _dst, alpha1, ff_alpha);
            hi = AlphaBlending<1>(_src, _dst, alpha1, ff_alpha);
            Store<align>(dst, vcombine_u8(lo, hi));
        }

        template <bool align, size_t channelCount> struct AlphaBlender2x
        {
            void operator()(const uint8_t* src0, uint8x16_t alpha0, const uint8_t* src1, uint8x16_t alpha1, uint8_t* dst);
        };

        template <bool align> struct AlphaBlender2x<align, 1>
        {
            SIMD_INLINE void operator()(const uint8_t* src0, uint8x16_t alpha0, const uint8_t* src1, uint8x16_t alpha1, uint8_t* dst)
            {
                AlphaBlending2x<align>(src0, alpha0, src1, alpha1, dst);
            }
        };

        template <bool align> struct AlphaBlender2x<align, 2>
        {
            SIMD_INLINE void operator()(const uint8_t* src0, uint8x16_t alpha0, const uint8_t* src1, uint8x16_t alpha1, uint8_t* dst)
            {
                uint8x16x2_t _alpha0 = vzipq_u8(alpha0, alpha0);
                uint8x16x2_t _alpha1 = vzipq_u8(alpha1, alpha1);
                AlphaBlending2x<align>(src0 + 0, _alpha0.val[0], src1 + 0, _alpha1.val[0], dst + 0);
                AlphaBlending2x<align>(src0 + A, _alpha0.val[1], src1 + A, _alpha1.val[1], dst + A);
            }
        };

        template <bool align> struct AlphaBlender2x<align, 3>
        {
            SIMD_INLINE void operator()(const uint8_t* src0, uint8x16_t alpha0, const uint8_t* src1, uint8x16_t alpha1, uint8_t* dst)
            {
                uint8x16x3_t _alpha0;
                _alpha0.val[0] = alpha0;
                _alpha0.val[1] = alpha0;
                _alpha0.val[2] = alpha0;
                Store3<align>((uint8_t*)&_alpha0, _alpha0);
                uint8x16x3_t _alpha1;
                _alpha1.val[0] = alpha1;
                _alpha1.val[1] = alpha1;
                _alpha1.val[2] = alpha1;
                Store3<align>((uint8_t*)&_alpha1, _alpha1);
                AlphaBlending2x<align>(src0 + 0 * A, _alpha0.val[0], src1 + 0 * A, _alpha1.val[0], dst + 0 * A);
                AlphaBlending2x<align>(src0 + 1 * A, _alpha0.val[1], src1 + 1 * A, _alpha1.val[1], dst + 1 * A);
                AlphaBlending2x<align>(src0 + 2 * A, _alpha0.val[2], src1 + 2 * A, _alpha1.val[2], dst + 2 * A);
            }
        };

        template <bool align> struct AlphaBlender2x<align, 4>
        {
            SIMD_INLINE void operator()(const uint8_t* src0, uint8x16_t alpha0, const uint8_t* src1, uint8x16_t alpha1, uint8_t* dst)
            {
                uint8x16x2_t _alpha0 = vzipq_u8(alpha0, alpha0);
                uint8x16x2_t _alpha1 = vzipq_u8(alpha1, alpha1);
                AlphaBlender2x<align, 2>()(src0 + A * 0, _alpha0.val[0], src1 + A * 0, _alpha1.val[0], dst + A * 0);
                AlphaBlender2x<align, 2>()(src0 + A * 2, _alpha0.val[1], src1 + A * 2, _alpha1.val[1], dst + A * 2);
            }
        };

        template <bool align, size_t channelCount> void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t step = channelCount * A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += step)
                {
                    uint8x16_t _alpha0 = Load<align>(alpha0 + col);
                    uint8x16_t _alpha1 = Load<align>(alpha1 + col);
                    AlphaBlender2x<align, channelCount>()(src0 + offset, _alpha0, src1 + offset, _alpha1, dst + offset);
                }
                if (alignedWidth != width)
                {
                    size_t col = width - A, offset = col * channelCount;
                    uint8x16_t _alpha0 = vandq_u8(Load<false>(alpha0 + col), tailMask);
                    uint8x16_t _alpha1 = vandq_u8(Load<false>(alpha1 + col), tailMask);
                    AlphaBlender2x<false, channelCount>()(src0 + offset, _alpha0, src1 + offset, _alpha1, dst + offset);
                }
                src0 += src0Stride;
                alpha0 += alpha0Stride;
                src1 += src1Stride;
                alpha1 += alpha1Stride;
                dst += dstStride;
            }
        }

        template <bool align> void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(src0) && Aligned(src0Stride));
                assert(Aligned(alpha0) && Aligned(alpha0Stride));
                assert(Aligned(src1) && Aligned(src1Stride));
                assert(Aligned(alpha1) && Aligned(alpha1Stride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            switch (channelCount)
            {
            case 1: AlphaBlending2x<align, 1>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 2: AlphaBlending2x<align, 2>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 3: AlphaBlending2x<align, 3>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 4: AlphaBlending2x<align, 4>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            if (Aligned(src0) && Aligned(src0Stride) && Aligned(alpha0) && Aligned(alpha0Stride) &&
                Aligned(src1) && Aligned(src1Stride) && Aligned(alpha1) && Aligned(alpha1Stride) && Aligned(dst) && Aligned(dstStride))
                AlphaBlending2x<true>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, channelCount, dst, dstStride);
            else
                AlphaBlending2x<false>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, channelCount, dst, dstStride);
        }

        //-----------------------------------------------------------------------

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
            uint8x16_t _alpha = vdupq_n_u8(alpha);
            uint8x16_t tail = vandq_u8(ShiftLeft(K8_FF, A - size + sizeA), _alpha);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t offs = 0; offs < sizeA; offs += A)
                    AlphaBlending<align>(src + offs, dst + offs, _alpha);
                if (sizeA != size)
                    AlphaBlending<false>(src + size - A, dst + size - A, tail);
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

        //-----------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AlphaFilling(uint8_t * dst, const uint8x16_t & channel, const uint8x16_t & alpha)
        {
            uint8x16_t _dst = Load<align>(dst);
            uint8x16_t ff_alpha = vsubq_u8(K8_FF, alpha);
            uint8x8_t lo = AlphaBlending<0>(channel, _dst, alpha, ff_alpha);
            uint8x8_t hi = AlphaBlending<1>(channel, _dst, alpha, ff_alpha);
            Store<align>(dst, vcombine_u8(lo, hi));
        }

        template <bool align, size_t channelCount> struct AlphaFiller
        {
            void operator() (uint8x16_t * dst, const uint8x16_t * channel, const uint8x16_t & alpha);
        };

        template <bool align> struct AlphaFiller<align, 1>
        {
            SIMD_INLINE void operator()(uint8_t * dst, const uint8x16_t * channel, const uint8x16_t & alpha)
            {
                AlphaFilling<align>(dst, channel[0], alpha);
            }
        };

        template <bool align> struct AlphaFiller<align, 2>
        {
            SIMD_INLINE void operator()(uint8_t * dst, const uint8x16_t * channel, const uint8x16_t & alpha)
            {
                uint8x16x2_t _alpha = vzipq_u8(alpha, alpha);
                AlphaFilling<align>(dst + 0 * A, channel[0], _alpha.val[0]);
                AlphaFilling<align>(dst + 1 * A, channel[1], _alpha.val[1]);
            }
        };

        template <bool align> struct AlphaFiller<align, 3>
        {
            SIMD_INLINE void operator()(uint8_t * dst, const uint8x16_t * channel, const uint8x16_t & alpha)
            {
                uint8x16x3_t _alpha;
                _alpha.val[0] = alpha;
                _alpha.val[1] = alpha;
                _alpha.val[2] = alpha;
                Store3<align>((uint8_t*)&_alpha, _alpha);
                AlphaFilling<align>(dst + 0 * A, channel[0], _alpha.val[0]);
                AlphaFilling<align>(dst + 1 * A, channel[1], _alpha.val[1]);
                AlphaFilling<align>(dst + 2 * A, channel[2], _alpha.val[2]);
            }
        };

        template <bool align> struct AlphaFiller<align, 4>
        {
            SIMD_INLINE void operator()(uint8_t * dst, const uint8x16_t * channel, const uint8x16_t & alpha)
            {
                uint8x16x2_t _alpha = vzipq_u8(alpha, alpha);
                AlphaFiller<align, 2>()(dst + A * 0, channel + 0, _alpha.val[0]);
                AlphaFiller<align, 2>()(dst + A * 2, channel + 2, _alpha.val[1]);
            }
        };

        template <bool align, size_t channelCount> void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8x16_t * channel, const uint8_t * alpha, size_t alphaStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t step = channelCount * A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += step)
                {
                    uint8x16_t _alpha = Load<align>(alpha + col);
                    AlphaFiller<align, channelCount>()(dst + offset, channel, _alpha);
                }
                if (alignedWidth != width)
                {
                    uint8x16_t _alpha = vandq_u8(Load<false>(alpha + width - A), tailMask);
                    AlphaFiller<false, channelCount>()(dst + (width - A)*channelCount, channel, _alpha);
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
            {
                uint8x16_t _channel = vdupq_n_u8(channel[0]);
                AlphaFilling<align, 1>(dst, dstStride, width, height, &_channel, alpha, alphaStride);
                break;
            }
            case 2:
            {
                uint8x16x2_t _channel;
                _channel.val[0] = vdupq_n_u8(channel[0]);
                _channel.val[1] = vdupq_n_u8(channel[1]);
                Store2<align>((uint8_t*)&_channel, _channel);
                AlphaFilling<align, 2>(dst, dstStride, width, height, _channel.val, alpha, alphaStride);
                break;
            }
            case 3:
            {
                uint8x16x3_t _channel;
                _channel.val[0] = vdupq_n_u8(channel[0]);
                _channel.val[1] = vdupq_n_u8(channel[1]);
                _channel.val[2] = vdupq_n_u8(channel[2]);
                Store3<align>((uint8_t*)&_channel, _channel);
                AlphaFilling<align, 3>(dst, dstStride, width, height, _channel.val, alpha, alphaStride);
                break;
            }
            case 4:
            {
                uint8x16x4_t _channel;
                _channel.val[0] = vdupq_n_u8(channel[0]);
                _channel.val[1] = vdupq_n_u8(channel[1]);
                _channel.val[2] = vdupq_n_u8(channel[2]);
                _channel.val[3] = vdupq_n_u8(channel[3]);
                Store4<align>((uint8_t*)&_channel, _channel);
                AlphaFilling<align, 4>(dst, dstStride, width, height, _channel.val, alpha, alphaStride);
                break;
            }
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

        //---------------------------------------------------------------------

        SIMD_INLINE uint16x8_t AlphaPremultiply(uint16x8_t value, uint16x8_t alpha)
        {
            uint16x8_t prem = vmulq_u16(value, alpha);
            return vshrq_n_u16(vaddq_u16(vaddq_u16(prem, K16_0001), vshrq_n_u16(prem, 8)), 8);
        }

        template<bool argb> void AlphaPremultiply2(const uint8_t* src, uint8_t* dst);

        template<> SIMD_INLINE void AlphaPremultiply2<false>(const uint8_t* src, uint8_t* dst)
        {
            uint8x8x4_t bgra = LoadHalf4<false>(src);
            uint16x8_t alpha = vmovl_u8(bgra.val[3]);
            bgra.val[0] = vmovn_u16(AlphaPremultiply(vmovl_u8(bgra.val[0]), alpha));
            bgra.val[1] = vmovn_u16(AlphaPremultiply(vmovl_u8(bgra.val[1]), alpha));
            bgra.val[2] = vmovn_u16(AlphaPremultiply(vmovl_u8(bgra.val[2]), alpha));
            Store4<false>(dst, bgra);
        }

        template<> SIMD_INLINE void AlphaPremultiply2<true>(const uint8_t* src, uint8_t* dst)
        {
            uint8x8x4_t argb = LoadHalf4<false>(src);
            uint16x8_t alpha = vmovl_u8(argb.val[0]);
            argb.val[1] = vmovn_u16(AlphaPremultiply(vmovl_u8(argb.val[1]), alpha));
            argb.val[2] = vmovn_u16(AlphaPremultiply(vmovl_u8(argb.val[2]), alpha));
            argb.val[3] = vmovn_u16(AlphaPremultiply(vmovl_u8(argb.val[3]), alpha));
            Store4<false>(dst, argb);
        }

        template<bool argb> void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t size = width * 4;
            size_t sizeA2 = AlignLo(size, A * 2);
            for (size_t row = 0; row < height; ++row)
            {
                size_t i = 0;
                for (; i < sizeA2; i += A * 2)
                    AlphaPremultiply2<argb>(src + i, dst + i);
                for (; i < size; i += 4)
                    Base::AlphaPremultiply<argb>(src + i, dst + i);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, SimdBool argb)
        {
            if (argb)
                AlphaPremultiply<true>(src, srcStride, width, height, dst, dstStride);
            else
                AlphaPremultiply<false>(src, srcStride, width, height, dst, dstStride);
        }

        //-----------------------------------------------------------------------

        template<bool argb> void AlphaUnpremultiply(const uint8_t* src, uint8_t* dst, float32x4_t _0, float32x4_t _255);

        template<> SIMD_INLINE void AlphaUnpremultiply<false>(const uint8_t* src, uint8_t* dst, float32x4_t _0, float32x4_t _255)
        {
            uint32x4_t _src = Load<false>((uint32_t*)src);
            uint32x4_t b = vandq_u32(_src, K32_000000FF);
            uint32x4_t g = vandq_u32(vshrq_n_u32(_src, 8), K32_000000FF);
            uint32x4_t r = vandq_u32(vshrq_n_u32(_src, 16), K32_000000FF);
            uint32x4_t a = vshrq_n_u32(_src, 24);
            float32x4_t k = vcvtq_f32_u32(a);
            k = vbslq_f32(vceqq_f32(k, _0), k, Div<-1>(_255, k));
            b = vcvtq_u32_f32(vminq_f32(vmulq_f32(vcvtq_f32_u32(b), k), _255));
            g = vcvtq_u32_f32(vminq_f32(vmulq_f32(vcvtq_f32_u32(g), k), _255));
            r = vcvtq_u32_f32(vminq_f32(vmulq_f32(vcvtq_f32_u32(r), k), _255));
            uint32x4_t _dst = vorrq_u32(b, vshlq_n_u32(g, 8));
            _dst = vorrq_u32(_dst, vshlq_n_u32(r, 16));
            _dst = vorrq_u32(_dst, vshlq_n_u32(a, 24));
            Store<false>((uint32_t*)dst, _dst);
        }

        template<> SIMD_INLINE void AlphaUnpremultiply<true>(const uint8_t* src, uint8_t* dst, float32x4_t _0, float32x4_t _255)
        {
            uint32x4_t _src = Load<false>((uint32_t*)src);
            uint32x4_t a = vandq_u32(_src, K32_000000FF);
            uint32x4_t r = vandq_u32(vshrq_n_u32(_src, 8), K32_000000FF);
            uint32x4_t g = vandq_u32(vshrq_n_u32(_src, 16), K32_000000FF);
            uint32x4_t b = vshrq_n_u32(_src, 24);
            float32x4_t k = vcvtq_f32_u32(a);
            k = vbslq_f32(vceqq_f32(k, _0), k, Div<-1>(_255, k));
            b = vcvtq_u32_f32(vminq_f32(vmulq_f32(vcvtq_f32_u32(b), k), _255));
            g = vcvtq_u32_f32(vminq_f32(vmulq_f32(vcvtq_f32_u32(g), k), _255));
            r = vcvtq_u32_f32(vminq_f32(vmulq_f32(vcvtq_f32_u32(r), k), _255));
            uint32x4_t _dst = vorrq_u32(a, vshlq_n_u32(r, 8));
            _dst = vorrq_u32(_dst, vshlq_n_u32(g, 16));
            _dst = vorrq_u32(_dst, vshlq_n_u32(b, 24));
            Store<false>((uint32_t*)dst, _dst);
        }

        template<bool argb> void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            float32x4_t _0 = vdupq_n_f32(0.0f);
            float32x4_t _255 = vdupq_n_f32(255.00001f);
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < sizeA; col += A)
                    AlphaUnpremultiply<argb>(src + col, dst + col, _0, _255);
                for (; col < size; col += 4)
                    Base::AlphaUnpremultiply<argb>(src + col, dst + col);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, SimdBool argb)
        {
            if (argb)
                AlphaUnpremultiply<true>(src, srcStride, width, height, dst, dstStride);
            else
                AlphaUnpremultiply<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
