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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdAlphaBlending.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i AlphaBlending16i(const __m512i &  src, const __m512i &  dst, const __m512i &  alpha)
        {
            return Divide16uBy255(_mm512_add_epi16(_mm512_mullo_epi16(src, alpha), _mm512_mullo_epi16(dst, _mm512_sub_epi16(K16_00FF, alpha))));
        }

        template <bool align, bool mask> SIMD_INLINE void AlphaBlending(const uint8_t * src, uint8_t * dst, const __m512i & alpha, __mmask64 m)
        {
            __m512i _src = Load<align, mask>(src, m);
            __m512i _dst = Load<align, mask>(dst, m);
            __m512i lo = AlphaBlending16i(UnpackU8<0>(_src), UnpackU8<0>(_dst), UnpackU8<0>(alpha));
            __m512i hi = AlphaBlending16i(UnpackU8<1>(_src), UnpackU8<1>(_dst), UnpackU8<1>(alpha));
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
            __m512i _alpha = _mm512_set1_epi8(alpha);
            __mmask64 tail = TailMask64(size - sizeA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t offs = 0;
                for (; offs < sizeA; offs += A)
                    AlphaBlending<align, false>(src + offs, dst + offs, _alpha, -1);
                if (offs < size)
                    AlphaBlending<align, true>(src + offs, dst + offs, _alpha, tail);
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

        template <bool align, bool mask> SIMD_INLINE void AlphaFilling(uint8_t * dst, __m512i channelLo, __m512i channelHi, __m512i alpha, __mmask64 m)
        {
            __m512i _dst = Load<align, mask>(dst, m);
            __m512i lo = AlphaBlending16i(channelLo, UnpackU8<0>(_dst), UnpackU8<0>(alpha));
            __m512i hi = AlphaBlending16i(channelHi, UnpackU8<1>(_dst), UnpackU8<1>(alpha));
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

        //---------------------------------------------------------------------

        SIMD_INLINE __m512i AlphaPremultiply16i(__m512i value, __m512i alpha)
        {
            return Divide16uBy255(_mm512_mullo_epi16(value, alpha));
        }

        const __m512i K8_SHUFFLE_BGRA_TO_A0A0 = SIMD_MM512_SETR_EPI8(
            0x3, -1, 0x3, -1, 0x7, -1, 0x7, -1, 0xB, -1, 0xB, -1, 0xF, -1, 0xF, -1,
            0x3, -1, 0x3, -1, 0x7, -1, 0x7, -1, 0xB, -1, 0xB, -1, 0xF, -1, 0xF, -1,
            0x3, -1, 0x3, -1, 0x7, -1, 0x7, -1, 0xB, -1, 0xB, -1, 0xF, -1, 0xF, -1,
            0x3, -1, 0x3, -1, 0x7, -1, 0x7, -1, 0xB, -1, 0xB, -1, 0xF, -1, 0xF, -1);

        SIMD_INLINE void AlphaPremultiply(const uint8_t* src, uint8_t* dst, __mmask64 tail = -1)
        {
            __m512i bgra = _mm512_maskz_loadu_epi8(tail, src);
            __m512i a0a0 = _mm512_shuffle_epi8(bgra, K8_SHUFFLE_BGRA_TO_A0A0);
            __m512i b0r0 = _mm512_and_si512(bgra, K16_00FF);
            __m512i g0f0 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi32(bgra, 8), K32_000000FF), K32_00FF0000);
            __m512i B0R0 = AlphaPremultiply16i(b0r0, a0a0);
            __m512i G0A0 = AlphaPremultiply16i(g0f0, a0a0);
            _mm512_mask_storeu_epi8(dst, tail, _mm512_or_si512(B0R0, _mm512_slli_epi32(G0A0, 8)));
        }

        void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            __mmask64 tail = TailMask64(size - sizeA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t i = 0;
                for (; i < sizeA; i += A)
                    AlphaPremultiply(src + i, dst + i);
                if (i < size)
                    AlphaPremultiply(src + i, dst + i, tail);
                src += srcStride;
                dst += dstStride;
            }
        }

        //---------------------------------------------------------------------

#if defined(_MSC_VER) && _MSC_VER < 1927
        void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            Avx2::AlphaUnpremultiply(src, srcStride, width, height, dst, dstStride);
        }
#else
        const __m512i K8_SHUFFLE_BGRA_TO_B = SIMD_MM512_SETR_EPI8(
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1,
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1,
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1,
            0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1);
        const __m512i K8_SHUFFLE_BGRA_TO_G = SIMD_MM512_SETR_EPI8(
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1,
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1,
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1,
            0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1);
        const __m512i K8_SHUFFLE_BGRA_TO_R = SIMD_MM512_SETR_EPI8(
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1,
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1,
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1,
            0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1);
        const __m512i K8_SHUFFLE_BGRA_TO_A = SIMD_MM512_SETR_EPI8(
            0x3, -1, -1, -1, 0x7, -1, -1, -1, 0xB, -1, -1, -1, 0xF, -1, -1, -1,
            0x3, -1, -1, -1, 0x7, -1, -1, -1, 0xB, -1, -1, -1, 0xF, -1, -1, -1,
            0x3, -1, -1, -1, 0x7, -1, -1, -1, 0xB, -1, -1, -1, 0xF, -1, -1, -1,
            0x3, -1, -1, -1, 0x7, -1, -1, -1, 0xB, -1, -1, -1, 0xF, -1, -1, -1);

        SIMD_INLINE void AlphaUnpremultiply(const uint8_t* src, uint8_t* dst, __m512 _255, __mmask64 tail = -1)
        {
            __m512i _src = _mm512_maskz_loadu_epi8(tail, src);
            __m512i b = _mm512_shuffle_epi8(_src, K8_SHUFFLE_BGRA_TO_B);
            __m512i g = _mm512_shuffle_epi8(_src, K8_SHUFFLE_BGRA_TO_G);
            __m512i r = _mm512_shuffle_epi8(_src, K8_SHUFFLE_BGRA_TO_R);
            __m512i a = _mm512_shuffle_epi8(_src, K8_SHUFFLE_BGRA_TO_A);
            __m512 k = _mm512_cvtepi32_ps(a);
            k = _mm512_maskz_div_ps(_mm512_cmp_ps_mask(k, _mm512_setzero_ps(), _CMP_NEQ_UQ), _255, k);
            b = _mm512_cvtps_epi32(_mm512_min_ps(_mm512_floor_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(b), k)), _255));
            g = _mm512_cvtps_epi32(_mm512_min_ps(_mm512_floor_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(g), k)), _255));
            r = _mm512_cvtps_epi32(_mm512_min_ps(_mm512_floor_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(r), k)), _255));
            __m512i _dst = _mm512_or_si512(b, _mm512_slli_epi32(g, 8));
            _dst = _mm512_or_si512(_dst, _mm512_slli_epi32(r, 16));
            _dst = _mm512_or_si512(_dst, _mm512_slli_epi32(a, 24));
            _mm512_mask_storeu_epi8(dst, tail, _dst);
        }

        void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            __m512 _255 = _mm512_set1_ps(255.00001f);
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            __mmask64 tail = TailMask64(size - sizeA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < sizeA; col += A)
                    AlphaUnpremultiply(src + col, dst + col, _255);
                if(col < size)
                    AlphaUnpremultiply(src + col, dst + col, _255, tail);
                src += srcStride;
                dst += dstStride;
            }
        }
#endif
    }
#endif// SIMD_AVX512BW_ENABLE
}
