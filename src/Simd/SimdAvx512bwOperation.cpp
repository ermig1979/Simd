/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:SimdOperationBinary8u
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
#include "Simd/SimdStore.h"
#include "Simd/SimdAlphaBlending.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <SimdOperationBinary8uType type> SIMD_INLINE __m512i OperationBinary8u(const __m512i & a, const __m512i & b);

        template <> SIMD_INLINE __m512i OperationBinary8u<SimdOperationBinary8uAverage>(const __m512i & a, const __m512i & b)
        {
            return _mm512_avg_epu8(a, b);
        }

        template <> SIMD_INLINE __m512i OperationBinary8u<SimdOperationBinary8uAnd>(const __m512i & a, const __m512i & b)
        {
            return _mm512_and_si512(a, b);
        }

        template <> SIMD_INLINE __m512i OperationBinary8u<SimdOperationBinary8uOr>(const __m512i & a, const __m512i & b)
        {
            return _mm512_or_si512(a, b);
        }

        template <> SIMD_INLINE __m512i OperationBinary8u<SimdOperationBinary8uMaximum>(const __m512i & a, const __m512i & b)
        {
            return _mm512_max_epu8(a, b);
        }

        template <> SIMD_INLINE __m512i OperationBinary8u<SimdOperationBinary8uMinimum>(const __m512i & a, const __m512i & b)
        {
            return _mm512_min_epu8(a, b);
        }

        template <> SIMD_INLINE __m512i OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(const __m512i & a, const __m512i & b)
        {
            return _mm512_subs_epu8(a, b);
        }

        template <> SIMD_INLINE __m512i OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(const __m512i & a, const __m512i & b)
        {
            return _mm512_adds_epu8(a, b);
        }

        template <bool align, bool mask, SimdOperationBinary8uType type> void OperationBinary8u(const uint8_t * a, const uint8_t * b, uint8_t * dst, size_t offset, __mmask64 m = -1)
        {
            const __m512i _a = Load<align, mask>(a + offset, m);
            const __m512i _b = Load<align, mask>(b + offset, m);
            Store<align, mask>(dst + offset, OperationBinary8u<type>(_a, _b), m);
        }

        template <bool align, SimdOperationBinary8uType type> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

            size_t size = channelCount*width;
            size_t fullAlignedSize = Simd::AlignLo(size, QA);
            size_t partialAlignedSize = Simd::AlignLo(size, A);
            __mmask64 tailMask = __mmask64(-1) >> (A + partialAlignedSize - size);
            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < fullAlignedSize; offset += QA)
                {
                    OperationBinary8u<align, false, type>(a, b, dst, offset);
                    OperationBinary8u<align, false, type>(a, b, dst, offset + A);
                    OperationBinary8u<align, false, type>(a, b, dst, offset + 2 * A);
                    OperationBinary8u<align, false, type>(a, b, dst, offset + 3 * A);
                }
                for (; offset < partialAlignedSize; offset += A)
                    OperationBinary8u<align, false, type>(a, b, dst, offset);
                for (; offset < size; offset += A)
                    OperationBinary8u<align, true, type>(a, b, dst, offset, tailMask);
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        template <bool align> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            switch (type)
            {
            case SimdOperationBinary8uAverage:
                return OperationBinary8u<align, SimdOperationBinary8uAverage>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uAnd:
                return OperationBinary8u<align, SimdOperationBinary8uAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uOr:
                return OperationBinary8u<align, SimdOperationBinary8uOr>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMaximum:
                return OperationBinary8u<align, SimdOperationBinary8uMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMinimum:
                return OperationBinary8u<align, SimdOperationBinary8uMinimum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedSubtraction:
                return OperationBinary8u<align, SimdOperationBinary8uSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedAddition:
                return OperationBinary8u<align, SimdOperationBinary8uSaturatedAddition>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            default:
                assert(0);
            }
        }

        void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
                OperationBinary8u<true>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
            else
                OperationBinary8u<false>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride, type);
        }

        template <SimdOperationBinary16iType type> SIMD_INLINE __m512i OperationBinary16i(const __m512i & a, const __m512i & b);

        template <> SIMD_INLINE __m512i OperationBinary16i<SimdOperationBinary16iAddition>(const __m512i & a, const __m512i & b)
        {
            return _mm512_add_epi16(a, b);
        }

        template <> SIMD_INLINE __m512i OperationBinary16i<SimdOperationBinary16iSubtraction>(const __m512i & a, const __m512i & b)
        {
            return _mm512_sub_epi16(a, b);
        }

        template <bool align, bool mask, SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t * a, const uint8_t * b, uint8_t * dst, size_t offset, __mmask32 m = -1)
        {
            const __m512i _a = Load<align, mask>((int16_t*)(a + offset), m);
            const __m512i _b = Load<align, mask>((int16_t*)(b + offset), m);
            Store<align, mask>((int16_t*)(dst + offset), OperationBinary16i<type>(_a, _b), m);
        }

        template <bool align, SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

            size_t size = width * sizeof(int16_t);
            size_t fullAlignedSize = Simd::AlignLo(size, QA);
            size_t partialAlignedSize = Simd::AlignLo(size, A);
            __mmask32 tailMask = __mmask32(-1) >> ((A + partialAlignedSize - size) / sizeof(int16_t));
            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < fullAlignedSize; offset += QA)
                {
                    OperationBinary16i<align, false, type>(a, b, dst, offset);
                    OperationBinary16i<align, false, type>(a, b, dst, offset + A);
                    OperationBinary16i<align, false, type>(a, b, dst, offset + 2 * A);
                    OperationBinary16i<align, false, type>(a, b, dst, offset + 3 * A);
                }
                for (; offset < partialAlignedSize; offset += A)
                    OperationBinary16i<align, false, type>(a, b, dst, offset);
                for (; offset < size; offset += A)
                    OperationBinary16i<align, true, type>(a, b, dst, offset, tailMask);
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        template <bool align> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
        {
            switch (type)
            {
            case SimdOperationBinary16iAddition:
                return OperationBinary16i<align, SimdOperationBinary16iAddition>(a, aStride, b, bStride, width, height, dst, dstStride);
            case SimdOperationBinary16iSubtraction:
                return OperationBinary16i<align, SimdOperationBinary16iSubtraction>(a, aStride, b, bStride, width, height, dst, dstStride);
            default:
                assert(0);
            }
        }

        void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
                OperationBinary16i<true>(a, aStride, b, bStride, width, height, dst, dstStride, type);
            else
                OperationBinary16i<false>(a, aStride, b, bStride, width, height, dst, dstStride, type);
        }

        SIMD_INLINE __m512i VectorProduct(const __m512i & vertical, const __m512i & horizontalLo, const __m512i & horizontalHi)
        {
            __m512i lo = Divide16uBy255(_mm512_mullo_epi16(vertical, horizontalLo));
            __m512i hi = Divide16uBy255(_mm512_mullo_epi16(vertical, horizontalHi));
            return _mm512_packus_epi16(lo, hi);
        }

        template <bool align, bool mask> SIMD_INLINE void VectorProduct2(const __m512i & vertical0, const __m512i & vertical1, const uint8_t * horizontal, uint8_t * dst, size_t stride, __mmask64 m = -1)
        {
            __m512i _horizontal = Load<align, mask>(horizontal, m);
            __m512i horizontalLo = UnpackU8<0>(_horizontal);
            __m512i horizontalHi = UnpackU8<1>(_horizontal);
            Store<align, mask>(dst + 0 * stride, VectorProduct(vertical0, horizontalLo, horizontalHi), m);
            Store<align, mask>(dst + 1 * stride, VectorProduct(vertical1, horizontalLo, horizontalHi), m);
        }

        template <bool align, bool mask> SIMD_INLINE void VectorProduct1(const __m512i & vertical, const uint8_t * horizontal, uint8_t * dst, __mmask64 m = -1)
        {
            __m512i _horizontal = Load<align, mask>(horizontal, m);
            Store<align, mask>(dst, VectorProduct(vertical, UnpackU8<0>(_horizontal), UnpackU8<1>(_horizontal)), m);
        }

        template <bool align> void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            if (align)
                assert(Aligned(horizontal) && Aligned(dst) && Aligned(stride));

            size_t alignedHeight = Simd::AlignLo(height, 2);
            size_t alignedWidth = Simd::AlignLo(width, A);
            __mmask64 tailMask = __mmask64(-1) >> (A + alignedWidth - width);
            size_t row = 0;
            for (; row < alignedHeight; row += 2)
            {
                __m512i vertical0 = _mm512_set1_epi16(vertical[row + 0]);
                __m512i vertical1 = _mm512_set1_epi16(vertical[row + 1]);
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    VectorProduct2<align, false>(vertical0, vertical1, horizontal + col, dst + col, stride);
                if (col < width)
                    VectorProduct2<false, true>(vertical0, vertical1, horizontal + col, dst + col, stride, tailMask);
                dst += 2 * stride;
            }
            for (; row < height; ++row)
            {
                __m512i _vertical = _mm512_set1_epi16(vertical[row]);
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    VectorProduct1<align, false>(_vertical, horizontal + col, dst + col);
                if (col < width)
                    VectorProduct1<false, true>(_vertical, horizontal + col, dst + col, tailMask);
                dst += stride;
            }
        }

        void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            if (Aligned(horizontal) && Aligned(dst) && Aligned(stride))
                VectorProduct<true>(vertical, horizontal, dst, stride, width, height);
            else
                VectorProduct<false>(vertical, horizontal, dst, stride, width, height);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
