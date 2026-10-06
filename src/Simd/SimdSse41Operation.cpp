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
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <SimdOperationBinary8uType type> SIMD_INLINE __m128i OperationBinary8u(const __m128i & a, const __m128i & b);

        template <> SIMD_INLINE __m128i OperationBinary8u<SimdOperationBinary8uAverage>(const __m128i & a, const __m128i & b)
        {
            return _mm_avg_epu8(a, b);
        }

        template <> SIMD_INLINE __m128i OperationBinary8u<SimdOperationBinary8uAnd>(const __m128i & a, const __m128i & b)
        {
            return _mm_and_si128(a, b);
        }

        template <> SIMD_INLINE __m128i OperationBinary8u<SimdOperationBinary8uOr>(const __m128i & a, const __m128i & b)
        {
            return _mm_or_si128(a, b);
        }

        template <> SIMD_INLINE __m128i OperationBinary8u<SimdOperationBinary8uMaximum>(const __m128i & a, const __m128i & b)
        {
            return _mm_max_epu8(a, b);
        }

        template <> SIMD_INLINE __m128i OperationBinary8u<SimdOperationBinary8uMinimum>(const __m128i & a, const __m128i & b)
        {
            return _mm_min_epu8(a, b);
        }

        template <> SIMD_INLINE __m128i OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(const __m128i & a, const __m128i & b)
        {
            return _mm_subs_epu8(a, b);
        }

        template <> SIMD_INLINE __m128i OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(const __m128i & a, const __m128i & b)
        {
            return _mm_adds_epu8(a, b);
        }

        template <SimdOperationBinary8uType type> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(width*channelCount >= A);

            size_t size = channelCount*width;
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t offset = 0; offset < alignedSize; offset += A)
                {
                    const __m128i a_ = _mm_loadu_si128((__m128i*)(a + offset));
                    const __m128i b_ = _mm_loadu_si128((__m128i*)(b + offset));
                    _mm_storeu_si128((__m128i*)(dst + offset), OperationBinary8u<type>(a_, b_));
                }
                if (alignedSize != size)
                {
                    const __m128i a_ = _mm_loadu_si128((__m128i*)(a + size - A));
                    const __m128i b_ = _mm_loadu_si128((__m128i*)(b + size - A));
                    _mm_storeu_si128((__m128i*)(dst + size - A), OperationBinary8u<type>(a_, b_));
                }
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            switch (type)
            {
            case SimdOperationBinary8uAverage:
                return OperationBinary8u<SimdOperationBinary8uAverage>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uAnd:
                return OperationBinary8u<SimdOperationBinary8uAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uOr:
                return OperationBinary8u<SimdOperationBinary8uOr>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMaximum:
                return OperationBinary8u<SimdOperationBinary8uMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMinimum:
                return OperationBinary8u<SimdOperationBinary8uMinimum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedSubtraction:
                return OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedAddition:
                return OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            default:
                assert(0);
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdOperationBinary16iType type> SIMD_INLINE __m128i OperationBinary16i(const __m128i & a, const __m128i & b);

        template <> SIMD_INLINE __m128i OperationBinary16i<SimdOperationBinary16iAddition>(const __m128i & a, const __m128i & b)
        {
            return _mm_add_epi16(a, b);
        }

        template <> SIMD_INLINE __m128i OperationBinary16i<SimdOperationBinary16iSubtraction>(const __m128i & a, const __m128i & b)
        {
            return _mm_sub_epi16(a, b);
        }

        template <SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(width * sizeof(uint16_t) >= A);

            size_t size = width * sizeof(int16_t);
            size_t alignedSize = Simd::AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t offset = 0; offset < alignedSize; offset += A)
                {
                    const __m128i a_ = _mm_loadu_si128((__m128i*)(a + offset));
                    const __m128i b_ = _mm_loadu_si128((__m128i*)(b + offset));
                    _mm_storeu_si128((__m128i*)(dst + offset), OperationBinary16i<type>(a_, b_));
                }
                if (alignedSize != size)
                {
                    const __m128i a_ = _mm_loadu_si128((__m128i*)(a + size - A));
                    const __m128i b_ = _mm_loadu_si128((__m128i*)(b + size - A));
                    _mm_storeu_si128((__m128i*)(dst + size - A), OperationBinary16i<type>(a_, b_));
                }
                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
        {
            switch (type)
            {
            case SimdOperationBinary16iAddition:
                return OperationBinary16i<SimdOperationBinary16iAddition>(a, aStride, b, bStride, width, height, dst, dstStride);
            case SimdOperationBinary16iSubtraction:
                return OperationBinary16i<SimdOperationBinary16iSubtraction>(a, aStride, b, bStride, width, height, dst, dstStride);
            default:
                assert(0);
            }
        }
    }
#endif
}
