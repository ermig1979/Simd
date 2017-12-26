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
#include "Simd/SimdMemory.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdStream.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template <bool inversion> __m128i Invert(__m128i value);

        template <> __m128i Invert<true>(__m128i value)
        {
            return _mm_sub_epi8(K_INV_ZERO, value);
        }

        template <> __m128i Invert<false>(__m128i value)
        {
            return value;
        }

        template <bool align, bool stream> void Convert(__m128i src, const __m128 &_1_255, float * dst)
        {
            Sse::Stream<align, stream>(dst + 0, _mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(src)), _1_255));
            Sse::Stream<align, stream>(dst + 4, _mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(src)), _1_255));
        }

        template <bool inversion, bool align, bool stream> void Convert(const uint8_t * src, const __m128 &_1_255, float * dst)
        {
            __m128i _src = Invert<inversion>(Load<align>((__m128i*)src));
            Convert<align, stream>(UnpackU8<0>(_src), _1_255, dst + 0);
            Convert<align, stream>(UnpackU8<1>(_src), _1_255, dst + 8);
        }

        template <bool inversion, bool align, bool stream> void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, A);
            __m128 _1_255 = _mm_set1_ps(1.0f / 255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    Convert<inversion, align, stream>(src + col, _1_255, dst + col);
                if (width != alignedWidth)
                    Convert<inversion, false, stream>(src + width - A, _1_255, dst + width - A);
                src += srcStride;
                dst += dstStride;
            }
            if (stream)
                _mm_mfence();
        }

        template <bool inversion> void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
            {
                if (width*height * sizeof(float) >= STREAM_SIZE_MIN)
                    NeuralConvert<inversion, true, true>(src, srcStride, width, height, dst, dstStride);
                else
                    NeuralConvert<inversion, true, false>(src, srcStride, width, height, dst, dstStride);
            }
            else
                NeuralConvert<inversion, false, false>(src, srcStride, width, height, dst, dstStride);
        }

        void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride, int inversion)
        {
            if (inversion)
                NeuralConvert<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralConvert<false>(src, srcStride, width, height, dst, dstStride);
        }

        class Power
        {
            __m128i _exponent, _mantissa;
            __m128 _one;

            void Init()
            {
                _exponent = _mm_set1_epi32(0x7F800000);
                _mantissa = _mm_set1_epi32(0x007FFFFF);
                _one = _mm_set1_ps(1.0f);
            }

            SIMD_INLINE __m128 Poly5(__m128 x, float a, float b, float c, float d, float e, float f)
            {
                __m128 p = _mm_set1_ps(f);
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(e));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(d));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(c));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(b));
                p = _mm_add_ps(_mm_mul_ps(x, p), _mm_set1_ps(a));
                return p;
            }

            SIMD_INLINE __m128 Exp2(__m128 x)
            {
                x = _mm_max_ps(_mm_min_ps(x, _mm_set1_ps(129.00000f)), _mm_set1_ps(-126.99999f));
                __m128i ipart = _mm_cvtps_epi32(_mm_sub_ps(x, _mm_set1_ps(0.5f)));
                __m128 fpart = _mm_sub_ps(x, _mm_cvtepi32_ps(ipart));
                __m128 expipart = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ipart, _mm_set1_epi32(127)), 23));
                __m128 expfpart = Poly5(fpart, 9.9999994e-1f, 6.9315308e-1f, 2.4015361e-1f, 5.5826318e-2f, 8.9893397e-3f, 1.8775767e-3f);
                return _mm_mul_ps(expipart, expfpart);
            }

            SIMD_INLINE __m128 Log2(__m128 x)
            {
                __m128i i = _mm_castps_si128(x);
                __m128 e = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_srli_epi32(_mm_and_si128(i, _exponent), 23), _mm_set1_epi32(127)));
                __m128 m = _mm_or_ps(_mm_castsi128_ps(_mm_and_si128(i, _mantissa)), _one);
                __m128 p = Poly5(m, 3.1157899f, -3.3241990f, 2.5988452f, -1.2315303f, 3.1821337e-1f, -3.4436006e-2f);
                return _mm_add_ps(_mm_mul_ps(p, _mm_sub_ps(m, _one)), e);
            }

            SIMD_INLINE __m128 Pow(__m128 basis, __m128 exponent)
            {
                return Exp2(_mm_mul_ps(Log2(basis), exponent));
            }

            template<bool align> void Run(const float * src, size_t size, const float * exponent, float * dst)
            {
                if (align)
                    assert(Aligned(src) && Aligned(dst));

                float e = exponent[0];
                size_t alignedSize = AlignLo(size, F);
                __m128 _e = _mm_set1_ps(e);
                size_t i = 0;
                for (; i < alignedSize; i += F)
                    Sse::Store<align>(dst + i, Pow(Sse::Load<align>(src + i), _e));
                for (; i < size; ++i)
                    dst[i] = Base::Pow(src[i], e);
            } 

        public:
            void Run(const float * src, size_t size, const float * exponent, float * dst)
            {
                Init();

                if (Aligned(src) && Aligned(dst))
                    Run<true>(src, size, exponent, dst);
                else
                    Run<false>(src, size, exponent, dst);
            }
        };

        void NeuralPow(const float * src, size_t size, const float * exponent, float * dst)
        {
            Power power;
            power.Run(src, size, exponent, dst);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
