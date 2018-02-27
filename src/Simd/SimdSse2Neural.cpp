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
#include "Simd/SimdMemory.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdStream.h"
#include "Simd/SimdPow.h"

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

        template<bool align> void NeuralPow(const float * src, size_t size, const float * exponent, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            float e = exponent[0];
            size_t alignedSize = AlignLo(size, F);
            __m128 _e = _mm_set1_ps(e);
            Pow pow;
            size_t i = 0;
            for (; i < alignedSize; i += F)
                Sse::Store<align>(dst + i, pow(Sse::Load<align>(src + i), _e));
            for (; i < size; ++i)
                dst[i] = Base::Pow(src[i], e);
        }

        void NeuralPow(const float * src, size_t size, const float * exponent, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralPow<true>(src, size, exponent, dst);
            else
                NeuralPow<false>(src, size, exponent, dst);
        }

        class ExpEstimator
        {
            __m128i _exponent, _mantissa, _127;
            __m128 _1_0, _0_5, _min, _max, _exp0, _exp1, _exp2, _exp3, _exp4, _exp5, _k;

            void Init(float k)
            {
                _exponent = _mm_set1_epi32(0x7F800000);
                _mantissa = _mm_set1_epi32(0x007FFFFF);
                _127 = _mm_set1_epi32(127);
                _1_0 = _mm_set1_ps(1.0f);
                _0_5 = _mm_set1_ps(0.5f);
                _min = _mm_set1_ps(-126.99999f);
                _max = _mm_set1_ps(129.00000f);
                _exp0 = _mm_set1_ps(9.9999994e-1f);
                _exp1 = _mm_set1_ps(6.9315308e-1f);
                _exp2 = _mm_set1_ps(2.4015361e-1f);
                _exp3 = _mm_set1_ps(5.5826318e-2f);
                _exp4 = _mm_set1_ps(8.9893397e-3f);
                _exp5 = _mm_set1_ps(1.8775767e-3f);
                _k = _mm_set1_ps(k / 0.69314718056f);
            }

            SIMD_INLINE __m128 Poly5(__m128 x)
            {
                __m128 p = _exp5;
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp4);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp3);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp2);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp1);
                p = _mm_add_ps(_mm_mul_ps(x, p), _exp0);
                return p;
            }

            SIMD_INLINE __m128 Exp2(__m128 x)
            {
                x = _mm_max_ps(_mm_min_ps(x, _max), _min);
                __m128i ipart = _mm_cvtps_epi32(_mm_sub_ps(x, _0_5));
                __m128 fpart = _mm_sub_ps(x, _mm_cvtepi32_ps(ipart));
                __m128 expipart = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(ipart, _127), 23));
                __m128 expfpart = Poly5(fpart);
                return _mm_mul_ps(expipart, expfpart);
            }

            SIMD_INLINE __m128 Sigmoid(__m128 value)
            {
                __m128 exp = Exp2(_mm_mul_ps(_k, value));
                return _mm_div_ps(_1_0, _mm_add_ps(_1_0, exp));
            }

            template<bool align> void Sigmoid(const float * src, size_t size, const float * slope, float * dst)
            {
                if (align)
                    assert(Aligned(src) && Aligned(dst));

                Init(-slope[0]);
                size_t alignedSize = AlignLo(size, F);
                size_t i = 0;
                for (; i < alignedSize; i += F)
                    Sse::Store<align>(dst + i, Sigmoid(Sse::Load<align>(src + i)));
                for (; i < size; ++i)
                    dst[i] = Base::Sigmoid(src[i] * slope[0]);
            }

            SIMD_INLINE __m128 Tanh(__m128 value)
            {
                __m128 exp = Exp2(_mm_mul_ps(_k, value));
                return _mm_div_ps(_mm_sub_ps(_1_0, exp), _mm_add_ps(_1_0, exp));
            }

            template<bool align> void Tanh(const float * src, size_t size, const float * slope, float * dst)
            {
                if (align)
                    assert(Aligned(src) && Aligned(dst));

                Init(-2.0f*slope[0]);
                size_t alignedSize = AlignLo(size, F);
                size_t i = 0;
                for (; i < alignedSize; i += F)
                    Sse::Store<align>(dst + i, Tanh(Sse::Load<align>(src + i)));
                for (; i < size; ++i)
                    dst[i] = Base::Tanh(src[i] * slope[0]);
            }

        public:
            void Sigmoid(const float * src, size_t size, const float * slope, float * dst)
            {
                if (Aligned(src) && Aligned(dst))
                    Sigmoid<true>(src, size, slope, dst);
                else
                    Sigmoid<false>(src, size, slope, dst);
            }

            void Tanh(const float * src, size_t size, const float * slope, float * dst)
            {
                if (Aligned(src) && Aligned(dst))
                    Tanh<true>(src, size, slope, dst);
                else
                    Tanh<false>(src, size, slope, dst);
            }
        };

        void NeuralSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            ExpEstimator estimator;
            estimator.Sigmoid(src, size, slope, dst);
        }

        void NeuralTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            ExpEstimator estimator;
            estimator.Tanh(src, size, slope, dst);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
