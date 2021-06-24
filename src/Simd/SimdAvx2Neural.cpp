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
#include "Simd/SimdMemory.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdStream.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdNeural.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool inversion> __m128i Invert(__m128i value);

        template <> __m128i Invert<true>(__m128i value)
        {
            return _mm_sub_epi8(Sse2::K_INV_ZERO, value);
        }

        template <> __m128i Invert<false>(__m128i value)
        {
            return value;
        }

        template <bool inversion, bool align, bool stream> void Convert(const uint8_t * src, const __m256 & _1_255, float * dst)
        {
            __m128i _src = Invert<inversion>(_mm_loadl_epi64((__m128i*)src));
            Avx::Stream<align, stream>(dst, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_src)), _1_255));
        }

        template <bool inversion, bool align, bool stream> void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            assert(width >= F);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, F);
            __m256 _1_255 = _mm256_set1_ps(1.0f / 255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                    Convert<inversion, align, stream>(src + col, _1_255, dst + col);
                if (width != alignedWidth)
                    Convert<inversion, false, stream>(src + width - F, _1_255, dst + width - F);
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

        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t offset, __m256 & sum)
        {
            __m256 _a = Load<align>(a + offset);
            __m256 _b = Load<align>(b + offset);
            sum = _mm256_fmadd_ps(_a, _b, sum);
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        NeuralProductSum<align>(a, b, i + F * 0, sums[0]);
                        NeuralProductSum<align>(a, b, i + F * 1, sums[1]);
                        NeuralProductSum<align>(a, b, i + F * 2, sums[2]);
                        NeuralProductSum<align>(a, b, i + F * 3, sums[3]);
                    }
                    sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += F)
                    NeuralProductSum<align>(a, b, i, sums[0]);
                *sum += Avx::ExtractSum(sums[0]);
            }
            for (; i < size; ++i)
                *sum += a[i] * b[i];
        }

        void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                NeuralProductSum<true>(a, b, size, sum);
            else
                NeuralProductSum<false>(a, b, size, sum);
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, const __m256 & value, float * dst)
        {
            Avx::Store<align>(dst, _mm256_fmadd_ps(value, Load<align>(src), Load<align>(dst)));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
        {
            size_t i = 0;
            if (partial)
            {
                __m256 _value = _mm256_set1_ps(value);
                for (; i < aligned; i += QF)
                {
                    AddMultiplied<align>(src + i + F * 0, _value, dst + i + 0);
                    AddMultiplied<align>(src + i + F * 1, _value, dst + i + 8);
                    AddMultiplied<align>(src + i + F * 2, _value, dst + i + 16);
                    AddMultiplied<align>(src + i + F * 3, _value, dst + i + 24);
                }
                for (; i < partial; i += F)
                    AddMultiplied<align>(src + i, _value, dst + i);
            }
            for (; i < full; ++i)
                dst[i] += src[i] * value;
        }

        void NeuralAddVectorMultipliedByValue(const float * src, size_t size, const float * value, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(src) && Aligned(dst))
                AddMultiplied<true>(src, aligned, partial, size, *value, dst);
            else
                AddMultiplied<false>(src, aligned, partial, size, *value, dst);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, const __m256 & k, const __m256 & o, const __m256 & m, float * dst)
        {
            __m256 _src = Load<align>(src);
            __m256 e1 = _mm256_max_ps(m, _mm256_fmadd_ps(_src, k, o));
            __m256 e2 = _mm256_mul_ps(e1, e1);
            __m256 e4 = _mm256_mul_ps(e2, e2);
            __m256 e8 = _mm256_mul_ps(e4, e4);
            __m256 e16 = _mm256_mul_ps(e8, e8);
            __m256 e32 = _mm256_mul_ps(e16, e16);
            __m256 e64 = _mm256_mul_ps(e32, e32);
            __m256 sigmoid = _mm256_rcp_ps(_mm256_fmadd_ps(e64, e64, o));
            Avx::Store<align>(dst, sigmoid);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            __m256 _k = _mm256_set1_ps(-(*slope)*0.0078125f);
            __m256 _1 = _mm256_set1_ps(1.0f);
            __m256 _05 = _mm256_set1_ps(0.5f);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                NeuralRoughSigmoid2<align>(src + i + 0 * F, _k, _1, _05, dst + i + 0 * F);
                NeuralRoughSigmoid2<align>(src + i + 1 * F, _k, _1, _05, dst + i + 1 * F);
                NeuralRoughSigmoid2<align>(src + i + 2 * F, _k, _1, _05, dst + i + 2 * F);
                NeuralRoughSigmoid2<align>(src + i + 3 * F, _k, _1, _05, dst + i + 3 * F);
            }
            for (; i < partialAlignedSize; i += F)
                NeuralRoughSigmoid2<align>(src + i, _k, _1, _05, dst + i);
            for (; i < size; ++i)
                dst[i] = Base::RoughSigmoid2(src[i] * slope[0]);
        }

        void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid2<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid2<false>(src, size, slope, dst);
        }

        template<bool align> void NeuralPow(const float * src, size_t size, const float * exponent, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            float e = exponent[0];
            size_t alignedSize = AlignLo(size, F);
            __m256 _e = _mm256_set1_ps(e);
            Pow pow;
            size_t i = 0;
            for (; i < alignedSize; i += F)
                Avx::Store<align>(dst + i, pow(Avx::Load<align>(src + i), _e));
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

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionForward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, F);
            __m256 tailMask = RightNotZero32f(width - alignedWidth);
            __m256 _weights[coreX*coreY];
            LoadWeightsForward<coreX*coreY>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += F)
                {
                    __m256 _dst = Avx::Load<align>(dst + col);
                    _dst = _mm256_add_ps(_dst, Convolution<coreX, coreY>::template Forward<align>(src + col, srcStride, _weights));
                    Avx::Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m256 _dst = Avx::Load<false>(dst + col);
                    _dst = _mm256_add_ps(_dst, _mm256_and_ps(tailMask, Convolution<coreX, coreY>::template Forward<false>(src + col, srcStride, _weights)));
                    Avx::Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution2x2Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution4x4Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template<bool condition> struct If
        {
            template<bool align> static SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
            {
                Avx2::AddMultiplied<align>(src, aligned, partial, full, value, dst);
            }
        };

        template<> struct If<false>
        {
            template<bool align> static SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
            {
            }
        };

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackwardSmall(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t aligned = AlignLo(width, QF);
            size_t partial = AlignLo(width, F);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < coreY; ++dy)
                {
                    const float * w = weights + dy * coreX;
                    float * d = dst + dy*dstStride;
                    If < 0 < coreX > ::template AddMultiplied<align>(src, aligned, partial, width, w[0], d + 0);
                    If < 1 < coreX > ::template AddMultiplied<false>(src, aligned, partial, width, w[1], d + 1);
                    If < 2 < coreX > ::template AddMultiplied<false>(src, aligned, partial, width, w[2], d + 2);
                    If < 3 < coreX > ::template AddMultiplied<false>(src, aligned, partial, width, w[3], d + 3);
                    If < 4 < coreX > ::template AddMultiplied<false>(src, aligned, partial, width, w[4], d + 4);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackwardLarge(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            ConvolutionBackwardBuffer<coreX> buffer(width, F);
            height += coreY - 1;
            width += coreX - 1;
            size_t alignedWidth = AlignLo(width, F);
            __m256 tailMask = RightNotZero32f(width - alignedWidth);
            __m256 _weights[coreX*coreY];
            LoadWeightsBackward<coreX*coreY>(weights, _weights);

            for (size_t row = 0; row < height; ++row)
            {
                buffer.Update(row <= height - coreY ? src : NULL);
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m256 _dst = Avx::Load<align>(dst + col);
                    _dst = _mm256_add_ps(_dst, Convolution<coreX, coreY>::template Backward<true>(buffer, col, _weights));
                    Avx::Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m256 _dst = Avx::Load<false>(dst + col);
                    _dst = _mm256_add_ps(_dst, _mm256_and_ps(tailMask, Convolution<coreX, coreY>::template Backward<false>(buffer, col, _weights)));
                    Avx::Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (width*height < 1024)
                NeuralAddConvolutionBackwardSmall<align, coreX, coreY>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackwardLarge<align, coreX, coreY>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution2x2Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution4x4Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <bool align, size_t coreX, size_t coreY> SIMD_INLINE void NeuralAddConvolutionSum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m256 tailMask = RightNotZero32f(width - alignedWidth);
            __m256 _sums[coreX*coreY];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m256 _dst = Avx::Load<align>(dst + col);
                    Convolution<coreX, coreY>::template Sum<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m256 _dst = _mm256_and_ps(tailMask, Avx::Load<false>(dst + col));
                    Convolution<coreX, coreY>::template Sum<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            size_t i = 0, n = Simd::AlignLo(coreX*coreY, F);
            for (; i < n; i += F)
                Add8ExtractedSums(_sums + i, sums + i);
            for (; i < coreX*coreY; ++i)
                sums[i] += Avx::ExtractSum(_sums[i]);
        }

        void NeuralAddConvolution2x2Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution4x4Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
        }

        template <bool align> SIMD_INLINE __m256 Pooling1x1Max3x1Body(const float * src)
        {
            return _mm256_max_ps(_mm256_max_ps(Avx::Load<false>(src - 1), Avx::Load<align>(src)), Avx::Load<false>(src + 1));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Body(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Body<align>(src - stride);
            __m256 src1 = Pooling1x1Max3x1Body<align>(src);
            __m256 src2 = Pooling1x1Max3x1Body<align>(src + stride);
            Avx::Store<align>(dst, _mm256_max_ps(_mm256_max_ps(src0, src1), src2));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Body(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Body<align>(src);
            __m256 src1 = Pooling1x1Max3x1Body<align>(src + stride);
            Avx::Store<align>(dst, _mm256_max_ps(src0, src1));
        }

        __m256i K32_PERMUTE_NOSE = SIMD_MM256_SETR_EPI32(0, 0, 1, 2, 3, 4, 5, 6);

        template <bool align> SIMD_INLINE __m256 Pooling1x1Max3x1Nose(const float * src)
        {
            __m256 src1 = Avx::Load<align>(src);
            __m256 src0 = _mm256_permutevar8x32_ps(src1, K32_PERMUTE_NOSE);
            __m256 src2 = Avx::Load<false>(src + 1);
            return _mm256_max_ps(_mm256_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Nose(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Nose<align>(src - stride);
            __m256 src1 = Pooling1x1Max3x1Nose<align>(src);
            __m256 src2 = Pooling1x1Max3x1Nose<align>(src + stride);
            Avx::Store<align>(dst, _mm256_max_ps(_mm256_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Nose(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Nose<align>(src);
            __m256 src1 = Pooling1x1Max3x1Nose<align>(src + stride);
            Avx::Store<align>(dst, _mm256_max_ps(src0, src1));
        }

        __m256i K32_PERMUTE_TAIL = SIMD_MM256_SETR_EPI32(1, 2, 3, 4, 5, 6, 7, 7);

        template <bool align> SIMD_INLINE __m256 Pooling1x1Max3x1Tail(const float * src)
        {
            __m256 src0 = Avx::Load<false>(src - 1);
            __m256 src1 = Avx::Load<align>(src);
            __m256 src2 = _mm256_permutevar8x32_ps(src1, K32_PERMUTE_TAIL);
            return _mm256_max_ps(_mm256_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Tail(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Tail<align>(src - stride);
            __m256 src1 = Pooling1x1Max3x1Tail<align>(src);
            __m256 src2 = Pooling1x1Max3x1Tail<align>(src + stride);
            Avx::Store<align>(dst, _mm256_max_ps(_mm256_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Tail(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Tail<align>(src);
            __m256 src1 = Pooling1x1Max3x1Tail<align>(src + stride);
            Avx::Store<align>(dst, _mm256_max_ps(src0, src1));
        }

        template <bool align> void NeuralPooling1x1Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            assert(width > F && height > 1);

            size_t alignedWidth = AlignHi(width, F) - F;
            height -= 1;

            Pooling1x1Max3x2Nose<align>(src, srcStride, dst);
            for (size_t col = F; col < alignedWidth; col += F)
                Pooling1x1Max3x2Body<align>(src + col, srcStride, dst + col);
            Pooling1x1Max3x2Tail<false>(src + width - F, srcStride, dst + width - F);

            for (size_t row = 1; row < height; ++row)
            {
                src += srcStride;
                dst += dstStride;
                Pooling1x1Max3x3Nose<align>(src, srcStride, dst);
                for (size_t col = F; col < alignedWidth; col += F)
                    Pooling1x1Max3x3Body<align>(src + col, srcStride, dst + col);
                Pooling1x1Max3x3Tail<false>(src + width - F, srcStride, dst + width - F);
            }

            dst += dstStride;
            Pooling1x1Max3x2Nose<align>(src, srcStride, dst);
            for (size_t col = F; col < alignedWidth; col += F)
                Pooling1x1Max3x2Body<align>(src + col, srcStride, dst + col);
            Pooling1x1Max3x2Tail<false>(src + width - F, srcStride, dst + width - F);
        }

        void NeuralPooling1x1Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling1x1Max3x3<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling1x1Max3x3<false>(src, srcStride, width, height, dst, dstStride);
        }

        SIMD_INLINE float Max2(const float * src)
        {
            return Simd::Max(src[0], src[1]);
        }

        SIMD_INLINE float Max2x2(const float * src, size_t stride)
        {
            return Simd::Max(Max2(src), Max2(src + stride));
        }

        SIMD_INLINE float Max2x3(const float * src, size_t stride)
        {
            return Simd::Max(Max2(src), Simd::Max(Max2(src + stride), Max2(src + 2 * stride)));
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max1x3(const float * src, size_t stride)
        {
            return _mm256_max_ps(_mm256_max_ps(Avx::Load<align>(src), Avx::Load<align>(src + stride)), Avx::Load<align>(src + 2 * stride));
        }

        SIMD_INLINE __m256 PermuteFor2x2(__m256 a)
        {
            return _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(a), 0xD8));
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max3x3(const float * src, size_t stride)
        {
            __m256 _01234567 = Pooling2x2Max1x3<align>(src, stride);
            __m256 _89abcdef = Pooling2x2Max1x3<align>(src + F, stride);
            __m256 _456789ab = _mm256_permute2f128_ps(_01234567, _89abcdef, 0x21);
            __m256 _12345678 = Alignr<1>(_01234567, _456789ab);
            __m256 _9abcdefg = Pooling2x2Max1x3<false>(src + F + 1, stride);
            __m256 _028a46ce = _mm256_shuffle_ps(_01234567, _89abcdef, 0x88);
            __m256 _139b57df = _mm256_shuffle_ps(_01234567, _89abcdef, 0xDD);
            __m256 _24ac68eg = _mm256_shuffle_ps(_12345678, _9abcdefg, 0xDD);
            return PermuteFor2x2(_mm256_max_ps(_mm256_max_ps(_028a46ce, _139b57df), _24ac68eg));
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max1x2(const float * src, size_t stride)
        {
            return _mm256_max_ps(Avx::Load<align>(src), Avx::Load<align>(src + stride));
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max3x2(const float * src, size_t stride)
        {
            __m256 _01234567 = Pooling2x2Max1x2<align>(src, stride);
            __m256 _89abcdef = Pooling2x2Max1x2<align>(src + F, stride);
            __m256 _456789ab = _mm256_permute2f128_ps(_01234567, _89abcdef, 0x21);
            __m256 _12345678 = Alignr<1>(_01234567, _456789ab);
            __m256 _9abcdefg = Pooling2x2Max1x2<false>(src + F + 1, stride);
            __m256 _028a46ce = _mm256_shuffle_ps(_01234567, _89abcdef, 0x88);
            __m256 _139b57df = _mm256_shuffle_ps(_01234567, _89abcdef, 0xDD);
            __m256 _24ac68eg = _mm256_shuffle_ps(_12345678, _9abcdefg, 0xDD);
            return PermuteFor2x2(_mm256_max_ps(_mm256_max_ps(_028a46ce, _139b57df), _24ac68eg));
        }

        template <bool align> void NeuralPooling2x2Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            size_t alignedWidth = AlignLo(width, DF);
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Avx::Store<align>(dst + (col >> 1), Pooling2x2Max3x3<align>(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Avx::Store<false>(dst + (col >> 1), Pooling2x2Max3x3<false>(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Max2x3(src + widthEven, srcStride);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Avx::Store<align>(dst + (col >> 1), Pooling2x2Max3x2<align>(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Avx::Store<false>(dst + (col >> 1), Pooling2x2Max3x2<false>(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Max2x2(src + widthEven, srcStride);
            }
        }

        void NeuralPooling2x2Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling2x2Max3x3<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling2x2Max3x3<false>(src, srcStride, width, height, dst, dstStride);
        }

        namespace Ncf
        {
            namespace Ver0
            {
                void PrepareB(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY,
                    size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, float * dst)
                {
                    const size_t K = kernelX*kernelY*srcDepth, N = dstHeight*dstWidth;
                    if (dilationX*dilationY*strideX*strideY != 1)
                    {
                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            size_t srcRow0 = dstRow*strideY - padY;
                            for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                            {
                                size_t srcCol0 = dstCol*strideX - padX;
                                for (size_t channel = 0; channel < srcDepth; ++channel)
                                {
                                    for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                    {
                                        size_t srcRow = srcRow0 + kernelRow*dilationY;
                                        if (srcRow < srcHeight)
                                        {
                                            const float * psrc = src + (channel*srcHeight + srcRow)*srcWidth;
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                            {
                                                size_t srcCol = srcCol0 + kernelCol*dilationX;
                                                if (srcCol < srcWidth)
                                                    *(dst++) = psrc[srcCol];
                                                else
                                                    *(dst++) = 0;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                                *(dst++) = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else if (kernelX*kernelY != 1)
                    {
                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            size_t srcRow0 = dstRow - padY;
                            for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                            {
                                size_t srcCol0 = dstCol - padX;
                                for (size_t channel = 0; channel < srcDepth; ++channel)
                                {
                                    for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                    {
                                        size_t srcRow = srcRow0 + kernelRow;
                                        if (srcRow < srcHeight)
                                        {
                                            const float * psrc = src + (channel*srcHeight + srcRow)*srcWidth;
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                            {
                                                size_t srcCol = srcCol0 + kernelCol;
                                                if (srcCol < srcWidth)
                                                    *(dst++) = psrc[srcCol];
                                                else
                                                    *(dst++) = 0;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                                *(dst++) = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < N; ++i)
                        {
                            for (size_t k = 0; k < K; ++k)
                                *(dst++) = src[k*N + i];
                        }
                    }
                }

                template <bool align> static SIMD_INLINE void Kernel1x4x8(const __m256 & a, size_t K, const float * b, __m256 * sums)
                {
                    sums[0] = _mm256_fmadd_ps(a, Avx::Load<align>(b + 0 * K), sums[0]);
                    sums[1] = _mm256_fmadd_ps(a, Avx::Load<align>(b + 1 * K), sums[1]);
                    sums[2] = _mm256_fmadd_ps(a, Avx::Load<align>(b + 2 * K), sums[2]);
                    sums[3] = _mm256_fmadd_ps(a, Avx::Load<align>(b + 3 * K), sums[3]);
                }

                template <bool align> static SIMD_INLINE void Kernel1x1x8(const __m256 & a, const float * b, __m256 & sum)
                {
                    sum = _mm256_fmadd_ps(a, Avx::Load<align>(b), sum);
                }

                SIMD_INLINE void Add4ExtractedSums(const __m256 * src, float * dst)
                {
                    __m256 sum256 = _mm256_hadd_ps(_mm256_hadd_ps(src[0], src[1]), _mm256_hadd_ps(src[2], src[3]));
                    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
                    _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), sum128));
                }

                template <bool align> static SIMD_INLINE void Kernel3x4x8(const __m256 * a, size_t K, const float * b, __m256 * sums)
                {
                    __m256 _b;
                    _b = Avx::Load<align>(b + 0 * K);
                    sums[0x0] = _mm256_fmadd_ps(a[0], _b, sums[0x0]);
                    sums[0x4] = _mm256_fmadd_ps(a[1], _b, sums[0x4]);
                    sums[0x8] = _mm256_fmadd_ps(a[2], _b, sums[0x8]);
                    _b = Avx::Load<align>(b + 1 * K);
                    sums[0x1] = _mm256_fmadd_ps(a[0], _b, sums[0x1]);
                    sums[0x5] = _mm256_fmadd_ps(a[1], _b, sums[0x5]);
                    sums[0x9] = _mm256_fmadd_ps(a[2], _b, sums[0x9]);
                    _b = Avx::Load<align>(b + 2 * K);
                    sums[0x2] = _mm256_fmadd_ps(a[0], _b, sums[0x2]);
                    sums[0x6] = _mm256_fmadd_ps(a[1], _b, sums[0x6]);
                    sums[0xA] = _mm256_fmadd_ps(a[2], _b, sums[0xA]);
                    _b = Avx::Load<align>(b + 3 * K);
                    sums[0x3] = _mm256_fmadd_ps(a[0], _b, sums[0x3]);
                    sums[0x7] = _mm256_fmadd_ps(a[1], _b, sums[0x7]);
                    sums[0xB] = _mm256_fmadd_ps(a[2], _b, sums[0xB]);
                }

                template <bool align> static SIMD_INLINE void Kernel3x1x8(const __m256 * a, const float * b, __m256 * sums)
                {
                    __m256 _b = Avx::Load<align>(b);
                    sums[0x0] = _mm256_fmadd_ps(a[0], _b, sums[0x0]);
                    sums[0x1] = _mm256_fmadd_ps(a[1], _b, sums[0x1]);
                    sums[0x2] = _mm256_fmadd_ps(a[2], _b, sums[0x2]);
                }

                template <bool align> void Execute(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    size_t M3 = M/3*3;
                    size_t N4 = Simd::AlignLo(N, 4);
                    size_t K8 = Simd::AlignLo(K, 8);
                    __m256 tailMask = RightNotZero32f(K - K8);
                    size_t i = 0;
                    for (; i < M3; i += 3)
                    {
                        const float * pa = a + i * K;
                        float * pc = c + i * N;
                        size_t j = 0;
                        for (; j < N4; j += 4)
                        {
                            const float * pb = b + j * K;
                            __m256 sums[12] = { 
                                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), 
                                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                            __m256 _a[3];
                            for (size_t k = 0; k < K8; k += 8)
                            {
                                _a[0] = Avx::Load<false>(pa + k + 0 * K);
                                _a[1] = Avx::Load<false>(pa + k + 1 * K);
                                _a[2] = Avx::Load<false>(pa + k + 2 * K);
                                Kernel3x4x8<align>(_a, K, pb + k, sums);
                            }
                            if (K8 < K)
                            {
                                size_t k = K - 8;
                                _a[0] = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k + 0 * K));
                                _a[1] = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k + 1 * K));
                                _a[2] = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k + 2 * K));
                                Kernel3x4x8<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0, pc + j + 0 * N);
                            Add4ExtractedSums(sums + 4, pc + j + 1 * N);
                            Add4ExtractedSums(sums + 8, pc + j + 2 * N);
                        }
                        for (; j < N; ++j)
                        {
                            const float * pb = b + j * K;
                            __m256 sums[3] = { _mm256_setzero_ps(), _mm256_setzero_ps() , _mm256_setzero_ps() };
                            __m256 _a[3];
                            for (size_t k = 0; k < K8; k += 8)
                            {
                                _a[0] = Avx::Load<false>(pa + k + 0 * K);
                                _a[1] = Avx::Load<false>(pa + k + 1 * K);
                                _a[2] = Avx::Load<false>(pa + k + 2 * K);
                                Kernel3x1x8<align>(_a, pb + k, sums);
                            }
                            if (K8 < K)
                            {
                                size_t k = K - 8;
                                _a[0] = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k + 0 * K));
                                _a[1] = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k + 1 * K));
                                _a[2] = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k + 2 * K));
                                Kernel3x1x8<false>(_a, pb + k, sums);
                            }
                            pc[j + 0 * N] += Avx::ExtractSum(sums[0]);
                            pc[j + 1 * N] += Avx::ExtractSum(sums[1]);
                            pc[j + 2 * N] += Avx::ExtractSum(sums[2]);
                        }
                    }
                    for (; i < M; ++i)
                    {
                        const float * pa = a + i*K;
                        float * pc = c + i*N;
                        size_t j = 0;
                        for (; j < N4; j += 4)
                        {
                            const float * pb = b + j*K;
                            __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                            for (size_t k = 0; k < K8; k += 8)
                            {
                                __m256 _a = Avx::Load<false>(pa + k);
                                Kernel1x4x8<align>(_a, K, pb + k, sums);
                            }
                            if (K8 < K)
                            {
                                size_t k = K - 8;
                                __m256 _a = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k));
                                Kernel1x4x8<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0, pc + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float * pb = b + j*K;
                            __m256 sum = _mm256_setzero_ps();
                            for (size_t k = 0; k < K8; k += 8)
                            {
                                __m256 _a = Avx::Load<false>(pa + k);
                                Kernel1x1x8<align>(_a, pb + k, sum);
                            }
                            if (K8 < K)
                            {
                                size_t k = K - 8;
                                __m256 _a = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k));
                                Kernel1x1x8<false>(_a, pb + k, sum);
                            }
                            pc[j] += Avx::ExtractSum(sum);
                        }
                    }
                }

                void Execute(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    if (Aligned(K, F))
                        Execute<true>(M, N, K, a, b, c);
                    else
                        Execute<false>(M, N, K, a, b, c);
                }
            }

            namespace Ver1
            {
                void PrepareA(const float * src, size_t M, size_t K, size_t cell, float * dst)
                {
                    size_t K4 = AlignLo(K, 4), K8 = AlignLo(K, 8);
                    for (size_t i = 0; i < M; i += cell)
                    {
                        size_t n = Simd::Min(cell, M - i), k = 0;
                        if (cell == 4 && n == 4)
                        {
                            for (; k < K8; k += 8)
                            {
                                const float * ps = src + k;
                                __m256 s0 = Avx::Load<false>(ps + 0 * K);
                                __m256 s1 = Avx::Load<false>(ps + 1 * K);
                                __m256 s2 = Avx::Load<false>(ps + 2 * K);
                                __m256 s3 = Avx::Load<false>(ps + 3 * K);
                                __m256 s00 = _mm256_unpacklo_ps(s0, s2);
                                __m256 s01 = _mm256_unpacklo_ps(s1, s3);
                                __m256 s10 = _mm256_unpackhi_ps(s0, s2);
                                __m256 s11 = _mm256_unpackhi_ps(s1, s3);
                                __m256 d0 = _mm256_unpacklo_ps(s00, s01);
                                __m256 d1 = _mm256_unpackhi_ps(s00, s01);
                                __m256 d2 = _mm256_unpacklo_ps(s10, s11);
                                __m256 d3 = _mm256_unpackhi_ps(s10, s11);
                                Avx::Store<false>(dst + 0, _mm256_permute2f128_ps(d0, d1, 0x20));
                                Avx::Store<false>(dst + 8, _mm256_permute2f128_ps(d2, d3, 0x20));
                                Avx::Store<false>(dst + 16, _mm256_permute2f128_ps(d0, d1, 0x31));
                                Avx::Store<false>(dst + 24, _mm256_permute2f128_ps(d2, d3, 0x31));
                                dst += 32;
                            }
                            for (; k < K4; k += 4)
                            {
                                const float * ps = src + k;
                                __m128 s0 = Sse2::Load<false>(ps + 0 * K);
                                __m128 s1 = Sse2::Load<false>(ps + 1 * K);
                                __m128 s2 = Sse2::Load<false>(ps + 2 * K);
                                __m128 s3 = Sse2::Load<false>(ps + 3 * K);
                                __m128 s00 = _mm_unpacklo_ps(s0, s2);
                                __m128 s01 = _mm_unpacklo_ps(s1, s3);
                                __m128 s10 = _mm_unpackhi_ps(s0, s2);
                                __m128 s11 = _mm_unpackhi_ps(s1, s3);
                                Sse2::Store<false>(dst + 0, _mm_unpacklo_ps(s00, s01));
                                Sse2::Store<false>(dst + 4, _mm_unpackhi_ps(s00, s01));
                                Sse2::Store<false>(dst + 8, _mm_unpacklo_ps(s10, s11));
                                Sse2::Store<false>(dst + 12, _mm_unpackhi_ps(s10, s11));
                                dst += 16;
                            }
                        }
                        for (; k < K; ++k)
                        {
                            for (size_t c = 0; c < n; ++c)
                                *(dst++) = src[c*K + k];
                        }
                        src += cell*K;
                    }
                }

                void PrepareB(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY,
                    size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t cell, float * tmp, float * dst)
                {
                    const size_t K = kernelX*kernelY*srcDepth, N = dstHeight*dstWidth;
                    if (kernelX*kernelY != 1)
                    {
                        float * dst = tmp;
                        size_t channelSize = srcHeight * srcWidth;
                        if (dilationX*dilationY*strideX*strideY != 1)
                        {
                            for (size_t channel = 0, k = 0; channel < srcDepth; ++channel, src += channelSize)
                            {
                                for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                {
                                    for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol, ++k)
                                    {
                                        size_t srcRow = kernelRow*dilationY - padY;
                                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                                        {
                                            if (srcRow < srcHeight)
                                            {
                                                size_t srcCol = kernelCol*dilationX - padX;
                                                for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = src[srcRow*srcWidth + srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                    srcCol += strideX;
                                                }
                                            }
                                            else
                                            {
                                                for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                                    *(dst++) = 0;
                                            }
                                            srcRow += strideY;
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            const size_t bodySize = dstWidth - padX * 2;
                            for (size_t channel = 0, k = 0; channel < srcDepth; ++channel, src += channelSize)
                            {
                                for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                {
                                    for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol, ++k)
                                    {
                                        size_t srcRow = kernelRow - padY;
                                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow, ++srcRow)
                                        {
                                            if (srcRow < srcHeight)
                                            {
                                                size_t srcCol = kernelCol - padX, dstCol = 0;
                                                const float * psrc = src + srcRow*srcWidth;
                                                for (; dstCol < padX; ++dstCol, ++srcCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = psrc[srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                }
                                                memcpy(dst, psrc + srcCol, bodySize * 4);
                                                dst += bodySize;
                                                dstCol += bodySize;
                                                srcCol += bodySize;
                                                for (; dstCol < dstWidth; ++dstCol, ++srcCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = psrc[srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                }
                                            }
                                            else
                                            {
                                                memset(dst, 0, dstWidth * 4);
                                                dst += dstWidth;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        src = tmp;
                    }
                    if (cell == 24)
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            if (n == cell)
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float * psrc = src + k * N;
                                    Avx::Store<false>(dst + 0x00, Avx::Load<false>(psrc + 0x00));
                                    Avx::Store<false>(dst + 0x08, Avx::Load<false>(psrc + 0x08));
                                    Avx::Store<false>(dst + 0x10, Avx::Load<false>(psrc + 0x10));
                                    dst += 24;
                                }
                            }
                            else
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float * psrc = src + k * N;
                                    size_t c = 0;
                                    for (; c < n; ++c)
                                        *(dst++) = *(psrc++);
                                    for (; c < cell; ++c)
                                        *(dst++) = 0;
                                }
                            }
                            src += cell;
                        }
                    }
                    else if (cell == 16)
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            if (n == cell)
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float * psrc = src + k*N;
                                    Avx::Store<false>(dst + 0, Avx::Load<false>(psrc + 0));
                                    Avx::Store<false>(dst + 8, Avx::Load<false>(psrc + 8));
                                    dst += 16;
                                }
                            }
                            else
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float * psrc = src + k*N;
                                    size_t c = 0;
                                    for (; c < n; ++c)
                                        *(dst++) = *(psrc++);
                                    for (; c < cell; ++c)
                                        *(dst++) = 0;
                                }
                            }
                            src += cell;
                        }
                    }
                    else
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            for (size_t k = 0; k < K; ++k)
                            {
                                const float * psrc = src + k*N;
                                size_t c = 0;
                                for (; c < n; ++c)
                                    *(dst++) = *(psrc++);
                                for (; c < cell; ++c)
                                    *(dst++) = 0;
                            }
                            src += cell;
                        }
                    }
                }

                SIMD_INLINE void AddSum(__m256 sum, float * dst)
                {
                    Avx::Store<false>(dst, _mm256_add_ps(Load<false>(dst), sum));
                }

                SIMD_INLINE void AddSums8(const __m256 * sums, size_t size, const float * mask, float * dst, size_t stride)
                {
                    if (mask)
                    {
                        __m256 _mask = _mm256_loadu_ps(mask);
                        for (size_t i = 0; i < size; ++i, dst += stride)
                            AddSum(_mm256_and_ps(_mask, sums[i]), dst);
                    }
                    else
                    {
                        for (size_t i = 0; i < size; ++i, dst += stride)
                            AddSum(sums[i], dst);
                    }
                }

                template <bool align> SIMD_INLINE void KernelMx8(size_t N, size_t K, const float * a, const float * b, float * c, const float * mask, size_t m)
                {
                    __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m256 b0 = Load<align>(b);
                        for (size_t s = 0; s < m; ++s)
                            sums[s] = _mm256_fmadd_ps(_mm256_set1_ps(a[s]), b0, sums[s]);
                        b += 8;
                        a += m;
                    }
                    AddSums8(sums, m, mask, c, N);
                }

                template <bool align> SIMD_INLINE void Kernel4x8(size_t N, size_t K, const float * a, const float * b, float * c, const float * mask)
                {
                    __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m256 b0 = Load<align>(b);
                        sums[0] = _mm256_fmadd_ps(_mm256_set1_ps(a[0]), b0, sums[0]);
                        sums[1] = _mm256_fmadd_ps(_mm256_set1_ps(a[1]), b0, sums[1]);
                        sums[2] = _mm256_fmadd_ps(_mm256_set1_ps(a[2]), b0, sums[2]);
                        sums[3] = _mm256_fmadd_ps(_mm256_set1_ps(a[3]), b0, sums[3]);
                        b += 8;
                        a += 4;
                    }
                    AddSums8(sums, 4, mask, c, N);
                }

                template <bool align> void Execute4x8(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t N8 = Simd::AlignLo(N, 8);
                    const int32_t mask[16] = { -1, -1, -1, -1,  -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
                    const float * tail = (float*)mask + 8 - N + N8;
                    size_t i = 0;
                    for (; i < M4; i += 4)
                    {
                        size_t j = 0;
                        for (; j < N8; j += 8)
                            Kernel4x8<align>(N, K, a + i*K, b + j*K, c + i*N + j, NULL);
                        if (N8 < N)
                            Kernel4x8<align>(N, K, a + i*K, b + j*K, c + i*N + j, tail);
                    }
                    if (M4 < M)
                    {
                        size_t j = 0;
                        for (; j < N8; j += 8)
                            KernelMx8<align>(N, K, a + i*K, b + j*K, c + i*N + j, NULL, M - M4);
                        if (N8 < N)
                            KernelMx8<align>(N, K, a + i*K, b + j*K, c + i*N + j, tail, M - M4);
                    }
                }

                SIMD_INLINE void AddSums16(const __m256 * sums, size_t size, const float * mask, float * dst, size_t stride)
                {
                    if (mask)
                    {
                        __m256 mask0 = _mm256_loadu_ps(mask + 0);
                        __m256 mask1 = _mm256_loadu_ps(mask + 8);
                        for (size_t i = 0; i < size; ++i, dst += stride)
                        {
                            AddSum(_mm256_and_ps(mask0, sums[i + 0]), dst + 0);
                            AddSum(_mm256_and_ps(mask1, sums[i + 4]), dst + 8);
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < size; ++i, dst += stride)
                        {
                            AddSum(sums[i + 0], dst + 0);
                            AddSum(sums[i + 4], dst + 8);
                        }
                    }
                }

                template <bool align> SIMD_INLINE void KernelMx16(size_t N, size_t K, const float * a, const float * b, float * c, const float * mask, size_t m)
                {
                    __m256 sums[8] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m256 b0 = Load<align>(b + 0);
                        __m256 b1 = Load<align>(b + 8);
                        for (size_t s = 0; s < m; ++s)
                        {
                            __m256 a0 = _mm256_set1_ps(a[s]);
                            sums[s + 0] = _mm256_fmadd_ps(b0, a0, sums[s + 0]);
                            sums[s + 4] = _mm256_fmadd_ps(b1, a0, sums[s + 4]);
                        }
                        b += 16;
                        a += m;
                    }
                    AddSums16(sums, m, mask, c, N);
                }

                template <bool align> SIMD_INLINE void Kernel4x16(size_t N, size_t K, const float * a, const float * b, float * c, const float * mask)
                {
                    __m256 sums[8] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m256 b0 = Load<align>(b + 0);
                        __m256 b1 = Load<align>(b + 8);
                        __m256 a0 = _mm256_set1_ps(a[0]);
                        sums[0] = _mm256_fmadd_ps(b0, a0, sums[0]);
                        sums[4] = _mm256_fmadd_ps(b1, a0, sums[4]);
                        __m256 a1 = _mm256_set1_ps(a[1]);
                        sums[1] = _mm256_fmadd_ps(b0, a1, sums[1]);
                        sums[5] = _mm256_fmadd_ps(b1, a1, sums[5]);
                        __m256 a2 = _mm256_set1_ps(a[2]);
                        sums[2] = _mm256_fmadd_ps(b0, a2, sums[2]);
                        sums[6] = _mm256_fmadd_ps(b1, a2, sums[6]);
                        __m256 a3 = _mm256_set1_ps(a[3]);
                        sums[3] = _mm256_fmadd_ps(b0, a3, sums[3]);
                        sums[7] = _mm256_fmadd_ps(b1, a3, sums[7]);
                        b += 16;
                        a += 4;
                    }
                    AddSums16(sums, 4, mask, c, N);
                }

                template <bool align> void Execute4x16(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t N16 = Simd::AlignLo(N, 16);
                    const int32_t mask[32] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                    const float * tail = (float*)mask + 16 - N + N16;
                    size_t i = 0;
                    for (; i < M4; i += 4)
                    {
                        size_t j = 0;
                        for (; j < N16; j += 16)
                            Kernel4x16<align>(N, K, a + i*K, b + j*K, c + i*N + j, NULL);
                        if (N16 < N)
                            Kernel4x16<align>(N, K, a + i*K, b + j*K, c + i*N + j, tail);
                    }
                    if (M4 < M)
                    {
                        size_t j = 0;
                        for (; j < N16; j += 16)
                            KernelMx16<align>(N, K, a + i*K, b + j*K, c + i*N + j, NULL, M - M4);
                        if (N16 < N)
                            KernelMx16<align>(N, K, a + i*K, b + j*K, c + i*N + j, tail, M - M4);
                    }
                }

                SIMD_INLINE void AddSums24(const __m256 * sums, size_t size, const float * mask, float * dst, size_t stride)
                {
                    if (mask)
                    {
                        __m256 mask0 = _mm256_loadu_ps(mask + 0 * F);
                        __m256 mask1 = _mm256_loadu_ps(mask + 1 * F);
                        __m256 mask2 = _mm256_loadu_ps(mask + 2 * F);
                        for (size_t i = 0; i < size; ++i, dst += stride)
                        {
                            AddSum(_mm256_and_ps(mask0, sums[i + 0]), dst + 0 * F);
                            AddSum(_mm256_and_ps(mask1, sums[i + 4]), dst + 1 * F);
                            AddSum(_mm256_and_ps(mask2, sums[i + 8]), dst + 2 * F);
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < size; ++i, dst += stride)
                        {
                            AddSum(sums[i + 0], dst + 0 * F);
                            AddSum(sums[i + 4], dst + 1 * F);
                            AddSum(sums[i + 8], dst + 2 * F);
                        }
                    }
                }

                template <bool align> SIMD_INLINE void KernelMx24(size_t N, size_t K, const float * a, const float * b, float * c, const float * mask, size_t m)
                {
                    __m256 sums[12] = { 
                        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                        _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                    __m256 _b[3];
                    for (size_t k = 0; k < K; ++k)
                    {   
                        _b[0] = Load<align>(b + 0 * F);
                        _b[1] = Load<align>(b + 1 * F);
                        _b[2] = Load<align>(b + 2 * F);
                        for (size_t s = 0; s < m; ++s)
                        {
                            __m256 _a = _mm256_set1_ps(a[s]);
                            sums[s + 0] = _mm256_fmadd_ps(_b[0], _a, sums[s + 0]);
                            sums[s + 4] = _mm256_fmadd_ps(_b[1], _a, sums[s + 4]);
                            sums[s + 8] = _mm256_fmadd_ps(_b[2], _a, sums[s + 8]);
                        }
                        b += 24;
                        a += m;
                    }
                    AddSums24(sums, m, mask, c, N);
                }

                void Kernel4x24(size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    __m256 _a, b0, b1, b2, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;

                    c00 = _mm256_setzero_ps();
                    c01 = _mm256_setzero_ps();
                    c02 = _mm256_setzero_ps();
                    c10 = _mm256_setzero_ps();
                    c11 = _mm256_setzero_ps();
                    c12 = _mm256_setzero_ps();
                    c20 = _mm256_setzero_ps();
                    c21 = _mm256_setzero_ps();
                    c22 = _mm256_setzero_ps();
                    c30 = _mm256_setzero_ps();
                    c31 = _mm256_setzero_ps();
                    c32 = _mm256_setzero_ps();

                    for (size_t k = 0; k < K; ++k)
                    {
                        b0 = _mm256_loadu_ps(b + 0 * F);
                        b1 = _mm256_loadu_ps(b + 1 * F);
                        b2 = _mm256_loadu_ps(b + 2 * F);
                        _a = _mm256_set1_ps(a[0]);
                        c00 = _mm256_fmadd_ps(b0, _a, c00);
                        c01 = _mm256_fmadd_ps(b1, _a, c01);
                        c02 = _mm256_fmadd_ps(b2, _a, c02);
                        _a = _mm256_set1_ps(a[1]);
                        c10 = _mm256_fmadd_ps(b0, _a, c10);
                        c11 = _mm256_fmadd_ps(b1, _a, c11);
                        c12 = _mm256_fmadd_ps(b2, _a, c12);
                        _a = _mm256_set1_ps(a[2]);
                        c20 = _mm256_fmadd_ps(b0, _a, c20);
                        c21 = _mm256_fmadd_ps(b1, _a, c21);
                        c22 = _mm256_fmadd_ps(b2, _a, c22);
                        _a = _mm256_set1_ps(a[3]);
                        c30 = _mm256_fmadd_ps(b0, _a, c30);
                        c31 = _mm256_fmadd_ps(b1, _a, c31);
                        c32 = _mm256_fmadd_ps(b2, _a, c32);
                        b += 24;
                        a += 4;
                    }

                    AddSum(c00, c + 0 * F);
                    AddSum(c01, c + 1 * F);
                    AddSum(c02, c + 2 * F);
                    c += N;
                    AddSum(c10, c + 0 * F);
                    AddSum(c11, c + 1 * F);
                    AddSum(c12, c + 2 * F);
                    c += N;
                    AddSum(c20, c + 0 * F);
                    AddSum(c21, c + 1 * F);
                    AddSum(c22, c + 2 * F);
                    c += N;
                    AddSum(c30, c + 0 * F);
                    AddSum(c31, c + 1 * F);
                    AddSum(c32, c + 2 * F);
                }

                template <bool align> void Execute4x24(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
                {
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t N24 = N/24*24;
                    const int32_t mask[48] = { 
                        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                    const float * tail = (float*)mask + 24 - N + N24;
                    if (M > N)
                    {
                        size_t i = 0;
                        for (; i < M4; i += 4)
                        {
                            size_t j = 0;
                            for (; j < N24; j += 24)
                                Kernel4x24(N, K, a + i * K, b + j * K, c + i * N + j);
                            if (N24 < N)
                                KernelMx24<align>(N, K, a + i * K, b + j * K, c + i * N + j, tail, 4);
                        }
                        if (M4 < M)
                        {
                            size_t j = 0;
                            for (; j < N24; j += 24)
                                KernelMx24<align>(N, K, a + i * K, b + j * K, c + i * N + j, NULL, M - M4);
                            if (N24 < N)
                                KernelMx24<align>(N, K, a + i * K, b + j * K, c + i * N + j, tail, M - M4);
                        }
                    }
                    else
                    {
                        size_t j = 0;
                        for (; j < N24; j += 24)
                        {
                            size_t i = 0;
                            for (; i < M4; i += 4)
                                Kernel4x24(N, K, a + i * K, b + j * K, c + i * N + j);
                            if (M4 < M)
                                KernelMx24<align>(N, K, a + i * K, b + j * K, c + i * N + j, NULL, M - M4);
                        }
                        if (N24 < N)
                        {
                            size_t i = 0;
                            for (; i < M4; i += 4)
                                KernelMx24<align>(N, K, a + i * K, b + j * K, c + i * N + j, tail, 4);
                            if (M4 < M)
                                KernelMx24<align>(N, K, a + i * K, b + j * K, c + i * N + j, tail, M - M4);
                        }
                    }
                }

                void Execute(size_t M, size_t N, size_t K, const float * a, const float * b, float * c, size_t cellA, size_t cellB)
                {
                    if (cellA == 4)
                    {
                        if (cellB == 8)
                            Execute4x8<false>(M, N, K, a, b, c);
                        if (cellB == 16)
                            Execute4x16<false>(M, N, K, a, b, c);
                        if (cellB == 24)
                            Execute4x24<false>(M, N, K, a, b, c);
                    }
                }
            }

            namespace Ver2
            {
                void PrepareB(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t padX, size_t padY, float * dst, size_t dstWidth, size_t dstHeight)
                {
                    for (size_t channel = 0; channel < srcDepth; ++channel)
                    {
                        const float * s = src;
                        float * d = dst;
                        memset(d, 0, padY*dstWidth * 4);
                        d += padY*dstWidth;
                        for (size_t row = padY; row < dstHeight - padY; ++row)
                        {
                            memset(d, 0, padX * 4);
                            memcpy(d + padX, s, srcWidth * 4);
                            memset(d + padX + srcWidth, 0, padX * 4);
                            d += dstWidth;
                            s += srcWidth;
                        }
                        memset(d, 0, padY*dstWidth * 4);
                        src += srcWidth*srcHeight;
                        dst += dstWidth*dstHeight;
                    }
                }

                template <bool align, size_t kernelX, size_t kernelY> void AddConvolution8x8(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float * weight, float * dst, size_t dstDepth)
                {
                    for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                    {
                        __m256 _dst[8];
                        float * pdst = dst;
                        for (size_t row = 0; row < 8; ++row, pdst += 8)
                            _dst[row] = Avx::Load<align>(pdst);
                        if (kernelY < 4)
                        {
                            __m256 _weight[kernelX*kernelY];
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float * psrc = src + srcWidth*srcHeight*srcChannel;
                                LoadWeightsForward<kernelX*kernelY>(weight, _weight);
                                for (size_t row = 0; row < 8; ++row)
                                {
                                    _dst[row] = _mm256_add_ps(_dst[row], Convolution<kernelX, kernelY>::template Forward<align>(psrc, srcWidth, _weight));
                                    psrc += srcWidth;
                                }
                                weight += kernelX*kernelY;
                            }
                        }
                        else
                        {
                            __m256 _weight[kernelX];
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float * psrc = src + srcWidth*srcHeight*srcChannel;
                                for (size_t dy = 0; dy < kernelY; dy++)
                                {
                                    const float * ps = psrc + dy*srcWidth;
                                    LoadWeightsForward<kernelX>(weight, _weight);
                                    for (size_t row = 0; row < 8; ++row)
                                    {
                                        _dst[row] = _mm256_add_ps(_dst[row], Convolution<kernelX, kernelY>::template RowConvolution<align>(ps, _weight));
                                        ps += srcWidth;
                                    }
                                    weight += kernelX;
                                }
                            }
                        }
                        for (size_t row = 0; row < 8; ++row, dst += 8)
                            Avx::Store<align>(dst, _dst[row]);
                    }
                }

                template <bool align, size_t kernelX, size_t kernelY> void AddConvolution(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float * weight, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    if (dstWidth == 8 && dstHeight == 8)
                    {
                        AddConvolution8x8<align, kernelX, kernelY>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstDepth);
                        return;
                    }
                    size_t alignedWidth = AlignLo(dstWidth, F);
                    __m256 tailMask = RightNotZero32f(dstWidth - alignedWidth);
                    __m256 _weight[kernelX*kernelY];
                    for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                    {
                        for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                        {
                            const float * psrc = src + srcWidth*srcHeight*srcChannel;
                            const float * pweight = weight + (dstChannel*srcDepth + srcChannel)*kernelX*kernelY;
                            float * pdst = dst + dstWidth*dstHeight*dstChannel;
                            LoadWeightsForward<kernelX*kernelY>(pweight, _weight);
                            for (size_t row = 0; row < dstHeight; ++row)
                            {
                                size_t col = 0;
                                for (; col < alignedWidth; col += F)
                                {
                                    __m256 _dst = Load<align>(pdst + col);
                                    _dst = _mm256_add_ps(_dst, Convolution<kernelX, kernelY>::template Forward<align>(psrc + col, srcWidth, _weight));
                                    Avx::Store<align>(pdst + col, _dst);
                                }
                                if (dstWidth - alignedWidth)
                                {
                                    size_t col = dstWidth - F;
                                    __m256 _dst = Load<false>(pdst + col);
                                    _dst = _mm256_add_ps(_dst, _mm256_and_ps(tailMask, Convolution<kernelX, kernelY>::template Forward<false>(psrc + col, srcWidth, _weight)));
                                    Avx::Store<false>(pdst + col, _dst);
                                }
                                psrc += srcWidth;
                                pdst += dstWidth;
                            }
                        }
                    }
                }

                void AddConvolution1x1x16(const float * src, size_t srcDepth, const float * weight, float * dst, size_t dstDepth)
                {
                    size_t dstDepth4 = dstDepth/4*4;
                    size_t dstChannel = 0;
                    for (; dstChannel < dstDepth4; dstChannel += 4)
                    {
                        __m256 dst00 = _mm256_loadu_ps(dst + 0 * F);
                        __m256 dst01 = _mm256_loadu_ps(dst + 1 * F);
                        __m256 dst10 = _mm256_loadu_ps(dst + 2 * F);
                        __m256 dst11 = _mm256_loadu_ps(dst + 3 * F);
                        __m256 dst20 = _mm256_loadu_ps(dst + 4 * F);
                        __m256 dst21 = _mm256_loadu_ps(dst + 5 * F);
                        __m256 dst30 = _mm256_loadu_ps(dst + 6 * F);
                        __m256 dst31 = _mm256_loadu_ps(dst + 7 * F);
                        const float * psrc = src;
                        const float * pw0 = weight;
                        const float * pw1 = pw0 + srcDepth;
                        const float * pw2 = pw1 + srcDepth;
                        const float * pw3 = pw2 + srcDepth;
                        for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                        {
                            __m256 _weight;
                            __m256 src0 = _mm256_loadu_ps(psrc + 0 * F);
                            __m256 src1 = _mm256_loadu_ps(psrc + 1 * F);
                            _weight = _mm256_set1_ps(pw0[srcChannel]);
                            dst00 = _mm256_fmadd_ps(_weight, src0, dst00);
                            dst01 = _mm256_fmadd_ps(_weight, src1, dst01);
                            _weight = _mm256_set1_ps(pw1[srcChannel]);
                            dst10 = _mm256_fmadd_ps(_weight, src0, dst10);
                            dst11 = _mm256_fmadd_ps(_weight, src1, dst11);
                            _weight = _mm256_set1_ps(pw2[srcChannel]);
                            dst20 = _mm256_fmadd_ps(_weight, src0, dst20);
                            dst21 = _mm256_fmadd_ps(_weight, src1, dst21);
                            _weight = _mm256_set1_ps(pw3[srcChannel]);
                            dst30 = _mm256_fmadd_ps(_weight, src0, dst30);
                            dst31 = _mm256_fmadd_ps(_weight, src1, dst31);
                            psrc += 16;
                        }
                        _mm256_storeu_ps(dst + 0 * F, dst00);
                        _mm256_storeu_ps(dst + 1 * F, dst01);
                        _mm256_storeu_ps(dst + 2 * F, dst10);
                        _mm256_storeu_ps(dst + 3 * F, dst11);
                        _mm256_storeu_ps(dst + 4 * F, dst20);
                        _mm256_storeu_ps(dst + 5 * F, dst21);
                        _mm256_storeu_ps(dst + 6 * F, dst30);
                        _mm256_storeu_ps(dst + 7 * F, dst31);
                        dst += 16*4;
                        weight += srcDepth * 4;
                    }
                    for (; dstChannel < dstDepth; ++dstChannel)
                    {
                        __m256 dst0 = _mm256_loadu_ps(dst + 0 * F);
                        __m256 dst1 = _mm256_loadu_ps(dst + 1 * F);
                        const float * psrc = src;
                        for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                        {
                            __m256 weight0 = _mm256_set1_ps(*weight++);
                            dst0 = _mm256_fmadd_ps(weight0, _mm256_loadu_ps(psrc + 0 * F), dst0);
                            dst1 = _mm256_fmadd_ps(weight0, _mm256_loadu_ps(psrc + 1 * F), dst1);
                            psrc += 16;
                        }
                        _mm256_storeu_ps(dst + 0 * F, dst0);
                        _mm256_storeu_ps(dst + 1 * F, dst1);
                        dst += 16;
                    }
                }

                void Execute(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float * weight, size_t kernelX, size_t kernelY, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    assert(kernelX == kernelY);
                    if (kernelX == 1 && dstWidth*dstHeight == 16)
                        AddConvolution1x1x16(src, srcDepth, weight, dst, dstDepth);
                    else if (kernelX == 2)
                        AddConvolution<false, 2, 2>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 3)
                        AddConvolution<false, 3, 3>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 4)
                        AddConvolution<false, 4, 4>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 5)
                        AddConvolution<false, 5, 5>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else
                        assert(0);
                }

                bool Preferable(size_t srcDepth, size_t kernelX, size_t kernelY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    if (kernelX == kernelY && strideX*strideY*dilationX*dilationY == 1 && dstWidth >= F)
                    {
                        if (kernelX >= 2 && kernelX <= 5 && dstWidth*dstHeight*kernelX*kernelY >= 8 * 8 * 3 * 3)
                            return true;
                        if (kernelX == 1 && (dstWidth*dstHeight == 16))// || dstWidth * dstHeight == 64))
                            return true;
                    }
                    return false;
                }
            }

            struct Opt
            {
                enum Alg
                {
                    None,
                    Ver0,
                    Ver1,
                    Ver2,
                } alg;

                size_t sizeA;
                size_t sizeB;
                size_t sizeT;

                size_t cellA;
                size_t cellB;

                size_t M, N, K;
                size_t strideB;
                size_t paddedW;
                size_t paddedH;

                Opt(size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    alg = None;
                    sizeA = 0;
                    sizeB = 0;
                    sizeT = 0;
                    cellA = 1;
                    cellB = 1;

                    M = dstDepth;
                    N = dstHeight*dstWidth;
                    K = kernelX*kernelY*srcDepth;

                    if (dstWidth*dstHeight / kernelX <= 1000)
                        alg = Ver0;
                    else
                        alg = Ver1;
                    if (Ver2::Preferable(srcDepth, kernelX, kernelY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth))
                        alg = Ver2;

                    switch (alg)
                    {
                    case Ver0:
                        sizeB = N*K;
                        break;
                    case Ver1:
                        cellA = 4;
                        cellB = 24;
                        sizeA = M*K;
                        strideB = (N + cellB - 1)/cellB*cellB;
                        sizeB = strideB*K;
                        if (kernelX*kernelY > 1)
                            sizeT = sizeB;
                        break;
                    case Ver2:
                        if (padX > 0 || padY > 0)
                        {
                            paddedW = Simd::AlignHi(srcWidth + 2 * padX, F);
                            paddedH = srcHeight + 2 * padY;
                            sizeB = paddedW*paddedH*srcDepth;
                        }
                        else
                        {
                            paddedW = srcWidth;
                            paddedH = srcHeight;
                        }
                        break;
                    default:
                        assert(0);
                        break;
                    }
                }
            };

            struct Data
            {
                float * a;
                float * b;
                float * t;

                Data(size_t sizeA, size_t sizeB, size_t sizeT, void * externalData, size_t * externalSize)
                    : a(0)
                    , b(0)
                    , _data(0)
                {
                    sizeA = AlignHi(sizeA, F);
                    sizeB = AlignHi(sizeB, F);
                    sizeT = AlignHi(sizeT, F);
                    size_t size = (sizeA + sizeB + sizeT) * sizeof(float);
                    if (size == 0)
                        return;
                    if (externalData != AlignHi(externalData, SIMD_ALIGN))
                        size += SIMD_ALIGN;
                    float * data = NULL;
                    if (externalData == NULL || externalSize == NULL || *externalSize < size)
                    {
                        _data = Simd::Allocate(size);
                        if (externalSize)
                            *externalSize = size;
                        data = (float*)_data;
                    }
                    else
                        data = (float*)AlignHi(externalData, SIMD_ALIGN);
                    if (sizeA)
                        a = data;
                    if (sizeB)
                        b = data + sizeA;
                    if (sizeT)
                        t = data + sizeA + sizeB;
                }

                ~Data()
                {
                    if (_data)
                        Simd::Free(_data);
                }

            private:
                void * _data;
            };
        }

        void NeuralConvolutionForward(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
            const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY,
            void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add)
        {
            using namespace Ncf;

            assert(dstWidth == (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1);
            assert(dstHeight == (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1);

            if (!add)
                memset(dst, 0, dstWidth*dstHeight*dstDepth * sizeof(float));

            Opt opt(srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth);

            Data data(opt.sizeA, opt.sizeB, opt.sizeT, buffer, size);

            if (opt.sizeA)
            {
                switch (opt.alg)
                {
                case Opt::Ver1: Ver1::PrepareA(weight, opt.M, opt.K, opt.cellA, data.a);
                default:
                    break;
                }
            }
            else
                data.a = (float*)weight;

            if (opt.sizeB)
            {
                switch (opt.alg)
                {
                case Opt::Ver0: Ver0::PrepareB(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, data.b); break;
                case Opt::Ver1: Ver1::PrepareB(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, opt.cellB, data.t, data.b); break;
                case Opt::Ver2: Ver2::PrepareB(src, srcWidth, srcHeight, srcDepth, padX, padY, data.b, opt.paddedW, opt.paddedH); break;
                default: break;
                }
            }
            else
                data.b = (float*)src;

            switch (opt.alg)
            {
            case Opt::Ver0: Ver0::Execute(opt.M, opt.N, opt.K, data.a, data.b, dst); break;
            case Opt::Ver1: Ver1::Execute(opt.M, opt.N, opt.K, data.a, data.b, dst, opt.cellA, opt.cellB); break;
            case Opt::Ver2: Ver2::Execute(data.b, opt.paddedW, opt.paddedH, srcDepth, weight, kernelX, kernelY, dst, dstWidth, dstHeight, dstDepth); break;
            default: break;
            }
        }
    }
#endif// SIMD_AVX2_ENABLE
}
