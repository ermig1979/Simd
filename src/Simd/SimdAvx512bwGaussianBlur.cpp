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
#include "Simd/SimdLoadBlock.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdGaussianBlur.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512 LoadAs32f(const uint8_t* src, __mmask16 tail = -1)
        {
            return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, src)));
        }

        SIMD_INLINE void BlurColsAny(const uint8_t* src, size_t size, __mmask16 tail, size_t channels, const float* weight, size_t kernel, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm512_fmadd_ps(_mm512_set1_ps(weight[k]), LoadAs32f(src + i + k * channels), sum);
                _mm512_storeu_ps(dst + i, sum);
            }
            if( i < size)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm512_fmadd_ps(_mm512_set1_ps(weight[k]), LoadAs32f(src + i + k * channels, tail), sum);
                _mm512_mask_storeu_ps(dst + i, tail, sum);
            }
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const __m512& f0, __mmask16 tail = -1)
        {
#if 0
            __m512i i0 = _mm512_cvtps_epi32(f0);
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(i0, K_ZERO), K_ZERO));
            _mm_mask_storeu_epi8(dst, tail, _mm512_extracti32x4_epi32(u8, 0));
#else
            _mm_mask_storeu_epi8(dst, tail, _mm512_cvtusepi32_epi8(_mm512_cvtps_epi32(f0)));
#endif
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const __m512& f0, const __m512& f1, const __m512& f2, const __m512& f3)
        {
            __m512i i0 = _mm512_cvtps_epi32(f0);
            __m512i i1 = _mm512_cvtps_epi32(f1);
            __m512i i2 = _mm512_cvtps_epi32(f2);
            __m512i i3 = _mm512_cvtps_epi32(f3);
            _mm512_storeu_si512(dst, _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(i0, i1), _mm512_packs_epi32(i2, i3))));
        }

        SIMD_INLINE void BlurRowsAny(const float* src, size_t size, __mmask16 tail, size_t stride, const float* weight, size_t kernel, uint8_t* dst)
        {
            size_t sizeA = AlignLo(size, A);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                __m512 sum0 = _mm512_setzero_ps();
                __m512 sum1 = _mm512_setzero_ps();
                __m512 sum2 = _mm512_setzero_ps();
                __m512 sum3 = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                {
                    __m512 w = _mm512_set1_ps(weight[k]);
                    const float* ps = src + i + k * stride;
                    sum0 = _mm512_fmadd_ps(w, _mm512_loadu_ps(ps + 0 * F), sum0);
                    sum1 = _mm512_fmadd_ps(w, _mm512_loadu_ps(ps + 1 * F), sum1);
                    sum2 = _mm512_fmadd_ps(w, _mm512_loadu_ps(ps + 2 * F), sum2);
                    sum3 = _mm512_fmadd_ps(w, _mm512_loadu_ps(ps + 3 * F), sum3);
                }
                StoreAs8u(dst + i, sum0, sum1, sum2, sum3);
            }
            for (; i < sizeF; i += F)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm512_fmadd_ps(_mm512_set1_ps(weight[k]), _mm512_loadu_ps(src + i + k * stride), sum);
                StoreAs8u(dst + i, sum);
            }
            if (i < size)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm512_fmadd_ps(_mm512_set1_ps(weight[k]), _mm512_loadu_ps(src + i + k * stride), sum);
                StoreAs8u(dst + i, sum, tail);
            }
        }

        template<int channels> void BlurImageAny(const BlurParam& p, const Base::AlgDefault& a, const uint8_t* src, size_t srcStride, uint8_t* cols, float* rows, uint8_t* dst, size_t dstStride)
        {
            __mmask16 tail = TailMask16(a.size - AlignLo(a.size, F));
            Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
            BlurColsAny(cols, a.size, tail, p.channels, a.weight.data, a.kernel, rows + a.half * a.stride);
            for (size_t row = 0; row < a.half; ++row)
                memcpy(rows + row * a.stride, rows + a.half * a.stride, a.size * sizeof(float));
            for (size_t row = 1; row < a.nose; ++row)
            {
                Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurColsAny(cols, a.size, tail, p.channels, a.weight.data, a.kernel, rows + (a.half + row) * a.stride);
            }
            for (size_t row = a.nose; row <= a.half; ++row)
                memcpy(rows + (a.half + row) * a.stride, rows + (a.half + a.nose - 1) * a.stride, a.size * sizeof(float));
            BlurRowsAny(rows, a.size, tail, a.stride, a.weight.data, a.kernel, dst), dst += dstStride;

            for (size_t row = 1, b = row % a.kernel + 2 * a.half, w = a.kernel - row % a.kernel; row < a.body; ++row, ++b, --w)
            {
                if (b >= a.kernel)
                    b -= a.kernel;
                if (w == 0)
                    w += a.kernel;
                Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurColsAny(cols, a.size, tail, p.channels, a.weight.data, a.kernel, rows + b * a.stride);
                BlurRowsAny(rows, a.size, tail, a.stride, a.weight.data + w, a.kernel, dst), dst += dstStride;
            }

            size_t last = (a.body + 2 * a.half - 1) % a.kernel;
            for (size_t row = a.body, b = row % a.kernel + 2 * a.half, w = a.kernel - row % a.kernel; row < p.height; ++row, ++b, --w)
            {
                if (b >= a.kernel)
                    b -= a.kernel;
                if (w == 0)
                    w += a.kernel;
                memcpy(rows + b * a.stride, rows + last * a.stride, a.size * sizeof(float));
                BlurRowsAny(rows, a.size, tail, a.stride, a.weight.data + w, a.kernel, dst), dst += dstStride;
            }
        }

        //---------------------------------------------------------------------

        template<int kernel> SIMD_INLINE void BlurCols(const uint8_t* src, size_t size, __mmask16 tail, size_t channels, const float* weight, float* dst)
        {
            __m512 w[kernel];
            for (size_t k = 0; k < kernel; ++k)
                w[k] = _mm512_set1_ps(weight[k]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm512_fmadd_ps(w[k], LoadAs32f(src + i + k * channels), sum);
                _mm512_storeu_ps(dst + i, sum);
            }
            if (i < size)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm512_fmadd_ps(w[k], LoadAs32f(src + i + k * channels, tail), sum);
                _mm512_mask_storeu_ps(dst + i, tail, sum);
            }
        }

        template<> SIMD_INLINE void BlurCols<3>(const uint8_t* src, size_t size, __mmask16 tail, size_t channels, const float* weight, float* dst)
        {
            __m512 w0 = _mm512_set1_ps(weight[0]);
            __m512 w1 = _mm512_set1_ps(weight[1]);
            __m512 w2 = _mm512_set1_ps(weight[2]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m512 sum = _mm512_mul_ps(w0, LoadAs32f(src + i + 0 * channels));
                sum = _mm512_fmadd_ps(w1, LoadAs32f(src + i + 1 * channels), sum);
                sum = _mm512_fmadd_ps(w2, LoadAs32f(src + i + 2 * channels), sum);
                _mm512_storeu_ps(dst + i, sum);
            }
            if (i < size)
            {
                __m512 sum = _mm512_mul_ps(w0, LoadAs32f(src + i + 0 * channels, tail));
                sum = _mm512_fmadd_ps(w1, LoadAs32f(src + i + 1 * channels, tail), sum);
                sum = _mm512_fmadd_ps(w2, LoadAs32f(src + i + 2 * channels, tail), sum);
                _mm512_mask_storeu_ps(dst + i, tail, sum);
            }
        }

        template<> SIMD_INLINE void BlurCols<5>(const uint8_t* src, size_t size, __mmask16 tail, size_t channels, const float* weight, float* dst)
        {
            __m512 w0 = _mm512_set1_ps(weight[0]);
            __m512 w1 = _mm512_set1_ps(weight[1]);
            __m512 w2 = _mm512_set1_ps(weight[2]);
            __m512 w3 = _mm512_set1_ps(weight[3]);
            __m512 w4 = _mm512_set1_ps(weight[4]);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m512 sum0 = _mm512_mul_ps(w0, LoadAs32f(src + i + 0 * channels));
                __m512 sum1 = _mm512_mul_ps(w1, LoadAs32f(src + i + 1 * channels));
                sum0 = _mm512_fmadd_ps(w2, LoadAs32f(src + i + 2 * channels), sum0);
                sum1 = _mm512_fmadd_ps(w3, LoadAs32f(src + i + 3 * channels), sum1);
                sum0 = _mm512_fmadd_ps(w4, LoadAs32f(src + i + 4 * channels), sum0);
                _mm512_storeu_ps(dst + i, _mm512_add_ps(sum0, sum1));
            }
            if (i < size)
            {
                __m512 sum0 = _mm512_mul_ps(w0, LoadAs32f(src + i + 0 * channels, tail));
                __m512 sum1 = _mm512_mul_ps(w1, LoadAs32f(src + i + 1 * channels, tail));
                sum0 = _mm512_fmadd_ps(w2, LoadAs32f(src + i + 2 * channels, tail), sum0);
                sum1 = _mm512_fmadd_ps(w3, LoadAs32f(src + i + 3 * channels, tail), sum1);
                sum0 = _mm512_fmadd_ps(w4, LoadAs32f(src + i + 4 * channels, tail), sum0);
                _mm512_mask_storeu_ps(dst + i, tail, _mm512_add_ps(sum0, sum1));
            }
        }

        template<int kernel> SIMD_INLINE void BlurRows(const float* src, size_t size, __mmask16 tail, size_t stride, const float* weight, uint8_t* dst)
        {
            __m512 w[kernel];
            for (size_t k = 0; k < kernel; ++k)
                w[k] = _mm512_set1_ps(weight[k]);
            size_t sizeA = AlignLo(size, A);
            size_t sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                __m512 sum0 = _mm512_setzero_ps();
                __m512 sum1 = _mm512_setzero_ps();
                __m512 sum2 = _mm512_setzero_ps();
                __m512 sum3 = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                {
                    const float* ps = src + i + k * stride;
                    sum0 = _mm512_fmadd_ps(w[k], _mm512_loadu_ps(ps + 0 * F), sum0);
                    sum1 = _mm512_fmadd_ps(w[k], _mm512_loadu_ps(ps + 1 * F), sum1);
                    sum2 = _mm512_fmadd_ps(w[k], _mm512_loadu_ps(ps + 2 * F), sum2);
                    sum3 = _mm512_fmadd_ps(w[k], _mm512_loadu_ps(ps + 3 * F), sum3);
                }
                StoreAs8u(dst + i, sum0, sum1, sum2, sum3);
            }
            for (; i < sizeF; i += F)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm512_fmadd_ps(w[k], _mm512_loadu_ps(src + i + k * stride), sum);
                StoreAs8u(dst + i, sum);
            }
            if (i < size)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t k = 0; k < kernel; ++k)
                    sum = _mm512_fmadd_ps(w[k], _mm512_maskz_loadu_ps(tail, src + i + k * stride), sum);
                StoreAs8u(dst + i, sum, tail);
            }
        }

        template<int channels, int kernel> void BlurImage(const BlurParam& p, const Base::AlgDefault& a, const uint8_t* src, size_t srcStride, uint8_t* cols, float* rows, uint8_t* dst, size_t dstStride)
        {
            __mmask16 tail = TailMask16(a.size - AlignLo(a.size, F));
            Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
            BlurCols<kernel>(cols, a.size, tail, p.channels, a.weight.data, rows + a.half * a.stride);
            for (size_t row = 0; row < a.half; ++row)
                memcpy(rows + row * a.stride, rows + a.half * a.stride, a.size * sizeof(float));
            for (size_t row = 1; row < a.nose; ++row)
            {
                Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurCols<kernel>(cols, a.size, tail, p.channels, a.weight.data, rows + (a.half + row) * a.stride);
            }
            for (size_t row = a.nose; row <= a.half; ++row)
                memcpy(rows + (a.half + row) * a.stride, rows + (a.half + a.nose - 1) * a.stride, a.size * sizeof(float));
            BlurRows<kernel>(rows, a.size, tail, a.stride, a.weight.data, dst), dst += dstStride;

            for (size_t row = 1, b = row % a.kernel + 2 * a.half, w = kernel - row % kernel; row < a.body; ++row, ++b, --w)
            {
                if (b >= kernel)
                    b -= kernel;
                if (w == 0)
                    w += kernel;
                Base::PadCols<channels>(src, a.half, a.size, cols), src += srcStride;
                BlurCols<kernel>(cols, a.size, tail, p.channels, a.weight.data, rows + b * a.stride);
                BlurRows<kernel>(rows, a.size, tail, a.stride, a.weight.data + w, dst), dst += dstStride;
            }

            size_t last = (a.body + 2 * a.half - 1) % kernel;
            for (size_t row = a.body, b = row % kernel + 2 * a.half, w = kernel - row % kernel; row < p.height; ++row, ++b, --w)
            {
                if (b >= kernel)
                    b -= kernel;
                if (w == 0)
                    w += kernel;
                memcpy(rows + b * a.stride, rows + last * a.stride, a.size * sizeof(float));
                BlurRows<kernel>(rows, a.size, tail, a.stride, a.weight.data + w, dst), dst += dstStride;
            }
        }

        //---------------------------------------------------------------------

        template<int channels> Base::BlurDefaultPtr GetBlurDefaultPtr(const BlurParam& p, const Base::AlgDefault& a)
        {
            switch (a.kernel)
            {
            case 3: return BlurImage<channels, 3>;
            case 5: return BlurImage<channels, 5>;
            case 7: return BlurImage<channels, 7>;
            case 9: return BlurImage<channels, 9>;
            default: return BlurImageAny<channels>;
            }
        }

        //---------------------------------------------------------------------

        GaussianBlurDefault::GaussianBlurDefault(const BlurParam& param)
            : Avx2::GaussianBlurDefault(param)
        {
            switch (_param.channels)
            {
            case 1: _blur = GetBlurDefaultPtr<1>(_param, _alg); break;
            case 2: _blur = GetBlurDefaultPtr<2>(_param, _alg); break;
            case 3: _blur = GetBlurDefaultPtr<3>(_param, _alg); break;
            case 4: _blur = GetBlurDefaultPtr<4>(_param, _alg); break;
            }
        }

        //---------------------------------------------------------------------

        void* GaussianBlurInit(size_t width, size_t height, size_t channels, const float* sigma, const float* epsilon)
        {
            BlurParam param(width, height, channels, sigma, epsilon, A);
            if (!param.Valid())
                return NULL;
            return new GaussianBlurDefault(param);
        }

        //---------------------------------------------------------------------

        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t) * 3 * width);
                    src0 = (uint16_t*)_p;
                    src1 = src0 + width;
                    src2 = src1 + width;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * src0;
                uint16_t * src1;
                uint16_t * src2;
            private:
                void * _p;
            };
        }

        SIMD_INLINE __m512i DivideBy16(__m512i value)
        {
            return _mm512_srli_epi16(_mm512_add_epi16(value, K16_0008), 4);
        }

        const __m512i K8_01_02 = SIMD_MM512_SET2_EPI8(0x01, 0x02);

        template<int part> SIMD_INLINE __m512i BinomialSumUnpackedU8(__m512i a[3])
        {
            return _mm512_add_epi16(_mm512_maddubs_epi16(UnpackU8<part>(a[0], a[1]), K8_01_02), UnpackU8<part>(a[2]));
        }

        template<bool align> SIMD_INLINE void BlurCol(__m512i a[3], uint16_t * b)
        {
            Store<align>(b + 00, BinomialSumUnpackedU8<0>(a));
            Store<align>(b + HA, BinomialSumUnpackedU8<1>(a));
        }

        template <bool align, size_t step> void BlurCol(const uint8_t * src, size_t aligned, size_t full, uint16_t * dst)
        {
            __m512i a[3];
            LoadNose3<align, step>(src, a);
            BlurCol<true>(a, dst);
            for (size_t col = A; col < aligned; col += A)
            {
                LoadBody3<align, step>(src + col, a);
                BlurCol<true>(a, dst + col);
            }
            LoadTail3<false, step>(src + full - A, a);
            BlurCol<true>(a, dst + aligned);
        }

        template<bool align> SIMD_INLINE __m512i BlurRow16(const Buffer & buffer, size_t offset)
        {
            return DivideBy16(BinomialSum16(
                Load<align>(buffer.src0 + offset),
                Load<align>(buffer.src1 + offset),
                Load<align>(buffer.src2 + offset)));
        }

        template<bool align> SIMD_INLINE __m512i BlurRow(const Buffer & buffer, size_t offset)
        {
            return _mm512_packus_epi16(BlurRow16<align>(buffer, offset), BlurRow16<align>(buffer, offset + HA));
        }

        template <bool align, size_t step> void GaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(step*(width - 1) >= Avx2::A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(step*width) && Aligned(dst) && Aligned(dstStride));

            size_t size = step*width;
            size_t bodySize = Simd::AlignHi(size, A) - A;

            Buffer buffer(Simd::AlignHi(size, A));
            BlurCol<align, step>(src, bodySize, size, buffer.src0);
            memcpy(buffer.src1, buffer.src0, sizeof(uint16_t)*(bodySize + A));

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                const uint8_t * src2 = src + srcStride*(row + 1);
                if (row >= height - 2)
                    src2 = src + srcStride*(height - 1);

                BlurCol<align, step>(src2, bodySize, size, buffer.src2);

                for (size_t col = 0; col < bodySize; col += A)
                    Store<align>(dst + col, BlurRow<true>(buffer, col));
                Store<false>(dst + size - A, BlurRow<true>(buffer, bodySize));

                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src0, buffer.src1);
            }
        }

        template <bool align> void GaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: GaussianBlur3x3<align, 1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: GaussianBlur3x3<align, 2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: GaussianBlur3x3<align, 3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: GaussianBlur3x3<align, 4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        void GaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(channelCount*width) && Aligned(dst) && Aligned(dstStride))
                GaussianBlur3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride);
            else
                GaussianBlur3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
