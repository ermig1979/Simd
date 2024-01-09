/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdStream.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdNeural.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <size_t size> SIMD_INLINE void LoadWeightsForward(const float* src, __m512* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm512_set1_ps(src[i]);
        }

        template <size_t size> SIMD_INLINE void LoadWeightsBackward(const float* src, __m512* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm512_set1_ps(src[size - i - 1]);
        }

        namespace
        {
            template<int count> struct Buffer
            {
                Buffer(size_t width)
                {
                    _size = width * sizeof(float);
                    size_t stride = AlignHi(width + 2 * (count - 1), F);
                    size_t full = count * stride * sizeof(float);
                    _ptr = Allocate(full);
                    memset(_ptr, 0, full);
                    rows[0] = (float*)_ptr;
                    for (size_t i = 1; i < count; ++i)
                        rows[i] = rows[i - 1] + stride;
                }

                void Update(const float* src)
                {
                    float* tmp = rows[0];
                    if (src == NULL)
                        memset(tmp + count - 1, 0, _size);
                    else
                        memcpy(tmp + count - 1, src, _size);
                    for (size_t i = 0; i < count - 1; ++i)
                        rows[i] = rows[i + 1];
                    rows[count - 1] = tmp;
                }

                ~Buffer()
                {
                    Free(_ptr);
                }

                float* rows[count];
            private:
                size_t _size;
                void* _ptr;
            };
        }

        template<size_t coreX, size_t coreY> struct Convolution
        {
            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1);

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<coreX>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1);

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1);

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums, __mmask16 m = -1);

            template <bool align, bool mask> static SIMD_INLINE void Sum1x2(const float* src0, size_t srcStride, const float* dst0, __m512* sums);

            template <bool align, bool mask> static SIMD_INLINE void Sum2x2(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums);
        };

        template<> struct Convolution<2, 2>
        {
            template <bool align, bool mask> static SIMD_INLINE __m512 RowConvolution(const float* src, const __m512* weights, __mmask16 m = -1)
            {
                __m512 src0 = Load<align, mask>(src, m);
                __m512 src1 = Load<false, mask>(src + 1, m);
                return _mm512_fmadd_ps(src0, weights[0], _mm512_mul_ps(src1, weights[1]));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(src, weights, m);
                __m512 row1 = RowConvolution<align, mask>(src + stride, weights + 2, m);
                return _mm512_add_ps(row0, row1);
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<2>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(buffer.rows[0] + offset, weights + 0, m);
                __m512 row1 = RowConvolution<align, mask>(buffer.rows[1] + offset, weights + 2, m);
                return _mm512_add_ps(row0, row1);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                sums[0] = _mm512_fmadd_ps(dst00, (Load<align, mask>(src0 + 0, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Load<false, mask>(src0 + 1, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Load<align, mask>(src1 + 0, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, (Load<false, mask>(src1 + 1, m)), sums[3]);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* dst1 = dst0 + dstStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align, mask>(src0, m);
                __m512 src01 = Load<false, mask>(src0 + 1, m);
                __m512 src10 = Load<align, mask>(src1, m);
                __m512 src11 = Load<false, mask>(src1 + 1, m);
                sums[0] = _mm512_fmadd_ps(dst00, src00, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, src01, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, src10, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, src11, sums[3]);
                __m512 dst10 = Load<align, mask>(dst1, m);
                __m512 src20 = Load<align, mask>(src2, m);
                __m512 src21 = Load<false, mask>(src2 + 1, m);
                sums[0] = _mm512_fmadd_ps(dst10, src10, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst10, src11, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst10, src20, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst10, src21, sums[3]);
            }

            template <bool align> static SIMD_INLINE void Sum1x2(const float* src0, size_t srcStride, const float* dst0, __m512* sums)
            {
                const float* src1 = src0 + srcStride;
                __m512 dst00 = Load<align>(dst0);
                __m512 src00 = Load<align>(src0);
                __m512 src01 = Load<align>(src0 + F);
                __m512 src10 = Load<align>(src1);
                __m512 src11 = Load<align>(src1 + F);
                sums[0] = _mm512_fmadd_ps(dst00, src00, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, Alignr<1>(src00, src01), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, src10, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, Alignr<1>(src10, src11), sums[3]);
                __m512 dst10 = Load<align>(dst0 + F);
                __m512 src02 = Load<false>(src0 + F + 1);
                __m512 src12 = Load<false>(src1 + F + 1);
                sums[0] = _mm512_fmadd_ps(dst10, src01, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst10, src02, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst10, src11, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst10, src12, sums[3]);
            }

            template <bool align> static SIMD_INLINE void Sum2x2(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* dst1 = dst0 + dstStride;

                __m512 dst00 = Load<align>(dst0);
                __m512 src000 = Load<align>(src0);
                __m512 src010 = Load<align>(src0 + F);
                __m512 src100 = Load<align>(src1);
                __m512 src110 = Load<align>(src1 + F);
                __m512 src101 = Alignr<1>(src100, src110);
                sums[0] = _mm512_fmadd_ps(dst00, src000, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, Alignr<1>(src000, src010), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, src100, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, src101, sums[3]);

                __m512 dst01 = Load<align>(dst0 + F);
                __m512 src011 = Load<false>(src0 + F + 1);
                __m512 src111 = Load<false>(src1 + F + 1);
                sums[0] = _mm512_fmadd_ps(dst01, src010, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst01, src011, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst01, src110, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst01, src111, sums[3]);

                __m512 dst10 = Load<align>(dst1);
                __m512 src200 = Load<align>(src2);
                __m512 src210 = Load<align>(src2 + F);
                sums[0] = _mm512_fmadd_ps(dst10, src100, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst10, src101, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst10, src200, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst10, Alignr<1>(src200, src210), sums[3]);

                __m512 dst11 = Load<align>(dst1 + F);
                __m512 src211 = Load<false>(src2 + F + 1);
                sums[0] = _mm512_fmadd_ps(dst11, src110, sums[0]);
                sums[1] = _mm512_fmadd_ps(dst11, src111, sums[1]);
                sums[2] = _mm512_fmadd_ps(dst11, src210, sums[2]);
                sums[3] = _mm512_fmadd_ps(dst11, src211, sums[3]);
            }
        };

        template<> struct Convolution<3, 3>
        {
            template <bool align, bool mask> static SIMD_INLINE __m512 RowConvolution(const float* src, const __m512* weights, __mmask16 m = -1)
            {
                __m512 src0 = Load<align, mask>(src, m);
                __m512 src1 = Load<false, mask>(src + 1, m);
                __m512 src2 = Load<false, mask>(src + 2, m);
                return _mm512_fmadd_ps(src0, weights[0], _mm512_fmadd_ps(src1, weights[1], _mm512_mul_ps(src2, weights[2])));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(src, weights, m);
                __m512 row1 = RowConvolution<align, mask>(src + stride, weights + 3, m);
                __m512 row2 = RowConvolution<align, mask>(src + 2 * stride, weights + 6, m);
                return _mm512_add_ps(_mm512_add_ps(row0, row1), row2);
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<3>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(buffer.rows[0] + offset, weights + 0, m);
                __m512 row1 = RowConvolution<align, mask>(buffer.rows[1] + offset, weights + 3, m);
                __m512 row2 = RowConvolution<align, mask>(buffer.rows[2] + offset, weights + 6, m);
                return _mm512_add_ps(_mm512_add_ps(row0, row1), row2);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[3] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src10, src1f, m)), sums[3]);
                sums[4] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src10, src1f, m)), sums[4]);
                sums[5] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src10, src1f, m)), sums[5]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[6] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src20, src2f, m)), sums[6]);
                sums[7] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src20, src2f, m)), sums[7]);
                sums[8] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src20, src2f, m)), sums[8]);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums, __mmask16 m = -1)
            {
                const float* dst1 = dst0 + dstStride;
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* src3 = src2 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                __m512 dst10 = Load<align, mask>(dst1, m);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[0] = _mm512_fmadd_ps(dst10, Mask<mask>(src10, m), sums[0]);
                sums[3] = _mm512_fmadd_ps(dst00, Mask<mask>(src10, m), sums[3]);
                __m512 src11 = Alignr<1, mask>(src10, src1f, m);
                sums[1] = _mm512_fmadd_ps(dst10, src11, sums[1]);
                sums[4] = _mm512_fmadd_ps(dst00, src11, sums[4]);
                __m512 src12 = Alignr<2, mask>(src10, src1f, m);
                sums[2] = _mm512_fmadd_ps(dst10, src12, sums[2]);
                sums[5] = _mm512_fmadd_ps(dst00, src12, sums[5]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[3] = _mm512_fmadd_ps(dst10, Mask<mask>(src20, m), sums[3]);
                sums[6] = _mm512_fmadd_ps(dst00, Mask<mask>(src20, m), sums[6]);
                __m512 src21 = Alignr<1, mask>(src20, src2f, m);
                sums[4] = _mm512_fmadd_ps(dst10, src21, sums[4]);
                sums[7] = _mm512_fmadd_ps(dst00, src21, sums[7]);
                __m512 src22 = Alignr<2, mask>(src20, src2f, m);
                sums[5] = _mm512_fmadd_ps(dst10, src22, sums[5]);
                sums[8] = _mm512_fmadd_ps(dst00, src22, sums[8]);
                __m512 src30 = Load<align>(src3);
                __m512 src3f = Load<align>(src3 + F);
                sums[6] = _mm512_fmadd_ps(dst10, (Alignr<0, mask>(src30, src3f, m)), sums[6]);
                sums[7] = _mm512_fmadd_ps(dst10, (Alignr<1, mask>(src30, src3f, m)), sums[7]);
                sums[8] = _mm512_fmadd_ps(dst10, (Alignr<2, mask>(src30, src3f, m)), sums[8]);
            }
        };

        template<> struct Convolution<4, 4>
        {
            template <bool align, bool mask> static SIMD_INLINE __m512 RowConvolution(const float* src, const __m512* weights, __mmask16 m = -1)
            {
                __m512 src0 = Load<align>(src);
                __m512 srcf = Load<align>(src + F);
                __m512 sum0 = _mm512_fmadd_ps(Alignr<0>(src0, srcf), weights[0], _mm512_mul_ps(Alignr<1>(src0, srcf), weights[1]));
                __m512 sum1 = _mm512_fmadd_ps(Alignr<2>(src0, srcf), weights[2], _mm512_mul_ps(Alignr<3>(src0, srcf), weights[3]));
                return _mm512_add_ps(sum0, sum1);
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(src, weights, m);
                __m512 row1 = RowConvolution<align, mask>(src + stride, weights + 4, m);
                __m512 row2 = RowConvolution<align, mask>(src + 2 * stride, weights + 8, m);
                __m512 row3 = RowConvolution<align, mask>(src + 3 * stride, weights + 12, m);
                return _mm512_add_ps(_mm512_add_ps(row0, row1), _mm512_add_ps(row2, row3));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<4>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1)
            {
                __m512 row0 = RowConvolution<align, mask>(buffer.rows[0] + offset, weights + 0, m);
                __m512 row1 = RowConvolution<align, mask>(buffer.rows[1] + offset, weights + 4, m);
                __m512 row2 = RowConvolution<align, mask>(buffer.rows[2] + offset, weights + 8, m);
                __m512 row3 = RowConvolution<align, mask>(buffer.rows[3] + offset, weights + 12, m);
                return _mm512_add_ps(_mm512_add_ps(row0, row1), _mm512_add_ps(row2, row3));
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* src3 = src2 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src00, src0f, m)), sums[3]);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[4] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src10, src1f, m)), sums[4]);
                sums[5] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src10, src1f, m)), sums[5]);
                sums[6] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src10, src1f, m)), sums[6]);
                sums[7] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src10, src1f, m)), sums[7]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[8] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src20, src2f, m)), sums[8]);
                sums[9] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src20, src2f, m)), sums[9]);
                sums[10] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src20, src2f, m)), sums[10]);
                sums[11] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src20, src2f, m)), sums[11]);
                __m512 src30 = Load<align>(src3);
                __m512 src3f = Load<align>(src3 + F);
                sums[12] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src30, src3f, m)), sums[12]);
                sums[13] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src30, src3f, m)), sums[13]);
                sums[14] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src30, src3f, m)), sums[14]);
                sums[15] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src30, src3f, m)), sums[15]);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src0, size_t srcStride, const float* dst0, size_t dstStride, __m512* sums, __mmask16 m = -1)
            {
                const float* dst1 = dst0 + dstStride;
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* src3 = src2 + srcStride;
                const float* src4 = src3 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src00, src0f, m)), sums[3]);
                __m512 dst10 = Load<align, mask>(dst1, m);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[0] = _mm512_fmadd_ps(dst10, Mask<mask>(src10, m), sums[0]);
                sums[4] = _mm512_fmadd_ps(dst00, Mask<mask>(src10, m), sums[4]);
                __m512 src11 = Alignr<1, mask>(src10, src1f, m);
                sums[1] = _mm512_fmadd_ps(dst10, src11, sums[1]);
                sums[5] = _mm512_fmadd_ps(dst00, src11, sums[5]);
                __m512 src12 = Alignr<2, mask>(src10, src1f, m);
                sums[2] = _mm512_fmadd_ps(dst10, src12, sums[2]);
                sums[6] = _mm512_fmadd_ps(dst00, src12, sums[6]);
                __m512 src13 = Alignr<3, mask>(src10, src1f, m);
                sums[3] = _mm512_fmadd_ps(dst10, src13, sums[3]);
                sums[7] = _mm512_fmadd_ps(dst00, src13, sums[7]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[4] = _mm512_fmadd_ps(dst10, Mask<mask>(src20, m), sums[4]);
                sums[8] = _mm512_fmadd_ps(dst00, Mask<mask>(src20, m), sums[8]);
                __m512 src21 = Alignr<1, mask>(src20, src2f, m);
                sums[5] = _mm512_fmadd_ps(dst10, src21, sums[5]);
                sums[9] = _mm512_fmadd_ps(dst00, src21, sums[9]);
                __m512 src22 = Alignr<2, mask>(src20, src2f, m);
                sums[6] = _mm512_fmadd_ps(dst10, src22, sums[6]);
                sums[10] = _mm512_fmadd_ps(dst00, src22, sums[10]);
                __m512 src23 = Alignr<3, mask>(src20, src2f, m);
                sums[7] = _mm512_fmadd_ps(dst10, src23, sums[7]);
                sums[11] = _mm512_fmadd_ps(dst00, src23, sums[11]);
                __m512 src30 = Load<align>(src3);
                __m512 src3f = Load<align>(src3 + F);
                sums[8] = _mm512_fmadd_ps(dst10, Mask<mask>(src30, m), sums[8]);
                sums[12] = _mm512_fmadd_ps(dst00, Mask<mask>(src30, m), sums[12]);
                __m512 src31 = Alignr<1, mask>(src30, src3f, m);
                sums[9] = _mm512_fmadd_ps(dst10, src31, sums[9]);
                sums[13] = _mm512_fmadd_ps(dst00, src31, sums[13]);
                __m512 src32 = Alignr<2, mask>(src30, src3f, m);
                sums[10] = _mm512_fmadd_ps(dst10, src32, sums[10]);
                sums[14] = _mm512_fmadd_ps(dst00, src32, sums[14]);
                __m512 src33 = Alignr<3, mask>(src30, src3f, m);
                sums[11] = _mm512_fmadd_ps(dst10, src33, sums[11]);
                sums[15] = _mm512_fmadd_ps(dst00, src33, sums[15]);
                __m512 src40 = Load<align>(src4);
                __m512 src4f = Load<align>(src4 + F);
                sums[12] = _mm512_fmadd_ps(dst10, (Alignr<0, mask>(src40, src4f, m)), sums[12]);
                sums[13] = _mm512_fmadd_ps(dst10, (Alignr<1, mask>(src40, src4f, m)), sums[13]);
                sums[14] = _mm512_fmadd_ps(dst10, (Alignr<2, mask>(src40, src4f, m)), sums[14]);
                sums[15] = _mm512_fmadd_ps(dst10, (Alignr<3, mask>(src40, src4f, m)), sums[15]);
            }
        };

        template<> struct Convolution<5, 5>
        {
            template <bool align, bool mask> static SIMD_INLINE __m512 RowConvolution(const float* src, const __m512* weights, __mmask16 m = -1)
            {
                __m512 src0 = Load<align>(src);
                __m512 srcf = Load<align>(src + F);
                __m512 sum0 = _mm512_fmadd_ps(Alignr<0>(src0, srcf), weights[0], _mm512_mul_ps(Alignr<1>(src0, srcf), weights[1]));
                __m512 sum1 = _mm512_fmadd_ps(Alignr<2>(src0, srcf), weights[2], _mm512_mul_ps(Alignr<3>(src0, srcf), weights[3]));
                return _mm512_fmadd_ps(Alignr<4>(src0, srcf), weights[4], _mm512_add_ps(sum0, sum1));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Forward(const float* src, size_t stride, const __m512* weights, __mmask16 m = -1)
            {
                return _mm512_add_ps((RowConvolution<align, mask>(src, weights, m)),
                    _mm512_add_ps(_mm512_add_ps((RowConvolution<align, mask>(src + stride, weights + 5, m)),
                        (RowConvolution<align, mask>(src + 2 * stride, weights + 10, m))),
                        _mm512_add_ps((RowConvolution<align, mask>(src + 3 * stride, weights + 15, m)),
                            (RowConvolution<align, mask>(src + 4 * stride, weights + 20, m)))));
            }

            template<bool align, bool mask> static SIMD_INLINE __m512 Backward(const Buffer<5>& buffer, size_t offset, const __m512* weights, __mmask16 m = -1)
            {
                return _mm512_add_ps((RowConvolution<align, mask>(buffer.rows[0] + offset, weights, m)),
                    _mm512_add_ps(_mm512_add_ps((RowConvolution<align, mask>(buffer.rows[1] + offset, weights + 5, m)),
                        (RowConvolution<align, mask>(buffer.rows[2] + offset, weights + 10, m))),
                        _mm512_add_ps((RowConvolution<align, mask>(buffer.rows[3] + offset, weights + 15, m)),
                            (RowConvolution<align, mask>(buffer.rows[4] + offset, weights + 20, m)))));
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum1x1(const float* src0, size_t srcStride, const float* dst0, __m512* sums, __mmask16 m = -1)
            {
                const float* src1 = src0 + srcStride;
                const float* src2 = src1 + srcStride;
                const float* src3 = src2 + srcStride;
                const float* src4 = src3 + srcStride;
                __m512 dst00 = Load<align, mask>(dst0, m);
                __m512 src00 = Load<align>(src0);
                __m512 src0f = Load<align>(src0 + F);
                sums[0] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src00, src0f, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src00, src0f, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src00, src0f, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src00, src0f, m)), sums[3]);
                sums[4] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src00, src0f, m)), sums[4]);
                __m512 src10 = Load<align>(src1);
                __m512 src1f = Load<align>(src1 + F);
                sums[5] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src10, src1f, m)), sums[5]);
                sums[6] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src10, src1f, m)), sums[6]);
                sums[7] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src10, src1f, m)), sums[7]);
                sums[8] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src10, src1f, m)), sums[8]);
                sums[9] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src10, src1f, m)), sums[9]);
                __m512 src20 = Load<align>(src2);
                __m512 src2f = Load<align>(src2 + F);
                sums[10] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src20, src2f, m)), sums[10]);
                sums[11] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src20, src2f, m)), sums[11]);
                sums[12] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src20, src2f, m)), sums[12]);
                sums[13] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src20, src2f, m)), sums[13]);
                sums[14] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src20, src2f, m)), sums[14]);
                __m512 src30 = Load<align>(src3);
                __m512 src3f = Load<align>(src3 + F);
                sums[15] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src30, src3f, m)), sums[15]);
                sums[16] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src30, src3f, m)), sums[16]);
                sums[17] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src30, src3f, m)), sums[17]);
                sums[18] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src30, src3f, m)), sums[18]);
                sums[19] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src30, src3f, m)), sums[19]);
                __m512 src40 = Load<align>(src4);
                __m512 src4f = Load<align>(src4 + F);
                sums[20] = _mm512_fmadd_ps(dst00, (Alignr<0, mask>(src40, src4f, m)), sums[20]);
                sums[21] = _mm512_fmadd_ps(dst00, (Alignr<1, mask>(src40, src4f, m)), sums[21]);
                sums[22] = _mm512_fmadd_ps(dst00, (Alignr<2, mask>(src40, src4f, m)), sums[22]);
                sums[23] = _mm512_fmadd_ps(dst00, (Alignr<3, mask>(src40, src4f, m)), sums[23]);
                sums[24] = _mm512_fmadd_ps(dst00, (Alignr<4, mask>(src40, src4f, m)), sums[24]);
            }

            template <bool align, bool mask> static SIMD_INLINE void SumRow1(const float* src, const __m512& dst, __m512* sums, __mmask16 m)
            {
                __m512 src0 = Load<align>(src + 0);
                __m512 srcf = Load<align>(src + F);
                sums[0] = _mm512_fmadd_ps(dst, (Alignr<0, mask>(src0, srcf, m)), sums[0]);
                sums[1] = _mm512_fmadd_ps(dst, (Alignr<1, mask>(src0, srcf, m)), sums[1]);
                sums[2] = _mm512_fmadd_ps(dst, (Alignr<2, mask>(src0, srcf, m)), sums[2]);
                sums[3] = _mm512_fmadd_ps(dst, (Alignr<3, mask>(src0, srcf, m)), sums[3]);
                sums[4] = _mm512_fmadd_ps(dst, (Alignr<4, mask>(src0, srcf, m)), sums[4]);
            }

            template <bool align, bool mask> static SIMD_INLINE void SumRow2(const float* src, const __m512& dst0, const __m512& dst1, __m512* sums, __mmask16 m)
            {
                __m512 src0 = Load<align>(src + 0);
                __m512 srcf = Load<align>(src + F);
                sums[0] = _mm512_fmadd_ps(dst1, Mask<mask>(src0, m), sums[0]);
                sums[5] = _mm512_fmadd_ps(dst0, Mask<mask>(src0, m), sums[5]);
                __m512 src1 = Alignr<1, mask>(src0, srcf, m);
                sums[1] = _mm512_fmadd_ps(dst1, src1, sums[1]);
                sums[6] = _mm512_fmadd_ps(dst0, src1, sums[6]);
                __m512 src2 = Alignr<2, mask>(src0, srcf, m);
                sums[2] = _mm512_fmadd_ps(dst1, src2, sums[2]);
                sums[7] = _mm512_fmadd_ps(dst0, src2, sums[7]);
                __m512 src3 = Alignr<3, mask>(src0, srcf, m);
                sums[3] = _mm512_fmadd_ps(dst1, src3, sums[3]);
                sums[8] = _mm512_fmadd_ps(dst0, src3, sums[8]);
                __m512 src4 = Alignr<4, mask>(src0, srcf, m);
                sums[4] = _mm512_fmadd_ps(dst1, src4, sums[4]);
                sums[9] = _mm512_fmadd_ps(dst0, src4, sums[9]);
            }

            template <bool align, bool mask> static SIMD_INLINE void Sum2x1(const float* src, size_t srcStride, const float* dst, size_t dstStride, __m512* sums, __mmask16 m = -1)
            {
                __m512 dst0 = Load<align, mask>(dst, m);
                SumRow1<align, mask>(src, dst0, sums + 0, m);
                __m512 dst1 = Load<align, mask>(dst + dstStride, m);
                SumRow2<align, mask>(src + srcStride, dst0, dst1, sums + 0, m);
                SumRow2<align, mask>(src + 2 * srcStride, dst0, dst1, sums + 5, m);
                SumRow2<align, mask>(src + 3 * srcStride, dst0, dst1, sums + 10, m);
                SumRow2<align, mask>(src + 4 * srcStride, dst0, dst1, sums + 15, m);
                SumRow1<align, mask>(src + 5 * srcStride, dst1, sums + 20, m);
            }
        };

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionForward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, F);
            __mmask16 tailMask = __mmask16(-1) >> (F + alignedWidth - width);
            __m512 _weights[coreX * coreY];
            LoadWeightsForward<coreX* coreY>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += F)
                {
                    __m512 sum = Convolution<coreX, coreY>::template Forward<align, false>(src + col, srcStride, _weights);
                    __m512 _dst = Load<align>(dst + col);
                    Store<align>(dst + col, _mm512_add_ps(_dst, sum));
                }
                if (col < width)
                {
                    __m512 sum = Convolution<coreX, coreY>::template Forward<align, true>(src + col, srcStride, _weights);
                    __m512 _dst = Load<align, true>(dst + col, tailMask);
                    Store<align, true>(dst + col, _mm512_add_ps(_dst, sum), tailMask);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution2x2Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution4x4Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template<bool condition> struct If
        {
            template<bool align> static SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
            {
                Avx512bw::AddMultiplied<align>(src, aligned, partial, full, value, dst);
            }
        };

        template<> struct If<false>
        {
            template<bool align> static SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
            {
            }
        };

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackwardSmall(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            size_t aligned = AlignLo(width, QF);
            size_t partial = AlignLo(width, F);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < coreY; ++dy)
                {
                    const float* w = weights + dy * coreX;
                    float* d = dst + dy * dstStride;
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

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackwardLarge(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            Buffer<coreX> buffer(width);
            height += coreY - 1;
            width += coreX - 1;
            size_t alignedWidth = AlignLo(width, F);
            __mmask16 tailMask = __mmask16(-1) >> (F + alignedWidth - width);
            __m512 _weights[coreX * coreY];
            LoadWeightsBackward<coreX* coreY>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                buffer.Update(row <= height - coreY ? src : NULL);
                size_t col = 0;
                for (; col < alignedWidth; col += F)
                {
                    __m512 sum = Convolution<coreX, coreY>::template Backward<align, false>(buffer, col, _weights);
                    __m512 _dst = Load<align>(dst + col);
                    Store<align>(dst + col, _mm512_add_ps(_dst, sum));
                }
                if (col < width)
                {
                    __m512 sum = Convolution<coreX, coreY>::template Backward<false, true>(buffer, col, _weights, tailMask);
                    __m512 _dst = Load<align, true>(dst + col, tailMask);
                    Store<align, true>(dst + col, _mm512_add_ps(_dst, sum), tailMask);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (width * height < 1024)
                NeuralAddConvolutionBackwardSmall<align, coreX, coreY>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackwardLarge<align, coreX, coreY>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution2x2Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution4x4Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
        }

        SIMD_INLINE __m128 PartialSum(const __m512& src)
        {
            __m128 lo = _mm_add_ps(_mm512_extractf32x4_ps(src, 0), _mm512_extractf32x4_ps(src, 1));
            __m128 hi = _mm_add_ps(_mm512_extractf32x4_ps(src, 2), _mm512_extractf32x4_ps(src, 3));
            return _mm_add_ps(lo, hi);
        }

        SIMD_INLINE void Add4ExtractedSums(const __m512* src, float* dst)
        {
            __m128 s0 = PartialSum(src[0]);
            __m128 s1 = PartialSum(src[1]);
            __m128 s2 = PartialSum(src[2]);
            __m128 s3 = PartialSum(src[3]);
            __m128 sums = _mm_hadd_ps(_mm_hadd_ps(s0, s1), _mm_hadd_ps(s2, s3));
            _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), sums));
        }

        template <bool align, size_t coreX, size_t coreY> SIMD_INLINE void NeuralAddConvolutionSum1x1(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __mmask16 tailMask = __mmask16(-1) >> (F + alignedWidth - width);
            __m512 _sums[coreX * coreY];
            memset(_sums, 0, sizeof(_sums));
            size_t row = 0;
            for (; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += F)
                    Convolution<coreX, coreY>::template Sum1x1<align, false>(src + col, srcStride, dst + col, _sums);
                if (col < width)
                    Convolution<coreX, coreY>::template Sum1x1<align, true>(src + col, srcStride, dst + col, _sums, tailMask);
                src += srcStride;
                dst += dstStride;
            }
            size_t i = 0, n = Simd::AlignLo(coreX * coreY, 4);
#ifndef _MSC_VER
            for (; i < n; i += 4)
                Add4ExtractedSums(_sums + i, sums + i);
#endif
            for (; i < coreX * coreY; ++i)
                sums[i] += ExtractSum(_sums[i]);
        }

        template <bool align, size_t coreX, size_t coreY> SIMD_INLINE void NeuralAddConvolutionSum2x1(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            size_t alignedHeight = Simd::AlignLo(height, 2);
            size_t alignedWidth = Simd::AlignLo(width, F);
            __mmask16 tailMask = __mmask16(-1) >> (F + alignedWidth - width);
            __m512 _sums[coreX * coreY];
            memset(_sums, 0, sizeof(_sums));
            size_t row = 0;
            for (; row < alignedHeight; row += 2)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += F)
                    Convolution<coreX, coreY>::template Sum2x1<align, false>(src + col, srcStride, dst + col, dstStride, _sums);
                if (col < width)
                    Convolution<coreX, coreY>::template Sum2x1<align, true>(src + col, srcStride, dst + col, dstStride, _sums, tailMask);
                src += 2 * srcStride;
                dst += 2 * dstStride;
            }
            for (; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += F)
                    Convolution<coreX, coreY>::template Sum1x1<align, false>(src + col, srcStride, dst + col, _sums);
                if (col < width)
                    Convolution<coreX, coreY>::template Sum1x1<align, true>(src + col, srcStride, dst + col, _sums, tailMask);
                src += srcStride;
                dst += dstStride;
            }
            size_t i = 0, n = Simd::AlignLo(coreX * coreY, 4);
#ifndef _MSC_VER
            for (; i < n; i += 4)
                Add4ExtractedSums(_sums + i, sums + i);
#endif
            for (; i < coreX * coreY; ++i)
                sums[i] += ExtractSum(_sums[i]);
        }

        template <bool align, size_t coreX, size_t coreY> SIMD_INLINE void NeuralAddConvolutionSum2x2(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            size_t alignedHeight = Simd::AlignLo(height, 2);
            size_t fullAlignedWidth = Simd::AlignLo(width - 1, DF);
            size_t partialAlignedWidth = Simd::AlignLo(width, F);
            __mmask16 tailMask = __mmask16(-1) >> (F + partialAlignedWidth - width);
            __m512 _sums[coreX * coreY];
            memset(_sums, 0, sizeof(_sums));
            size_t row = 0;
            for (; row < alignedHeight; row += 2)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += DF)
                    Convolution<coreX, coreY>::template Sum2x2<align>(src + col, srcStride, dst + col, dstStride, _sums);
                for (; col < partialAlignedWidth; col += F)
                    Convolution<coreX, coreY>::template Sum2x1<align, false>(src + col, srcStride, dst + col, dstStride, _sums);
                if (col < width)
                    Convolution<coreX, coreY>::template Sum2x1<align, true>(src + col, srcStride, dst + col, dstStride, _sums, tailMask);
                src += 2 * srcStride;
                dst += 2 * dstStride;
            }
            for (; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += DF)
                    Convolution<coreX, coreY>::template Sum1x2<align>(src + col, srcStride, dst + col, _sums);
                for (; col < partialAlignedWidth; col += F)
                    Convolution<coreX, coreY>::template Sum1x1<align, false>(src + col, srcStride, dst + col, _sums);
                if (col < width)
                    Convolution<coreX, coreY>::template Sum1x1<align, true>(src + col, srcStride, dst + col, _sums, tailMask);
                src += srcStride;
                dst += dstStride;
            }
            size_t i = 0, n = Simd::AlignLo(coreX * coreY, 4);
#ifndef _MSC_VER
            for (; i < n; i += 4)
                Add4ExtractedSums(_sums + i, sums + i);
#endif
            for (; i < coreX * coreY; ++i)
                sums[i] += ExtractSum(_sums[i]);
        }

        void NeuralAddConvolution2x2Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum2x2<true, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum2x2<false, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution3x3Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum2x1<true, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum2x1<false, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution4x4Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum2x1<true, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum2x1<false, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution5x5Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum2x1<true, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum2x1<false, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
        }

        namespace Ncf
        {
            namespace Ver0
            {
                void PrepareB(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY,
                    size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, float* dst)
                {
                    const size_t K = kernelX * kernelY * srcDepth, N = dstHeight * dstWidth;
                    if (dilationX * dilationY * strideX * strideY != 1)
                    {
                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            size_t srcRow0 = dstRow * strideY - padY;
                            for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                            {
                                size_t srcCol0 = dstCol * strideX - padX;
                                for (size_t channel = 0; channel < srcDepth; ++channel)
                                {
                                    for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                    {
                                        size_t srcRow = srcRow0 + kernelRow * dilationY;
                                        if (srcRow < srcHeight)
                                        {
                                            const float* psrc = src + (channel * srcHeight + srcRow) * srcWidth;
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                            {
                                                size_t srcCol = srcCol0 + kernelCol * dilationX;
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
                    else if (kernelX * kernelY != 1)
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
                                            const float* psrc = src + (channel * srcHeight + srcRow) * srcWidth;
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
                                *(dst++) = src[k * N + i];
                        }
                    }
                }

                template <bool align> static SIMD_INLINE void Kernel1x4x16(const __m512& a, size_t K, const float* b, __m512* sums)
                {
                    sums[0] = _mm512_fmadd_ps(a, Load<align>(b + 0 * K), sums[0]);
                    sums[1] = _mm512_fmadd_ps(a, Load<align>(b + 1 * K), sums[1]);
                    sums[2] = _mm512_fmadd_ps(a, Load<align>(b + 2 * K), sums[2]);
                    sums[3] = _mm512_fmadd_ps(a, Load<align>(b + 3 * K), sums[3]);
                }

                template <bool align> static SIMD_INLINE void Kernel1x1x16(const __m512& a, const float* b, __m512& sum)
                {
                    sum = _mm512_fmadd_ps(a, Load<align>(b), sum);
                }

                SIMD_INLINE void Add4ExtractedSums(const __m512* src, float* dst)
                {
                    __m512 sum02 = _mm512_add_ps(_mm512_unpacklo_ps(src[0], src[2]), _mm512_unpackhi_ps(src[0], src[2]));
                    __m512 sum13 = _mm512_add_ps(_mm512_unpacklo_ps(src[1], src[3]), _mm512_unpackhi_ps(src[1], src[3]));
                    __m512 sum512 = _mm512_add_ps(_mm512_unpacklo_ps(sum02, sum13), _mm512_unpackhi_ps(sum02, sum13));
                    __m128 sum128 = _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(sum512, 0), _mm512_extractf32x4_ps(sum512, 1)),
                        _mm_add_ps(_mm512_extractf32x4_ps(sum512, 2), _mm512_extractf32x4_ps(sum512, 3)));
                    _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), sum128));
                }

                template <bool align> static SIMD_INLINE void Kernel6x4x16(const __m512* a, size_t K, const float* b, __m512* sums)
                {
                    __m512 _b;
                    _b = Load<align>(b + 0 * K);
                    sums[0x00] = _mm512_fmadd_ps(a[0], _b, sums[0x00]);
                    sums[0x04] = _mm512_fmadd_ps(a[1], _b, sums[0x04]);
                    sums[0x08] = _mm512_fmadd_ps(a[2], _b, sums[0x08]);
                    sums[0x0C] = _mm512_fmadd_ps(a[3], _b, sums[0x0C]);
                    sums[0x10] = _mm512_fmadd_ps(a[4], _b, sums[0x10]);
                    sums[0x14] = _mm512_fmadd_ps(a[5], _b, sums[0x14]);
                    _b = Load<align>(b + 1 * K);
                    sums[0x01] = _mm512_fmadd_ps(a[0], _b, sums[0x01]);
                    sums[0x05] = _mm512_fmadd_ps(a[1], _b, sums[0x05]);
                    sums[0x09] = _mm512_fmadd_ps(a[2], _b, sums[0x09]);
                    sums[0x0D] = _mm512_fmadd_ps(a[3], _b, sums[0x0D]);
                    sums[0x11] = _mm512_fmadd_ps(a[4], _b, sums[0x11]);
                    sums[0x15] = _mm512_fmadd_ps(a[5], _b, sums[0x15]);
                    _b = Load<align>(b + 2 * K);
                    sums[0x02] = _mm512_fmadd_ps(a[0], _b, sums[0x02]);
                    sums[0x06] = _mm512_fmadd_ps(a[1], _b, sums[0x06]);
                    sums[0x0A] = _mm512_fmadd_ps(a[2], _b, sums[0x0A]);
                    sums[0x0E] = _mm512_fmadd_ps(a[3], _b, sums[0x0E]);
                    sums[0x12] = _mm512_fmadd_ps(a[4], _b, sums[0x12]);
                    sums[0x16] = _mm512_fmadd_ps(a[5], _b, sums[0x16]);
                    _b = Load<align>(b + 3 * K);
                    sums[0x03] = _mm512_fmadd_ps(a[0], _b, sums[0x03]);
                    sums[0x07] = _mm512_fmadd_ps(a[1], _b, sums[0x07]);
                    sums[0x0B] = _mm512_fmadd_ps(a[2], _b, sums[0x0B]);
                    sums[0x0F] = _mm512_fmadd_ps(a[3], _b, sums[0x0F]);
                    sums[0x13] = _mm512_fmadd_ps(a[4], _b, sums[0x13]);
                    sums[0x17] = _mm512_fmadd_ps(a[5], _b, sums[0x17]);
                }

                template <bool align> static SIMD_INLINE void Kernel6x1x16(const __m512* a, const float* b, __m512* sums)
                {
                    __m512 b0 = Load<align>(b);
                    sums[0] = _mm512_fmadd_ps(a[0], b0, sums[0]);
                    sums[1] = _mm512_fmadd_ps(a[1], b0, sums[1]);
                    sums[2] = _mm512_fmadd_ps(a[2], b0, sums[2]);
                    sums[3] = _mm512_fmadd_ps(a[3], b0, sums[3]);
                    sums[4] = _mm512_fmadd_ps(a[4], b0, sums[4]);
                    sums[5] = _mm512_fmadd_ps(a[5], b0, sums[5]);
                }

                template <bool align> static SIMD_INLINE void Kernel3x4x16(const __m512* a, size_t K, const float* b, __m512* sums)
                {
                    __m512 _b;
                    _b = Load<align>(b + 0 * K);
                    sums[0x0] = _mm512_fmadd_ps(a[0], _b, sums[0x0]);
                    sums[0x4] = _mm512_fmadd_ps(a[1], _b, sums[0x4]);
                    sums[0x8] = _mm512_fmadd_ps(a[2], _b, sums[0x8]);
                    _b = Load<align>(b + 1 * K);
                    sums[0x1] = _mm512_fmadd_ps(a[0], _b, sums[0x1]);
                    sums[0x5] = _mm512_fmadd_ps(a[1], _b, sums[0x5]);
                    sums[0x9] = _mm512_fmadd_ps(a[2], _b, sums[0x9]);
                    _b = Load<align>(b + 2 * K);
                    sums[0x2] = _mm512_fmadd_ps(a[0], _b, sums[0x2]);
                    sums[0x6] = _mm512_fmadd_ps(a[1], _b, sums[0x6]);
                    sums[0xA] = _mm512_fmadd_ps(a[2], _b, sums[0xA]);
                    _b = Load<align>(b + 3 * K);
                    sums[0x3] = _mm512_fmadd_ps(a[0], _b, sums[0x3]);
                    sums[0x7] = _mm512_fmadd_ps(a[1], _b, sums[0x7]);
                    sums[0xB] = _mm512_fmadd_ps(a[2], _b, sums[0xB]);
                }

                template <bool align> static SIMD_INLINE void Kernel3x1x16(const __m512* a, const float* b, __m512* sums)
                {
                    __m512 _b = Load<align>(b);
                    sums[0x0] = _mm512_fmadd_ps(a[0], _b, sums[0x0]);
                    sums[0x1] = _mm512_fmadd_ps(a[1], _b, sums[0x1]);
                    sums[0x2] = _mm512_fmadd_ps(a[2], _b, sums[0x2]);
                }

                template <bool align, bool mask> static SIMD_INLINE void Load6(const float* p, __m512* a, size_t step, __mmask16 tail = -1)
                {
                    a[0] = Load<align, mask>(p + 0 * step, tail);
                    a[1] = Load<align, mask>(p + 1 * step, tail);
                    a[2] = Load<align, mask>(p + 2 * step, tail);
                    a[3] = Load<align, mask>(p + 3 * step, tail);
                    a[4] = Load<align, mask>(p + 4 * step, tail);
                    a[5] = Load<align, mask>(p + 5 * step, tail);
                }

                template <bool align, bool mask> static SIMD_INLINE void Load3(const float* p, __m512* a, size_t step, __mmask16 tail = -1)
                {
                    a[0] = Load<align, mask>(p + 0 * step, tail);
                    a[1] = Load<align, mask>(p + 1 * step, tail);
                    a[2] = Load<align, mask>(p + 2 * step, tail);
                }

                template <bool align> void Execute(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    size_t M3 = M / 3 * 3;
                    size_t M6 = M / 6 * 6;
                    size_t N4 = Simd::AlignLo(N, 4);
                    size_t K16 = Simd::AlignLo(K, 16);
                    __mmask16 tailMask = TailMask16(K - K16);
                    size_t i = 0;
#if SIMD_ZMM_COUNT == 32
                    for (; i < M6; i += 6)
                    {
                        const float* pa = a + i * K;
                        float* pc = c + i * N;
                        size_t j = 0;
                        __m512 _a[6];
                        for (; j < N4; j += 4)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[24] = {
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                Load6<false, false>(pa + k, _a, K);
                                Kernel6x4x16<align>(_a, K, pb + k, sums);
                            }
                            if (k < K)
                            {
                                Load6<false, true>(pa + k, _a, K, tailMask);
                                Kernel6x4x16<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0x00, pc + 0 * N + j);
                            Add4ExtractedSums(sums + 0x04, pc + 1 * N + j);
                            Add4ExtractedSums(sums + 0x08, pc + 2 * N + j);
                            Add4ExtractedSums(sums + 0x0C, pc + 3 * N + j);
                            Add4ExtractedSums(sums + 0x10, pc + 4 * N + j);
                            Add4ExtractedSums(sums + 0x14, pc + 5 * N + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[6] = {
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                Load6<false, false>(pa + k, _a, K);
                                Kernel6x1x16<align>(_a, pb + k, sums);
                            }
                            if (k < K)
                            {
                                Load6<false, true>(pa + k, _a, K, tailMask);
                                Kernel6x1x16<false>(_a, pb + k, sums);
                            }
                            pc[0 * N + j] += ExtractSum(sums[0]);
                            pc[1 * N + j] += ExtractSum(sums[1]);
                            pc[2 * N + j] += ExtractSum(sums[2]);
                            pc[3 * N + j] += ExtractSum(sums[3]);
                            pc[4 * N + j] += ExtractSum(sums[4]);
                            pc[5 * N + j] += ExtractSum(sums[5]);
                        }
                    }
#endif
                    for (; i < M3; i += 3)
                    {
                        const float* pa = a + i * K;
                        float* pc = c + i * N;
                        size_t j = 0;
                        __m512 _a[3];
                        for (; j < N4; j += 4)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[12] = {
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                Load3<false, false>(pa + k, _a, K);
                                Kernel3x4x16<align>(_a, K, pb + k, sums);
                            }
                            if (k < K)
                            {
                                Load3<false, true>(pa + k, _a, K, tailMask);
                                Kernel3x4x16<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0x0, pc + 0 * N + j);
                            Add4ExtractedSums(sums + 0x4, pc + 1 * N + j);
                            Add4ExtractedSums(sums + 0x8, pc + 2 * N + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[3] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                Load3<false, false>(pa + k, _a, K);
                                Kernel3x1x16<align>(_a, pb + k, sums);
                            }
                            if (k < K)
                            {
                                Load3<false, true>(pa + k, _a, K, tailMask);
                                Kernel3x1x16<false>(_a, pb + k, sums);
                            }
                            pc[0 * N + j] += ExtractSum(sums[0]);
                            pc[1 * N + j] += ExtractSum(sums[1]);
                            pc[2 * N + j] += ExtractSum(sums[2]);
                        }
                    }
                    for (; i < M; ++i)
                    {
                        const float* pa = a + i * K;
                        float* pc = c + i * N;
                        size_t j = 0;
                        for (; j < N4; j += 4)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                __m512 _a = Load<false>(pa + k);
                                Kernel1x4x16<align>(_a, K, pb + k, sums);
                            }
                            if (k < K)
                            {
                                __m512 _a = Load<false, true>(pa + k, tailMask);
                                Kernel1x4x16<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0, pc + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float* pb = b + j * K;
                            __m512 sum = _mm512_setzero_ps();
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                __m512 _a = Load<false>(pa + k);
                                Kernel1x1x16<align>(_a, pb + k, sum);
                            }
                            if (k < K)
                            {
                                __m512 _a = Load<false, true>(pa + k, tailMask);
                                Kernel1x1x16<false>(_a, pb + k, sum);
                            }
                            pc[j] += ExtractSum(sum);
                        }
                    }
                }

                void Execute(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    if (Aligned(K, F))
                        Execute<true>(M, N, K, a, b, c);
                    else
                        Execute<false>(M, N, K, a, b, c);
                }
            }

            namespace Ver1
            {
                void PrepareA(const float* src, size_t M, size_t K, size_t cell, float* dst)
                {
                    size_t K4 = AlignLo(K, 4), K8 = AlignLo(K, 8);
                    for (size_t i = 0; i < M; i += cell)
                    {
                        size_t n = Simd::Min(cell, M - i), k = 0;
                        if (cell == 4 && n == 4)
                        {
                            for (; k < K8; k += 8)
                            {
                                const float* ps = src + k;
                                __m256 s0 = Avx2::Load<false>(ps + 0 * K);
                                __m256 s1 = Avx2::Load<false>(ps + 1 * K);
                                __m256 s2 = Avx2::Load<false>(ps + 2 * K);
                                __m256 s3 = Avx2::Load<false>(ps + 3 * K);
                                __m256 s00 = _mm256_unpacklo_ps(s0, s2);
                                __m256 s01 = _mm256_unpacklo_ps(s1, s3);
                                __m256 s10 = _mm256_unpackhi_ps(s0, s2);
                                __m256 s11 = _mm256_unpackhi_ps(s1, s3);
                                __m256 d0 = _mm256_unpacklo_ps(s00, s01);
                                __m256 d1 = _mm256_unpackhi_ps(s00, s01);
                                __m256 d2 = _mm256_unpacklo_ps(s10, s11);
                                __m256 d3 = _mm256_unpackhi_ps(s10, s11);
                                Avx2::Store<false>(dst + 0, _mm256_permute2f128_ps(d0, d1, 0x20));
                                Avx2::Store<false>(dst + 8, _mm256_permute2f128_ps(d2, d3, 0x20));
                                Avx2::Store<false>(dst + 16, _mm256_permute2f128_ps(d0, d1, 0x31));
                                Avx2::Store<false>(dst + 24, _mm256_permute2f128_ps(d2, d3, 0x31));
                                dst += 32;
                            }
                            for (; k < K4; k += 4)
                            {
                                const float* ps = src + k;
                                __m128 s0 = Sse41::Load<false>(ps + 0 * K);
                                __m128 s1 = Sse41::Load<false>(ps + 1 * K);
                                __m128 s2 = Sse41::Load<false>(ps + 2 * K);
                                __m128 s3 = Sse41::Load<false>(ps + 3 * K);
                                __m128 s00 = _mm_unpacklo_ps(s0, s2);
                                __m128 s01 = _mm_unpacklo_ps(s1, s3);
                                __m128 s10 = _mm_unpackhi_ps(s0, s2);
                                __m128 s11 = _mm_unpackhi_ps(s1, s3);
                                Sse41::Store<false>(dst + 0, _mm_unpacklo_ps(s00, s01));
                                Sse41::Store<false>(dst + 4, _mm_unpackhi_ps(s00, s01));
                                Sse41::Store<false>(dst + 8, _mm_unpacklo_ps(s10, s11));
                                Sse41::Store<false>(dst + 12, _mm_unpackhi_ps(s10, s11));
                                dst += 16;
                            }
                        }
                        for (; k < K; ++k)
                        {
                            for (size_t c = 0; c < n; ++c)
                                *(dst++) = src[c * K + k];
                        }
                        src += cell * K;
                    }
                }

                void PrepareB(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY,
                    size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t cell, float* tmp, float* dst)
                {
                    const size_t K = kernelX * kernelY * srcDepth, N = dstHeight * dstWidth;
                    if (kernelX * kernelY != 1)
                    {
                        float* dst = tmp;
                        size_t channelSize = srcHeight * srcWidth;
                        if (dilationX * dilationY * strideX * strideY != 1)
                        {
                            for (size_t channel = 0, k = 0; channel < srcDepth; ++channel, src += channelSize)
                            {
                                for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                {
                                    for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol, ++k)
                                    {
                                        size_t srcRow = kernelRow * dilationY - padY;
                                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                                        {
                                            if (srcRow < srcHeight)
                                            {
                                                size_t srcCol = kernelCol * dilationX - padX;
                                                for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = src[srcRow * srcWidth + srcCol];
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
                                                const float* psrc = src + srcRow * srcWidth;
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
                    if (cell == 48)
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            if (n == cell)
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float* psrc = src + k * N;
                                    Store<false>(dst + 0 * F, Load<false>(psrc + 0 * F));
                                    Store<false>(dst + 1 * F, Load<false>(psrc + 1 * F));
                                    Store<false>(dst + 2 * F, Load<false>(psrc + 2 * F));
                                    dst += 48;
                                }
                            }
                            else
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float* psrc = src + k * N;
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
                                    const float* psrc = src + k * N;
                                    Store<false>(dst, Load<false>(psrc));
                                    dst += 16;
                                }
                            }
                            else
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float* psrc = src + k * N;
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
                                const float* psrc = src + k * N;
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

                SIMD_INLINE void AddSum(__m512 sum, float* dst)
                {
                    _mm512_storeu_ps(dst, _mm512_add_ps(_mm512_loadu_ps(dst), sum));
                }

                template<bool mask> SIMD_INLINE void AddSum(__m512 sum, float* dst, __mmask16 tail = -1)
                {
                    Store<false, mask>(dst, _mm512_add_ps((Load<false, mask>(dst, tail)), sum), tail);
                }

                template<bool mask> SIMD_INLINE void AddSums16(const __m512* sums, size_t size, float* dst, size_t stride, __mmask16 tail = -1)
                {
                    for (size_t i = 0; i < size; ++i, dst += stride)
                        AddSum<mask>(sums[i], dst, tail);
                }

                template <bool align, bool mask> SIMD_INLINE void KernelMx16(size_t N, size_t K, const float* a, const float* b, float* c, size_t m, __mmask16 tail = -1)
                {
                    __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m512 b0 = Load<align>(b);
                        for (size_t s = 0; s < m; ++s)
                        {
                            __m512 a0 = _mm512_set1_ps(a[s]);
                            sums[s] = _mm512_fmadd_ps(b0, a0, sums[s]);
                        }
                        b += 16;
                        a += m;
                    }
                    AddSums16<mask>(sums, m, c, N, tail);
                }

                template <bool align, bool mask> SIMD_INLINE void Kernel4x16(size_t N, size_t K, const float* a, const float* b, float* c, __mmask16 tail = -1)
                {
                    __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m512 b0 = Load<align>(b);
                        __m512 a0 = _mm512_set1_ps(a[0]);
                        sums[0] = _mm512_fmadd_ps(b0, a0, sums[0]);
                        __m512 a1 = _mm512_set1_ps(a[1]);
                        sums[1] = _mm512_fmadd_ps(b0, a1, sums[1]);
                        __m512 a2 = _mm512_set1_ps(a[2]);
                        sums[2] = _mm512_fmadd_ps(b0, a2, sums[2]);
                        __m512 a3 = _mm512_set1_ps(a[3]);
                        sums[3] = _mm512_fmadd_ps(b0, a3, sums[3]);
                        b += 16;
                        a += 4;
                    }
                    AddSums16<mask>(sums, 4, c, N, tail);
                }

                template <bool align> void Execute4x16(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t N16 = Simd::AlignLo(N, 16);
                    __mmask16 tailMask = TailMask16(N - N16);
                    size_t i = 0;
                    for (; i < M4; i += 4)
                    {
                        size_t j = 0;
                        for (; j < N16; j += 16)
                            Kernel4x16<align, false>(N, K, a + i * K, b + j * K, c + i * N + j);
                        if (j < N)
                            Kernel4x16<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, tailMask);
                    }
                    if (i < M)
                    {
                        size_t j = 0;
                        for (; j < N16; j += 16)
                            KernelMx16<align, false>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4);
                        if (j < N)
                            KernelMx16<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMask);
                    }
                }

                template <bool align, bool mask> SIMD_INLINE void KernelMx48(size_t N, size_t K, const float* a, const float* b, float* c, size_t m, const __mmask16* tails)
                {
                    __m512 sums[12] = {
                        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m512 b0 = Load<align>(b + 00);
                        __m512 b1 = Load<align>(b + 16);
                        __m512 b2 = Load<align>(b + 32);
                        for (size_t s = 0; s < m; ++s)
                        {
                            __m512 a0 = _mm512_set1_ps(a[s]);
                            sums[s + 0] = _mm512_fmadd_ps(b0, a0, sums[s + 0]);
                            sums[s + 4] = _mm512_fmadd_ps(b1, a0, sums[s + 4]);
                            sums[s + 8] = _mm512_fmadd_ps(b2, a0, sums[s + 8]);
                        }
                        b += 48;
                        a += m;
                    }
                    for (size_t i = 0; i < m; ++i, c += N)
                    {
                        AddSum<mask>(sums[i + 0], c + 00, tails[0]);
                        AddSum<mask>(sums[i + 4], c + 16, tails[1]);
                        AddSum<mask>(sums[i + 8], c + 32, tails[2]);
                    }
                }

                void Kernel4x48(size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    __m512 _a, b0, b1, b2, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;

                    c00 = _mm512_setzero_ps();
                    c01 = _mm512_setzero_ps();
                    c02 = _mm512_setzero_ps();
                    c10 = _mm512_setzero_ps();
                    c11 = _mm512_setzero_ps();
                    c12 = _mm512_setzero_ps();
                    c20 = _mm512_setzero_ps();
                    c21 = _mm512_setzero_ps();
                    c22 = _mm512_setzero_ps();
                    c30 = _mm512_setzero_ps();
                    c31 = _mm512_setzero_ps();
                    c32 = _mm512_setzero_ps();

                    for (size_t k = 0; k < K; ++k)
                    {
                        b0 = _mm512_loadu_ps(b + 0 * F);
                        b1 = _mm512_loadu_ps(b + 1 * F);
                        b2 = _mm512_loadu_ps(b + 2 * F);
                        _a = _mm512_set1_ps(a[0]);
                        c00 = _mm512_fmadd_ps(b0, _a, c00);
                        c01 = _mm512_fmadd_ps(b1, _a, c01);
                        c02 = _mm512_fmadd_ps(b2, _a, c02);
                        _a = _mm512_set1_ps(a[1]);
                        c10 = _mm512_fmadd_ps(b0, _a, c10);
                        c11 = _mm512_fmadd_ps(b1, _a, c11);
                        c12 = _mm512_fmadd_ps(b2, _a, c12);
                        _a = _mm512_set1_ps(a[2]);
                        c20 = _mm512_fmadd_ps(b0, _a, c20);
                        c21 = _mm512_fmadd_ps(b1, _a, c21);
                        c22 = _mm512_fmadd_ps(b2, _a, c22);
                        _a = _mm512_set1_ps(a[3]);
                        c30 = _mm512_fmadd_ps(b0, _a, c30);
                        c31 = _mm512_fmadd_ps(b1, _a, c31);
                        c32 = _mm512_fmadd_ps(b2, _a, c32);
                        b += 48;
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

                template <bool align> void Execute4x48(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t N48 = N / 48 * 48;
                    __mmask16 tailMasks[3];
                    for (size_t i = 0; i < 3; ++i)
                        tailMasks[i] = TailMask16(N - N48 - F * i);
                    if (M > N)
                    {
                        size_t i = 0;
                        for (; i < M4; i += 4)
                        {
                            size_t j = 0;
                            for (; j < N48; j += 48)
                                Kernel4x48(N, K, a + i * K, b + j * K, c + i * N + j);
                            if (j < N)
                                KernelMx48<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, 4, tailMasks);
                        }
                        if (i < M)
                        {
                            size_t j = 0;
                            for (; j < N48; j += 48)
                                KernelMx48<align, false>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMasks);
                            if (j < N)
                                KernelMx48<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMasks);
                        }
                    }
                    else
                    {
                        size_t j = 0;
                        for (; j < N48; j += 48)
                        {
                            size_t i = 0;
                            for (; i < M4; i += 4)
                                Kernel4x48(N, K, a + i * K, b + j * K, c + i * N + j);
                            if (M4 < M)
                                KernelMx48<align, false>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMasks);
                        }
                        if (N48 < N)
                        {
                            size_t i = 0;
                            for (; i < M4; i += 4)
                                KernelMx48<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, 4, tailMasks);
                            if (M4 < M)
                                KernelMx48<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMasks);
                        }
                    }
                }

                void Execute(size_t M, size_t N, size_t K, const float* a, const float* b, float* c, size_t cellA, size_t cellB)
                {
                    if (cellA == 4)
                    {
                        if (cellB == 16)
                            Execute4x16<false>(M, N, K, a, b, c);
                        if (cellB == 48)
                            Execute4x48<false>(M, N, K, a, b, c);
                    }
                }
            }

            namespace Ver2
            {
                void PrepareB(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t padX, size_t padY, float* dst, size_t dstWidth, size_t dstHeight)
                {
                    for (size_t channel = 0; channel < srcDepth; ++channel)
                    {
                        const float* s = src;
                        float* d = dst;
                        memset(d, 0, padY * dstWidth * 4);
                        d += padY * dstWidth;
                        for (size_t row = padY; row < dstHeight - padY; ++row)
                        {
                            memset(d, 0, padX * 4);
                            memcpy(d + padX, s, srcWidth * 4);
                            memset(d + padX + srcWidth, 0, padX * 4);
                            d += dstWidth;
                            s += srcWidth;
                        }
                        memset(d, 0, padY * dstWidth * 4);
                        src += srcWidth * srcHeight;
                        dst += dstWidth * dstHeight;
                    }
                }

                template <bool align, size_t kernelX, size_t kernelY> void AddConvolution8x8(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstDepth)
                {
                    __m256 _weight[kernelX * kernelY];
                    for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                    {
                        __m256 _dst[8];
                        float* pdst = dst;
                        for (size_t row = 0; row < 8; ++row, pdst += 8)
                            _dst[row] = Avx2::Load<align>(pdst);
                        if (kernelY < 4)
                        {
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float* psrc = src + srcWidth * srcHeight * srcChannel;
                                Avx2::LoadWeightsForward<kernelX* kernelY>(weight, _weight);
                                for (size_t row = 0; row < 8; ++row)
                                {
                                    _dst[row] = _mm256_add_ps(_dst[row], Avx2::Convolution<kernelX, kernelY>::template Forward<align>(psrc, srcWidth, _weight));
                                    psrc += srcWidth;
                                }
                                weight += kernelX * kernelY;
                            }
                        }
                        else
                        {
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float* psrc = src + srcWidth * srcHeight * srcChannel;
                                for (size_t dy = 0; dy < kernelY; dy++)
                                {
                                    const float* ps = psrc + dy * srcWidth;
                                    Avx2::LoadWeightsForward<kernelX>(weight, _weight);
                                    for (size_t row = 0; row < 8; ++row)
                                    {
                                        _dst[row] = _mm256_add_ps(_dst[row], Avx2::Convolution<kernelX, kernelY>::template RowConvolution<align>(ps, _weight));
                                        ps += srcWidth;
                                    }
                                    weight += kernelX;
                                }
                            }
                        }
                        for (size_t row = 0; row < 8; ++row, dst += 8)
                            Avx2::Store<align>(dst, _dst[row]);
                    }
                }

                template <bool align, size_t kernelX, size_t kernelY> void AddConvolution16x16(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstDepth)
                {
                    __m512 _weight[kernelX * kernelY];
                    for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                    {
                        __m512 _dst[16];
                        float* pdst = dst;
                        for (size_t row = 0; row < 16; ++row, pdst += 16)
                            _dst[row] = Load<align>(pdst);
                        if (kernelY < 4)
                        {
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float* psrc = src + srcWidth * srcHeight * srcChannel;
                                LoadWeightsForward<kernelX* kernelY>(weight, _weight);
                                for (size_t row = 0; row < 16; ++row)
                                {
                                    _dst[row] = _mm512_add_ps(_dst[row], (Convolution<kernelX, kernelY>::template Forward<align, false>(psrc, srcWidth, _weight)));
                                    psrc += srcWidth;
                                }
                                weight += kernelX * kernelY;
                            }
                        }
                        else
                        {
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float* psrc = src + srcWidth * srcHeight * srcChannel;
                                for (size_t dy = 0; dy < kernelY; dy++)
                                {
                                    const float* ps = psrc + dy * srcWidth;
                                    LoadWeightsForward<kernelX>(weight, _weight);
                                    for (size_t row = 0; row < 16; ++row)
                                    {
                                        _dst[row] = _mm512_add_ps(_dst[row], (Convolution<kernelX, kernelY>::template RowConvolution<align, false>(ps, _weight)));
                                        ps += srcWidth;
                                    }
                                    weight += kernelX;
                                }
                            }
                        }
                        for (size_t row = 0; row < 16; ++row, dst += 16)
                            Store<align>(dst, _dst[row]);
                    }
                }

                template <bool align, size_t kernelX, size_t kernelY> void AddConvolution(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    if (dstWidth == 8 && dstHeight == 8)
                    {
                        AddConvolution8x8<align, kernelX, kernelY>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstDepth);
                        return;
                    }
                    if (dstWidth == 16 && dstHeight == 16)
                    {
                        AddConvolution16x16<align, kernelX, kernelY>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstDepth);
                        return;
                    }
                    size_t alignedWidth = AlignLo(dstWidth, F);
                    __mmask16 tailMask = TailMask16(dstWidth - alignedWidth);
                    __m512 _weight[kernelX * kernelY];
                    for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                    {
                        for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                        {
                            const float* psrc = src + srcWidth * srcHeight * srcChannel;
                            const float* pweight = weight + (dstChannel * srcDepth + srcChannel) * kernelX * kernelY;
                            float* pdst = dst + dstWidth * dstHeight * dstChannel;
                            LoadWeightsForward<kernelX* kernelY>(pweight, _weight);
                            for (size_t row = 0; row < dstHeight; ++row)
                            {
                                size_t col = 0;
                                for (; col < alignedWidth; col += F)
                                {
                                    __m512 _dst = Load<align>(pdst + col);
                                    _dst = _mm512_add_ps(_dst, (Convolution<kernelX, kernelY>::template Forward<align, false>(psrc + col, srcWidth, _weight)));
                                    Store<align>(pdst + col, _dst);
                                }
                                if (col < dstWidth)
                                {
                                    __m512 _dst = Load<align, true>(pdst + col, tailMask);
                                    _dst = _mm512_add_ps(_dst, (Convolution<kernelX, kernelY>::template Forward<align, true>(psrc + col, srcWidth, _weight, tailMask)));
                                    Store<align, true>(pdst + col, _dst, tailMask);
                                }
                                psrc += srcWidth;
                                pdst += dstWidth;
                            }
                        }
                    }
                }

                void AddConvolution1x1x16(const float* src, size_t srcDepth, const float* weight, float* dst, size_t dstDepth)
                {
                    size_t dstDepth4 = dstDepth / 4 * 4;
                    size_t dstChannel = 0;
                    for (; dstChannel < dstDepth4; dstChannel += 4)
                    {
                        __m512 dst00 = _mm512_loadu_ps(dst + 0 * F);
                        __m512 dst10 = _mm512_loadu_ps(dst + 1 * F);
                        __m512 dst20 = _mm512_loadu_ps(dst + 2 * F);
                        __m512 dst30 = _mm512_loadu_ps(dst + 3 * F);
                        const float* psrc = src;
                        const float* pw0 = weight;
                        const float* pw1 = pw0 + srcDepth;
                        const float* pw2 = pw1 + srcDepth;
                        const float* pw3 = pw2 + srcDepth;
                        for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                        {
                            __m512 _weight;
                            __m512 src0 = _mm512_loadu_ps(psrc + 0 * F);
                            _weight = _mm512_set1_ps(pw0[srcChannel]);
                            dst00 = _mm512_fmadd_ps(_weight, src0, dst00);
                            _weight = _mm512_set1_ps(pw1[srcChannel]);
                            dst10 = _mm512_fmadd_ps(_weight, src0, dst10);
                            _weight = _mm512_set1_ps(pw2[srcChannel]);
                            dst20 = _mm512_fmadd_ps(_weight, src0, dst20);
                            _weight = _mm512_set1_ps(pw3[srcChannel]);
                            dst30 = _mm512_fmadd_ps(_weight, src0, dst30);
                            psrc += 16;
                        }
                        _mm512_storeu_ps(dst + 0 * F, dst00);
                        _mm512_storeu_ps(dst + 1 * F, dst10);
                        _mm512_storeu_ps(dst + 2 * F, dst20);
                        _mm512_storeu_ps(dst + 3 * F, dst30);
                        dst += 16 * 4;
                        weight += srcDepth * 4;
                    }
                    for (; dstChannel < dstDepth; ++dstChannel)
                    {
                        __m512 dst0 = _mm512_loadu_ps(dst + 0 * F);
                        const float* psrc = src;
                        for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                        {
                            __m512 weight0 = _mm512_set1_ps(*weight++);
                            dst0 = _mm512_fmadd_ps(weight0, _mm512_loadu_ps(psrc + 0 * F), dst0);
                            psrc += 16;
                        }
                        _mm512_storeu_ps(dst + 0 * F, dst0);
                        dst += 16;
                    }
                }

                void Execute(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, size_t kernelX, size_t kernelY, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    assert(kernelX == kernelY);
                    if (kernelX == 1 && dstWidth * dstHeight == 16)
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
                    if (kernelX == kernelY && strideX * strideY * dilationX * dilationY == 1)
                    {
                        if (kernelX >= 2 && kernelX <= 5)// && dstWidth*dstHeight*kernelX*kernelY >= 8 * 8 * 3 * 3)
                            return true;
                        if (kernelX == 1 && (dstWidth * dstHeight == 16))// || dstWidth * dstHeight == 64))
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
                    N = dstHeight * dstWidth;
                    K = kernelX * kernelY * srcDepth;

                    if (dstWidth * dstHeight / kernelX <= 1000)
                        alg = Ver0;
                    else
                        alg = Ver1;
                    if (Ver2::Preferable(srcDepth, kernelX, kernelY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth))
                        alg = Ver2;

                    switch (alg)
                    {
                    case Ver0:
                        sizeB = N * K;
                        break;
                    case Ver1:
                        cellA = 4;
                        cellB = 48;
                        sizeA = M * K;
                        strideB = (N + cellB - 1) / cellB * cellB;
                        sizeB = strideB * K;
                        if (kernelX * kernelY > 1)
                            sizeT = sizeB;
                        break;
                    case Ver2:
                        if (padX > 0 || padY > 0)
                        {
                            paddedW = Simd::AlignHi(srcWidth + 2 * padX, F);
                            paddedH = srcHeight + 2 * padY;
                            sizeB = paddedW * paddedH * srcDepth;
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
                float* a;
                float* b;
                float* t;

                Data(size_t sizeA, size_t sizeB, size_t sizeT, void* externalData, size_t* externalSize)
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
                    float* data = NULL;
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
                void* _data;
            };
        }

        void NeuralConvolutionForward(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
            const float* weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY,
            void* buffer, size_t* size, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add)
        {
            using namespace Ncf;

            assert(dstWidth == (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1);
            assert(dstHeight == (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1);

            if (dstWidth < F && srcDepth <= 32)
            {
                Avx2::NeuralConvolutionForward(src, srcWidth, srcHeight, srcDepth, weight, kernelX, kernelY, padX, padY,
                    strideX, strideY, dilationX, dilationY, buffer, size, dst, dstWidth, dstHeight, dstDepth, add);
                return;
            }

            if (!add)
                memset(dst, 0, dstWidth * dstHeight * dstDepth * sizeof(float));

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
#endif
}
