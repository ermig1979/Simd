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
#include "Simd/SimdResizer.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE 
    namespace Avx2
    {
        ResizerByteBilinear::ResizerByteBilinear(const ResParam & param)
            : Ssse3::ResizerByteBilinear(param)
        {
        }

        void ResizerByteBilinear::EstimateParams()
        {
            if (_ax.data)
                return;
            if (_param.channels == 1 && _param.srcW < 4 * _param.dstW)
                _blocks = BlockCountMax(A);
            float scale = (float)_param.srcW / _param.dstW;
            _ax.Resize(AlignHi(_param.dstW, A) * _param.channels * 2, false, _param.align);
            uint8_t * alphas = _ax.data;
            if (_blocks)
            {
                _ixg.Resize(_blocks);
                int block = 0;
                _ixg[0].src = 0;
                _ixg[0].dst = 0;
                for (int dstIndex = 0; dstIndex < (int)_param.dstW; ++dstIndex)
                {
                    float alpha = (float)((dstIndex + 0.5)*scale - 0.5);
                    int srcIndex = (int)::floor(alpha);
                    alpha -= srcIndex;

                    if (srcIndex < 0)
                    {
                        srcIndex = 0;
                        alpha = 0;
                    }

                    if (srcIndex > (int)_param.srcW - 2)
                    {
                        srcIndex = (int)_param.srcW - 2;
                        alpha = 1;
                    }

                    int dst = 2 * dstIndex - _ixg[block].dst;
                    int src = srcIndex - _ixg[block].src;
                    if (src >= A - 1 || dst >= A)
                    {
                        block++;
                        _ixg[block].src = Simd::Min(srcIndex, int(_param.srcW - A));
                        _ixg[block].dst = 2 * dstIndex;
                        dst = 0;
                        src = srcIndex - _ixg[block].src;
                    }
                    _ixg[block].shuffle[dst] = src;
                    _ixg[block].shuffle[dst + 1] = src + 1;

                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    alphas += 2;
                }
                _blocks = block + 1;
            }
            else
            {
                _ix.Resize(AlignHi(_param.dstW, _param.align/4), true, _param.align);
                for (size_t i = 0; i < _param.dstW; ++i)
                {
                    float alpha = (float)((i + 0.5)*scale - 0.5);
                    ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                    alpha -= index;

                    if (index < 0)
                    {
                        index = 0;
                        alpha = 0;
                    }

                    if (index >(ptrdiff_t)_param.srcW - 2)
                    {
                        index = _param.srcW - 2;
                        alpha = 1;
                    }

                    _ix[i] = (int)index;
                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    for (size_t channel = 1; channel < _param.channels; channel++)
                        ((uint16_t*)alphas)[channel] = *(uint16_t*)alphas;
                    alphas += 2 * _param.channels;
                }
            }
            size_t size = AlignHi(_param.dstW, _param.align)*_param.channels * 2;
            _bx[0].Resize(size, false, _param.align);
            _bx[1].Resize(size, false, _param.align);
        }

        template <size_t channelCount> void ResizerByteBilinearInterpolateX(const __m256i * alpha, __m256i * buffer);

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<1>(const __m256i * alpha, __m256i * buffer)
        {
#ifdef SIMD_MADDUBS_ERROR
            __m256i _buffer = _mm256_or_si256(K_ZERO, _mm256_load_si256(buffer));
#else
            __m256i _buffer = _mm256_load_si256(buffer);
#endif
            _mm256_store_si256(buffer, _mm256_maddubs_epi16(_buffer, _mm256_load_si256(alpha)));
        }

        const __m256i K8_SHUFFLE_X2 = SIMD_MM256_SETR_EPI8(0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

        SIMD_INLINE void ResizerByteBilinearInterpolateX2(const __m256i * alpha, __m256i * buffer)
        {
            __m256i src = _mm256_shuffle_epi8(_mm256_load_si256(buffer), K8_SHUFFLE_X2);
            _mm256_store_si256(buffer, _mm256_maddubs_epi16(src, _mm256_load_si256(alpha)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<2>(const __m256i * alpha, __m256i * buffer)
        {
            ResizerByteBilinearInterpolateX2(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX2(alpha + 1, buffer + 1);
        }

        const __m256i K8_SHUFFLE_X3_00 = SIMD_MM256_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i K8_SHUFFLE_X3_01 = SIMD_MM256_SETR_EPI8(0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1,
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1);
        const __m256i K8_SHUFFLE_X3_02 = SIMD_MM256_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1);

        const __m256i K8_SHUFFLE_X3_10 = SIMD_MM256_SETR_EPI8(0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i K8_SHUFFLE_X3_11 = SIMD_MM256_SETR_EPI8(-1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF,
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1);
        const __m256i K8_SHUFFLE_X3_12 = SIMD_MM256_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0);

        const __m256i K8_SHUFFLE_X3_20 = SIMD_MM256_SETR_EPI8(0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i K8_SHUFFLE_X3_21 = SIMD_MM256_SETR_EPI8(-1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1,
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF);
        const __m256i K8_SHUFFLE_X3_22 = SIMD_MM256_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<3>(const __m256i * alpha, __m256i * buffer)
        {
            __m256i src[3], shuffled;
            src[0] = _mm256_load_si256(buffer + 0);
            src[1] = _mm256_load_si256(buffer + 1);
            src[2] = _mm256_load_si256(buffer + 2);

            shuffled = _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[0], src[0], 0x21), K8_SHUFFLE_X3_00);
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(src[0], K8_SHUFFLE_X3_01));
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[0], src[1], 0x21), K8_SHUFFLE_X3_02));
            _mm256_store_si256(buffer + 0, _mm256_maddubs_epi16(shuffled, _mm256_load_si256(alpha + 0)));

            shuffled = _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[0], src[1], 0x21), K8_SHUFFLE_X3_10);
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(src[1], K8_SHUFFLE_X3_11));
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[1], src[2], 0x21), K8_SHUFFLE_X3_12));
            _mm256_store_si256(buffer + 1, _mm256_maddubs_epi16(shuffled, _mm256_load_si256(alpha + 1)));

            shuffled = _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[1], src[2], 0x21), K8_SHUFFLE_X3_20);
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(src[2], K8_SHUFFLE_X3_21));
            shuffled = _mm256_or_si256(shuffled, _mm256_shuffle_epi8(_mm256_permute2x128_si256(src[2], src[2], 0x21), K8_SHUFFLE_X3_22));
            _mm256_store_si256(buffer + 2, _mm256_maddubs_epi16(shuffled, _mm256_load_si256(alpha + 2)));
        }

        const __m256i K8_SHUFFLE_X4 = SIMD_MM256_SETR_EPI8(0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

        SIMD_INLINE void ResizerByteBilinearInterpolateX4(const __m256i * alpha, __m256i * buffer)
        {
            __m256i src = _mm256_shuffle_epi8(_mm256_load_si256(buffer), K8_SHUFFLE_X4);
            _mm256_store_si256(buffer, _mm256_maddubs_epi16(src, _mm256_load_si256(alpha)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<4>(const __m256i * alpha, __m256i * buffer)
        {
            ResizerByteBilinearInterpolateX4(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX4(alpha + 1, buffer + 1);
            ResizerByteBilinearInterpolateX4(alpha + 2, buffer + 2);
            ResizerByteBilinearInterpolateX4(alpha + 3, buffer + 3);
        }

        const __m256i K16_FRACTION_ROUND_TERM = SIMD_MM256_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        template<bool align> SIMD_INLINE __m256i ResizerByteBilinearInterpolateY(const __m256i * pbx0, const __m256i * pbx1, __m256i alpha[2])
        {
            __m256i sum = _mm256_add_epi16(_mm256_mullo_epi16(Load<align>(pbx0), alpha[0]), _mm256_mullo_epi16(Load<align>(pbx1), alpha[1]));
            return _mm256_srli_epi16(_mm256_add_epi16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool align> SIMD_INLINE void ResizerByteBilinearInterpolateY(const uint8_t * bx0, const uint8_t * bx1, __m256i alpha[2], uint8_t * dst)
        {
            __m256i lo = ResizerByteBilinearInterpolateY<align>((__m256i*)bx0 + 0, (__m256i*)bx1 + 0, alpha);
            __m256i hi = ResizerByteBilinearInterpolateY<align>((__m256i*)bx0 + 1, (__m256i*)bx1 + 1, alpha);
            Store<false>((__m256i*)dst, PackI16ToU8(lo, hi));
        }

        template<size_t N> void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            struct One { uint8_t val[N * 1]; };
            struct Two { uint8_t val[N * 2]; };

            size_t size = 2 * _param.dstW*N;
            size_t aligned = AlignHi(size, DA) - DA;
            const size_t step = A * N;
            size_t dstW = _param.dstW;
            ptrdiff_t previous = -2;
            __m256i a[2];
            uint8_t * bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t * ax = _ax.data;
            const int32_t * ix = _ix.data;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = _mm256_set1_epi16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = _mm256_set1_epi16(int16_t(_ay[yDst]));

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    Two * pb = (Two *)bx[k];
                    const One * psrc = (const One *)(src + (sy + k)*srcStride);
                    for (size_t x = 0; x < dstW; x++)
                        pb[x] = *(Two *)(psrc + ix[x]);

                    uint8_t * pbx = bx[k];
                    for (size_t i = 0; i < size; i += step)
                        ResizerByteBilinearInterpolateX<N>((__m256i*)(ax + i), (__m256i*)(pbx + i));
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(bx[0] + ib, bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(bx[0] + i, bx[1] + i, a, dst + i / 2);
            }
        }

        void ResizerByteBilinear::RunG(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t bufW = AlignHi(_param.dstW, A) * 2;
            size_t size = 2 * _param.dstW;
            size_t aligned = AlignHi(size, DA) - DA;
            size_t blocks = _blocks;
            ptrdiff_t previous = -2;
            __m256i a[2];
            uint8_t * bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t * ax = _ax.data;
            const Idx * ixg = _ixg.data;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = _mm256_set1_epi16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = _mm256_set1_epi16(int16_t(_ay[yDst]));

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    const uint8_t * psrc = src + (sy + k)*srcStride;
                    uint8_t * pdst = bx[k];
                    for (size_t i = 0; i < blocks; ++i)
                        ResizerByteBilinearLoadGrayInterpolated(psrc, ixg[i], ax, pdst);
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(bx[0] + ib, bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(bx[0] + i, bx[1] + i, a, dst + i / 2);
            }
        }

        void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            assert(_param.dstW >= A);

            EstimateParams();
            switch (_param.channels)
            {
            case 1:
                if (_blocks)
                    RunG(src, srcStride, dst, dstStride);
                else
                    Run<1>(src, srcStride, dst, dstStride);
                break;
            case 2: Run<2>(src, srcStride, dst, dstStride); break;
            case 3: Run<3>(src, srcStride, dst, dstStride); break;
            case 4: Run<4>(src, srcStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        ResizerByteArea::ResizerByteArea(const ResParam & param)
            : Sse41::ResizerByteArea(param)
        {
        }

        SIMD_INLINE __m256i SaveLoadTail(const uint8_t * ptr, size_t tail)
        {
            uint8_t buffer[DA];
            _mm256_storeu_si256((__m256i*)(buffer), _mm256_loadu_si256((__m256i*)(ptr + tail - A)));
            return _mm256_loadu_si256((__m256i*)(buffer + A - tail));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t size, int32_t a, int32_t * dst)
        {
            __m256i alpha = SetInt16(a, a);
            size_t sizeA = AlignLo(size, A);
            size_t i = 0;
            for (; i < sizeA; i += A, dst += A)
            {
                __m256i s0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(src0 + i)), K32_TWO_UNPACK_PERMUTE);
                __m256i i0 = UnpackU8<0>(s0);
                __m256i i1 = UnpackU8<1>(s0);
                Update<update, true>(dst + 0 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
            if (i < size)
            {
                __m256i s0 = _mm256_permutevar8x32_epi32(SaveLoadTail(src0 + i, size - i), K32_TWO_UNPACK_PERMUTE);
                __m256i i0 = UnpackU8<0>(s0);
                __m256i i1 = UnpackU8<1>(s0);
                Update<update, true>(dst + 0 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t stride, size_t size, int32_t a0, int32_t a1, int32_t * dst)
        {
            __m256i alpha = SetInt16(a0, a1);
            const uint8_t * src1 = src0 + stride;
            size_t sizeA = AlignLo(size, A);
            size_t i = 0;
            for (; i < sizeA; i += A, dst += A)
            {
                __m256i s0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(src0 + i)), K32_TWO_UNPACK_PERMUTE);
                __m256i s1 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(src1 + i)), K32_TWO_UNPACK_PERMUTE);
                __m256i i0 = UnpackU8<0>(s0, s1);
                __m256i i1 = UnpackU8<1>(s0, s1);
                Update<update, true>(dst + 0 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
            if (i < size)
            {
                __m256i s0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(src0 + i)), K32_TWO_UNPACK_PERMUTE);
                __m256i s1 = _mm256_permutevar8x32_epi32(SaveLoadTail(src1 + i, size - i), K32_TWO_UNPACK_PERMUTE);
                __m256i i0 = UnpackU8<0>(s0, s1);
                __m256i i1 = UnpackU8<1>(s0, s1);
                Update<update, true>(dst + 0 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
        }

        SIMD_INLINE void ResizerByteAreaRowSum(const uint8_t * src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, int32_t * dst)
        {
            if (count)
            {
                size_t i = 0;
                ResizerByteAreaRowUpdate<UpdateSet>(src, stride, size, curr, count == 1 ? zero - next : zero, dst), src += 2 * stride, i += 2;
                for (; i < count; i += 2, src += 2 * stride)
                    ResizerByteAreaRowUpdate<UpdateAdd>(src, stride, size, zero, i == count - 1 ? zero - next : zero, dst);
                if (i == count)
                    ResizerByteAreaRowUpdate<UpdateAdd>(src, size, zero - next, dst);
            }
            else
                ResizerByteAreaRowUpdate<UpdateSet>(src, size, curr - next, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaSet(const int32_t * src, int32_t value, int32_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = src[c] * value;
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaAdd(const int32_t * src, int32_t value, int32_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] += src[c] * value;
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaRes(const int32_t * src, uint8_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = uint8_t((src[c] + Base::AREA_ROUND) >> Base::AREA_SHIFT);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaResult(const int32_t * src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t * dst)
        {
            int32_t sum[N];
            ResizerByteAreaSet<N>(src, curr, sum);
            for (size_t i = 0; i < count; ++i)
                src += N, ResizerByteAreaAdd<N>(src, zero, sum);
            ResizerByteAreaAdd<N>(src, -next, sum);
            ResizerByteAreaRes<N>(sum, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaResult34(const int32_t * src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t * dst)
        {
            __m128i sum = _mm_mullo_epi32(_mm_loadu_si128((__m128i*)src), _mm_set1_epi32(curr));
            for (size_t i = 0; i < count; ++i)
                src += N, sum = _mm_add_epi32(sum, _mm_mullo_epi32(_mm_loadu_si128((__m128i*)src), _mm_set1_epi32(zero)));
            sum = _mm_add_epi32(sum, _mm_mullo_epi32(_mm_loadu_si128((__m128i*)src), _mm_set1_epi32(-next)));
            __m128i res = _mm_srai_epi32(_mm_add_epi32(sum, _mm_set1_epi32(Base::AREA_ROUND)), Base::AREA_SHIFT);
            *(int32_t*)dst = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(res, Sse2::K_ZERO), Sse2::K_ZERO));
        }

        template<> SIMD_INLINE void ResizerByteAreaResult<4>(const int32_t * src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t * dst)
        {
            ResizerByteAreaResult34<4>(src, count, curr, zero, next, dst);
        }

        template<> SIMD_INLINE void ResizerByteAreaResult<3>(const int32_t * src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t * dst)
        {
            ResizerByteAreaResult34<3>(src, count, curr, zero, next, dst);
        }

        template<size_t N> void ResizerByteArea::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t dstW = _param.dstW, rowSize = _param.srcW*N, rowRest = dstStride - dstW * N;
            const int32_t * iy = _iy.data, *ix = _ix.data, *ay = _ay.data, *ax = _ax.data;
            int32_t ay0 = ay[0], ax0 = ax[0];
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t * buf = _by.data;
                size_t yn = iy[dy + 1] - iy[dy];
                ResizerByteAreaRowSum(src, srcStride, yn, rowSize, ay[dy], ay0, ay[dy + 1], buf), src += yn * srcStride;
                for (size_t dx = 0; dx < dstW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
                }
            }
        }

        void ResizerByteArea::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            switch (_param.channels)
            {
            case 1: Run<1>(src, srcStride, dst, dstStride); return;
            case 2: Run<2>(src, srcStride, dst, dstStride); return;
            case 3: Run<3>(src, srcStride, dst, dstStride); return;
            case 4: Run<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        ResizerFloatBilinear::ResizerFloatBilinear(const ResParam & param)
            : Base::ResizerFloatBilinear(param)
        {
        }

        void ResizerFloatBilinear::Run(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            size_t cn = _param.channels;
            size_t rs = _param.dstW * cn;
            float * pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rsa = AlignLo(rs, Avx::F);
            size_t rsh = AlignLo(rs, Sse::F);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    float * pb = pbx[k];
                    const float * ps = src + (sy + k)*srcStride;
                    size_t dx = 0;
                    if (cn == 1)
                    {
                        __m256 _1 = _mm256_set1_ps(1.0f);
                        for (; dx < rsa; dx += Avx::F)
                        {
                            __m256i idx = Avx2::LoadPermuted<true>((__m256i*)(_ix.data + dx));
                            __m256 s0145 = _mm256_castpd_ps(_mm256_i32gather_pd((double*)ps, _mm256_extracti128_si256(idx, 0), 4));
                            __m256 s2367 = _mm256_castpd_ps(_mm256_i32gather_pd((double*)ps, _mm256_extracti128_si256(idx, 1), 4));
                            __m256 fx1 = _mm256_load_ps(_ax.data + dx);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            __m256 s0 = _mm256_shuffle_ps(s0145, s2367, 0x88);
                            __m256 s1 = _mm256_shuffle_ps(s0145, s2367, 0xDD);
                            _mm256_store_ps(pb + dx, _mm256_fmadd_ps(s0, fx0, _mm256_mul_ps(s1, fx1)));
                        }
                        for (; dx < rsh; dx += Sse::F)
                        {
                            __m128 s01 = Sse::Load(ps + _ix[dx + 0], ps + _ix[dx + 1]);
                            __m128 s23 = Sse::Load(ps + _ix[dx + 2], ps + _ix[dx + 3]);
                            __m128 fx1 = _mm_load_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            __m128 m0 = _mm_mul_ps(fx0, _mm_shuffle_ps(s01, s23, 0x88));
                            __m128 m1 = _mm_mul_ps(fx1, _mm_shuffle_ps(s01, s23, 0xDD));
                            _mm_store_ps(pb + dx, _mm_add_ps(m0, m1));
                        }
                    }
                    if (cn == 3 && rs > 3)
                    {
                        __m256 _1 = _mm256_set1_ps(1.0f);
                        size_t rs3 = rs - 3;
                        size_t rs6 = AlignLoAny(rs3, 6);
                        for (; dx < rs6; dx += 6)
                        {
                            __m256 s0 = Avx::Load<false>(ps + _ix[dx + 0] + 0, ps + _ix[dx + 3] + 0);
                            __m256 s1 = Avx::Load<false>(ps + _ix[dx + 0] + 3, ps + _ix[dx + 3] + 3);
                            __m256 fx1 = Avx::Load<false>(_ax.data + dx + 0, _ax.data + dx + 3);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            Avx::Store<false>(pb + dx + 0, pb + dx + 3, _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)));
                        }
                        for (; dx < rs3; dx += 3)
                        {
                            __m128 s0 = _mm_loadu_ps(ps + _ix[dx] + 0);
                            __m128 s1 = _mm_loadu_ps(ps + _ix[dx] + 3);
                            __m128 fx1 = _mm_set1_ps(_ax.data[dx]);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, s0), _mm_mul_ps(fx1, s1)));
                        }
                    }
                    else
                    {
                        __m256 _1 = _mm256_set1_ps(1.0f);
                        __m256i _cn = _mm256_set1_epi32((int)cn);
                        for (; dx < rsa; dx += Avx::F)
                        {
                            __m256i i0 = _mm256_load_si256((__m256i*)(_ix.data + dx));
                            __m256i i1 = _mm256_add_epi32(i0, _cn);
                            __m256 s0 = _mm256_i32gather_ps(ps, i0, 4);
                            __m256 s1 = _mm256_i32gather_ps(ps, i1, 4);
                            __m256 fx1 = _mm256_load_ps(_ax.data + dx);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            _mm256_store_ps(pb + dx, _mm256_fmadd_ps(s0, fx0, _mm256_mul_ps(s1, fx1)));
                        }
                    }
                    for (; dx < rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        float fx = _ax[dx];
                        pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + cn] * fx;
                    }
                }  

                size_t dx = 0;
                __m256 _fy0 = _mm256_set1_ps(fy0);
                __m256 _fy1 = _mm256_set1_ps(fy1);
                for (; dx < rsa; dx += Avx::F)
                {
                    __m256 b0 = _mm256_load_ps(pbx[0] + dx);
                    __m256 b1 = _mm256_load_ps(pbx[1] + dx);
                    _mm256_storeu_ps(dst + dx, _mm256_fmadd_ps(b0, _fy0, _mm256_mul_ps(b1, _fy1)));
                }
                for (; dx < rsh; dx += Sse::F)
                {
                    __m128 m0 = _mm_mul_ps(_mm_load_ps(pbx[0] + dx), _mm256_castps256_ps128(_fy0));
                    __m128 m1 = _mm_mul_ps(_mm_load_ps(pbx[1] + dx), _mm256_castps256_ps128(_fy1));
                    _mm_storeu_ps(dst + dx, _mm_add_ps(m0, m1));
                }
                for (; dx < rs; dx++)
                    dst[dx] = pbx[0][dx] * fy0 + pbx[1][dx] * fy1;
            }
        }

        //---------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
        {
            ResParam param(srcX, srcY, dstX, dstY, channels, type, method, sizeof(__m256i));
            if (param.IsByteBilinear() && dstX >= A)
                return new ResizerByteBilinear(param);
            else if (param.IsByteArea())
                return new ResizerByteArea(param);
            else if (param.IsFloatBilinear())
                return new ResizerFloatBilinear(param);
            else
                return Avx::ResizerInit(srcX, srcY, dstX, dstY, channels, type, method);
        }
    }
#endif //SIMD_AVX2_ENABLE 
}

