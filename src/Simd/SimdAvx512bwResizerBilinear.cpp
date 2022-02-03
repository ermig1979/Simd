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
#include "Simd/SimdMemory.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE 
    namespace Avx512bw
    {
        ResizerByteBilinear::ResizerByteBilinear(const ResParam & param)
            : Avx2::ResizerByteBilinear(param)
        {
        }

        template <size_t N> void ResizerByteBilinearInterpolateX(const uint8_t * alpha, uint8_t * buffer);

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<1>(const uint8_t * alpha, uint8_t * buffer)
        {
            __m512i _buffer = Load<true>(buffer);
            Store<true>(buffer, _mm512_maddubs_epi16(_buffer, Load<true>(alpha)));
        }

        const __m512i K8_SHUFFLE_X2 = SIMD_MM512_SETR_EPI8(
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

        SIMD_INLINE void ResizerByteBilinearInterpolateX2(const uint8_t * alpha, uint8_t * buffer)
        {
            __m512i _buffer = _mm512_shuffle_epi8(Load<true>(buffer), K8_SHUFFLE_X2);
            Store<true>(buffer, _mm512_maddubs_epi16(_buffer, Load<true>(alpha)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<2>(const uint8_t * alpha, uint8_t * buffer)
        {
            ResizerByteBilinearInterpolateX2(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX2(alpha + A, buffer + A);
        }

        const __m512i K8_SHUFFLE_X3_00 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m512i K8_SHUFFLE_X3_01 = SIMD_MM512_SETR_EPI8(
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1,
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1,
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF,
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1);
        const __m512i K8_SHUFFLE_X3_02 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0);

        const __m512i K8_SHUFFLE_X3_10 = SIMD_MM512_SETR_EPI8(
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m512i K8_SHUFFLE_X3_11 = SIMD_MM512_SETR_EPI8(
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1,
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF,
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1,
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1);
        const __m512i K8_SHUFFLE_X3_12 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1);

        const __m512i K8_SHUFFLE_X3_20 = SIMD_MM512_SETR_EPI8(
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m512i K8_SHUFFLE_X3_21 = SIMD_MM512_SETR_EPI8(
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF,
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1,
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1,
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF);
        const __m512i K8_SHUFFLE_X3_22 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<3>(const uint8_t * alpha, uint8_t * buffer)
        {
            __m512i src[3], shuffled;
            src[0] = Load<true>(buffer + 0 * A);
            src[1] = Load<true>(buffer + 1 * A);
            src[2] = Load<true>(buffer + 2 * A);

            shuffled = _mm512_shuffle_epi8(_mm512_alignr_epi32(src[0], src[0], 12), K8_SHUFFLE_X3_00);
            shuffled = _mm512_or_si512(shuffled, _mm512_shuffle_epi8(src[0], K8_SHUFFLE_X3_01));
            shuffled = _mm512_or_si512(shuffled, _mm512_shuffle_epi8(_mm512_alignr_epi32(src[1], src[0], 4), K8_SHUFFLE_X3_02));
            Store<true>(buffer + 0 * A, _mm512_maddubs_epi16(shuffled, Load<true>(alpha + 0 * A)));

            shuffled = _mm512_shuffle_epi8(_mm512_alignr_epi32(src[1], src[0], 12), K8_SHUFFLE_X3_10);
            shuffled = _mm512_or_si512(shuffled, _mm512_shuffle_epi8(src[1], K8_SHUFFLE_X3_11));
            shuffled = _mm512_or_si512(shuffled, _mm512_shuffle_epi8(_mm512_alignr_epi32(src[2], src[1], 4), K8_SHUFFLE_X3_12));
            Store<true>(buffer + 1 * A, _mm512_maddubs_epi16(shuffled, Load<true>(alpha + 1 * A)));

            shuffled = _mm512_shuffle_epi8(_mm512_alignr_epi32(src[2], src[1], 12), K8_SHUFFLE_X3_20);
            shuffled = _mm512_or_si512(shuffled, _mm512_shuffle_epi8(src[2], K8_SHUFFLE_X3_21));
            shuffled = _mm512_or_si512(shuffled, _mm512_shuffle_epi8(_mm512_alignr_epi32(src[2], src[2], 4), K8_SHUFFLE_X3_22));
            Store<true>(buffer + 2 * A, _mm512_maddubs_epi16(shuffled, Load<true>(alpha + 2 * A)));
        }

        const __m512i K8_SHUFFLE_X4 = SIMD_MM512_SETR_EPI8(
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

        SIMD_INLINE void ResizerByteBilinearInterpolateX4(const uint8_t * alpha, uint8_t * buffer)
        {
            __m512i _buffer = _mm512_shuffle_epi8(Load<true>(buffer), K8_SHUFFLE_X4);
            Store<true>(buffer, _mm512_maddubs_epi16(_buffer, Load<true>(alpha)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<4>(const uint8_t * alpha, uint8_t * buffer)
        {
            ResizerByteBilinearInterpolateX4(alpha + 0 * A, buffer + 0 * A);
            ResizerByteBilinearInterpolateX4(alpha + 1 * A, buffer + 1 * A);
            ResizerByteBilinearInterpolateX4(alpha + 2 * A, buffer + 2 * A);
            ResizerByteBilinearInterpolateX4(alpha + 3 * A, buffer + 3 * A);
        }

        const __m512i K16_FRACTION_ROUND_TERM = SIMD_MM512_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        template<bool align> SIMD_INLINE __m512i ResizerByteBilinearInterpolateY(const uint8_t * pbx0, const uint8_t * pbx1, __m512i alpha[2])
        {
            __m512i sum = _mm512_add_epi16(_mm512_mullo_epi16(Load<align>(pbx0), alpha[0]), _mm512_mullo_epi16(Load<align>(pbx1), alpha[1]));
            return _mm512_srli_epi16(_mm512_add_epi16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool align> SIMD_INLINE void ResizerByteBilinearInterpolateY(const uint8_t * bx0, const uint8_t * bx1, __m512i alpha[2], uint8_t * dst)
        {
            __m512i lo = ResizerByteBilinearInterpolateY<align>(bx0 + 0, bx1 + 0, alpha);
            __m512i hi = ResizerByteBilinearInterpolateY<align>(bx0 + A, bx1 + A, alpha);
            Store<false>(dst, _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(lo, hi)));
        }

        template <size_t N> SIMD_INLINE void ResizerByteBilinearGather(const uint8_t * src, const int * idx, size_t size, uint8_t * dst)
        {
            struct Src { uint8_t channels[N * 1]; };
            struct Dst { uint8_t channels[N * 2]; };
            const Src * s = (const Src *)src;
            Dst * d = (Dst*)dst;
            for (size_t i = 0; i < size; i++)
                d[i] = *(Dst *)(s + idx[i]);
        }

        template <> SIMD_INLINE void ResizerByteBilinearGather<2>(const uint8_t * src, const int * idx, size_t size, uint8_t * dst)
        {
            for (size_t i = 0; i < size; i += 16)
            {
#if defined(__GNUC__) &&  __GNUC__ < 6
                _mm512_storeu_si512(dst + 4 * i, _mm512_i32gather_epi32(_mm512_loadu_si512(idx + i), (const int *)src, 2));
#else
                _mm512_storeu_si512(dst + 4 * i, _mm512_i32gather_epi32(_mm512_loadu_si512(idx + i), src, 2));
#endif
            }
        }

        template <> SIMD_INLINE void ResizerByteBilinearGather<4>(const uint8_t * src, const int * idx, size_t size, uint8_t * dst)
        {
            for (size_t i = 0; i < size; i += 8)
            {
#if defined(__GNUC__) &&  __GNUC__ < 6
                _mm512_storeu_si512(dst + 8 * i, _mm512_i32gather_epi64(_mm256_loadu_si256((__m256i*)(idx + i)), (const long long int*)src, 4));
#else
                _mm512_storeu_si512(dst + 8 * i, _mm512_i32gather_epi64(_mm256_loadu_si256((__m256i*)(idx + i)), src, 4));
#endif
            }
        }

        template<size_t N> void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            struct One { uint8_t val[N * 1]; };
            struct Two { uint8_t val[N * 2]; };

            size_t size = 2 * _param.dstW*N;
            size_t aligned = AlignHi(size, DA) - DA;
            const size_t step = A * N;
            ptrdiff_t previous = -2;
            __m512i a[2];
            uint8_t * bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t * ax = _ax.data;
            const int32_t * ix = _ix.data;
            size_t dstW = _param.dstW;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = _mm512_set1_epi16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = _mm512_set1_epi16(int16_t(_ay[yDst]));

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
                    ResizerByteBilinearGather<N>(src + (sy + k)*srcStride, ix, dstW, bx[k]);

                    uint8_t * pbx = bx[k];
                    for (size_t i = 0; i < size; i += step)
                        ResizerByteBilinearInterpolateX<N>(ax + i, pbx + i);
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
            __m512i a[2];
            uint8_t * bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t * ax = _ax.data;
            const Idx * ixg = _ixg.data;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = _mm512_set1_epi16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = _mm512_set1_epi16(int16_t(_ay[yDst]));

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
                        Avx2::ResizerByteBilinearLoadGrayInterpolated(psrc, ixg[i], ax, pdst);
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

        //---------------------------------------------------------------------------------------------

        ResizerShortBilinear::ResizerShortBilinear(const ResParam& param)
            : Avx2::ResizerShortBilinear(param)
        {
        }

        const __m512i RSB_1_0 = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, -1, -1, 0x4, 0x5, -1, -1, 0x8, 0x9, -1, -1, 0xC, 0xD, -1, -1,
            0x0, 0x1, -1, -1, 0x4, 0x5, -1, -1, 0x8, 0x9, -1, -1, 0xC, 0xD, -1, -1,
            0x0, 0x1, -1, -1, 0x4, 0x5, -1, -1, 0x8, 0x9, -1, -1, 0xC, 0xD, -1, -1,
            0x0, 0x1, -1, -1, 0x4, 0x5, -1, -1, 0x8, 0x9, -1, -1, 0xC, 0xD, -1, -1);
        const __m512i RSB_1_1 = SIMD_MM512_SETR_EPI8(
            0x2, 0x3, -1, -1, 0x6, 0x7, -1, -1, 0xA, 0xB, -1, -1, 0xE, 0xF, -1, -1,
            0x2, 0x3, -1, -1, 0x6, 0x7, -1, -1, 0xA, 0xB, -1, -1, 0xE, 0xF, -1, -1,
            0x2, 0x3, -1, -1, 0x6, 0x7, -1, -1, 0xA, 0xB, -1, -1, 0xE, 0xF, -1, -1,
            0x2, 0x3, -1, -1, 0x6, 0x7, -1, -1, 0xA, 0xB, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m512 BilColS1(const uint16_t* src, const int32_t* idx, __m512 fx0, __m512 fx1, __mmask16 tail = -1)
        {
            __m512i s = _mm512_mask_i32gather_epi32(K_ZERO, tail, _mm512_maskz_loadu_epi32(tail, idx), (int*)src, 2);
            __m512 m0 = _mm512_mul_ps(fx0, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_1_0)));
            __m512 m1 = _mm512_mul_ps(fx1, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_1_1)));
            return _mm512_add_ps(m0, m1);
        }

        const __m512i RSB_2_0 = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1);
        const __m512i RSB_2_1 = SIMD_MM512_SETR_EPI8(
            0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1,
            0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1,
            0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1,
            0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m512 BilColS2(const uint16_t* src, const int32_t* idx, __m512 fx0, __m512 fx1)
        {
            __m512i s = _mm512_setr_epi64(
                *(uint64_t*)(src + idx[0]), *(uint64_t*)(src + idx[2]),
                *(uint64_t*)(src + idx[4]), *(uint64_t*)(src + idx[6]),
                *(uint64_t*)(src + idx[8]), *(uint64_t*)(src + idx[10]),
                *(uint64_t*)(src + idx[12]), *(uint64_t*)(src + idx[14]));
            __m512 m0 = _mm512_mul_ps(fx0, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_2_0)));
            __m512 m1 = _mm512_mul_ps(fx1, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_2_1)));
            return _mm512_add_ps(m0, m1);
        }

        SIMD_INLINE __m512 BilColS2(const uint16_t* src, const int32_t* idx, __m512 fx0, __m512 fx1, __mmask16 tail)
        {
            __m512i s = _mm512_i64gather_epi64(_mm512_and_epi32(_mm512_maskz_loadu_epi32(tail, idx), K64_00000000FFFFFFFF), (long long int*)src, 2);
            __m512 m0 = _mm512_mul_ps(fx0, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_2_0)));
            __m512 m1 = _mm512_mul_ps(fx1, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_2_1)));
            return _mm512_add_ps(m0, m1);
        }

        const __m512i RSB_3_0 = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1, -1, -1);
        const __m512i RSB_3_1 = SIMD_MM512_SETR_EPI8(
            0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1,
            0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1,
            0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1,
            0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1);
        const __m512i RSB_3_P1 = SIMD_MM512_SETR_EPI32(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 15, 15, 15);

        SIMD_INLINE __m512 BilColS3(const uint16_t* src, const int32_t* idx, __m512 fx0, __m512 fx1)
        {
            __m512i s = Load<false>((__m128i*)(src + idx[0]), (__m128i*)(src + idx[3]), (__m128i*)(src + idx[6]), (__m128i*)(src + idx[9]));
            __m512 m0 = _mm512_mul_ps(fx0, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_3_0)));
            __m512 m1 = _mm512_mul_ps(fx1, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_3_1)));
            return _mm512_permutexvar_ps(RSB_3_P1, _mm512_add_ps(m0, m1));
        }

        const __m512i RSB_4_0 = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1,
            0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1);
        const __m512i RSB_4_1 = SIMD_MM512_SETR_EPI8(
            0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1,
            0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1,
            0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1,
            0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m512 BilColS4(const uint16_t* src, const int32_t* idx, __m512 fx0, __m512 fx1)
        {
            __m512i s = Load<false>((__m128i*)(src + idx[0]), (__m128i*)(src + idx[4]), (__m128i*)(src + idx[8]), (__m128i*)(src + idx[12]));
            __m512 m0 = _mm512_mul_ps(fx0, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_4_0)));
            __m512 m1 = _mm512_mul_ps(fx1, _mm512_cvtepi32_ps(_mm512_shuffle_epi8(s, RSB_4_1)));
            return _mm512_add_ps(m0, m1);
        }

        template<size_t N> void ResizerShortBilinear::RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N;
            float* pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rs12 = AlignLoAny(rs - 1, 12);
            size_t rs16 = AlignLo(rs, 16);
            size_t rs32 = AlignLo(rs, 32);
            __mmask16 tail16 = TailMask16(rs - rs16);
            __m512 _1 = _mm512_set1_ps(1.0f);
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
                    float* pb = pbx[k];
                    const uint16_t* ps = src + (sy + k) * srcStride;
                    size_t dx = 0;
                    if (N == 1)
                    {
                        for (; dx < rs16; dx += 16)
                        {
                            __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            _mm512_storeu_ps(pb + dx, BilColS1(ps, _ix.data + dx, fx0, fx1));
                        }
                        if (dx < rs)
                        {
                            __m512 fx1 = _mm512_maskz_loadu_ps(tail16, _ax.data + dx);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            _mm512_mask_storeu_ps(pb + dx, tail16, BilColS1(ps, _ix.data + dx, fx0, fx1, tail16));
                        }
                    }
                    if (N == 2)
                    {
                        for (; dx < rs16; dx += 16)
                        {
                            __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            _mm512_storeu_ps(pb + dx, BilColS2(ps, _ix.data + dx, fx0, fx1));
                        }
                        if (dx < rs)
                        {
                            __m512 fx1 = _mm512_maskz_loadu_ps(tail16, _ax.data + dx);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            _mm512_mask_storeu_ps(pb + dx, tail16, BilColS2(ps, _ix.data + dx, fx0, fx1, tail16));
                        }
                    }
                    if (N == 3)
                    {
                        for (; dx < rs12; dx += 12)
                        {
                            __m512 fx1 = Load<false>(_ax.data + dx, _ax.data + dx + 3, _ax.data + dx + 6, _ax.data + dx + 9);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            _mm512_storeu_ps(pb + dx, BilColS3(ps, _ix.data + dx, fx0, fx1));
                        }
                        for (; dx < rs; dx += 3)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_mm512_castps512_ps128(_1), fx1);
                            _mm_storeu_ps(pb + dx, Sse41::BilColS3(ps + _ix[dx], fx0, fx1));
                        }
                    }
                    if (N == 4)
                    {
                        for (; dx < rs16; dx += 16)
                        {
                            __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            _mm512_storeu_ps(pb + dx, BilColS4(ps, _ix.data + dx, fx0, fx1));
                        }
                        for (; dx < rs; dx += 4)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_mm512_castps512_ps128(_1), fx1);
                            _mm_storeu_ps(pb + dx, Sse41::BilColS4(ps + _ix[dx], fx0, fx1));
                        }
                    }
                }

                size_t dx = 0;
                __m512 _fy0 = _mm512_set1_ps(fy0);
                __m512 _fy1 = _mm512_set1_ps(fy1);
                for (; dx < rs32; dx += 32)
                {
                    __m512 m00 = _mm512_mul_ps(_mm512_loadu_ps(pbx[0] + dx + 0), _fy0);
                    __m512 m01 = _mm512_mul_ps(_mm512_loadu_ps(pbx[1] + dx + 0), _fy1);
                    __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m00, m01));
                    __m512 m10 = _mm512_mul_ps(_mm512_loadu_ps(pbx[0] + dx + 16), _fy0);
                    __m512 m11 = _mm512_mul_ps(_mm512_loadu_ps(pbx[1] + dx + 16), _fy1);
                    __m512i i1 = _mm512_cvttps_epi32(_mm512_add_ps(m10, m11));
                    _mm512_storeu_si512(dst + dx, PackU32ToI16(i0, i1));
                }
                for (; dx < rs16; dx += 16)
                {
                    __m512 m0 = _mm512_mul_ps(_mm512_loadu_ps(pbx[0] + dx), _fy0);
                    __m512 m1 = _mm512_mul_ps(_mm512_loadu_ps(pbx[1] + dx), _fy1);
                    __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m0, m1));
                    _mm256_storeu_si256((__m256i*)(dst + dx), _mm512_castsi512_si256(PackU32ToI16(i0)));
                }
                if (dx < rs)
                {
                    __m512 m0 = _mm512_mul_ps(_mm512_maskz_loadu_ps(tail16, pbx[0] + dx), _fy0);
                    __m512 m1 = _mm512_mul_ps(_mm512_maskz_loadu_ps(tail16, pbx[1] + dx), _fy1);
                    __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m0, m1));
                    _mm256_mask_storeu_epi16((__m256i*)(dst + dx), tail16, _mm512_castsi512_si256(PackU32ToI16(i0)));
                }
            }
        }

        const __m512i RSB_3_P2 = SIMD_MM512_SETR_EPI32(0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15);

        SIMD_INLINE __m512i PackU32ToI16Rsb3(__m512i lo, __m512i hi)
        {
            return _mm512_permutexvar_epi32(RSB_3_P2, _mm512_packus_epi32(lo, hi));
        }

        template<size_t N> void ResizerShortBilinear::RunS(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N;
            size_t rs12 = AlignLoAny(rs - 1, 12);
            size_t rs24 = AlignLoAny(rs - 1, 24);
            size_t rs16 = AlignLo(rs, 16);
            size_t rs32 = AlignLo(rs, 32);
            __mmask16 tail16 = TailMask16(rs - rs16);
            __m512 _1 = _mm512_set1_ps(1.0f);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                const uint16_t* ps0 = src + (sy + 0) * srcStride;
                const uint16_t* ps1 = src + (sy + 1) * srcStride;
                size_t dx = 0;
                __m512 _fy0 = _mm512_set1_ps(fy0);
                __m512 _fy1 = _mm512_set1_ps(fy1);
                if (N == 1)
                {
                    for (; dx < rs32; dx += 32)
                    {
                        __m512 fx01 = _mm512_loadu_ps(_ax.data + dx + 0);
                        __m512 fx00 = _mm512_sub_ps(_1, fx01);
                        __m512 m00 = _mm512_mul_ps(BilColS1(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        __m512 m01 = _mm512_mul_ps(BilColS1(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m00, m01));
                        __m512 fx11 = _mm512_loadu_ps(_ax.data + dx + 16);
                        __m512 fx10 = _mm512_sub_ps(_1, fx11);
                        __m512 m10 = _mm512_mul_ps(BilColS1(ps0, _ix.data + dx + 16, fx10, fx11), _fy0);
                        __m512 m11 = _mm512_mul_ps(BilColS1(ps1, _ix.data + dx + 16, fx10, fx11), _fy1);
                        __m512i i1 = _mm512_cvttps_epi32(_mm512_add_ps(m10, m11));
                        _mm512_storeu_si512(dst + dx, PackU32ToI16(i0, i1));
                    }
                    for (; dx < rs16; dx += 16)
                    {
                        __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                        __m512 fx0 = _mm512_sub_ps(_1, fx1);
                        __m512 m0 = _mm512_mul_ps(BilColS1(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        __m512 m1 = _mm512_mul_ps(BilColS1(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m0, m1));
                        _mm256_storeu_si256((__m256i*)(dst + dx), _mm512_castsi512_si256(PackU32ToI16(i0)));
                    }
                    if (dx < rs)
                    {
                        __m512 fx1 = _mm512_maskz_loadu_ps(tail16, _ax.data + dx);
                        __m512 fx0 = _mm512_sub_ps(_1, fx1);
                        __m512 m0 = _mm512_mul_ps(BilColS1(ps0, _ix.data + dx, fx0, fx1, tail16), _fy0);
                        __m512 m1 = _mm512_mul_ps(BilColS1(ps1, _ix.data + dx, fx0, fx1, tail16), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m0, m1));
                        _mm256_mask_storeu_epi16(dst + dx, tail16, _mm512_castsi512_si256(PackU32ToI16(i0)));
                    }
                }
                if (N == 2)
                {
                    for (; dx < rs32; dx += 32)
                    {
                        __m512 fx01 = _mm512_loadu_ps(_ax.data + dx + 0);
                        __m512 fx00 = _mm512_sub_ps(_1, fx01);
                        __m512 m00 = _mm512_mul_ps(BilColS2(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        __m512 m01 = _mm512_mul_ps(BilColS2(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m00, m01));
                        __m512 fx11 = _mm512_loadu_ps(_ax.data + dx + 16);
                        __m512 fx10 = _mm512_sub_ps(_1, fx11);
                        __m512 m10 = _mm512_mul_ps(BilColS2(ps0, _ix.data + dx + 16, fx10, fx11), _fy0);
                        __m512 m11 = _mm512_mul_ps(BilColS2(ps1, _ix.data + dx + 16, fx10, fx11), _fy1);
                        __m512i i1 = _mm512_cvttps_epi32(_mm512_add_ps(m10, m11));
                        _mm512_storeu_si512(dst + dx, PackU32ToI16(i0, i1));
                    }
                    for (; dx < rs16; dx += 16)
                    {
                        __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                        __m512 fx0 = _mm512_sub_ps(_1, fx1);
                        __m512 m0 = _mm512_mul_ps(BilColS2(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        __m512 m1 = _mm512_mul_ps(BilColS2(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m0, m1));
                        _mm256_storeu_si256((__m256i*)(dst + dx), _mm512_castsi512_si256(PackU32ToI16(i0)));
                    }
                    if (dx < rs)
                    {
                        __m512 fx1 = _mm512_maskz_loadu_ps(tail16, _ax.data + dx);
                        __m512 fx0 = _mm512_sub_ps(_1, fx1);
                        __m512 m0 = _mm512_mul_ps(BilColS2(ps0, _ix.data + dx, fx0, fx1, tail16), _fy0);
                        __m512 m1 = _mm512_mul_ps(BilColS2(ps1, _ix.data + dx, fx0, fx1, tail16), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m0, m1));
                        _mm256_mask_storeu_epi16(dst + dx, tail16, _mm512_castsi512_si256(PackU32ToI16(i0)));
                    }
                }
                if (N == 3)
                {
                    for (; dx < rs24; dx += 24)
                    {
                        __m512 fx01 = Load<false>(_ax.data + dx + 0, _ax.data + dx + 3, _ax.data + dx + 6, _ax.data + dx + 9);
                        __m512 fx00 = _mm512_sub_ps(_1, fx01);
                        __m512 m00 = _mm512_mul_ps(BilColS3(ps0, _ix.data + dx, fx00, fx01), _fy0);
                        __m512 m01 = _mm512_mul_ps(BilColS3(ps1, _ix.data + dx, fx00, fx01), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m00, m01));
                        __m512 fx11 = Load<false>(_ax.data + dx + 12, _ax.data + dx + 15, _ax.data + dx + 18, _ax.data + dx + 21);
                        __m512 fx10 = _mm512_sub_ps(_1, fx11);
                        __m512 m10 = _mm512_mul_ps(BilColS3(ps0, _ix.data + dx + 12, fx10, fx11), _fy0);
                        __m512 m11 = _mm512_mul_ps(BilColS3(ps1, _ix.data + dx + 12, fx10, fx11), _fy1);
                        __m512i i1 = _mm512_cvttps_epi32(_mm512_add_ps(m10, m11));
                        _mm512_storeu_si512((__m512i*)(dst + dx), PackU32ToI16Rsb3(i0, i1));
                    }
                    for (; dx < rs12; dx += 12)
                    {
                        __m512 fx1 = Load<false>(_ax.data + dx, _ax.data + dx + 3, _ax.data + dx + 6, _ax.data + dx + 9);
                        __m512 fx0 = _mm512_sub_ps(_1, fx1);
                        __m512 m0 = _mm512_mul_ps(BilColS3(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        __m512 m1 = _mm512_mul_ps(BilColS3(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m0, m1));
                        _mm256_storeu_si256((__m256i*)(dst + dx), _mm512_castsi512_si256(PackU32ToI16(i0)));
                    }
                    for (; dx < rs; dx += 3)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_mm512_castps512_ps128(_1), fx1);
                        __m128 m0 = _mm_mul_ps(Sse41::BilColS3(ps0 + _ix[dx], fx0, fx1), _mm512_castps512_ps128(_fy0));
                        __m128 m1 = _mm_mul_ps(Sse41::BilColS3(ps1 + _ix[dx], fx0, fx1), _mm512_castps512_ps128(_fy1));
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, Sse2::K_ZERO));
                    }
                }
                if (N == 4)
                {
                    for (; dx < rs32; dx += 32)
                    {
                        __m512 fx01 = _mm512_loadu_ps(_ax.data + dx + 0);
                        __m512 fx00 = _mm512_sub_ps(_1, fx01);
                        __m512 m00 = _mm512_mul_ps(BilColS4(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        __m512 m01 = _mm512_mul_ps(BilColS4(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m00, m01));
                        __m512 fx11 = _mm512_loadu_ps(_ax.data + dx + 16);
                        __m512 fx10 = _mm512_sub_ps(_1, fx11);
                        __m512 m10 = _mm512_mul_ps(BilColS4(ps0, _ix.data + dx + 16, fx10, fx11), _fy0);
                        __m512 m11 = _mm512_mul_ps(BilColS4(ps1, _ix.data + dx + 16, fx10, fx11), _fy1);
                        __m512i i1 = _mm512_cvttps_epi32(_mm512_add_ps(m10, m11));
                        _mm512_storeu_si512((__m512i*)(dst + dx), PackU32ToI16(i0, i1));
                    }
                    for (; dx < rs16; dx += 16)
                    {
                        __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                        __m512 fx0 = _mm512_sub_ps(_1, fx1);
                        __m512 m0 = _mm512_mul_ps(BilColS4(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        __m512 m1 = _mm512_mul_ps(BilColS4(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        __m512i i0 = _mm512_cvttps_epi32(_mm512_add_ps(m0, m1));
                        _mm256_storeu_si256((__m256i*)(dst + dx), _mm512_castsi512_si256(PackU32ToI16(i0)));
                    }
                    for (; dx < rs; dx += 4)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_mm512_castps512_ps128(_1), fx1);
                        __m128 m0 = _mm_mul_ps(Sse41::BilColS4(ps0 + _ix[dx], fx0, fx1), _mm512_castps512_ps128(_fy0));
                        __m128 m1 = _mm_mul_ps(Sse41::BilColS4(ps1 + _ix[dx], fx0, fx1), _mm512_castps512_ps128(_fy1));
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, Sse2::K_ZERO));
                    }
                }
            }
        }

        void ResizerShortBilinear::Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            bool sparse = _param.dstH * 2.0 <= _param.srcH;
            switch (_param.channels)
            {
            case 1: sparse ? RunS<1>(src, srcStride, dst, dstStride) : RunB<1>(src, srcStride, dst, dstStride); return;
            case 2: sparse ? RunS<2>(src, srcStride, dst, dstStride) : RunB<2>(src, srcStride, dst, dstStride); return;
            case 3: sparse ? RunS<3>(src, srcStride, dst, dstStride) : RunB<3>(src, srcStride, dst, dstStride); return;
            case 4: sparse ? RunS<4>(src, srcStride, dst, dstStride) : RunB<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }
    }
#endif //SIMD_AVX512BW_ENABLE 
}

