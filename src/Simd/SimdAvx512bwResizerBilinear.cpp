/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdBFloat16.h"

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

        //-----------------------------------------------------------------------------------------

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
                        _mm_mask_storeu_epi16((__m128i*)(dst + dx), 0x7, _mm_packus_epi32(i0, Sse41::K_ZERO));
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
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, Sse41::K_ZERO));
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

        //-----------------------------------------------------------------------------------------

        ResizerFloatBilinear::ResizerFloatBilinear(const ResParam& param)
            : Avx2::ResizerFloatBilinear(param)
        {
            size_t cn = _param.channels, rs = _param.dstW * cn, fs = AlignLoAny(F, cn);
            _fastLoad1 = (_rowBuf && cn <= 8), _fastLoad2 = (_rowBuf && cn <= 8 && _param.dstW <= _param.srcW);
            if (_fastLoad1)
            {
                for (size_t dx = 0; dx < rs && _fastLoad1; dx += fs)
                {
                    for(size_t i = dx, n = Simd::Min(dx + fs, rs); i < n; ++i)
                        if (_ix[i] + cn - _ix[dx] >= F)
                            _fastLoad1 = false;
                }
            }
        }

        void ResizerFloatBilinear::Run(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            size_t cn = _param.channels, cnF = AlignLo(cn, F), cnT = cn - cnF;
            size_t dw = _param.dstW, dw2 = AlignLo(dw, 2), dw4 = AlignLo(dw, 4), dw8 = AlignLo(dw, 8), dw1 = dw - 1;
            __mmask16 cnMF = TailMask16(cn - cnF);
            __m512 _1 = _mm512_set1_ps(1.0f);
            __m512i _cn = _mm512_set1_epi32((int)cn);
            if (_rowBuf)
            {
                size_t rs = _param.dstW * cn, rs3 = rs - 3, rs6 = AlignLoAny(rs3, 6);
                float* pbx[2] = { _bx[0].data, _bx[1].data };
                int32_t prev = -2;
                size_t rsF = AlignLo(rs, F);
                __mmask16 rsMF = TailMask16(rs - rsF);
                size_t fs = AlignLoAny(F, cn), rsFS = (_fastLoad1 || _fastLoad2) ? AlignLoAny(rs, fs) : rs;
                __mmask16 rsMSM = TailMask16(fs), rsMST = TailMask16(rs - rsFS), rsMSTS = TailMask16(_param.srcW*cn - _ix[rsFS]);
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
                        const float* ps = src + (sy + k) * srcStride;
                        size_t dx = 0;
                        if (_fastLoad1)
                        {
                            for (; dx < rsFS; dx += fs)
                            {
                                size_t sx0 = _ix[dx];
                                __m512i idx = _mm512_sub_epi32(_mm512_loadu_si512(_ix.data + dx), _mm512_set1_epi32(sx0));
                               __m512 _src = _mm512_loadu_ps(ps + sx0);
                                __m512 s0 = _mm512_permutexvar_ps(idx, _src);
                                __m512 s1 = _mm512_permutexvar_ps(_mm512_add_epi32(idx, _cn), _src);
                                __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_mask_storeu_ps(pb + dx, rsMSM, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                            }
                            if (dx < rs)
                            {
                                size_t sx0 = _ix[dx];
                                __m512i idx = _mm512_sub_epi32(_mm512_maskz_loadu_epi32(rsMST, _ix.data + dx), _mm512_set1_epi32(sx0));
                                __m512 _src = _mm512_maskz_loadu_ps(rsMSTS, ps + sx0);
                                __m512 s0 = _mm512_permutexvar_ps(idx, _src);
                                __m512 s1 = _mm512_permutexvar_ps(_mm512_add_epi32(idx, _cn), _src);
                                __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_mask_storeu_ps(pb + dx, rsMST, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                            }
                        } 
                        else  if (_fastLoad2)
                        {
                            for (; dx < rsFS; dx += fs)
                            {
                                size_t sx0 = _ix[dx];
                                __m512i idx = _mm512_sub_epi32(_mm512_loadu_si512(_ix.data + dx), _mm512_set1_epi32(sx0));
                                __m512 s0 = _mm512_permutexvar_ps(idx, _mm512_loadu_ps(ps + sx0));
                                __m512 s1 = _mm512_permutexvar_ps(idx, _mm512_loadu_ps(ps + sx0 + cn));
                                __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_mask_storeu_ps(pb + dx, rsMSM, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                            }
                            if (dx < rs)
                            {
                                size_t sx0 = _ix[dx];
                                __m512i idx = _mm512_sub_epi32(_mm512_maskz_loadu_epi32(rsMST, _ix.data + dx), _mm512_set1_epi32(sx0));
                                __m512 s0 = _mm512_permutexvar_ps(idx, _mm512_maskz_loadu_ps(rsMSTS, ps + sx0));
                                __m512 s1 = _mm512_permutexvar_ps(idx, _mm512_maskz_loadu_ps(rsMSTS, ps + sx0 + cn));
                                __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_mask_storeu_ps(pb + dx, rsMST, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                            }
                        }
                        else if(cn > 8)
                        {
                            for (; dx < rs;)
                            {
                                const float* ps0 = ps + _ix[dx];
                                __m512 fx1 = _mm512_set1_ps(_ax[dx]);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                for (size_t eF = dx + cnF; dx < eF; dx += F, ps0 += F)
                                    _mm512_storeu_ps(pb + dx, _mm512_fmadd_ps(fx0, _mm512_loadu_ps(ps0), _mm512_mul_ps(fx1, _mm512_loadu_ps(ps0 + cn))));
                                if (cnMF)
                                {
                                    _mm512_mask_storeu_ps(pb + dx, cnMF, _mm512_fmadd_ps(fx0, _mm512_maskz_loadu_ps(cnMF, ps0), _mm512_mul_ps(fx1, _mm512_maskz_loadu_ps(cnMF, ps0 + cn))));
                                    dx += cnT;
                                }
                            }
                        }
                        else
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m512i i0 = _mm512_loadu_si512(_ix.data + dx);
                                __m512i i1 = _mm512_add_epi32(i0, _cn);
                                __m512 s0 = _mm512_i32gather_ps(i0, ps, 4);
                                __m512 s1 = _mm512_i32gather_ps(i1, ps, 4);
                                __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_storeu_ps(pb + dx, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                            }
                            if (dx < rs)
                            {
                                __m512i i0 = _mm512_maskz_loadu_epi32(rsMF, _ix.data + dx);
                                __m512i i1 = _mm512_add_epi32(i0, _cn);
                                __m512 s0 = _mm512_i32gather_ps(i0, ps, 4);
                                __m512 s1 = _mm512_i32gather_ps(i1, ps, 4);
                                __m512 fx1 = _mm512_maskz_loadu_ps(rsMF, _ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_mask_storeu_ps(pb + dx, rsMF, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                            }
                        }
                    }
                    size_t dx = 0;
                    __m512 _fy0 = _mm512_set1_ps(fy0);
                    __m512 _fy1 = _mm512_set1_ps(fy1);
                    for (; dx < rsF; dx += F)
                    {
                        __m512 b0 = _mm512_loadu_ps(pbx[0] + dx);
                        __m512 b1 = _mm512_loadu_ps(pbx[1] + dx);
                        _mm512_storeu_ps(dst + dx, _mm512_fmadd_ps(b0, _fy0, _mm512_mul_ps(b1, _fy1)));
                    }
                    if (dx < rs)
                    {
                        __m512 b0 = _mm512_maskz_loadu_ps(rsMF, pbx[0] + dx);
                        __m512 b1 = _mm512_maskz_loadu_ps(rsMF, pbx[1] + dx);
                        _mm512_mask_storeu_ps(dst + dx, rsMF, _mm512_fmadd_ps(b0, _fy0, _mm512_mul_ps(b1, _fy1)));
                    }
                }
            }
            else
            {
                if (cn <= 8)
                {
                    Avx2::ResizerFloatBilinear::Run(src, srcStride, dst, dstStride);
                    return;
                }
                for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                {
                    __m512 fy1 = _mm512_set1_ps(_ay[dy]);
                    __m512 fy0 = _mm512_sub_ps(_1, fy1);
                    const float* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                    {
                        for (size_t dx = 0; dx < dw; dx++)
                        {
                            size_t os = _ix[dx], eF = os + cnF, od = dx * cn;
                            __m512 fx1 = _mm512_set1_ps(_ax[dx]);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            for (; os < eF; os += F, od += F)
                            {
                                __m512 r0 = _mm512_fmadd_ps(_mm512_loadu_ps(src0 + os), fx0, _mm512_mul_ps(_mm512_loadu_ps(src0 + os + cn), fx1));
                                __m512 r1 = _mm512_fmadd_ps(_mm512_loadu_ps(src1 + os), fx0, _mm512_mul_ps(_mm512_loadu_ps(src1 + os + cn), fx1));
                                _mm512_storeu_ps(dst + od, _mm512_fmadd_ps(r0, fy0, _mm512_mul_ps(r1, fy1)));
                            }
                            if (cnT)
                            {
                                __m512 r0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(cnMF, src0 + os), fx0, _mm512_mul_ps(_mm512_maskz_loadu_ps(cnMF, src0 + os + cn), fx1));
                                __m512 r1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(cnMF, src1 + os), fx0, _mm512_mul_ps(_mm512_maskz_loadu_ps(cnMF, src1 + os + cn), fx1));
                                _mm512_mask_storeu_ps(dst + od, cnMF, _mm512_fmadd_ps(r0, fy0, _mm512_mul_ps(r1, fy1)));
                            }
                        }
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        ResizerBf16Bilinear::ResizerBf16Bilinear(const ResParam& param)
            : Avx2::ResizerBf16Bilinear(param)
        {
        }

        __m512i K8_IDX_20 = SIMD_MM512_SETR_EPI8(
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB,
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB,
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB,
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB);
        __m512i K8_IDX_21 = SIMD_MM512_SETR_EPI8(
            -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF,
            -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF,
            -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF,
            -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF);

        const __m512i K8_IDX_30 = SIMD_MM512_SETR_EPI8(
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1,
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1,
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1,
            -1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1);
        const __m512i K8_IDX_31 = SIMD_MM512_SETR_EPI8(
            -1, -1, 0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1,
            -1, -1, 0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1,
            -1, -1, 0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1,
            -1, -1, 0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1);

        SIMD_INLINE __m128 BilinearRowSumBf16(const uint16_t* src, size_t channels, __m128 fx0, __m128 fx1)
        {
            __m128 s0 = Sse41::BFloat16ToFloat32(Sse41::UnpackU16<0>(_mm_loadl_epi64((__m128i*)src)));
            __m128 s1 = Sse41::BFloat16ToFloat32(Sse41::UnpackU16<0>(_mm_loadl_epi64((__m128i*)(src + channels))));
            return _mm_fmadd_ps(fx0, s0, _mm_mul_ps(fx1, s1));
        }

        SIMD_INLINE __m256 BilinearRowSumBf16(const uint16_t* src, __mmask8 mask, size_t channels, __m256 fx0, __m256 fx1)
        {
            __m256 s0 = Avx2::BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_maskz_loadu_epi16(mask, src)));
            __m256 s1 = Avx2::BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_maskz_loadu_epi16(mask, src + channels)));
            return _mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1));
        }

        SIMD_INLINE __m512 BilinearRowSumBf16(const uint16_t* src, __mmask16 mask, size_t channels, __m512 fx0, __m512 fx1)
        {
            __m512 s0 = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask, src)));
            __m512 s1 = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask, src + channels)));
            return _mm512_fmadd_ps(fx0, s0, _mm512_mul_ps(fx1, s1));
        }

        SIMD_INLINE __m512 BilinearRowSumBf16(const uint16_t* src0, const uint16_t* src1, __mmask8 mask, size_t channels, __m512 fx0, __m512 fx1)
        {
            __m256i _src0 = _mm256_inserti32x4(_mm256_castsi128_si256(_mm_maskz_loadu_epi16(mask, src0)), _mm_maskz_loadu_epi16(mask, src1), 1);
            __m256i _src1 = _mm256_inserti32x4(_mm256_castsi128_si256(_mm_maskz_loadu_epi16(mask, src0 + channels)), _mm_maskz_loadu_epi16(mask, src1 + channels), 1);
            __m512 s0 = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_src0));
            __m512 s1 = BFloat16ToFloat32(_mm512_cvtepu16_epi32(_src1));
            return _mm512_fmadd_ps(fx0, s0, _mm512_mul_ps(fx1, s1));
        }

        void ResizerBf16Bilinear::Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t cn = _param.channels, cnD = AlignLo(cn, DF), cnF = AlignLo(cn, F), cnH = AlignLo(cn, F / 2);
            __mmask16 cnMF = TailMask16(cn - cnF);
            __mmask8 cnMH = TailMask8(cn - cnH);
            __m512 _1 = _mm512_set1_ps(1.0f);
            if (_rowBuf)
            {
                size_t rs = _param.dstW * cn, rsQ = AlignLo(rs, Sse41::F), rsH = AlignLo(rs, Avx2::F), rsF = AlignLo(rs, F), rsD = AlignLo(rs, DF);
                size_t rs3 = AlignLoAny(rs - 1, 3), rs6 = AlignLoAny(rs - 1, 6), rs12 = AlignLoAny(rs - 1, 12);
                __mmask16 rsMF = TailMask16(rs - rsF);
                float* pbx[2] = { _bx[0].data, _bx[1].data };
                int32_t prev = -2;
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
                        if (cn == 1)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m512i _src = _mm512_i32gather_epi32(_mm512_loadu_si512(_ix.data + dx), (int*)ps, 2);
                                __m512 s0 = BFloat16ToFloat32Even(_src);
                                __m512 s1 = BFloat16ToFloat32Odd(_src);
                                __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_storeu_ps(pb + dx, _mm512_fmadd_ps(fx0, s0, _mm512_mul_ps(fx1, s1)));
                            }
                            if (dx < rs)
                            {
                                __m512i _src = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), rsMF, _mm512_maskz_loadu_epi32(rsMF, _ix.data + dx), (int*)ps, 2);
                                __m512 s0 = BFloat16ToFloat32Even(_src);
                                __m512 s1 = BFloat16ToFloat32Odd(_src);
                                __m512 fx1 = _mm512_maskz_loadu_ps(rsMF, _ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_mask_storeu_ps(pb + dx, rsMF, _mm512_fmadd_ps(fx0, s0, _mm512_mul_ps(fx1, s1)));
                            }
                        }
                        else if (cn == 2)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m512i _src = _mm512_i64gather_epi64(_mm512_and_si512(_mm512_loadu_si512(_ix.data + dx), K64_00000000FFFFFFFF), (int*)ps, 2);
                                __m512 s0 = _mm512_castsi512_ps(_mm512_shuffle_epi8(_src, K8_IDX_20));
                                __m512 s1 = _mm512_castsi512_ps(_mm512_shuffle_epi8(_src, K8_IDX_21));
                                __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_storeu_ps(pb + dx, _mm512_fmadd_ps(fx0, s0, _mm512_mul_ps(fx1, s1)));
                            }
                            if (dx < rs)
                            {
                                __m512i _src = _mm512_i64gather_epi64(_mm512_and_si512(_mm512_maskz_loadu_epi32(rsMF, _ix.data + dx), K64_00000000FFFFFFFF), (int*)ps, 2);
                                __m512 s0 = _mm512_castsi512_ps(_mm512_shuffle_epi8(_src, K8_IDX_20));
                                __m512 s1 = _mm512_castsi512_ps(_mm512_shuffle_epi8(_src, K8_IDX_21));
                                __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_storeu_ps(pb + dx, _mm512_fmadd_ps(fx0, s0, _mm512_mul_ps(fx1, s1)));
                            }
                        }
                        else if (cn == 3 && rs >= 3)
                        {
                            for (; dx < rs12; dx += 12)
                            {
                                const float *pax = _ax.data + dx;
                                __m512 fx1 = Load<false>(pax + 0, pax + 3, pax + 6, pax + 9);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                __m512i _src = Load<false>((__m128i*)(ps + _ix[dx + 0]), (__m128i*)(ps + _ix[dx + 3]), (__m128i*)(ps + _ix[dx + 6]), (__m128i*)(ps + _ix[dx + 9]));
                                __m512 s0 = _mm512_castsi512_ps(_mm512_shuffle_epi8(_src, K8_IDX_30));
                                __m512 s1 = _mm512_castsi512_ps(_mm512_shuffle_epi8(_src, K8_IDX_31));
                                _mm512_storeu_ps(pb + dx, _mm512_permutexvar_ps(RSB_3_P1, _mm512_fmadd_ps(fx0, s0, _mm512_mul_ps(fx1, s1))));
                            }
                            for (; dx < rs6; dx += 6)
                            {
                                __m256 fx1 = Avx2::Load<false>(_ax.data + dx, _ax.data + dx + 3);
                                __m256 fx0 = _mm256_sub_ps(_mm512_castps512_ps256(_1), fx1);
                                __m256i _src = Avx2::Load<false>((__m128i*)(ps + _ix[dx + 0]), (__m128i*)(ps + _ix[dx + 3]));
                                __m256 s0 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_src, _mm512_castsi512_si256(K8_IDX_30)));
                                __m256 s1 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_src, _mm512_castsi512_si256(K8_IDX_31)));
                                _mm256_storeu_ps(pb + dx, _mm256_permutevar8x32_ps(_mm256_fmadd_ps(fx0, s0, _mm256_mul_ps(fx1, s1)), _mm512_castsi512_si256(RSB_3_P1)));
                            }
                            for (; dx < rs3; dx += 3)
                            {
                                __m128 fx1 = _mm_set1_ps(_ax.data[dx]);
                                __m128 fx0 = _mm_sub_ps(_mm512_castps512_ps128(_1), fx1);
                                _mm_storeu_ps(pb + dx, BilinearRowSumBf16(ps + _ix[dx], cn, fx0, fx1));
                            }
                            for (; dx < rs; dx++)
                            {
                                int32_t sx = _ix[dx];
                                float fx = _ax[dx];
                                pb[dx] = Base::BFloat16ToFloat32(ps[sx]) * (1.0f - fx) + Base::BFloat16ToFloat32(ps[sx + cn]) * fx;
                            }
                        }
                        else if (cn == 4)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m512i _src = Load<false>((__m128i*)(ps + _ix[dx + 0]), (__m128i*)(ps + _ix[dx + 4]), (__m128i*)(ps + _ix[dx + 8]), (__m128i*)(ps + _ix[dx + 12]));
                                __m512 fx1 = _mm512_loadu_ps(_ax.data + dx);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                _mm512_storeu_ps(pb + dx, _mm512_fmadd_ps(fx0, _mm512_castsi512_ps(UnpackU16<0>(K_ZERO, _src)), _mm512_mul_ps(fx1, _mm512_castsi512_ps(UnpackU16<1>(K_ZERO, _src)))));
                            }
                            for (; dx < rs; dx += 4)
                            {
                                __m128 fx1 = _mm_set1_ps(_ax[dx]);
                                __m128 fx0 = _mm_sub_ps(_mm512_castps512_ps128(_1), fx1);
                                __m128i _src = _mm_loadu_si128((__m128i*)(ps + _ix[dx]));
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, Sse41::BFloat16ToFloat32<0>(_src)), _mm_mul_ps(fx1, Sse41::BFloat16ToFloat32<1>(_src))));
                            }
                        }
                        else if (cn <= 8)
                        {
                            for (; dx < rs; dx += cn)
                            {
                                __m256 fx1 = _mm256_set1_ps(_ax[dx]);
                                __m256 fx0 = _mm256_sub_ps(_mm512_castps512_ps256(_1), fx1);
                                _mm256_mask_storeu_ps(pb + dx, cnMH, BilinearRowSumBf16(ps + _ix[dx], cnMH, cn, fx0, fx1));
                            }
                        }
                        else
                        {
                            size_t cnT = cn - cnF;
                            for (; dx < rs;)
                            {
                                const uint16_t* ps0 = ps + _ix[dx];
                                __m512 fx1 = _mm512_set1_ps(_ax[dx]);
                                __m512 fx0 = _mm512_sub_ps(_1, fx1);
                                for (size_t eF = dx + cnF; dx < eF; dx += F, ps0 += F)
                                    _mm512_storeu_ps(pb + dx, BilinearRowSumBf16(ps0, -1, cn, fx0, fx1));
                                if (cnMF)
                                    _mm512_mask_storeu_ps(pb + dx, cnMF, BilinearRowSumBf16(ps0, cnMF, cn, fx0, fx1)), dx += cnT;
                            }
                        }
                    }

                    size_t dx = 0;
                    __m512 _fy0 = _mm512_set1_ps(fy0);
                    __m512 _fy1 = _mm512_set1_ps(fy1);
                    for (; dx < rsD; dx += DF)
                    {
                        __m512i d0 = Float32ToBFloat16(_mm512_fmadd_ps(_mm512_loadu_ps(pbx[0] + dx + 0), _fy0, _mm512_mul_ps(_mm512_loadu_ps(pbx[1] + dx + 0), _fy1)));
                        __m512i d1 = Float32ToBFloat16(_mm512_fmadd_ps(_mm512_loadu_ps(pbx[0] + dx + F), _fy0, _mm512_mul_ps(_mm512_loadu_ps(pbx[1] + dx + F), _fy1)));
                        _mm512_storeu_si512((__m512i*)(dst + dx), _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, d1)));
                    }
                    for (; dx < rsF; dx += F)
                    {
                        __m512i d0 = Float32ToBFloat16(_mm512_fmadd_ps(_mm512_loadu_ps(pbx[0] + dx + 0), _fy0, _mm512_mul_ps(_mm512_loadu_ps(pbx[1] + dx + 0), _fy1)));
                        _mm256_storeu_si256((__m256i*)(dst + dx), _mm512_castsi512_si256(_mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, K_ZERO))));
                    }
                    if (dx < rs)
                    {
                        __m512i d0 = Float32ToBFloat16(_mm512_fmadd_ps(_mm512_loadu_ps(pbx[0] + dx + 0), _fy0, _mm512_mul_ps(_mm512_loadu_ps(pbx[1] + dx + 0), _fy1)));
                        _mm256_mask_storeu_epi16(dst + dx, rsMF, _mm512_castsi512_si256(_mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, K_ZERO))));
                    }
                }
            }
            else
            {
                if (cn >= DF)
                {
                    for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                    {
                        __m512 fy1 = _mm512_set1_ps(_ay[dy]);
                        __m512 fy0 = _mm512_sub_ps(_1, fy1);
                        const uint16_t* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                        for (size_t dx = 0; dx < _param.dstW; dx++)
                        {
                            size_t os = _ix[dx], eF = os + cnF, eD = os + cnD, od = dx * cn;
                            __m512 fx1 = _mm512_set1_ps(_ax[dx]);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            for (; os < eD; os += DF, od += DF)
                            {
                                __m512i s00 = _mm512_loadu_si512((__m512i*)(src0 + os));
                                __m512i s01 = _mm512_loadu_si512((__m512i*)(src0 + os + cn));
                                __m512i s10 = _mm512_loadu_si512((__m512i*)(src1 + os));
                                __m512i s11 = _mm512_loadu_si512((__m512i*)(src1 + os + cn));

                                __m512 r0e = _mm512_fmadd_ps(fx0, BFloat16ToFloat32Even(s00), _mm512_mul_ps(fx1, BFloat16ToFloat32Even(s01)));
                                __m512 r1e = _mm512_fmadd_ps(fx0, BFloat16ToFloat32Even(s10), _mm512_mul_ps(fx1, BFloat16ToFloat32Even(s11)));
                                __m512 even = _mm512_fmadd_ps(r0e, fy0, _mm512_mul_ps(r1e, fy1));

                                __m512 r0o = _mm512_fmadd_ps(fx0, BFloat16ToFloat32Odd(s00), _mm512_mul_ps(fx1, BFloat16ToFloat32Odd(s01)));
                                __m512 r1o = _mm512_fmadd_ps(fx0, BFloat16ToFloat32Odd(s10), _mm512_mul_ps(fx1, BFloat16ToFloat32Odd(s11)));
                                __m512 odd = _mm512_fmadd_ps(r0o, fy0, _mm512_mul_ps(r1o, fy1));

                                _mm512_storeu_si512((__m512i*)(dst + od), Float32ToBFloat16Interlived(even, odd));
                            }
                            for (; os < eF; os += F, od += F)
                            {
                                __m512 r0 = BilinearRowSumBf16(src0 + os, -1, cn, fx0, fx1);
                                __m512 r1 = BilinearRowSumBf16(src1 + os, -1, cn, fx0, fx1);
                                __m512i d0 = Float32ToBFloat16(_mm512_fmadd_ps(r0, fy0, _mm512_mul_ps(r1, fy1)));
                                _mm256_mask_storeu_epi16(dst + od, -1, _mm512_castsi512_si256(_mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, K_ZERO))));
                            }
                            if (cnMF)
                            {
                                __m512 r0 = BilinearRowSumBf16(src0 + os, cnMF, cn, fx0, fx1);
                                __m512 r1 = BilinearRowSumBf16(src1 + os, cnMF, cn, fx0, fx1);
                                __m512i d0 = Float32ToBFloat16(_mm512_fmadd_ps(r0, fy0, _mm512_mul_ps(r1, fy1)));
                                _mm256_mask_storeu_epi16(dst + od, cnMF, _mm512_castsi512_si256(_mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, K_ZERO))));
                            }
                        }
                    }
                }
                else if (cn <= HF)
                {
                    for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                    {
                        __m256 fy1 = _mm256_set1_ps(_ay[dy]);
                        __m256 fy0 = _mm256_sub_ps(_mm512_castps512_ps256(_1), fy1);
                        const uint16_t* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                        for (size_t dx = 0; dx < _param.dstW; dx++)
                        {
                            size_t os = _ix[dx], od = dx * cn;
                            __m256 fx1 = _mm256_set1_ps(_ax[dx]);
                            __m256 fx0 = _mm256_sub_ps(_mm512_castps512_ps256(_1), fx1);
                            __m256 r0 = BilinearRowSumBf16(src0 + os, cnMH, cn, fx0, fx1);
                            __m256 r1 = BilinearRowSumBf16(src1 + os, cnMH, cn, fx0, fx1);
                            __m256i d0 = Avx2::Float32ToBFloat16(_mm256_fmadd_ps(r0, fy0, _mm256_mul_ps(r1, fy1)));
                            _mm_mask_storeu_epi16(dst + od, cnMH, _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(d0, Avx2::K_ZERO), 0xD8)));
                        }
                    }
                }
                else
                {
                    for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                    {
                        __m512 fy1 = _mm512_set1_ps(_ay[dy]);
                        __m512 fy0 = _mm512_sub_ps(_1, fy1);
                        const uint16_t* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                        for (size_t dx = 0; dx < _param.dstW; dx++)
                        {
                            size_t os = _ix[dx], eF = os + cnF, od = dx * cn;
                            __m512 fx1 = _mm512_set1_ps(_ax[dx]);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            for (; os < eF; os += F, od += F)
                            {
                                __m512 r0 = BilinearRowSumBf16(src0 + os, -1, cn, fx0, fx1);
                                __m512 r1 = BilinearRowSumBf16(src1 + os, -1, cn, fx0, fx1);
                                __m512i d0 = Float32ToBFloat16(_mm512_fmadd_ps(r0, fy0, _mm512_mul_ps(r1, fy1)));
                                _mm256_mask_storeu_epi16(dst + od, -1, _mm512_castsi512_si256(_mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, K_ZERO))));
                            }
                            if (cnMF)
                            {
                                __m512 r0 = BilinearRowSumBf16(src0 + os, cnMF, cn, fx0, fx1);
                                __m512 r1 = BilinearRowSumBf16(src1 + os, cnMF, cn, fx0, fx1);
                                __m512i d0 = Float32ToBFloat16(_mm512_fmadd_ps(r0, fy0, _mm512_mul_ps(r1, fy1)));
                                _mm256_mask_storeu_epi16(dst + od, cnMF, _mm512_castsi512_si256(_mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi32(d0, K_ZERO))));
                            }
                        }
                    }
                }
            }
        }
    }
#endif 
}

