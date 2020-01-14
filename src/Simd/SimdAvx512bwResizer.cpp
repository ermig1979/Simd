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


        //---------------------------------------------------------------------

        ResizerByteArea::ResizerByteArea(const ResParam & param)
            : Avx2::ResizerByteArea(param)
        {
        }

        template<UpdateType update, bool mask> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, __m512i alpha, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i s0 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src0, tail)));
            __m512i i0 = UnpackU8<0>(s0);
            __m512i i1 = UnpackU8<1>(s0);
            Update<update, true>(dst + 0 * F, _mm512_madd_epi16(alpha, UnpackU8<0>(i0)));
            Update<update, true>(dst + 1 * F, _mm512_madd_epi16(alpha, UnpackU8<1>(i0)));
            Update<update, true>(dst + 2 * F, _mm512_madd_epi16(alpha, UnpackU8<0>(i1)));
            Update<update, true>(dst + 3 * F, _mm512_madd_epi16(alpha, UnpackU8<1>(i1)));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t size, size_t aligned, int32_t a, int32_t * dst, __mmask64 tail)
        {
            __m512i alpha = SetInt16(a, a);
            size_t i = 0;
            for (; i < aligned; i += A, dst += A, src0 += A)
                ResizerByteAreaRowUpdate<update, false>(src0, alpha, dst);
            if(i < size)
                ResizerByteAreaRowUpdate<update, true>(src0, alpha, dst, tail);
        }

        template<UpdateType update, bool mask> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, const uint8_t * src1, __m512i alpha, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i s0 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src0, tail)));
            __m512i s1 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src1, tail)));
            __m512i i0 = UnpackU8<0>(s0, s1);
            __m512i i1 = UnpackU8<1>(s0, s1);
            Update<update, true>(dst + 0 * F, _mm512_madd_epi16(alpha, UnpackU8<0>(i0)));
            Update<update, true>(dst + 1 * F, _mm512_madd_epi16(alpha, UnpackU8<1>(i0)));
            Update<update, true>(dst + 2 * F, _mm512_madd_epi16(alpha, UnpackU8<0>(i1)));
            Update<update, true>(dst + 3 * F, _mm512_madd_epi16(alpha, UnpackU8<1>(i1)));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t stride, size_t size, size_t aligned, int32_t a0, int32_t a1, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i alpha = SetInt16(a0, a1);
            const uint8_t * src1 = src0 + stride;
            size_t i = 0;
            for (; i < aligned; i += A, dst += A)
                ResizerByteAreaRowUpdate<update, false>(src0 + i, src1 + i, alpha, dst);
            if (i < size)
                ResizerByteAreaRowUpdate<update, true>(src0 + i, src1 + i, alpha, dst, tail);
        }

        template<UpdateType update, bool mask> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, const uint8_t * src1, 
            const uint8_t * src2, const uint8_t * src3, __m512i a01, __m512i a23, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i s0 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src0, tail)));
            __m512i s1 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src1, tail)));
            __m512i t010 = _mm512_maddubs_epi16(UnpackU8<0>(s0, s1), a01);
            __m512i t011 = _mm512_maddubs_epi16(UnpackU8<1>(s0, s1), a01);
            __m512i s2 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src2, tail)));
            __m512i s3 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src3, tail)));
            __m512i t230 = _mm512_maddubs_epi16(UnpackU8<0>(s2, s3), a23);
            __m512i t231 = _mm512_maddubs_epi16(UnpackU8<1>(s2, s3), a23);
            Update<update, true>(dst + 0 * F, _mm512_madd_epi16(K16_0001, UnpackU16<0>(t010, t230)));
            Update<update, true>(dst + 1 * F, _mm512_madd_epi16(K16_0001, UnpackU16<1>(t010, t230)));
            Update<update, true>(dst + 2 * F, _mm512_madd_epi16(K16_0001, UnpackU16<0>(t011, t231)));
            Update<update, true>(dst + 3 * F, _mm512_madd_epi16(K16_0001, UnpackU16<1>(t011, t231)));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t stride, size_t size, size_t aligned, int32_t a0, int32_t a12, int32_t a3, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i a01 = SetInt8(a0, a12);
            __m512i a23 = SetInt8(a12, a3);
            const uint8_t * src1 = src0 + stride;
            const uint8_t * src2 = src1 + stride;
            const uint8_t * src3 = src2 + stride;
            size_t i = 0;
            for (; i < aligned; i += A, dst += A)
                ResizerByteAreaRowUpdate<update, false>(src0 + i, src1 + i, src2 + i, src3 + i, a01, a23, dst);
            if (i < size)
                ResizerByteAreaRowUpdate<update, true>(src0 + i, src1 + i, src2 + i, src3 + i, a01, a23, dst, tail);
        }

        SIMD_INLINE void ResizerByteAreaRowSum(const uint8_t * src, size_t stride, size_t count, size_t size, size_t aligned, int32_t curr, int32_t zero, int32_t next, int32_t * dst, __mmask64 tail)
        {
            if (count)
            {
                size_t i = 0;
                ResizerByteAreaRowUpdate<UpdateSet>(src, stride, size, aligned, curr, count == 1 ? zero - next : zero, dst, tail), src += 2 * stride, i += 2;
                for (; i < count; i += 2, src += 2 * stride)
                    ResizerByteAreaRowUpdate<UpdateAdd>(src, stride, size, aligned, zero, i == count - 1 ? zero - next : zero, dst, tail);
                if (i == count)
                    ResizerByteAreaRowUpdate<UpdateAdd>(src, size, aligned, zero - next, dst, tail);
            }
            else
                ResizerByteAreaRowUpdate<UpdateSet>(src, size, aligned, curr - next, dst, tail);
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
            size_t rowSizeA = AlignLo(rowSize, A);
            __mmask64 tail = TailMask64(rowSize - rowSizeA);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t * buf = _by.data;
                size_t yn = iy[dy + 1] - iy[dy];
                ResizerByteAreaRowSum(src, srcStride, yn, rowSize, rowSizeA, ay[dy], ay0, ay[dy + 1], buf, tail), src += yn * srcStride;
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

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
        {
            ResParam param(srcX, srcY, dstX, dstY, channels, type, method, sizeof(__m512i));
            if (param.IsByteBilinear() && dstX >= A)
                return new ResizerByteBilinear(param);
            else if (param.IsByteArea())
                return new ResizerByteArea(param);
            else
                return Avx512f::ResizerInit(srcX, srcY, dstX, dstY, channels, type, method);
        }
    }
#endif //SIMD_AVX512BW_ENABLE 
}

