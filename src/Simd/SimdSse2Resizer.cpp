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
#include "Simd/SimdStore.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        ResizerByteBilinear::ResizerByteBilinear(const ResParam & param)
            : Base::ResizerByteBilinear(param)
        {
        }

        void ResizerByteBilinear::EstimateParams()
        {
            if (_ax.data)
                return;
            _ix.Resize(_param.dstW);
            _ax.Resize(AlignHi(_param.dstW, A) * 2);
            float scale = (float)_param.srcW / _param.dstW;
            for (size_t dx = 0; dx < _param.dstW; ++dx)
            {
                float a = (float)((dx + 0.5)*scale - 0.5);
                ptrdiff_t i = (ptrdiff_t)::floor(a);
                a -= i;
                if (i < 0)
                {
                    i = 0;
                    a = 0;
                }
                if (i > (ptrdiff_t)_param.srcW - 2)
                {
                    i = _param.srcW - 2;
                    a = 1;
                }
                _ix.data[dx] = (int32_t)i;
                _ax.data[2 * dx + 1] = (int16_t)(a * Base::FRACTION_RANGE + 0.5);
                _ax.data[2 * dx + 0] = (int16_t)(Base::FRACTION_RANGE - _ax.data[2 * dx + 1]);
            }
            size_t size = AlignHi(_param.dstW, A)*_param.channels * 2;
            _bx[0].Resize(size);
            _bx[1].Resize(size);
        }

        template <size_t N> void ResizerByteBilinearInterpolateX(const __m128i * alpha, __m128i * buffer);

        SIMD_INLINE void ResizerByteBilinearInterpolateX1(const __m128i * alpha, __m128i * buffer)
        {
            __m128i src = _mm_load_si128(buffer);
            __m128i lo = _mm_madd_epi16(_mm_unpacklo_epi8(src, K_ZERO), _mm_load_si128(alpha + 0));
            __m128i hi = _mm_madd_epi16(_mm_unpackhi_epi8(src, K_ZERO), _mm_load_si128(alpha + 1));
            _mm_store_si128(buffer, _mm_packs_epi32(lo, hi));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<1>(const __m128i * alpha, __m128i * buffer)
        {
            ResizerByteBilinearInterpolateX1(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX1(alpha + 2, buffer + 1);
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateX2(const __m128i * alpha, __m128i * buffer)
        {
            __m128i src = _mm_load_si128(buffer);
            __m128i a = _mm_load_si128(alpha);
            __m128i u = _mm_madd_epi16(_mm_and_si128(src, K16_00FF), a);
            __m128i v = _mm_madd_epi16(_mm_and_si128(_mm_srli_si128(src, 1), K16_00FF), a);
            _mm_store_si128(buffer, _mm_or_si128(u, _mm_slli_si128(v, 2)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<2>(const __m128i * alpha, __m128i * buffer)
        {
            ResizerByteBilinearInterpolateX2(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX2(alpha + 1, buffer + 1);
        }

        const __m128i K16_FRACTION_ROUND_TERM = SIMD_MM_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        template<bool align> SIMD_INLINE __m128i ResizerByteBilinearInterpolateY(const __m128i * pbx0, const __m128i * pbx1, __m128i alpha[2])
        {
            __m128i sum = _mm_add_epi16(_mm_mullo_epi16(Load<align>(pbx0), alpha[0]), _mm_mullo_epi16(Load<align>(pbx1), alpha[1]));
            return _mm_srli_epi16(_mm_add_epi16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool align> SIMD_INLINE void ResizerByteBilinearInterpolateY(const uint8_t * bx0, const uint8_t * bx1, __m128i alpha[2], uint8_t * dst)
        {
            __m128i lo = ResizerByteBilinearInterpolateY<align>((__m128i*)bx0 + 0, (__m128i*)bx1 + 0, alpha);
            __m128i hi = ResizerByteBilinearInterpolateY<align>((__m128i*)bx0 + 1, (__m128i*)bx1 + 1, alpha);
            Store<false>((__m128i*)dst, _mm_packus_epi16(lo, hi));
        }

        template<size_t N> void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            assert(_param.dstW >= A);

            struct One { uint8_t val[N * 1]; };
            struct Two { uint8_t val[N * 2]; };

            size_t size = 2 * _param.dstW*N;
            size_t aligned = AlignHi(size, DA) - DA;
            const size_t stepB = A / N;
            const size_t stepA = DA / N;
            size_t bufW = AlignHi(_param.dstW, stepB);

            ptrdiff_t previous = -2;
            __m128i a[2];
            uint8_t * pbx[2] = { _bx[0].data, _bx[1].data };

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = _mm_set1_epi16(int16_t(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = _mm_set1_epi16(int16_t(_ay[yDst]));

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    Two * pb = (Two *)pbx[k];
                    const One * ps = (const One *)(src + (sy + k)*srcStride);
                    for (size_t x = 0; x < _param.dstW; x++)
                        pb[x] = *(Two *)(ps + _ix[x]);

                    for (size_t ib = 0, ia = 0; ib < bufW; ib += stepB, ia += stepA)
                        ResizerByteBilinearInterpolateX<N>((__m128i*)(_ax.data + ia), (__m128i*)(pb + ib));
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(pbx[0] + ib, pbx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(pbx[0] + i, pbx[1] + i, a, dst + i / 2);
            }
        }

        void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            EstimateParams();
            switch (_param.channels)
            {
            case 1: Run<1>(src, srcStride, dst, dstStride); break;
            case 2: Run<2>(src, srcStride, dst, dstStride); break;
            default:
                assert(0);
            }        
        }

        //---------------------------------------------------------------------

        ResizerByteArea::ResizerByteArea(const ResParam & param)
            : Base::ResizerByteArea(param)
        {
            _by.Resize(AlignHi(_param.srcW*_param.channels, _param.align), false, _param.align);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t size, int32_t a, int32_t * dst)
        {
            __m128i alpha = SetInt16(a, a);
            for (size_t i = 0; i < size; i += A, dst += A, src0 += A)
            {
                __m128i s0 = _mm_loadu_si128((__m128i*)src0);
                __m128i i0 = UnpackU8<0>(s0);
                __m128i i1 = UnpackU8<1>(s0);
                Update<update, true>(dst + 0 * F, _mm_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t stride, size_t size, int32_t a0, int32_t a1, int32_t * dst)
        {
            __m128i alpha = SetInt16(a0, a1);
            const uint8_t * src1 = src0 + stride;
            for (size_t i = 0; i < size; i += A, dst += A)
            {
                __m128i s0 = _mm_loadu_si128((__m128i*)(src0 + i));
                __m128i s1 = _mm_loadu_si128((__m128i*)(src1 + i));
                __m128i i0 = UnpackU8<0>(s0, s1);
                __m128i i1 = UnpackU8<1>(s0, s1);
                Update<update, true>(dst + 0 * F, _mm_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
        }

        SIMD_INLINE void ResizerByteAreaRowSum(const uint8_t * src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, int32_t * dst)
        {
            if (count)
            {
                ResizerByteAreaRowUpdate<UpdateSet>(src, stride, size, curr, count == 1 ? zero - next : zero, dst), src += 2 * stride;
                for (size_t i = 2; i < count; i += 2, src += 2 * stride)
                    ResizerByteAreaRowUpdate<UpdateAdd>(src, stride, size, zero, i == count - 1 ? zero - next : zero, dst);
                if (!(count & 1))
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

        template<size_t N> void ResizerByteArea::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t dstW = _param.dstW, rowSize = _param.srcW*N, rowRest = dstStride - dstW*N;
            const int32_t * iy = _iy.data, * ix = _ix.data, * ay = _ay.data, * ax = _ax.data;
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

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
        {
            ResParam param(srcX, srcY, dstX, dstY, channels, type, method, sizeof(__m128i));
            if (param.IsByteBilinear() && (channels == 1 || channels == 2) && dstX >= A)
                return new ResizerByteBilinear(param);
            else if (param.IsByteArea())
                return new ResizerByteArea(param);
            else
                return Sse::ResizerInit(srcX, srcY, dstX, dstY, channels, type, method);
        }
    }
#endif//SIMD_SSE2_ENABLE
}

