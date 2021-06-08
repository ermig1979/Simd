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
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        ResizerByteArea::ResizerByteArea(const ResParam & param)
            : Sse2::ResizerByteArea(param)
        {
        }

        SIMD_INLINE __m128i SaveLoadTail(const uint8_t * ptr, size_t tail)
        {
            uint8_t buffer[DA];
            _mm_storeu_si128((__m128i*)(buffer), _mm_loadu_si128((__m128i*)(ptr + tail - A)));
            return _mm_loadu_si128((__m128i*)(buffer + A - tail));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaRowUpdate(const uint8_t * src0, size_t size, int32_t a, int32_t * dst)
        {
            __m128i alpha = SetInt16(a, a);
            size_t sizeA = AlignLo(size, A);
            size_t i = 0;
            for (; i < sizeA; i += A, dst += A)
            {
                __m128i s0 = _mm_loadu_si128((__m128i*)(src0 + i));
                __m128i i0 = UnpackU8<0>(s0);
                __m128i i1 = UnpackU8<1>(s0);
                Update<update, true>(dst + 0 * F, _mm_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
            if (i < size)
            {
                __m128i s0 = SaveLoadTail(src0 + i, size - i);
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
            size_t sizeA = AlignLo(size, A);
            size_t i = 0;
            for (; i < sizeA; i += A, dst += A)
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
            if (i < size)
            {
                __m128i s0 = _mm_loadu_si128((__m128i*)(src0 + i));
                __m128i s1 = SaveLoadTail(src1 + i, size - i);
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
                size_t i = 0;
                ResizerByteAreaRowUpdate<UpdateSet>(src, stride, size, curr, count == 1 ? zero - next : zero, dst), src += 2 * stride, i +=2;
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
            *(int32_t*)dst = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(res, K_ZERO), K_ZERO));
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

        ResizerShortBilinear::ResizerShortBilinear(const ResParam& param)
            : Base::ResizerShortBilinear(param)
        {
        }

        const __m128i RSB_4_0 = SIMD_MM_SETR_EPI8(0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1);
        const __m128i RSB_4_1 = SIMD_MM_SETR_EPI8(0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m128 BilColS4(const uint16_t* src, __m128 fx0, __m128 fx1)
        {
            __m128i s = _mm_loadu_si128((__m128i*)src);
            __m128 m0 = _mm_mul_ps(fx0, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_4_0)));
            __m128 m1 = _mm_mul_ps(fx1, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_4_1)));
            return _mm_add_ps(m0, m1);
        }

        template<size_t N> void ResizerShortBilinear::RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N;
            float* pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rs4 = AlignLo(rs, 4);
            size_t rs8 = AlignLo(rs, 8);
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
                    if (N == 4)
                    {
                        __m128 _1 = _mm_set1_ps(1.0f);
                        for (; dx < rs4; dx += 4)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            _mm_store_ps(pb + dx, BilColS4(ps + _ix[dx], fx0, fx1));
                        }
                    }
                    for (; dx < rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        float fx = _ax[dx];
                        pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + N] * fx;
                    }
                }

                size_t dx = 0;
                __m128 _fy0 = _mm_set1_ps(fy0);
                __m128 _fy1 = _mm_set1_ps(fy1);
                for (; dx < rs8; dx += 8)
                {
                    __m128 m00 = _mm_mul_ps(_mm_loadu_ps(pbx[0] + dx + 0), _fy0);
                    __m128 m01 = _mm_mul_ps(_mm_loadu_ps(pbx[1] + dx + 0), _fy1);
                    __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m00, m01));
                    __m128 m10 = _mm_mul_ps(_mm_loadu_ps(pbx[0] + dx + 4), _fy0);
                    __m128 m11 = _mm_mul_ps(_mm_loadu_ps(pbx[1] + dx + 4), _fy1);
                    __m128i i1 = _mm_cvttps_epi32(_mm_add_ps(m10, m11));
                    _mm_storeu_si128((__m128i*)(dst + dx), _mm_packus_epi32(i0, i1));
                }
                for (; dx < rs4; dx += 4)
                {
                    __m128 m0 = _mm_mul_ps(_mm_loadu_ps(pbx[0] + dx), _fy0);
                    __m128 m1 = _mm_mul_ps(_mm_loadu_ps(pbx[1] + dx), _fy1);
                    __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                    _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, K_ZERO));
                }
                for (; dx < rs; dx++)
                    dst[dx] = Round(pbx[0][dx] * fy0 + pbx[1][dx] * fy1);
            }
        }

        template<size_t N> void ResizerShortBilinear::RunS(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N;
            size_t rs4 = AlignLo(rs, 4);
            size_t rs8 = AlignLo(rs, 8);
            __m128 _1 = _mm_set1_ps(1.0f);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                const uint16_t* ps0 = src + (sy + 0) * srcStride;
                const uint16_t* ps1 = src + (sy + 1) * srcStride;
                size_t dx = 0;
                __m128 _fy0 = _mm_set1_ps(fy0);
                __m128 _fy1 = _mm_set1_ps(fy1);
                if (N == 4)
                {
                    for (; dx < rs8; dx += 8)
                    {
                        __m128 fx01 = _mm_loadu_ps(_ax.data + dx + 0);
                        __m128 fx00 = _mm_sub_ps(_1, fx01);
                        __m128 m00 = _mm_mul_ps(BilColS4(ps0 + _ix[dx + 0], fx00, fx01), _fy0);
                        __m128 m01 = _mm_mul_ps(BilColS4(ps1 + _ix[dx + 0], fx00, fx01), _fy1);
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m00, m01));
                        __m128 fx11 = _mm_loadu_ps(_ax.data + dx + 4);
                        __m128 fx10 = _mm_sub_ps(_1, fx11);
                        __m128 m10 = _mm_mul_ps(BilColS4(ps0 + _ix[dx + 4], fx10, fx11), _fy0);
                        __m128 m11 = _mm_mul_ps(BilColS4(ps1 + _ix[dx + 4], fx10, fx11), _fy1);
                        __m128i i1 = _mm_cvttps_epi32(_mm_add_ps(m10, m11));
                        _mm_storeu_si128((__m128i*)(dst + dx), _mm_packus_epi32(i0, i1));
                    }
                    for (; dx < rs4; dx += 4)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_1, fx1);
                        __m128 m0 = _mm_mul_ps(BilColS4(ps0 + _ix[dx], fx0, fx1), _fy0);
                        __m128 m1 = _mm_mul_ps(BilColS4(ps1 + _ix[dx], fx0, fx1), _fy1);
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, K_ZERO));
                    }
                }
                for (; dx < rs; dx++)
                {
                    int32_t sx = _ix[dx];
                    float fx1 = _ax[dx];
                    float fx0 = 1.0f - fx1;
                    float r0 = ps0[sx] * fx0 + ps0[sx + N] * fx1;
                    float r1 = ps1[sx] * fx0 + ps1[sx + N] * fx1;
                    dst[dx] = Round(r0 * fy0 + r1 * fy1);
                }
            }
        }

        void ResizerShortBilinear::Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            bool sparse = _param.dstH * 2.0 <= _param.srcH;
            switch (_param.channels)
            {
            //case 1: Run<1>(src, srcStride, dst, dstStride); return;
            //case 2: Run<2>(src, srcStride, dst, dstStride); return;
            //case 3: Run<3>(src, srcStride, dst, dstStride); return;
            case 4: sparse ? RunS<4>(src, srcStride, dst, dstStride) : RunB<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
        {
            ResParam param(srcX, srcY, dstX, dstY, channels, type, method, sizeof(__m128i));
            if (param.IsByteArea())
                return new ResizerByteArea(param);
            else if (param.IsShortBilinear() && channels == 4)
                return new ResizerShortBilinear(param);
            else
                return Ssse3::ResizerInit(srcX, srcY, dstX, dstY, channels, type, method);
        }
    }
#endif//SIMD_SSE41_ENABLE
}

