/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdWarpAffine.h"
#include "Simd/SimdWarpAffineCommon.h"
#include "Simd/SimdCopyPixel.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"

#include "Simd/SimdPoint.hpp"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
#if !defined(SIMD_AVX512_FLOOR_CEIL_ABSENT)
        template<int N> SIMD_INLINE void FillBorder(uint8_t* dst, int count, const __m512i& bv, const uint8_t* bs)
        {
            int i = 0, size = count * N, size64 = (int)AlignLo(size, 64);
            for (; i < size64; i += 64)
                _mm512_storeu_si512((__m512i*)(dst + i), bv);
            if (i < size)
            {
                __mmask64 mask = TailMask64(size - size64);
                _mm512_mask_storeu_epi8(dst + i, mask, bv);
            }
        }

        template<> SIMD_INLINE void FillBorder<3>(uint8_t* dst, int count, const __m512i& bv, const uint8_t* bs)
        {
            int i = 0, size = count * 3, size3 = size - 3;
            for (; i < size3; i += 3)
                Base::CopyPixel<4>(bs, dst + i);
            for (; i < size; i += 3)
                Base::CopyPixel<3>(bs, dst + i);
        }

        template<int N> SIMD_INLINE __m512i InitBorder(const uint8_t* border)
        {
            switch (N)
            {
            case 1: return _mm512_set1_epi8(*border);
            case 2: return _mm512_set1_epi16(*(uint16_t*)border);
            case 3: return _mm512_setzero_si512();
            case 4: return _mm512_set1_epi32(*(uint32_t*)border);
            }
            return _mm512_setzero_si512();
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m512i NearestOffset(__m512 x, __m512 y, const __m512* m, __m512i w, __m512i h, __m512i n, __m512i s)
        {
            __m512 dx = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, m[0]), _mm512_mul_ps(y, m[1])), m[2]);
            __m512 dy = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, m[3]), _mm512_mul_ps(y, m[4])), m[5]);
            __m512i ix = _mm512_min_epi32(_mm512_max_epi32(_mm512_cvtps_epi32(dx), _mm512_setzero_si512()), w);
            __m512i iy = _mm512_min_epi32(_mm512_max_epi32(_mm512_cvtps_epi32(dy), _mm512_setzero_si512()), h);
            return _mm512_add_epi32(_mm512_mullo_epi32(ix, n), _mm512_mullo_epi32(iy, s));
        }

        //-----------------------------------------------------------------------------------------

        template<int N, bool soft> SIMD_INLINE void NearestGather(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            int i = 0;
            for (; i < count; i++, dst += N)
                Base::CopyPixel<N>(src + offset[i], dst);
        }

        template<> SIMD_INLINE void NearestGather<3, true>(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            int i = 0, count1 = count - 1;
            for (; i < count1; i++, dst += 3)
                Base::CopyPixel<4>(src + offset[i], dst);
            if (i < count)
                Base::CopyPixel<3>(src + offset[i], dst);
        }

        template<> SIMD_INLINE void NearestGather<1, false>(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
                0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0x0, 0x4, 0x8, 0xC, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            int i = 0, count16 = (int)AlignLo(count, 16);
            for (; i < count16; i += 16, dst += 16)
            {
                __m512i _offs = _mm512_loadu_si512((__m512i*)(offset + i));
                __m512i _dst = _mm512_i32gather_epi32(_offs, src, 1);
                _dst = _mm512_permutexvar_epi32(PERMUTE, _mm512_shuffle_epi8(_dst, SHUFFLE));
                _mm_storeu_si128((__m128i*)dst, _mm512_castsi512_si128(_dst));
            }
            if(i < count)
            {
                __mmask16 mask = TailMask16(count - count16);
                __m512i _offs = _mm512_maskz_loadu_epi32(mask, offset + i);
                __m512i _dst = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _offs, src, 1);
                _dst = _mm512_permutexvar_epi32(PERMUTE, _mm512_shuffle_epi8(_dst, SHUFFLE));
                _mm_mask_storeu_epi8(dst, mask, _mm512_castsi512_si128(_dst));
            }
        }

        template<> SIMD_INLINE void NearestGather<2, false>(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI64(0x0, 0x2, 0x4, 0x6, 0, 0, 0, 0);
            int i = 0, count16 = (int)AlignLo(count, 16);
            for (; i < count16; i += 16, dst += 32)
            {
                __m512i _offs = _mm512_loadu_si512((__m512i*)(offset + i));
                __m512i _dst = _mm512_i32gather_epi32(_offs, src, 1);
                _dst = _mm512_permutexvar_epi64(PERMUTE, _mm512_shuffle_epi8(_dst, SHUFFLE));
                _mm256_storeu_si256((__m256i*)dst, _mm512_castsi512_si256(_dst));
            }
            if (i < count)
            {
                __mmask16 mask = TailMask16(count - count16);
                __m512i _offs = _mm512_maskz_loadu_epi32(mask, offset + i);
                __m512i _dst = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _offs, src, 1);
                _dst = _mm512_permutexvar_epi64(PERMUTE, _mm512_shuffle_epi8(_dst, SHUFFLE));
                _mm256_mask_storeu_epi16(dst, mask, _mm512_castsi512_si256(_dst));
            }
        }

        template<> SIMD_INLINE void NearestGather<3, false>(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, 0, 0, 0, 0);
            int i = 0, count16 = (int)AlignLo(count, 16), count1 = count - 1;
            for (; i < count16; i += 16, dst += 48)
            {
                __m512i _offs = _mm512_loadu_si512((__m512i*)(offset + i));
                __m512i _dst = _mm512_i32gather_epi32(_offs, src, 1);
                _dst = _mm512_permutexvar_epi32(PERMUTE, _mm512_shuffle_epi8(_dst, SHUFFLE));
                _mm512_mask_storeu_epi8(dst, 0x0000FFFFFFFFFFFF, _dst);
            }
            if (i < count)
            {
                __mmask16 srcMask = TailMask16(count - count16);
                __mmask64 dstMask = TailMask64((count - count16) * 3);
                __m512i _offs = _mm512_maskz_loadu_epi32(srcMask, offset + i);
                __m512i _dst = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), srcMask, _offs, src, 1);
                _dst = _mm512_permutexvar_epi32(PERMUTE, _mm512_shuffle_epi8(_dst, SHUFFLE));
                _mm512_mask_storeu_epi8(dst, dstMask, _dst);
            }
        }

        template<> SIMD_INLINE void NearestGather<4, false>(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            int i = 0, count16 = (int)AlignLo(count, 16);
            for (; i < count16; i += 16, dst += 64)
            {
                __m512i _offs = _mm512_loadu_si512((__m512i*)(offset + i));
                __m512i _dst = _mm512_i32gather_epi32(_offs, src, 1);
                _mm512_storeu_si512((__m512i*)dst, _dst);
            }
            if (i < count)
            {
                __mmask16 mask = TailMask16(count - count16);
                __m512i _offs = _mm512_maskz_loadu_epi32(mask, offset + i);
                __m512i _dst = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _offs, src, 1);
                _mm512_mask_storeu_epi32(dst, mask, _dst);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<int N, bool soft> void NearestRun(const WarpAffParam& p, int yBeg, int yEnd, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1;
            const __m512 _16 = _mm512_set1_ps(16.0f);
            static const __m512i _0123 = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF);
            __m512 _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = _mm512_set1_ps(p.inv[i]);
            __m512i _w = _mm512_set1_epi32(w);
            __m512i _h = _mm512_set1_epi32(h);
            __m512i _n = _mm512_set1_epi32(N);
            __m512i _s = _mm512_set1_epi32(s);
            __m512i _border = InitBorder<N>(p.border);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int nose = beg[y], tail = end[y];
                {
                    int x = nose;
                    __m512 _y = _mm512_cvtepi32_ps(_mm512_set1_epi32(y));
                    __m512 _x = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(x), _0123));
                    for (; x < tail; x += 16)
                    {
                        _mm512_storeu_si512((__m512i*)(buf + x), NearestOffset(_x, _y, _m, _w, _h, _n, _s));
                        _x = _mm512_add_ps(_x, _16);
                    }
                }
                if (fill)
                    FillBorder<N>(dst, nose, _border, p.border);
                NearestGather<N, soft>(src, buf + nose, tail - nose, dst + N * nose);
                if (fill)
                    FillBorder<N>(dst + tail * N, width - tail, _border, p.border);
                dst += p.dstS;
            }
        }

        //-------------------------------------------------------------------------------------------------

        WarpAffineNearest::WarpAffineNearest(const WarpAffParam& param)
            : Avx2::WarpAffineNearest(param)
        {
            bool soft = Avx2::SlowGather;
            switch (_param.channels)
            {
            case 1: _run = soft ? NearestRun<1, true> : NearestRun<1, false>; break;
            case 2: _run = soft ? NearestRun<2, true> : NearestRun<2, false>; break;
            case 3: _run = soft ? NearestRun<3, true> : NearestRun<3, false>; break;
            case 4: _run = soft ? NearestRun<4, true> : NearestRun<4, false>; break;
            }
        }

        void WarpAffineNearest::SetRange(const Base::Point* points)
        {
            const WarpAffParam& p = _param;
            int w = (int)p.dstW, h = (int)p.dstH, h16 = (int)AlignLo(h, 16);
            static const __m512i _0123 = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF);
            __m512i _w = _mm512_set1_epi32(w), _1 = _mm512_set1_epi32(1);
            int y = 0;
            for (; y < h16; y += 16)
            {
                _mm512_storeu_si512((__m512i*)(_beg.data + y), _w);
                _mm512_storeu_si512((__m512i*)(_end.data + y), _mm512_setzero_si512());
            }
            if(y < h)
            {
                __mmask16 tail = TailMask16(h - h16);
                _mm512_mask_storeu_epi32(_beg.data + y, tail, _w);
                _mm512_mask_storeu_epi32(_end.data + y, tail, _mm512_setzero_si512());
            }
            for (int v = 0; v < 4; ++v)
            {
                const Base::Point& curr = points[v];
                const Base::Point& next = points[(v + 1) & 3];
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), 0.0f);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), (float)p.dstH);
                int yBeg = Round(yMin);
                int yEnd = Round(yMax);
                int yEnd16 = (int)AlignLo(yEnd - yBeg, 16) + yBeg;
                __mmask16 tail = TailMask16(yEnd - yEnd16);
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                __m512 _a = _mm512_set1_ps(a);
                __m512 _b = _mm512_set1_ps(b);
                if (abs(a) <= 1.0f)
                {
                    int y = yBeg;
                    for (; y < yEnd16; y += 16)
                    {
                        __m512 _y = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(y), _0123));
                        __m512i _x = _mm512_cvtps_epi32(_mm512_add_ps(_mm512_mul_ps(_y, _a), _b));
                        __m512i xBeg = _mm512_loadu_si512((__m512i*)(_beg.data + y));
                        __m512i xEnd = _mm512_loadu_si512((__m512i*)(_end.data + y));
                        xBeg = _mm512_min_epi32(xBeg, _mm512_max_epi32(_x, _mm512_setzero_si512()));
                        xEnd = _mm512_max_epi32(xEnd, _mm512_min_epi32(_mm512_add_epi32(_x, _1), _w));
                        _mm512_storeu_si512((__m512i*)(_beg.data + y), xBeg);
                        _mm512_storeu_si512((__m512i*)(_end.data + y), xEnd);
                    }
                    if(y < yEnd)
                    {
                        __m512 _y = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(y), _0123));
                        __m512i _x = _mm512_cvtps_epi32(_mm512_add_ps(_mm512_mul_ps(_y, _a), _b));
                        __m512i xBeg = _mm512_loadu_si512((__m512i*)(_beg.data + y));
                        __m512i xEnd = _mm512_loadu_si512((__m512i*)(_end.data + y));
                        xBeg = _mm512_min_epi32(xBeg, _mm512_max_epi32(_x, _mm512_setzero_si512()));
                        xEnd = _mm512_max_epi32(xEnd, _mm512_min_epi32(_mm512_add_epi32(_x, _1), _w));
                        _mm512_mask_storeu_epi32(_beg.data + y, tail, xBeg);
                        _mm512_mask_storeu_epi32(_end.data + y, tail, xEnd);
                    }
                }
                else
                {
                    int y = yBeg;
                    __m512 _05 = _mm512_set1_ps(0.5f);
                    __m512 _yMin = _mm512_set1_ps(yMin);
                    __m512 _yMax = _mm512_set1_ps(yMax);
                    for (; y < yEnd16; y += 16)
                    {
                        __m512 _y = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(y), _0123));
                        __m512 yM = _mm512_min_ps(_mm512_max_ps(_mm512_sub_ps(_y, _05), _yMin), _yMax);
                        __m512 yP = _mm512_min_ps(_mm512_max_ps(_mm512_add_ps(_y, _05), _yMin), _yMax);
                        __m512 xM = _mm512_add_ps(_mm512_mul_ps(yM, _a), _b);
                        __m512 xP = _mm512_add_ps(_mm512_mul_ps(yP, _a), _b);
                        __m512i xBeg = _mm512_loadu_si512((__m512i*)(_beg.data + y));
                        __m512i xEnd = _mm512_loadu_si512((__m512i*)(_end.data + y));
                        xBeg = _mm512_min_epi32(xBeg, _mm512_max_epi32(_mm512_cvtps_epi32(_mm512_min_ps(xM, xP)), _mm512_setzero_si512()));
                        xEnd = _mm512_max_epi32(xEnd, _mm512_min_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(_mm512_max_ps(xM, xP)), _1), _w));
                        _mm512_storeu_si512((__m512i*)(_beg.data + y), xBeg);
                        _mm512_storeu_si512((__m512i*)(_end.data + y), xEnd);
                    }
                    if (y < yEnd)
                    {
                        __m512 _y = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(y), _0123));
                        __m512 yM = _mm512_min_ps(_mm512_max_ps(_mm512_sub_ps(_y, _05), _yMin), _yMax);
                        __m512 yP = _mm512_min_ps(_mm512_max_ps(_mm512_add_ps(_y, _05), _yMin), _yMax);
                        __m512 xM = _mm512_add_ps(_mm512_mul_ps(yM, _a), _b);
                        __m512 xP = _mm512_add_ps(_mm512_mul_ps(yP, _a), _b);
                        __m512i xBeg = _mm512_loadu_si512((__m512i*)(_beg.data + y));
                        __m512i xEnd = _mm512_loadu_si512((__m512i*)(_end.data + y));
                        xBeg = _mm512_min_epi32(xBeg, _mm512_max_epi32(_mm512_cvtps_epi32(_mm512_min_ps(xM, xP)), _mm512_setzero_si512()));
                        xEnd = _mm512_max_epi32(xEnd, _mm512_min_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(_mm512_max_ps(xM, xP)), _1), _w));
                        _mm512_mask_storeu_epi32(_beg.data + y, tail, xBeg);
                        _mm512_mask_storeu_epi32(_end.data + y, tail, xEnd);
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        const __m512i K32_WA_FRACTION_RANGE = SIMD_MM512_SET1_EPI32(Base::WA_FRACTION_RANGE);
        const __m512i K32_WA_BILINEAR_ROUND_TERM = SIMD_MM512_SET1_EPI32(Base::WA_BILINEAR_ROUND_TERM);

        SIMD_INLINE void ByteBilinearPrepMain16(__m512 x, __m512 y, const __m512* m, __m512i n, __m512i s, uint32_t* offs, uint8_t* fx, uint16_t* fy)
        {
            __m512 dx = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, m[0]), _mm512_mul_ps(y, m[1])), m[2]);
            __m512 dy = _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(x, m[3]), _mm512_mul_ps(y, m[4])), m[5]);
            __m512 ix = _mm512_floor_ps(dx);
            __m512 iy = _mm512_floor_ps(dy);
            __m512 range = _mm512_cvtepi32_ps(K32_WA_FRACTION_RANGE);
            __m512i _fx = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_sub_ps(dx, ix), range));
            __m512i _fy = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_sub_ps(dy, iy), range));
            _mm512_storeu_si512((__m512i*)offs, _mm512_add_epi32(_mm512_mullo_epi32(_mm512_cvtps_epi32(ix), n), _mm512_mullo_epi32(_mm512_cvtps_epi32(iy), s)));
            _fx = _mm512_or_si512(_mm512_sub_epi32(K32_WA_FRACTION_RANGE, _fx), _mm512_slli_epi32(_fx, 16));
            _fy = _mm512_or_si512(_mm512_sub_epi32(K32_WA_FRACTION_RANGE, _fy), _mm512_slli_epi32(_fy, 16));
            _mm256_storeu_si256((__m256i*)fx, _mm512_castsi512_si256(PackI16ToU8(_fx, _mm512_setzero_si512())));
            _mm512_storeu_si512((__m512i*)fy, _fy);
        }

        //-------------------------------------------------------------------------------------------------

        template<int N> SIMD_INLINE void ByteBilinearInterpEdge(int x, __m128 sy, const __m128* me, __m128i wh, __m128i ns, int s, const uint8_t* src, const uint8_t* brd, uint8_t* dst)
        {
            static const __m128i FX = SIMD_MM_SETR_EPI8(0x4, 0x0, 0x4, 0x0, 0x4, 0x0, 0x4, 0x0, 0x4, 0x0, 0x4, 0x0, 0x4, 0x0, 0x4, 0x0);
            static const __m128i FY = SIMD_MM_SETR_EPI8(0xC, -1, 0x8, -1, 0xC, -1, 0x8, -1, 0xC, -1, 0x8, -1, 0xC, -1, 0x8, -1);
            static const __m128i SRC = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);

            __m128 sx = _mm_cvtepi32_ps(_mm_set1_epi32(x));
            __m128 dxy = _mm_add_ps(_mm_add_ps(_mm_mul_ps(sx, me[0]), _mm_mul_ps(sy, me[1])), me[2]);
            __m128 fixy = _mm_floor_ps(dxy);
            __m128 range = _mm_cvtepi32_ps(_mm512_castsi512_si128(K32_WA_FRACTION_RANGE));
            __m128i fxy = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(dxy, fixy), range));
            fxy = _mm_unpacklo_epi32(fxy, _mm_sub_epi32(_mm512_castsi512_si128(K32_WA_FRACTION_RANGE), fxy));
            __m128i ixy = _mm_cvtps_epi32(fixy);

            __m128i offs = _mm_mullo_epi32(ixy, _mm_srli_si128(ns, 4));
            offs = _mm_add_epi32(ns, _mm_shuffle_epi32(_mm_hadd_epi32(offs, offs), 0));
            __m128i xy01 = _mm_unpacklo_epi32(_mm_cmplt_epi32(ixy, _mm_setzero_si128()), _mm_cmpgt_epi32(ixy, wh));
            __m128i mask = _mm_or_si128(_mm_shuffle_epi32(xy01, 0x44), _mm_shuffle_epi32(xy01, 0xFA));
            __m128i _src = _mm_mask_i32gather_epi32(_mm_set1_epi32(*(int*)brd), (int*)src, offs, _mm_andnot_si128(mask, Sse41::K_INV_ZERO), 1);

            _src = _mm_shuffle_epi8(_src, SRC);
            __m128i sum = _mm_maddubs_epi16(_src, _mm_shuffle_epi8(fxy, FX));
            sum = _mm_madd_epi16(sum, _mm_shuffle_epi8(fxy, FY));
            __m128i _dst = _mm_srli_epi32(_mm_add_epi32(sum, _mm512_castsi512_si128(K32_WA_BILINEAR_ROUND_TERM)), Base::WA_BILINEAR_SHIFT);
            _mm_mask_storeu_epi8(dst, (__mmask16(-1) >> (16 - N)), _mm_shuffle_epi8(_dst, SRC));
        }

        //-------------------------------------------------------------------------------------------------

        template<int N, bool soft> SIMD_INLINE void ByteBilinearGather(const uint8_t* src0, const uint8_t* src1, uint32_t* offset, int count, uint8_t* dst0, uint8_t* dst1)
        {
            int i = 0;
            for (; i < count; i++, dst0 += 2 * N, dst1 += 2 * N)
            {
                int offs = offset[i];
                Base::CopyPixel<N * 2>(src0 + offs, dst0);
                Base::CopyPixel<N * 2>(src1 + offs, dst1);
            }
        }

        template<> SIMD_INLINE void ByteBilinearGather<1, false>(const uint8_t* src0, const uint8_t* src1, uint32_t* offset, int count, uint8_t* dst0, uint8_t* dst1)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI64(0x0, 0x2, 0x4, 0x6, 0, 0, 0, 0);
            int i = 0, count16 = (int)AlignLo(count, 16);
            for (; i < count16; i += 16, dst0 += 32, dst1 += 32)
            {
                __m512i _offs = _mm512_loadu_si512((__m512i*)(offset + i));
                __m512i _dst0 = _mm512_shuffle_epi8(_mm512_i32gather_epi32(_offs, src0, 1), SHUFFLE);
                _mm256_storeu_si256((__m256i*)dst0, _mm512_castsi512_si256(_mm512_permutexvar_epi64(PERMUTE, _dst0)));
                __m512i _dst1 = _mm512_shuffle_epi8(_mm512_i32gather_epi32(_offs, src1, 1), SHUFFLE);
                _mm256_storeu_si256((__m256i*)dst1, _mm512_castsi512_si256(_mm512_permutexvar_epi64(PERMUTE, _dst1)));
            }
            if (i < count)
            {
                __mmask16 mask = __mmask16(-1) >> (16 + count16 - count);
                __m512i _offs = _mm512_maskz_loadu_epi32(mask, offset + i);
                __m512i _dst0 = _mm512_shuffle_epi8(_mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _offs, src0, 1), SHUFFLE);
                _mm256_mask_storeu_epi16(dst0, mask, _mm512_castsi512_si256(_mm512_permutexvar_epi64(PERMUTE, _dst0)));
                __m512i _dst1 = _mm512_shuffle_epi8(_mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _offs, src1, 1), SHUFFLE);
                _mm256_mask_storeu_epi16(dst1, mask, _mm512_castsi512_si256(_mm512_permutexvar_epi64(PERMUTE, _dst1)));
            }
        }

        template<> SIMD_INLINE void ByteBilinearGather<2, false>(const uint8_t* src0, const uint8_t* src1, uint32_t* offset, int count, uint8_t* dst0, uint8_t* dst1)
        {
            int i = 0, count16 = (int)AlignLo(count, 16);
            for (; i < count16; i += 16, dst0 += 64, dst1 += 64)
            {
                __m512i _offs = _mm512_loadu_si512((__m512i*)(offset + i));
                _mm512_storeu_si512((__m512i*)dst0, _mm512_i32gather_epi32(_offs, src0, 1));
                _mm512_storeu_si512((__m512i*)dst1, _mm512_i32gather_epi32(_offs, src1, 1));
            }
            if(i < count)
            {
                __mmask16 mask = __mmask16(-1) >> (16 + count16 - count);
                __m512i _offs = _mm512_maskz_loadu_epi32(mask, offset + i);
                _mm512_mask_storeu_epi32(dst0, mask, _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _offs, src0, 1));
                _mm512_mask_storeu_epi32(dst1, mask, _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _offs, src1, 1));
            }
        }

        template<> SIMD_INLINE void ByteBilinearGather<4, false>(const uint8_t* src0, const uint8_t* src1, uint32_t* offset, int count, uint8_t* dst0, uint8_t* dst1)
        {
            int i = 0, count8 = (int)AlignLo(count, 8);
            for (; i < count8; i += 8, dst0 += 64, dst1 += 64)
            {
                __m256i _offs = _mm256_loadu_si256((__m256i*)(offset + i));
                _mm512_storeu_si512((__m512i*)dst0, _mm512_i32gather_epi64(_offs, src0, 1));
                _mm512_storeu_si512((__m512i*)dst1, _mm512_i32gather_epi64(_offs, src1, 1));
            }
            if (i < count)
            {
                __mmask8 mask = __mmask8(-1) >> (8 + count8 - count);
                __m256i _offs = _mm256_maskz_loadu_epi32(mask, offset + i);
                _mm512_mask_storeu_epi64(dst0, mask, _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), mask, _offs, src0, 1));
                _mm512_mask_storeu_epi64(dst1, mask, _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), mask, _offs, src1, 1));
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int N> void ByteBilinearInterpMainN(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst, int count);

        template<> SIMD_INLINE void ByteBilinearInterpMainN<1>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst, int count)
        {
            __m512i fx0 = _mm512_loadu_si512((__m512i*)fx + 0);
            __m512i fx1 = _mm512_loadu_si512((__m512i*)fx + 1);
            __m512i r00 = _mm512_maddubs_epi16(_mm512_loadu_si512((__m512i*)src0 + 0), fx0);
            __m512i r01 = _mm512_maddubs_epi16(_mm512_loadu_si512((__m512i*)src0 + 1), fx1);
            __m512i r10 = _mm512_maddubs_epi16(_mm512_loadu_si512((__m512i*)src1 + 0), fx0);
            __m512i r11 = _mm512_maddubs_epi16(_mm512_loadu_si512((__m512i*)src1 + 1), fx1);

            __m512i s0 = _mm512_madd_epi16(UnpackU16<0>(r00, r10), Load<false>((__m128i*)fy + 0x0, (__m128i*)fy + 0x2, (__m128i*)fy + 0x4, (__m128i*)fy + 0x6));
            __m512i d0 = _mm512_srli_epi32(_mm512_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i s1 = _mm512_madd_epi16(UnpackU16<1>(r00, r10), Load<false>((__m128i*)fy + 0x1, (__m128i*)fy + 0x3, (__m128i*)fy + 0x5, (__m128i*)fy + 0x7));
            __m512i d1 = _mm512_srli_epi32(_mm512_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i s2 = _mm512_madd_epi16(UnpackU16<0>(r01, r11), Load<false>((__m128i*)fy + 0x8, (__m128i*)fy + 0xA, (__m128i*)fy + 0xC, (__m128i*)fy + 0xE));
            __m512i d2 = _mm512_srli_epi32(_mm512_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i s3 = _mm512_madd_epi16(UnpackU16<1>(r01, r11), Load<false>((__m128i*)fy + 0x9, (__m128i*)fy + 0xB, (__m128i*)fy + 0xD, (__m128i*)fy + 0xF));
            __m512i d3 = _mm512_srli_epi32(_mm512_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __mmask64 mask = __mmask64(-1) >> (64 - count * 1);
            _mm512_mask_storeu_epi8(dst, mask, PackI16ToU8(_mm512_packus_epi32(d0, d1), _mm512_packus_epi32(d2, d3)));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<2>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst, int count)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

            __m512i _fx = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_loadu_si512((__m512i*)fx));
            __m512i fx0 = UnpackU16<0>(_fx, _fx);
            __m512i fx1 = UnpackU16<1>(_fx, _fx);
            __m512i r00 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src0 + 0), SHUFFLE), fx0);
            __m512i r01 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src0 + 1), SHUFFLE), fx1);
            __m512i r10 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src1 + 0), SHUFFLE), fx0);
            __m512i r11 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src1 + 1), SHUFFLE), fx1);

            __m512i fy0 = _mm512_loadu_si512((__m512i*)fy + 0);
            __m512i s0 = _mm512_madd_epi16(UnpackU16<0>(r00, r10), UnpackU32<0>(fy0, fy0));
            __m512i d0 = _mm512_srli_epi32(_mm512_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i s1 = _mm512_madd_epi16(UnpackU16<1>(r00, r10), UnpackU32<1>(fy0, fy0));
            __m512i d1 = _mm512_srli_epi32(_mm512_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i fy1 = _mm512_loadu_si512((__m512i*)fy + 1);
            __m512i s2 = _mm512_madd_epi16(UnpackU16<0>(r01, r11), UnpackU32<0>(fy1, fy1));
            __m512i d2 = _mm512_srli_epi32(_mm512_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i s3 = _mm512_madd_epi16(UnpackU16<1>(r01, r11), UnpackU32<1>(fy1, fy1));
            __m512i d3 = _mm512_srli_epi32(_mm512_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __mmask64 mask = __mmask64(-1) >> (64 - count * 2);
            _mm512_mask_storeu_epi8(dst, mask, PackI16ToU8(_mm512_packus_epi32(d0, d1), _mm512_packus_epi32(d2, d3)));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<3>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst, int count)
        {
            static const __m512i SRC_SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x3, 0x1, 0x4, 0x2, 0x5, -1, -1, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, -1, -1,
                0x0, 0x3, 0x1, 0x4, 0x2, 0x5, -1, -1, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, -1, -1,
                0x0, 0x3, 0x1, 0x4, 0x2, 0x5, -1, -1, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, -1, -1,
                0x0, 0x3, 0x1, 0x4, 0x2, 0x5, -1, -1, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, -1, -1);
            static const __m512i DST_SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);
            static const __m512i DST_PERMUTE = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, 0x0, 0x0, 0x0, 0x0);

            __m512i _fx = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_loadu_si512((__m512i*)fx));
            _fx = UnpackU16<0>(_fx, _fx);
            __m512i fx0 = UnpackU16<0>(_fx, _fx);
            __m512i fx1 = UnpackU16<1>(_fx, _fx);
            __m512i r00 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src0 + 0), SRC_SHUFFLE), fx0);
            __m512i r01 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src0 + 1), SRC_SHUFFLE), fx1);
            __m512i r10 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src1 + 0), SRC_SHUFFLE), fx0);
            __m512i r11 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src1 + 1), SRC_SHUFFLE), fx1);

            __m512i _fy = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_loadu_si512((__m512i*)fy));
            __m512i fy0 = UnpackU32<0>(_fy, _fy);
            __m512i s0 = _mm512_madd_epi16(UnpackU16<0>(r00, r10), UnpackU32<0>(fy0, fy0));
            __m512i d0 = _mm512_srli_epi32(_mm512_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i s1 = _mm512_madd_epi16(UnpackU16<1>(r00, r10), UnpackU32<1>(fy0, fy0));
            __m512i d1 = _mm512_srli_epi32(_mm512_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i fy1 = UnpackU32<1>(_fy, _fy);
            __m512i s2 = _mm512_madd_epi16(UnpackU16<0>(r01, r11), UnpackU32<0>(fy1, fy1));
            __m512i d2 = _mm512_srli_epi32(_mm512_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i s3 = _mm512_madd_epi16(UnpackU16<1>(r01, r11), UnpackU32<1>(fy1, fy1));
            __m512i d3 = _mm512_srli_epi32(_mm512_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __mmask64 mask = __mmask64(-1) >> (64 - count * 3);
            __m512i _dst = PackI16ToU8(_mm512_packus_epi32(d0, d1), _mm512_packus_epi32(d2, d3));
            _mm512_mask_storeu_epi8(dst, mask, _mm512_permutexvar_epi32(DST_PERMUTE, _mm512_shuffle_epi8(_dst, DST_SHUFFLE)));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<4>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst, int count)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

            __m512i _fx = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_loadu_si512((__m512i*)fx));
            _fx = UnpackU16<0>(_fx, _fx);
            __m512i fx0 = UnpackU16<0>(_fx, _fx);
            __m512i fx1 = UnpackU16<1>(_fx, _fx);
            __m512i r00 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src0 + 0), SHUFFLE), fx0);
            __m512i r01 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src0 + 1), SHUFFLE), fx1);
            __m512i r10 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src1 + 0), SHUFFLE), fx0);
            __m512i r11 = _mm512_maddubs_epi16(_mm512_shuffle_epi8(_mm512_loadu_si512((__m512i*)src1 + 1), SHUFFLE), fx1);

            __m512i _fy = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_loadu_si512((__m512i*)fy));
            __m512i fy0 = UnpackU32<0>(_fy, _fy);
            __m512i s0 = _mm512_madd_epi16(UnpackU16<0>(r00, r10), UnpackU32<0>(fy0, fy0));
            __m512i d0 = _mm512_srli_epi32(_mm512_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i s1 = _mm512_madd_epi16(UnpackU16<1>(r00, r10), UnpackU32<1>(fy0, fy0));
            __m512i d1 = _mm512_srli_epi32(_mm512_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i fy1 = UnpackU32<1>(_fy, _fy);
            __m512i s2 = _mm512_madd_epi16(UnpackU16<0>(r01, r11), UnpackU32<0>(fy1, fy1));
            __m512i d2 = _mm512_srli_epi32(_mm512_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m512i s3 = _mm512_madd_epi16(UnpackU16<1>(r01, r11), UnpackU32<1>(fy1, fy1));
            __m512i d3 = _mm512_srli_epi32(_mm512_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __mmask64 mask = __mmask64(-1) >> (64 - count * 4);
            _mm512_mask_storeu_epi8(dst, mask, PackI16ToU8(_mm512_packus_epi32(d0, d1), _mm512_packus_epi32(d2, d3)));
        }

        //-------------------------------------------------------------------------------------------------

        template<int N, bool soft> void ByteBilinearRun(const WarpAffParam& p, int yBeg, int yEnd, const int* ib, const int* ie, const int* ob, const int* oe, const uint8_t* src, uint8_t* dst, uint8_t* buf)
        {
            constexpr int M = (N == 3 ? 4 : N);
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 2, h = (int)p.srcH - 2, n = A / M;
            size_t wa = AlignHi(p.dstW, p.align) + p.align;
            uint32_t* offs = (uint32_t*)buf;
            uint8_t* fx = (uint8_t*)(offs + wa);
            uint16_t* fy = (uint16_t*)(fx + wa * 2);
            uint8_t* rb0 = (uint8_t*)(fy + wa * 2);
            uint8_t* rb1 = (uint8_t*)(rb0 + wa * M * 2);
            const __m512 _16 = _mm512_set1_ps(16.0f);
            static const __m512i _0123 = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF);
            __m512 _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = _mm512_set1_ps(p.inv[i]);
            __m512i _n = _mm512_set1_epi32(N);
            __m512i _s = _mm512_set1_epi32(s);
            __m512i _border = InitBorder<N>(p.border);
            __m128 _me[3];
            for (int i = 0; i < 3; ++i)
                _me[i] = Sse41::SetFloat(p.inv[i + 0], p.inv[i + 3]);
            __m128i _wh = Sse41::SetInt32(w, h);
            __m128i _ns = _mm_setr_epi32(0, N, s, s + N);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int iB = ib[y], iE = ie[y], oB = ob[y], oE = oe[y];
                __m512 _y = _mm512_cvtepi32_ps(_mm512_set1_epi32(y));
                if (fill)
                {
                    FillBorder<N>(dst, oB, _border, p.border);
                   for (int x = oB; x < iB; ++x)
                       ByteBilinearInterpEdge<N>(x, _mm512_castps512_ps128(_y), _me, _wh, _ns, s, src, p.border, dst + x * N);
                }
                else
                {
                    for (int x = oB; x < iB; ++x)
                        ByteBilinearInterpEdge<N>(x, _mm512_castps512_ps128(_y), _me, _wh, _ns, s, src, dst + x * N, dst + x * N);
                }
                {
                    int x = iB, iEn = (int)AlignLo(iE - iB, n) + iB;
                    __m512 _x = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(x), _0123));
                    for (; x < iE; x += 16)
                    {
                        ByteBilinearPrepMain16(_x, _y, _m, _n, _s, offs + x, fx + 2 * x, fy + 2 * x);
                        _x = _mm512_add_ps(_x, _16);
                    }
                    ByteBilinearGather<M, soft>(src, src + s, offs + iB, iE - iB, rb0 + 2 * M * iB, rb1 + 2 * M * iB);
                    for (x = iB; x < iEn; x += n)
                        ByteBilinearInterpMainN<N>(rb0 + x * M * 2, rb1 + x * M * 2, fx + 2 * x, fy + 2 * x, dst + x * N, n);
                    if(x < iE)
                        ByteBilinearInterpMainN<N>(rb0 + x * M * 2, rb1 + x * M * 2, fx + 2 * x, fy + 2 * x, dst + x * N, iE - iEn);
                }
                if (fill)
                {
                    for (int x = iE; x < oE; ++x)
                        ByteBilinearInterpEdge<N>(x, _mm512_castps512_ps128(_y), _me, _wh, _ns, s, src, p.border, dst + x * N);
                    FillBorder<N>(dst + oE * N, width - oE, _border, p.border);
                }
                else
                {
                    for (int x = iE; x < oE; ++x)
                        ByteBilinearInterpEdge<N>(x, _mm512_castps512_ps128(_y), _me, _wh, _ns, s, src, dst + x * N, dst + x * N);
                }
                dst += p.dstS;
            }
        }

        //-------------------------------------------------------------------------------------------------

        WarpAffineByteBilinear::WarpAffineByteBilinear(const WarpAffParam& param)
            : Avx2::WarpAffineByteBilinear(param)
        {
            bool soft = Avx2::SlowGather;
            switch (_param.channels)
            {
            case 1: _run = soft ? ByteBilinearRun<1, true> : ByteBilinearRun<1, false>; break;
            case 2: _run = soft ? ByteBilinearRun<2, true> : ByteBilinearRun<2, false>; break;
            case 3: _run = soft ? ByteBilinearRun<3, true> : ByteBilinearRun<3, false>; break;
            case 4: _run = soft ? ByteBilinearRun<4, true> : ByteBilinearRun<4, false>; break;
            }
        }

        void WarpAffineByteBilinear::SetRange(const Base::Point* rect, int* beg, int* end, const int* lo, const int* hi)
        {
            const WarpAffParam& p = _param;
            float* min = (float*)_buf.data;
            float* max = min + p.dstH;
            float w = (float)p.dstW, h = (float)p.dstH, z = 0.0f;
            static const __m512i _0123 = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF);
            __m512 _w = _mm512_set1_ps(w), _z = _mm512_set1_ps(z);
            int y = 0, dH = (int)p.dstH, dH16 = (int)AlignLo(dH, 16);
            for (; y < dH16; y += 16)
            {
                _mm512_storeu_ps(min + y, _w);
                _mm512_storeu_ps(max + y, _mm512_setzero_ps());
            }
            if (y < dH)
            {
                __mmask16 tail = TailMask16(dH - dH16);
                _mm512_mask_storeu_ps(min + y, tail, _w);
                _mm512_mask_storeu_ps(max + y, tail, _mm512_setzero_ps());
            }
            for (int v = 0; v < 4; ++v)
            {
                const Base::Point& curr = rect[v];
                const Base::Point& next = rect[(v + 1) & 3];
                if (next.y == curr.y)
                    continue;
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), z);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), h);
                int yBeg = (int)ceil(yMin);
                int yEnd = (int)ceil(yMax);
                int yEnd16 = (int)AlignLo(yEnd - yBeg, 16) + yBeg;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                __m512 _a = _mm512_set1_ps(a);
                __m512 _b = _mm512_set1_ps(b);
                __m512 _yMin = _mm512_set1_ps(yMin);
                __m512 _yMax = _mm512_set1_ps(yMax);
                for (y = yBeg; y < yEnd16; y += 16)
                {
                    __m512 _y = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(y), _0123));
                    _y = _mm512_min_ps(_yMax, _mm512_max_ps(_y, _yMin));
                    __m512 _x = _mm512_add_ps(_mm512_mul_ps(_y, _a), _b);
                    _mm512_storeu_ps(min + y, _mm512_min_ps(_mm512_loadu_ps(min + y), _mm512_max_ps(_x, _z)));
                    _mm512_storeu_ps(max + y, _mm512_max_ps(_mm512_loadu_ps(max + y), _mm512_min_ps(_x, _w)));
                }
                if(y < yEnd)
                {
                    __mmask16 tail = TailMask16(yEnd - yEnd16);
                    __m512 _y = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(y), _0123));
                    _y = _mm512_min_ps(_yMax, _mm512_max_ps(_y, _yMin));
                    __m512 _x = _mm512_add_ps(_mm512_mul_ps(_y, _a), _b);
                    _mm512_mask_storeu_ps(min + y, tail, _mm512_min_ps(_mm512_loadu_ps(min + y), _mm512_max_ps(_x, _z)));
                    _mm512_mask_storeu_ps(max + y, tail, _mm512_max_ps(_mm512_loadu_ps(max + y), _mm512_min_ps(_x, _w)));
                }
            }
            for (y = 0; y < dH16; y += 16)
            {
                __m512i _beg = _mm512_cvtps_epi32(_mm512_ceil_ps(_mm512_loadu_ps(min + y)));
                __m512i _end = _mm512_cvtps_epi32(_mm512_ceil_ps(_mm512_loadu_ps(max + y)));
                _mm512_storeu_si512((__m512i*)(beg + y), _beg);
                _mm512_storeu_si512((__m512i*)(end + y), _mm512_max_epi32(_beg, _end));
            }
            if (y < dH)
            {
                __mmask16 tail = TailMask16(dH - dH16);
                __m512i _beg = _mm512_cvtps_epi32(_mm512_ceil_ps(_mm512_maskz_loadu_ps(tail, min + y)));
                __m512i _end = _mm512_cvtps_epi32(_mm512_ceil_ps(_mm512_maskz_loadu_ps(tail, max + y)));
                _mm512_mask_storeu_epi32(beg + y, tail, _beg);
                _mm512_mask_storeu_epi32(end + y, tail, _mm512_max_epi32(_beg, _end));
            }
            if (hi)
            {
                for (y = 0; y < dH16; y += 16)
                {
                    __m512i _hi = _mm512_loadu_si512((__m512i*)(hi + y));
                    _mm512_storeu_si512((__m512i*)(beg + y), _mm512_min_epi32(_mm512_loadu_si512((__m512i*)(beg + y)), _hi));
                    _mm512_storeu_si512((__m512i*)(end + y), _mm512_min_epi32(_mm512_loadu_si512((__m512i*)(end + y)), _hi));
                }
                if (y < dH)
                {
                    __mmask16 tail = TailMask16(dH - dH16);
                    __m512i _hi = _mm512_maskz_loadu_epi32(tail, hi + y);
                    _mm512_mask_storeu_epi32(beg + y, tail, _mm512_min_epi32(_mm512_maskz_loadu_epi32(tail, beg + y), _hi));
                    _mm512_mask_storeu_epi32(end + y, tail, _mm512_min_epi32(_mm512_maskz_loadu_epi32(tail, end + y), _hi));
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border)
        {
            WarpAffParam param(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border, A);
            if (!param.Valid())
                return NULL;
            if (param.IsNearest())
                return new WarpAffineNearest(param);
            else if (param.IsByteBilinear())
                return new WarpAffineByteBilinear(param);
            else
                return NULL;
        }
#else
        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border)
        {
            WarpAffParam param(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border, A);
            if (!param.Valid())
                return NULL;
            if (param.IsNearest())
                return new Avx2::WarpAffineNearest(param);
            else if (param.IsByteBilinear())
                return new Avx2::WarpAffineByteBilinear(param);
            else
                return NULL;
        }
#endif
    }
#endif
}
