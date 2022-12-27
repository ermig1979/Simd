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
#include "Simd/SimdWarpAffine.h"
#include "Simd/SimdWarpAffineCommon.h"
#include "Simd/SimdCopyPixel.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdEnable.h"

#include "Simd/SimdPoint.hpp"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
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
                __m512i _dst = _mm512_mask_i32gather_epi32(_offs, mask, _offs, src, 1);
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
                __m512i _dst = _mm512_mask_i32gather_epi32(_offs, mask, _offs, src, 1);
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
                __m512i _dst = _mm512_mask_i32gather_epi32(_offs, srcMask, _offs, src, 1);
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
                __m512i _dst = _mm512_mask_i32gather_epi32(_offs, mask, _offs, src, 1);
                _mm512_mask_storeu_epi32(dst, mask, _dst);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<int N, bool soft> void NearestRun(const WarpAffParam& p, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
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
            for (int y = 0; y < (int)p.dstH; ++y)
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

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border)
        {
            WarpAffParam param(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border, A);
            if (!param.Valid())
                return NULL;
            if (param.IsNearest())
                return new WarpAffineNearest(param);
            else if (param.IsByteBilinear())
                return new Avx2::WarpAffineByteBilinear(param);
            else
                return NULL;
        }
    }
#endif
}
