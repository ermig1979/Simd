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
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        template<int N> SIMD_INLINE void FillBorder(uint8_t* dst, int count, const __m256i& bv, const uint8_t* bs)
        {
            int i = 0, size = count * N, size16 = (int)AlignLo(size, 16), size32 = (int)AlignLo(size, 32);
            for (; i < size32; i += 32)
                _mm256_storeu_si256((__m256i*)(dst + i), bv);
            for (; i < size16; i += 16)
                _mm_storeu_si128((__m128i*)(dst + i), _mm256_castsi256_si128(bv));
            for (; i < size; i += N)
                Base::CopyPixel<N>(bs, dst + i);
        }

        template<> SIMD_INLINE void FillBorder<3>(uint8_t* dst, int count, const __m256i& bv, const uint8_t* bs)
        {
            int i = 0, size = count * 3, size3 = size - 3;
            for (; i < size3; i += 3)
                Base::CopyPixel<4>(bs, dst + i);
            for (; i < size; i += 3)
                Base::CopyPixel<3>(bs, dst + i);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m256i NearestOffset(__m256 x, __m256 y, const __m256* m, __m256i w, __m256i h, __m256i n, __m256i s)
        {
            __m256 dx = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, m[0]), _mm256_mul_ps(y, m[1])), m[2]);
            __m256 dy = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, m[3]), _mm256_mul_ps(y, m[4])), m[5]);
            __m256i ix = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(dx), _mm256_setzero_si256()), w);
            __m256i iy = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(dy), _mm256_setzero_si256()), h);
            return _mm256_add_epi32(_mm256_mullo_epi32(ix, n), _mm256_mullo_epi32(iy, s));
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
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m256i PERMUTE = SIMD_MM256_SETR_EPI32(0, 4, 0, 0, 0, 0, 0, 0);
            int i = 0, count8 = AlignLo(count, 8);
            for (; i < count8; i += 8, dst += 8)
            {
                __m256i _offs = _mm256_loadu_si256((__m256i*)(offset + i));
                __m256i _dst = _mm256_i32gather_epi32((int*)src, _offs, 1);
                _dst = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_dst, SHUFFLE), PERMUTE);
                Sse41::StoreHalf<false>((__m128i*)dst, _mm256_castsi256_si128(_dst));
            }
            for (; i < count; i++, dst += 1)
                Base::CopyPixel<1>(src + offset[i], dst);
        }

        template<> SIMD_INLINE void NearestGather<2, false>(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1);
            int i = 0, count8 = AlignLo(count, 8);
            for (; i < count8; i += 8, dst += 16)
            {
                __m256i _offs = _mm256_loadu_si256((__m256i*)(offset + i));
                __m256i _dst = _mm256_i32gather_epi32((int*)src, _offs, 1);
                _dst = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(_dst, SHUFFLE), 0x08);
                Sse41::Store<false>((__m128i*)dst, _mm256_castsi256_si128(_dst));
            }
            for (; i < count; i++, dst += 2)
                Base::CopyPixel<2>(src + offset[i], dst);
        }

        template<> SIMD_INLINE void NearestGather<3, false>(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);
            static const __m256i PERMUTE = SIMD_MM256_SETR_EPI32(0, 1, 2, 4, 5, 6, 0, 0);
            int i = 0, count8 = AlignLo(count, 8), count1 = count - 1;
            for (; i < count8; i += 8, dst += 24)
            {
                __m256i _offs = _mm256_loadu_si256((__m256i*)(offset + i));
                __m256i _dst = _mm256_i32gather_epi32((int*)src, _offs, 1);
                _dst = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_dst, SHUFFLE), PERMUTE);
                Store24<false>(dst, _dst);
            }
            for (; i < count1; i++, dst += 3)
                Base::CopyPixel<4>(src + offset[i], dst);
            if (i < count)
                Base::CopyPixel<3>(src + offset[i], dst);
        }

        template<> SIMD_INLINE void NearestGather<4, false>(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            int i = 0, count8 = AlignLo(count, 8);
            for (; i < count8; i += 8, dst += 32)
            {
                __m256i _offs = _mm256_loadu_si256((__m256i*)(offset + i));
                __m256i _dst = _mm256_i32gather_epi32((int*)src, _offs, 1);
                _mm256_storeu_si256((__m256i*)dst, _dst);
            }
            for (; i < count; i++, dst += 4)
                Base::CopyPixel<4>(src + offset[i], dst);
        }

        //-----------------------------------------------------------------------------------------

        template<int N, bool soft> void NearestRun(const WarpAffParam& p, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1, n32 = 32 / N, n16 = 16 / N;
            const __m256 _8 = _mm256_set1_ps(8.0f);
            static const __m256i _01234567 = SIMD_MM256_SETR_EPI32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256 _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = _mm256_set1_ps(p.inv[i]);
            __m256i _w = _mm256_set1_epi32(w);
            __m256i _h = _mm256_set1_epi32(h);
            __m256i _n = _mm256_set1_epi32(N);
            __m256i _s = _mm256_set1_epi32(s);
            __m256i _border;
            switch (N)
            {
            case 1: _border = _mm256_set1_epi8(*p.border); break;
            case 2: _border = _mm256_set1_epi16(*(uint16_t*)p.border); break;
            case 4: _border = _mm256_set1_epi32(*(uint32_t*)p.border); break;
            }
            for (int y = 0; y < (int)p.dstH; ++y)
            {
                int nose = beg[y], tail = end[y];
                {
                    int x = nose;
                    __m256 _y = _mm256_cvtepi32_ps(_mm256_set1_epi32(y));
                    __m256 _x = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(x), _01234567));
                    for (; x < tail; x += 8)
                    {
                        _mm256_storeu_si256((__m256i*)(buf + x), NearestOffset(_x, _y, _m, _w, _h, _n, _s));
                        _x = _mm256_add_ps(_x, _8);
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
            : Sse41::WarpAffineNearest(param)
        {
            bool soft = SlowGather;
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
            int w = (int)p.dstW, h = (int)p.dstH, h8 = (int)AlignLo(h, 8);
            static const __m256i _01234567 = SIMD_MM256_SETR_EPI32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i _w = _mm256_set1_epi32(w), _1 = _mm256_set1_epi32(1);
            int y = 0;
            for (; y < h8; y += 8)
            {
                _mm256_storeu_si256((__m256i*)(_beg.data + y), _w);
                _mm256_storeu_si256((__m256i*)(_end.data + y), _mm256_setzero_si256());
            }
            for (; y < h; ++y)
            {
                _beg[y] = w;
                _end[y] = 0;
            }
            for (int v = 0; v < 4; ++v)
            {
                const Base::Point& curr = points[v];
                const Base::Point& next = points[(v + 1) & 3];
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), 0.0f);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), (float)p.dstH);
                int yBeg = Round(yMin);
                int yEnd = Round(yMax);
                int yEnd8 = (int)AlignLo(yEnd - yBeg, 8) + yBeg;
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                __m256 _a = _mm256_set1_ps(a);
                __m256 _b = _mm256_set1_ps(b);
                if (abs(a) <= 1.0f)
                {
                    int y = yBeg;
                    for (; y < yEnd8; y += 8)
                    {
                        __m256 _y = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(y), _01234567));
                        __m256i _x = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(_y, _a), _b));
                        __m256i xBeg = _mm256_loadu_si256((__m256i*)(_beg.data + y));
                        __m256i xEnd = _mm256_loadu_si256((__m256i*)(_end.data + y));
                        xBeg = _mm256_min_epi32(xBeg, _mm256_max_epi32(_x, _mm256_setzero_si256()));
                        xEnd = _mm256_max_epi32(xEnd, _mm256_min_epi32(_mm256_add_epi32(_x, _1), _w));
                        _mm256_storeu_si256((__m256i*)(_beg.data + y), xBeg);
                        _mm256_storeu_si256((__m256i*)(_end.data + y), xEnd);
                    }
                    for (; y < yEnd; ++y)
                    {
                        int x = Round(y * a + b);
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(x, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(x + 1, w));
                    }
                }
                else
                {
                    int y = yBeg;
                    __m256 _05 = _mm256_set1_ps(0.5f);
                    __m256 _yMin = _mm256_set1_ps(yMin);
                    __m256 _yMax = _mm256_set1_ps(yMax);
                    for (; y < yEnd8; y += 8)
                    {
                        __m256 _y = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(y), _01234567));
                        __m256 yM = _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(_y, _05), _yMin), _yMax);
                        __m256 yP = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(_y, _05), _yMin), _yMax);
                        __m256 xM = _mm256_add_ps(_mm256_mul_ps(yM, _a), _b);
                        __m256 xP = _mm256_add_ps(_mm256_mul_ps(yP, _a), _b);
                        __m256i xBeg = _mm256_loadu_si256((__m256i*)(_beg.data + y));
                        __m256i xEnd = _mm256_loadu_si256((__m256i*)(_end.data + y));
                        xBeg = _mm256_min_epi32(xBeg, _mm256_max_epi32(_mm256_cvtps_epi32(_mm256_min_ps(xM, xP)), _mm256_setzero_si256()));
                        xEnd = _mm256_max_epi32(xEnd, _mm256_min_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(_mm256_max_ps(xM, xP)), _1), _w));
                        _mm256_storeu_si256((__m256i*)(_beg.data + y), xBeg);
                        _mm256_storeu_si256((__m256i*)(_end.data + y), xEnd);
                    }
                    for (; y < yEnd; ++y)
                    {
                        float xM = b + Simd::RestrictRange(float(y) - 0.5f, yMin, yMax) * a;
                        float xP = b + Simd::RestrictRange(float(y) + 0.5f, yMin, yMax) * a;
                        int xBeg = Round(Simd::Min(xM, xP));
                        int xEnd = Round(Simd::Max(xM, xP));
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(xBeg, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(xEnd + 1, w));
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
                return new Sse41::WarpAffineByteBilinear(param);
            else
                return NULL;
        }
    }
#endif
}
