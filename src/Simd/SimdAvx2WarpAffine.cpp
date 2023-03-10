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

        template<int N> SIMD_INLINE __m256i InitBorder(const uint8_t* border)
        {
            switch (N)
            {
            case 1: return _mm256_set1_epi8(*border);
            case 2: return _mm256_set1_epi16(*(uint16_t*)border);
            case 3: return _mm256_setzero_si256();
            case 4: return _mm256_set1_epi32(*(uint32_t*)border);
            }
            return _mm256_setzero_si256();
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m256i NearestOffset(__m256 x, __m256 y, const __m256* m, __m256i w, const __m256i & h, const __m256i & n, const __m256i & s)
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
            int i = 0, count8 = (int)AlignLo(count, 8);
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
            int i = 0, count8 = (int)AlignLo(count, 8);
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
            int i = 0, count8 = (int)AlignLo(count, 8), count1 = count - 1;
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
            int i = 0, count8 = (int)AlignLo(count, 8);
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

        template<int N, bool soft> void NearestRun(const WarpAffParam& p, int yBeg, int yEnd, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1;
            const __m256 _8 = _mm256_set1_ps(8.0f);
            static const __m256i _01234567 = SIMD_MM256_SETR_EPI32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256 _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = _mm256_set1_ps(p.inv[i]);
            __m256i _w = _mm256_set1_epi32(w);
            __m256i _h = _mm256_set1_epi32(h);
            __m256i _n = _mm256_set1_epi32(N);
            __m256i _s = _mm256_set1_epi32(s);
            __m256i _border = InitBorder<N>(p.border);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
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

        const __m256i K32_WA_FRACTION_RANGE = SIMD_MM256_SET1_EPI32(Base::WA_FRACTION_RANGE);

        SIMD_INLINE void ByteBilinearPrepMain8(__m256 x, __m256 y, const __m256* m, __m256i n, const __m256i & s, uint32_t* offs, uint8_t* fx, uint16_t* fy)
        {
            __m256 dx = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, m[0]), _mm256_mul_ps(y, m[1])), m[2]);
            __m256 dy = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(x, m[3]), _mm256_mul_ps(y, m[4])), m[5]);
            __m256 ix = _mm256_floor_ps(dx);
            __m256 iy = _mm256_floor_ps(dy);
            __m256 range = _mm256_cvtepi32_ps(K32_WA_FRACTION_RANGE);
            __m256i _fx = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(dx, ix), range));
            __m256i _fy = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(dy, iy), range));
            _mm256_storeu_si256((__m256i*)offs, _mm256_add_epi32(_mm256_mullo_epi32(_mm256_cvtps_epi32(ix), n), _mm256_mullo_epi32(_mm256_cvtps_epi32(iy), s)));
            _fx = _mm256_or_si256(_mm256_sub_epi32(K32_WA_FRACTION_RANGE, _fx), _mm256_slli_epi32(_fx, 16));
            _fy = _mm256_or_si256(_mm256_sub_epi32(K32_WA_FRACTION_RANGE, _fy), _mm256_slli_epi32(_fy, 16));
            _mm_storeu_si128((__m128i*)fx, _mm256_castsi256_si128(PackI16ToU8(_fx, _mm256_setzero_si256())));
            _mm256_storeu_si256((__m256i*)fy, _fy);
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
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                0x0, 0x1, 0x4, 0x5, 0x8, 0x9, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1);
            int i = 0, count8 = (int)AlignLo(count, 8);
            for (; i < count8; i += 8, dst0 += 16, dst1 += 16)
            {
                __m256i _offs = _mm256_loadu_si256((__m256i*)(offset + i));
                __m256i _dst0 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((int*)src0, _offs, 1), SHUFFLE);
                _mm_storeu_si128((__m128i*)dst0, _mm256_castsi256_si128(_mm256_permute4x64_epi64(_dst0, 0x08)));
                __m256i _dst1 = _mm256_shuffle_epi8(_mm256_i32gather_epi32((int*)src1, _offs, 1), SHUFFLE);
                _mm_storeu_si128((__m128i*)dst1, _mm256_castsi256_si128(_mm256_permute4x64_epi64(_dst1, 0x08)));
            }
            for (; i < count; i++, dst0 += 2, dst1 += 2)
            {
                int offs = offset[i];
                Base::CopyPixel<2>(src0 + offs, dst0);
                Base::CopyPixel<2>(src1 + offs, dst1);
            }
        }

        template<> SIMD_INLINE void ByteBilinearGather<2, false>(const uint8_t* src0, const uint8_t* src1, uint32_t* offset, int count, uint8_t* dst0, uint8_t* dst1)
        {
            int i = 0, count8 = (int)AlignLo(count, 8);
            for (; i < count8; i += 8, dst0 += 32, dst1 += 32)
            {
                __m256i _offs = _mm256_loadu_si256((__m256i*)(offset + i));
                _mm256_storeu_si256((__m256i*)dst0, _mm256_i32gather_epi32((int*)src0, _offs, 1));
                _mm256_storeu_si256((__m256i*)dst1, _mm256_i32gather_epi32((int*)src1, _offs, 1));
            }
            for (; i < count; i++, dst0 += 4, dst1 += 4)
            {
                int offs = offset[i];
                Base::CopyPixel<4>(src0 + offs, dst0);
                Base::CopyPixel<4>(src1 + offs, dst1);
            }
        }

        template<> SIMD_INLINE void ByteBilinearGather<4, false>(const uint8_t* src0, const uint8_t* src1, uint32_t* offset, int count, uint8_t* dst0, uint8_t* dst1)
        {
            int i = 0, count4 = (int)AlignLo(count, 4);
            for (; i < count4; i += 4, dst0 += 32, dst1 += 32)
            {
                __m128i _offs = _mm_loadu_si128((__m128i*)(offset + i));
                _mm256_storeu_si256((__m256i*)dst0, _mm256_i32gather_epi64((long long*)src0, _offs, 1));
                _mm256_storeu_si256((__m256i*)dst1, _mm256_i32gather_epi64((long long*)src1, _offs, 1));
            }
            for (; i < count; i++, dst0 += 8, dst1 += 8)
            {
                int offs = offset[i];
                Base::CopyPixel<8>(src0 + offs, dst0);
                Base::CopyPixel<8>(src1 + offs, dst1);
            }
        }

        //-------------------------------------------------------------------------------------------------

        const __m256i K32_WA_BILINEAR_ROUND_TERM = SIMD_MM256_SET1_EPI32(Base::WA_BILINEAR_ROUND_TERM);

        template<int N> void ByteBilinearInterpMainN(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst);

        template<> SIMD_INLINE void ByteBilinearInterpMainN<1>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            __m256i fx0 = _mm256_loadu_si256((__m256i*)fx + 0);
            __m256i fx1 = _mm256_loadu_si256((__m256i*)fx + 1);
            __m256i r00 = _mm256_maddubs_epi16(_mm256_loadu_si256((__m256i*)src0 + 0), fx0);
            __m256i r01 = _mm256_maddubs_epi16(_mm256_loadu_si256((__m256i*)src0 + 1), fx1);
            __m256i r10 = _mm256_maddubs_epi16(_mm256_loadu_si256((__m256i*)src1 + 0), fx0);
            __m256i r11 = _mm256_maddubs_epi16(_mm256_loadu_si256((__m256i*)src1 + 1), fx1);

            __m256i s0 = _mm256_madd_epi16(UnpackU16<0>(r00, r10), Load<false>((__m128i*)fy + 0, (__m128i*)fy + 2));
            __m256i d0 = _mm256_srli_epi32(_mm256_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i s1 = _mm256_madd_epi16(UnpackU16<1>(r00, r10), Load<false>((__m128i*)fy + 1, (__m128i*)fy + 3));
            __m256i d1 = _mm256_srli_epi32(_mm256_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i s2 = _mm256_madd_epi16(UnpackU16<0>(r01, r11), Load<false>((__m128i*)fy + 4, (__m128i*)fy + 6));
            __m256i d2 = _mm256_srli_epi32(_mm256_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i s3 = _mm256_madd_epi16(UnpackU16<1>(r01, r11), Load<false>((__m128i*)fy + 5, (__m128i*)fy + 7));
            __m256i d3 = _mm256_srli_epi32(_mm256_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(_mm256_packus_epi32(d0, d1), _mm256_packus_epi32(d2, d3)));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<2>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

            __m256i _fx = LoadPermuted<false>((__m256i*)fx);
            __m256i fx0 = UnpackU16<0>(_fx, _fx);
            __m256i fx1 = UnpackU16<1>(_fx, _fx);
            __m256i r00 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src0 + 0), SHUFFLE), fx0);
            __m256i r01 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src0 + 1), SHUFFLE), fx1);
            __m256i r10 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src1 + 0), SHUFFLE), fx0);
            __m256i r11 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src1 + 1), SHUFFLE), fx1);

            __m256i fy0 = _mm256_loadu_si256((__m256i*)fy + 0);
            __m256i s0 = _mm256_madd_epi16(UnpackU16<0>(r00, r10), UnpackU32<0>(fy0, fy0));
            __m256i d0 = _mm256_srli_epi32(_mm256_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i s1 = _mm256_madd_epi16(UnpackU16<1>(r00, r10), UnpackU32<1>(fy0, fy0));
            __m256i d1 = _mm256_srli_epi32(_mm256_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i fy1 = _mm256_loadu_si256((__m256i*)fy + 1);
            __m256i s2 = _mm256_madd_epi16(UnpackU16<0>(r01, r11), UnpackU32<0>(fy1, fy1));
            __m256i d2 = _mm256_srli_epi32(_mm256_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i s3 = _mm256_madd_epi16(UnpackU16<1>(r01, r11), UnpackU32<1>(fy1, fy1));
            __m256i d3 = _mm256_srli_epi32(_mm256_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(_mm256_packus_epi32(d0, d1), _mm256_packus_epi32(d2, d3)));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<3>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            static const __m256i SRC_SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x3, 0x1, 0x4, 0x2, 0x5, -1, -1, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, -1, -1,
                0x0, 0x3, 0x1, 0x4, 0x2, 0x5, -1, -1, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, -1, -1);
            static const __m256i DST_SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1,
                0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);
            static const __m256i DST_PERMUTE = SIMD_MM256_SETR_EPI32(0, 1, 2, 4, 5, 6, 0, 0);

            __m256i _fx = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)fx), K32_TWO_UNPACK_PERMUTE);
            _fx = UnpackU16<0>(_fx, _fx);
            __m256i fx0 = UnpackU16<0>(_fx, _fx);
            __m256i fx1 = UnpackU16<1>(_fx, _fx);
            __m256i r00 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src0 + 0), SRC_SHUFFLE), fx0);
            __m256i r01 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src0 + 1), SRC_SHUFFLE), fx1);
            __m256i r10 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src1 + 0), SRC_SHUFFLE), fx0);
            __m256i r11 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src1 + 1), SRC_SHUFFLE), fx1);

            __m256i _fy = LoadPermuted<false>((__m256i*)fy);
            __m256i fy0 = UnpackU32<0>(_fy, _fy);
            __m256i s0 = _mm256_madd_epi16(UnpackU16<0>(r00, r10), UnpackU32<0>(fy0, fy0));
            __m256i d0 = _mm256_srli_epi32(_mm256_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i s1 = _mm256_madd_epi16(UnpackU16<1>(r00, r10), UnpackU32<1>(fy0, fy0));
            __m256i d1 = _mm256_srli_epi32(_mm256_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i fy1 = UnpackU32<1>(_fy, _fy);
            __m256i s2 = _mm256_madd_epi16(UnpackU16<0>(r01, r11), UnpackU32<0>(fy1, fy1));
            __m256i d2 = _mm256_srli_epi32(_mm256_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i s3 = _mm256_madd_epi16(UnpackU16<1>(r01, r11), UnpackU32<1>(fy1, fy1));
            __m256i d3 = _mm256_srli_epi32(_mm256_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i _dst = PackI16ToU8(_mm256_packus_epi32(d0, d1), _mm256_packus_epi32(d2, d3));
            Store24<false>(dst, _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_dst, DST_SHUFFLE), DST_PERMUTE));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<4>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

            __m256i _fx = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)fx), K32_TWO_UNPACK_PERMUTE);
            _fx = UnpackU16<0>(_fx, _fx);
            __m256i fx0 = UnpackU16<0>(_fx, _fx);
            __m256i fx1 = UnpackU16<1>(_fx, _fx);
            __m256i r00 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src0 + 0), SHUFFLE), fx0);
            __m256i r01 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src0 + 1), SHUFFLE), fx1);
            __m256i r10 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src1 + 0), SHUFFLE), fx0);
            __m256i r11 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)src1 + 1), SHUFFLE), fx1);

            __m256i _fy = LoadPermuted<false>((__m256i*)fy);
            __m256i fy0 = UnpackU32<0>(_fy, _fy);
            __m256i s0 = _mm256_madd_epi16(UnpackU16<0>(r00, r10), UnpackU32<0>(fy0, fy0));
            __m256i d0 = _mm256_srli_epi32(_mm256_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i s1 = _mm256_madd_epi16(UnpackU16<1>(r00, r10), UnpackU32<1>(fy0, fy0));
            __m256i d1 = _mm256_srli_epi32(_mm256_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i fy1 = UnpackU32<1>(_fy, _fy);
            __m256i s2 = _mm256_madd_epi16(UnpackU16<0>(r01, r11), UnpackU32<0>(fy1, fy1));
            __m256i d2 = _mm256_srli_epi32(_mm256_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m256i s3 = _mm256_madd_epi16(UnpackU16<1>(r01, r11), UnpackU32<1>(fy1, fy1));
            __m256i d3 = _mm256_srli_epi32(_mm256_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(_mm256_packus_epi32(d0, d1), _mm256_packus_epi32(d2, d3)));
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
            const __m256 _8 = _mm256_set1_ps(8.0f);
            static const __m256i _01234567 = SIMD_MM256_SETR_EPI32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256 _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = _mm256_set1_ps(p.inv[i]);
            __m256i _n = _mm256_set1_epi32(N);
            __m256i _s = _mm256_set1_epi32(s);
            __m256i _border = InitBorder<N>(p.border);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int iB = ib[y], iE = ie[y], oB = ob[y], oE = oe[y];
                if (fill)
                {
                    FillBorder<N>(dst, oB, _border, p.border);
                    for (int x = oB; x < iB; ++x)
                        Base::ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, p.border, dst + x * N);
                }
                else
                {
                    for (int x = oB; x < iB; ++x)
                        Base::ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, dst + x * N, dst + x * N);
                }
                {
                    int x = iB, iEn = (int)AlignLo(iE - iB, n) + iB;
                    __m256 _y = _mm256_cvtepi32_ps(_mm256_set1_epi32(y));
                    __m256 _x = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(x), _01234567));
                    for (; x < iE; x += 8)
                    {
                        ByteBilinearPrepMain8(_x, _y, _m, _n, _s, offs + x, fx + 2 * x, fy + 2 * x);
                        _x = _mm256_add_ps(_x, _8);
                    }
                    ByteBilinearGather<M, soft>(src, src + s, offs + iB, iE - iB, rb0 + 2 * M * iB, rb1 + 2 * M * iB);
                    for (x = iB; x < iEn; x += n)
                        ByteBilinearInterpMainN<N>(rb0 + x * M * 2, rb1 + x * M * 2, fx + 2 * x, fy + 2 * x, dst + x * N);
                    for (; x < iE; ++x)
                        Base::ByteBilinearInterpMain<N>(rb0 + x * M * 2, rb1 + x * M * 2, fx + 2 * x, fy + 2 * x, dst + x * N);
                }
                if (fill)
                {
                    for (int x = iE; x < oE; ++x)
                        Base::ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, p.border, dst + x * N);
                    FillBorder<N>(dst + oE * N, width - oE, _border, p.border);
                }
                else
                {
                    for (int x = iE; x < oE; ++x)
                        Base::ByteBilinearInterpEdge<N>(x, y, p.inv, w, h, s, src, dst + x * N, dst + x * N);
                }
                dst += p.dstS;
            }
        }

        //-------------------------------------------------------------------------------------------------
        
        WarpAffineByteBilinear::WarpAffineByteBilinear(const WarpAffParam& param)
            : Sse41::WarpAffineByteBilinear(param)
        {
            bool soft = SlowGather;
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
            static const __m256i _01234567 = SIMD_MM256_SETR_EPI32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256 _w = _mm256_set1_ps(w), _z = _mm256_set1_ps(z);
            int y = 0, dH = (int)p.dstH, dH8 = (int)AlignLo(dH, 8);
            for (; y < dH8; y += 8)
            {
                _mm256_storeu_ps(min + y, _w);
                _mm256_storeu_ps(max + y, _mm256_setzero_ps());
            }
            for (; y < dH; ++y)
            {
                min[y] = w;
                max[y] = 0;
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
                int yEnd8 = (int)AlignLo(yEnd - yBeg, 8) + yBeg;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                __m256 _a = _mm256_set1_ps(a);
                __m256 _b = _mm256_set1_ps(b);
                __m256 _yMin = _mm256_set1_ps(yMin);
                __m256 _yMax = _mm256_set1_ps(yMax);
                for (y = yBeg; y < yEnd8; y += 8)
                {
                    __m256 _y = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(y), _01234567));
                    _y = _mm256_min_ps(_yMax, _mm256_max_ps(_y, _yMin));
                    __m256 _x = _mm256_add_ps(_mm256_mul_ps(_y, _a), _b);
                    _mm256_storeu_ps(min + y, _mm256_min_ps(_mm256_loadu_ps(min + y), _mm256_max_ps(_x, _z)));
                    _mm256_storeu_ps(max + y, _mm256_max_ps(_mm256_loadu_ps(max + y), _mm256_min_ps(_x, _w)));
                }
                for (; y < yEnd; ++y)
                {
                    float x = Simd::RestrictRange(float(y), yMin, yMax) * a + b;
                    min[y] = Simd::Min(min[y], Simd::Max(x, z));
                    max[y] = Simd::Max(max[y], Simd::Min(x, w));
                }
            }
            for (y = 0; y < dH8; y += 8)
            {
                __m256i _beg = _mm256_cvtps_epi32(_mm256_ceil_ps(_mm256_loadu_ps(min + y)));
                __m256i _end = _mm256_cvtps_epi32(_mm256_ceil_ps(_mm256_loadu_ps(max + y)));
                _mm256_storeu_si256((__m256i*)(beg + y), _beg);
                _mm256_storeu_si256((__m256i*)(end + y), _mm256_max_epi32(_beg, _end));
            }
            for (; y < dH; ++y)
            {
                beg[y] = (int)ceil(min[y]);
                end[y] = (int)ceil(max[y]);
                end[y] = Simd::Max(beg[y], end[y]);
            }
            if (hi)
            {
                for (y = 0; y < dH8; y += 8)
                {
                    __m256i _hi = _mm256_loadu_si256((__m256i*)(hi + y));
                    _mm256_storeu_si256((__m256i*)(beg + y), _mm256_min_epi32(_mm256_loadu_si256((__m256i*)(beg + y)), _hi));
                    _mm256_storeu_si256((__m256i*)(end + y), _mm256_min_epi32(_mm256_loadu_si256((__m256i*)(end + y)), _hi));
                }
                for (; y < dH; ++y)
                {
                    beg[y] = Simd::Min(beg[y], hi[y]);
                    end[y] = Simd::Min(end[y], hi[y]);
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
    }
#endif
}
