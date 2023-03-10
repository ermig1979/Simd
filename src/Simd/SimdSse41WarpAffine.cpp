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

#include "Simd/SimdPoint.hpp"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        template<int N> SIMD_INLINE void FillBorder(uint8_t* dst, int count, const __m128i & bv, const uint8_t * bs)
        {
            int i = 0, size = count * N, size16 = (int)AlignLo(size, 16);
            for (; i < size16; i += 16)
                _mm_storeu_si128((__m128i*)(dst + i), bv);
            for (; i < size; i += N)
                Base::CopyPixel<N>(bs, dst + i);
        }

        template<> SIMD_INLINE void FillBorder<3>(uint8_t* dst, int count, const __m128i& bv, const uint8_t* bs)
        {
            int i = 0, size = count * 3, size3 = size - 3;
            for (; i < size3; i += 3)
                Base::CopyPixel<4>(bs, dst + i);
            for (; i < size; i += 3)
                Base::CopyPixel<3>(bs, dst + i);
        }

        template<int N> SIMD_INLINE __m128i InitBorder(const uint8_t* border)
        {
            switch (N)
            {
            case 1: return _mm_set1_epi8(*border);
            case 2: return _mm_set1_epi16(*(uint16_t*)border);
            case 3: return _mm_setzero_si128();
            case 4: return _mm_set1_epi32(*(uint32_t*)border);
            }
            return _mm_setzero_si128();
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m128i NearestOffset(__m128 x, __m128 y, const __m128* m, __m128i w, const __m128i & h, const __m128i & n, const __m128i & s)
        {
            __m128 dx = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x, m[0]), _mm_mul_ps(y, m[1])), m[2]);
            __m128 dy = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x, m[3]), _mm_mul_ps(y, m[4])), m[5]);
            __m128i ix = _mm_min_epi32(_mm_max_epi32(_mm_cvtps_epi32(dx), _mm_setzero_si128()), w);
            __m128i iy = _mm_min_epi32(_mm_max_epi32(_mm_cvtps_epi32(dy), _mm_setzero_si128()), h);
            return _mm_add_epi32(_mm_mullo_epi32(ix, n), _mm_mullo_epi32(iy, s));
        }

        //-----------------------------------------------------------------------------------------

        template<int N> void NearestRun(const WarpAffParam& p, int yBeg, int yEnd, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1;
            const __m128 _4 = _mm_set1_ps(4.0f);
            static const __m128i _0123 = SIMD_MM_SETR_EPI32(0, 1, 2, 3);
            __m128 _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = _mm_set1_ps(p.inv[i]);
            __m128i _w = _mm_set1_epi32(w);
            __m128i _h = _mm_set1_epi32(h);
            __m128i _n = _mm_set1_epi32(N);
            __m128i _s = _mm_set1_epi32(s);
            __m128i _border = InitBorder<N>(p.border);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int nose = beg[y], tail = end[y];
                {
                    int x = nose;
                    __m128 _y = _mm_cvtepi32_ps(_mm_set1_epi32(y));
                    __m128 _x = _mm_cvtepi32_ps(_mm_add_epi32(_mm_set1_epi32(x), _0123));
                    for (; x < tail; x += 4)
                    {
                        _mm_storeu_si128((__m128i*)(buf + x), NearestOffset(_x, _y, _m, _w, _h, _n, _s));
                        _x = _mm_add_ps(_x, _4);
                    }
                }
				if (fill)
                    FillBorder<N>(dst, nose, _border, p.border);
                Base::NearestGather<N>(src, buf + nose, tail - nose, dst + N * nose);
                if (fill)
                    FillBorder<N>(dst + tail * N, width - tail, _border, p.border);
                dst += p.dstS;
            }
        }

        //-------------------------------------------------------------------------------------------------

        WarpAffineNearest::WarpAffineNearest(const WarpAffParam& param)
            : Base::WarpAffineNearest(param)
        {
            switch (_param.channels)
            {
            case 1: _run = NearestRun<1>; break;
            case 2: _run = NearestRun<2>; break;
            case 3: _run = NearestRun<3>; break;
            case 4: _run = NearestRun<4>; break;
            }
        }

        void WarpAffineNearest::SetRange(const Base::Point* points)
        {
            const WarpAffParam& p = _param;
            int w = (int)p.dstW, h = (int)p.dstH, h4 = (int)AlignLo(h, 4);
            static const __m128i _0123 = SIMD_MM_SETR_EPI32(0, 1, 2, 3);
            __m128i _w = _mm_set1_epi32(w), _1 = _mm_set1_epi32(1);
            int y = 0;
            for (; y < h4; y += 4)
            {
                _mm_storeu_si128((__m128i*)(_beg.data + y), _w);
                _mm_storeu_si128((__m128i*)(_end.data + y), _mm_setzero_si128());
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
                int yEnd4 = (int)AlignLo(yEnd - yBeg, 4) + yBeg;
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                __m128 _a = _mm_set1_ps(a);
                __m128 _b = _mm_set1_ps(b);
                if (abs(a) <= 1.0f)
                {
                    int y = yBeg;
                    for (; y < yEnd4; y += 4)
                    {
                        __m128 _y = _mm_cvtepi32_ps(_mm_add_epi32(_mm_set1_epi32(y), _0123));
                        __m128i _x = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(_y, _a), _b));
                        __m128i xBeg = _mm_loadu_si128((__m128i*)(_beg.data + y));
                        __m128i xEnd = _mm_loadu_si128((__m128i*)(_end.data + y));
                        xBeg = _mm_min_epi32(xBeg, _mm_max_epi32(_x, _mm_setzero_si128()));
                        xEnd = _mm_max_epi32(xEnd, _mm_min_epi32(_mm_add_epi32(_x, _1), _w));
                        _mm_storeu_si128((__m128i*)(_beg.data + y), xBeg);
                        _mm_storeu_si128((__m128i*)(_end.data + y), xEnd);
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
                    __m128 _05 = _mm_set1_ps(0.5f);
                    __m128 _yMin = _mm_set1_ps(yMin);
                    __m128 _yMax = _mm_set1_ps(yMax);
                    for (; y < yEnd4; y += 4)
                    {
                        __m128 _y = _mm_cvtepi32_ps(_mm_add_epi32(_mm_set1_epi32(y), _0123));
                        __m128 yM = _mm_min_ps(_mm_max_ps(_mm_sub_ps(_y, _05), _yMin), _yMax);
                        __m128 yP = _mm_min_ps(_mm_max_ps(_mm_add_ps(_y, _05), _yMin), _yMax);
                        __m128 xM = _mm_add_ps(_mm_mul_ps(yM, _a), _b);
                        __m128 xP = _mm_add_ps(_mm_mul_ps(yP, _a), _b);
                        __m128i xBeg = _mm_loadu_si128((__m128i*)(_beg.data + y));
                        __m128i xEnd = _mm_loadu_si128((__m128i*)(_end.data + y));
                        xBeg = _mm_min_epi32(xBeg, _mm_max_epi32(_mm_cvtps_epi32(_mm_min_ps(xM, xP)), _mm_setzero_si128()));
                        xEnd = _mm_max_epi32(xEnd, _mm_min_epi32(_mm_add_epi32(_mm_cvtps_epi32(_mm_max_ps(xM, xP)), _1), _w));
                        _mm_storeu_si128((__m128i*)(_beg.data + y), xBeg);
                        _mm_storeu_si128((__m128i*)(_end.data + y), xEnd);
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

        const __m128i K32_WA_FRACTION_RANGE = SIMD_MM_SET1_EPI32(Base::WA_FRACTION_RANGE);

        SIMD_INLINE void ByteBilinearPrepMain4(__m128 x, __m128 y, const __m128* m, __m128i n, const __m128i & s, uint32_t* offs, uint8_t* fx, uint16_t* fy)
        {
            __m128 dx = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x, m[0]), _mm_mul_ps(y, m[1])), m[2]);
            __m128 dy = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x, m[3]), _mm_mul_ps(y, m[4])), m[5]);
            __m128 ix = _mm_floor_ps(dx);
            __m128 iy = _mm_floor_ps(dy);
            __m128 range = _mm_cvtepi32_ps(K32_WA_FRACTION_RANGE);
            __m128i _fx = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(dx, ix), range));
            __m128i _fy = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(dy, iy), range));
            _mm_storeu_si128((__m128i*)offs, _mm_add_epi32(_mm_mullo_epi32(_mm_cvtps_epi32(ix), n), _mm_mullo_epi32(_mm_cvtps_epi32(iy), s)));
            _fx = _mm_or_si128(_mm_sub_epi32(K32_WA_FRACTION_RANGE, _fx), _mm_slli_epi32(_fx, 16));
            _fy = _mm_or_si128(_mm_sub_epi32(K32_WA_FRACTION_RANGE, _fy), _mm_slli_epi32(_fy, 16));
            _mm_storel_epi64((__m128i*)fx, _mm_packus_epi16(_fx, _mm_setzero_si128()));
            _mm_storeu_si128((__m128i*)fy, _fy);
        }

        //-------------------------------------------------------------------------------------------------

        const __m128i K32_WA_BILINEAR_ROUND_TERM = SIMD_MM_SET1_EPI32(Base::WA_BILINEAR_ROUND_TERM);

        template<int N> void ByteBilinearInterpMainN(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst);

        template<> SIMD_INLINE void ByteBilinearInterpMainN<1>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            __m128i fx0 = _mm_loadu_si128((__m128i*)fx + 0);
            __m128i fx1 = _mm_loadu_si128((__m128i*)fx + 1);
            __m128i r00 = _mm_maddubs_epi16(_mm_loadu_si128((__m128i*)src0 + 0), fx0);
            __m128i r01 = _mm_maddubs_epi16(_mm_loadu_si128((__m128i*)src0 + 1), fx1);
            __m128i r10 = _mm_maddubs_epi16(_mm_loadu_si128((__m128i*)src1 + 0), fx0);
            __m128i r11 = _mm_maddubs_epi16(_mm_loadu_si128((__m128i*)src1 + 1), fx1);

            __m128i s0 = _mm_madd_epi16(UnpackU16<0>(r00, r10), _mm_loadu_si128((__m128i*)fy + 0));
            __m128i d0 = _mm_srli_epi32(_mm_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i s1 = _mm_madd_epi16(UnpackU16<1>(r00, r10), _mm_loadu_si128((__m128i*)fy + 1));
            __m128i d1 = _mm_srli_epi32(_mm_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i s2 = _mm_madd_epi16(UnpackU16<0>(r01, r11), _mm_loadu_si128((__m128i*)fy + 2));
            __m128i d2 = _mm_srli_epi32(_mm_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i s3 = _mm_madd_epi16(UnpackU16<1>(r01, r11), _mm_loadu_si128((__m128i*)fy + 3));
            __m128i d3 = _mm_srli_epi32(_mm_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packus_epi32(d0, d1), _mm_packus_epi32(d2, d3)));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<2>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            static const __m128i SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

            __m128i _fx = _mm_loadu_si128((__m128i*)fx);
            __m128i fx0 = UnpackU16<0>(_fx, _fx);
            __m128i fx1 = UnpackU16<1>(_fx, _fx);
            __m128i r00 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src0 + 0), SHUFFLE), fx0);
            __m128i r01 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src0 + 1), SHUFFLE), fx1);
            __m128i r10 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src1 + 0), SHUFFLE), fx0);
            __m128i r11 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src1 + 1), SHUFFLE), fx1);

            __m128i fy0 = _mm_loadu_si128((__m128i*)fy + 0);
            __m128i s0 = _mm_madd_epi16(UnpackU16<0>(r00, r10), UnpackU32<0>(fy0, fy0));
            __m128i d0 = _mm_srli_epi32(_mm_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i s1 = _mm_madd_epi16(UnpackU16<1>(r00, r10), UnpackU32<1>(fy0, fy0));
            __m128i d1 = _mm_srli_epi32(_mm_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i fy1 = _mm_loadu_si128((__m128i*)fy + 1);
            __m128i s2 = _mm_madd_epi16(UnpackU16<0>(r01, r11), UnpackU32<0>(fy1, fy1));
            __m128i d2 = _mm_srli_epi32(_mm_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i s3 = _mm_madd_epi16(UnpackU16<1>(r01, r11), UnpackU32<1>(fy1, fy1));
            __m128i d3 = _mm_srli_epi32(_mm_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packus_epi32(d0, d1), _mm_packus_epi32(d2, d3)));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<3>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            static const __m128i SRC_SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x3, 0x1, 0x4, 0x2, 0x5, -1, -1, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, -1, -1);
            static const __m128i DST_SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);

            __m128i _fx = _mm_loadu_si128((__m128i*)fx);
            _fx = UnpackU16<0>(_fx, _fx);
            __m128i fx0 = UnpackU16<0>(_fx, _fx);
            __m128i fx1 = UnpackU16<1>(_fx, _fx);
            __m128i r00 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src0 + 0), SRC_SHUFFLE), fx0);
            __m128i r01 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src0 + 1), SRC_SHUFFLE), fx1);
            __m128i r10 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src1 + 0), SRC_SHUFFLE), fx0);
            __m128i r11 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src1 + 1), SRC_SHUFFLE), fx1);

            __m128i _fy = _mm_loadu_si128((__m128i*)fy);
            __m128i fy0 = UnpackU32<0>(_fy, _fy);
            __m128i s0 = _mm_madd_epi16(UnpackU16<0>(r00, r10), UnpackU32<0>(fy0, fy0));
            __m128i d0 = _mm_srli_epi32(_mm_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i s1 = _mm_madd_epi16(UnpackU16<1>(r00, r10), UnpackU32<1>(fy0, fy0));
            __m128i d1 = _mm_srli_epi32(_mm_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i fy1 = UnpackU32<1>(_fy, _fy);
            __m128i s2 = _mm_madd_epi16(UnpackU16<0>(r01, r11), UnpackU32<0>(fy1, fy1));
            __m128i d2 = _mm_srli_epi32(_mm_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i s3 = _mm_madd_epi16(UnpackU16<1>(r01, r11), UnpackU32<1>(fy1, fy1));
            __m128i d3 = _mm_srli_epi32(_mm_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            Store12(dst, _mm_shuffle_epi8(_mm_packus_epi16(_mm_packus_epi32(d0, d1), _mm_packus_epi32(d2, d3)), DST_SHUFFLE));
        }

        template<> SIMD_INLINE void ByteBilinearInterpMainN<4>(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t* dst)
        {
            static const __m128i SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

            __m128i _fx = _mm_loadu_si128((__m128i*)fx);
            _fx = UnpackU16<0>(_fx, _fx);
            __m128i fx0 = UnpackU16<0>(_fx, _fx);
            __m128i fx1 = UnpackU16<1>(_fx, _fx);
            __m128i r00 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src0 + 0), SHUFFLE), fx0);
            __m128i r01 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src0 + 1), SHUFFLE), fx1);
            __m128i r10 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src1 + 0), SHUFFLE), fx0);
            __m128i r11 = _mm_maddubs_epi16(_mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src1 + 1), SHUFFLE), fx1);

            __m128i _fy = _mm_loadu_si128((__m128i*)fy);
            __m128i fy0 = UnpackU32<0>(_fy, _fy);
            __m128i s0 = _mm_madd_epi16(UnpackU16<0>(r00, r10), UnpackU32<0>(fy0, fy0));
            __m128i d0 = _mm_srli_epi32(_mm_add_epi32(s0, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i s1 = _mm_madd_epi16(UnpackU16<1>(r00, r10), UnpackU32<1>(fy0, fy0));
            __m128i d1 = _mm_srli_epi32(_mm_add_epi32(s1, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i fy1 = UnpackU32<1>(_fy, _fy);
            __m128i s2 = _mm_madd_epi16(UnpackU16<0>(r01, r11), UnpackU32<0>(fy1, fy1));
            __m128i d2 = _mm_srli_epi32(_mm_add_epi32(s2, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            __m128i s3 = _mm_madd_epi16(UnpackU16<1>(r01, r11), UnpackU32<1>(fy1, fy1));
            __m128i d3 = _mm_srli_epi32(_mm_add_epi32(s3, K32_WA_BILINEAR_ROUND_TERM), Base::WA_BILINEAR_SHIFT);

            _mm_storeu_si128((__m128i*)dst, _mm_packus_epi16(_mm_packus_epi32(d0, d1), _mm_packus_epi32(d2, d3)));
        }

        //-------------------------------------------------------------------------------------------------

        template<int N> void ByteBilinearRun(const WarpAffParam& p, int yBeg, int yEnd, const int* ib, const int* ie, const int* ob, const int* oe, const uint8_t* src, uint8_t* dst, uint8_t* buf)
        {
            constexpr int M = (N == 3 ? 4 : N);
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 2, h = (int)p.srcH - 2, n = A / M;
            size_t wa = AlignHi(p.dstW, p.align) + p.align;
            uint32_t* offs = (uint32_t*)buf;
            uint8_t* fx = (uint8_t *)(offs + wa);
            uint16_t* fy = (uint16_t*)(fx + wa * 2);
            uint8_t* rb0 = (uint8_t*)(fy + wa * 2);
            uint8_t* rb1 = (uint8_t*)(rb0 + wa * M * 2);
            const __m128 _4 = _mm_set1_ps(4.0f);
            static const __m128i _0123 = SIMD_MM_SETR_EPI32(0, 1, 2, 3);
            __m128 _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = _mm_set1_ps(p.inv[i]);
            __m128i _n = _mm_set1_epi32(N);
            __m128i _s = _mm_set1_epi32(s);
            __m128i _border = InitBorder<N>(p.border);
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
                    __m128 _y = _mm_cvtepi32_ps(_mm_set1_epi32(y));
                    __m128 _x = _mm_cvtepi32_ps(_mm_add_epi32(_mm_set1_epi32(x), _0123));
                    for (; x < iE; x += 4)
                    {
                        ByteBilinearPrepMain4(_x, _y, _m, _n, _s, offs + x, fx + 2 * x, fy + 2 * x);
                        _x = _mm_add_ps(_x, _4);
                    }
                    Base::ByteBilinearGather<M>(src, src + s, offs + iB, iE - iB, rb0 + 2 * M * iB, rb1 + 2 * M * iB);
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
            : Base::WarpAffineByteBilinear(param)
        {
            switch (_param.channels)
            {
            case 1: _run = ByteBilinearRun<1>; break;
            case 2: _run = ByteBilinearRun<2>; break;
            case 3: _run = ByteBilinearRun<3>; break;
            case 4: _run = ByteBilinearRun<4>; break;
            }
        }

        void WarpAffineByteBilinear::SetRange(const Base::Point* rect, int* beg, int* end, const int* lo, const int* hi)
        {
            const WarpAffParam& p = _param;
            float* min = (float*)_buf.data;
            float* max = min + p.dstH;
            float w = (float)p.dstW, h = (float)p.dstH, z = 0.0f;
            static const __m128i _0123 = SIMD_MM_SETR_EPI32(0, 1, 2, 3);
            __m128 _w = _mm_set1_ps(w), _z = _mm_set1_ps(z);
            int y = 0, dH = (int)p.dstH, dH4 = (int)AlignLo(dH, 4);
            for (; y < dH4; y += 4)
            {
                _mm_storeu_ps(min + y, _w);
                _mm_storeu_ps(max + y, _mm_setzero_ps());
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
                int yEnd4 = (int)AlignLo(yEnd - yBeg, 4) + yBeg;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                __m128 _a = _mm_set1_ps(a);
                __m128 _b = _mm_set1_ps(b);
                __m128 _yMin = _mm_set1_ps(yMin);
                __m128 _yMax = _mm_set1_ps(yMax);
                for (y = yBeg; y < yEnd4; y += 4)
                {
                    __m128 _y = _mm_cvtepi32_ps(_mm_add_epi32(_mm_set1_epi32(y), _0123));
                    _y = _mm_min_ps(_yMax, _mm_max_ps(_y, _yMin));
                    __m128 _x = _mm_add_ps(_mm_mul_ps(_y, _a), _b);
                    _mm_storeu_ps(min + y, _mm_min_ps(_mm_loadu_ps(min + y), _mm_max_ps(_x, _z)));
                    _mm_storeu_ps(max + y, _mm_max_ps(_mm_loadu_ps(max + y), _mm_min_ps(_x, _w)));
                }
                for (; y < yEnd; ++y)
                {
                    float x = Simd::RestrictRange(float(y), yMin, yMax) * a + b;
                    min[y] = Simd::Min(min[y], Simd::Max(x, z));
                    max[y] = Simd::Max(max[y], Simd::Min(x, w));
                }
            }
            for (y = 0; y < dH4; y += 4)
            {
                __m128i _beg = _mm_cvtps_epi32(_mm_ceil_ps(_mm_loadu_ps(min + y)));
                __m128i _end = _mm_cvtps_epi32(_mm_ceil_ps(_mm_loadu_ps(max + y)));
                _mm_storeu_si128((__m128i*)(beg + y), _beg);
                _mm_storeu_si128((__m128i*)(end + y), _mm_max_epi32(_beg, _end));
            }
            for (; y < dH; ++y)
            {
                beg[y] = (int)ceil(min[y]);
                end[y] = (int)ceil(max[y]);
                end[y] = Simd::Max(beg[y], end[y]);
            }
            if (hi)
            {
                for (y = 0; y < dH4; y += 4)
                {
                    __m128i _hi = _mm_loadu_si128((__m128i*)(hi + y));
                    _mm_storeu_si128((__m128i*)(beg + y), _mm_min_epi32(_mm_loadu_si128((__m128i*)(beg + y)), _hi));
                    _mm_storeu_si128((__m128i*)(end + y), _mm_min_epi32(_mm_loadu_si128((__m128i*)(end + y)), _hi));
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
