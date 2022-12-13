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
#include "Simd/SimdCopyPixel.h"

#include "Simd/SimdPoint.hpp"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        SIMD_INLINE __m128i NearestOffset(__m128 x, __m128 y, const __m128* m, __m128i w, __m128i h, __m128i n, __m128i s)
        {
            __m128 dx = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x, m[0]), _mm_mul_ps(y, m[1])), m[2]);
            __m128 dy = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x, m[3]), _mm_mul_ps(y, m[4])), m[5]);
            __m128i ix = _mm_min_epi32(_mm_max_epi32(_mm_cvtps_epi32(dx), _mm_setzero_si128()), w);
            __m128i iy = _mm_min_epi32(_mm_max_epi32(_mm_cvtps_epi32(dy), _mm_setzero_si128()), h);
            return _mm_add_epi32(_mm_mullo_epi32(ix, n), _mm_mullo_epi32(iy, s));
        }

        template<int N> void NearestRun(const WarpAffParam& p, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1, n = A / N;
            const __m128 _4 = _mm_set1_ps(4.0f);
            static const __m128i _0123 = SIMD_MM_SETR_EPI32(0, 1, 2, 3);
            __m128 _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = _mm_set1_ps(p.inv[i]);
            __m128i _w = _mm_set1_epi32(w);
            __m128i _h = _mm_set1_epi32(h);
            __m128i _n = _mm_set1_epi32(N);
            __m128i _s = _mm_set1_epi32(s);
            __m128i _border;
            switch (N)
            {
            case 1: _border = _mm_set1_epi8(*p.border); break;
            case 2: _border = _mm_set1_epi16(*(uint16_t*)p.border); break;
            case 4: _border = _mm_set1_epi32(*(uint32_t*)p.border); break;
            }
            for (int y = 0; y < (int)p.dstH; ++y)
            {
                int nose = beg[y], tail = end[y], tail4 = (int)AlignLo(tail - nose, 4) + nose;
                {
                    int x = nose;
                    __m128 _y = _mm_cvtepi32_ps(_mm_set1_epi32(y));
                    __m128 _x = _mm_cvtepi32_ps(_mm_add_epi32(_mm_set1_epi32(x), _0123));
                    for (; x < tail4; x += 4)
                    {
                        _mm_storeu_si128((__m128i*)(buf + x), NearestOffset(_x, _y, _m, _w, _h, _n, _s));
                        _x = _mm_add_ps(_x, _4);
                    }
                    for (; x < tail; ++x)
                        buf[x] = Base::NearestOffset<N>(x, y, p.inv, w, h, s);
                }
                if (N == 3)
                {
					if (fill)
					{
						int x = 0, nose1 = nose - 1;
						for (; x < nose1; ++x)
							Base::CopyPixel<4>(p.border, dst + x * 3);
						for (; x < nose; ++x)
							Base::CopyPixel<3>(p.border, dst + x * 3);
					}
					{
						int x = nose, tail1 = tail - 1;
						for (; x < tail1; ++x)
							Base::CopyPixel<4>(src + buf[x], dst + x * 3);
						for (; x < tail; ++x)
							Base::CopyPixel<3>(src + buf[x], dst + x * 3);
					}
					if (fill)
					{
						int x = tail, width1 = width - 1;
						for (; x < width1; ++x)
							Base::CopyPixel<4>(p.border, dst + x * 3);
						for (; x < width; ++x)
							Base::CopyPixel<3>(p.border, dst + x * 3);
					}
                }
                else
                {
                    if (fill)
                    {
                        int x = 0, noseN = (int)AlignLo(nose, n);
                        for (; x < noseN; x += n)
                            _mm_storeu_si128((__m128i*)(dst + x * N), _border);
                        for (; x < nose; ++x)
                            Base::CopyPixel<N>(p.border, dst + x * N);
                    }
                    {
                        for (int x = nose; x < tail; ++x)
                            Base::CopyPixel<N>(src + buf[x], dst + x * N);
                    }
                    if (fill)
                    {
                        int x = tail, widthN = (int)AlignLo(width - tail, n) + tail;
                        for (; x < widthN; x += n)
                            _mm_storeu_si128((__m128i*)(dst + x * N), _border);
                        for(; x < width; ++x)
                            Base::CopyPixel<N>(p.border, dst + x * N);
                    }
                }
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
                _mm_store_si128((__m128i*)(_beg.data + y), _w);
                _mm_store_si128((__m128i*)(_end.data + y), _mm_setzero_si128());
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
                int beg = Round(Simd::Max(Simd::Min(curr.y, next.y), 0.0f));
                int end = Round(Simd::Min(Simd::Max(curr.y, next.y), (float)p.dstH));
                int end4 = AlignLo(end - beg, 4) + beg;
                if (next.y == curr.y)
                    continue;
                float k = (next.x - curr.x) / (next.y - curr.y);
                __m128 _k = _mm_set1_ps(k);
                __m128 _y0 = _mm_set1_ps(curr.y);
                __m128 _x0 = _mm_set1_ps(curr.x);
                int y = beg;
                for (; y < end4; y += 4)
                {
                    __m128 _y = _mm_cvtepi32_ps(_mm_add_epi32(_mm_set1_epi32(y), _0123));
                    __m128i _x = _mm_cvtps_epi32(_mm_add_ps(_x0, _mm_mul_ps(_mm_sub_ps(_y, _y0), _k)));
                    __m128i _b = _mm_loadu_si128((__m128i*)(_beg.data + y));
                    __m128i _e = _mm_loadu_si128((__m128i*)(_end.data + y));
                    _b = _mm_min_epi32(_b, _mm_max_epi32(_x, _mm_setzero_si128()));
                    _e = _mm_max_epi32(_e, _mm_min_epi32(_mm_add_epi32(_x, _1), _w));
                    _mm_storeu_si128((__m128i*)(_beg.data + y), _b);
                    _mm_storeu_si128((__m128i*)(_end.data + y), _e);
                }
                for (; y < end; ++y)
                {
                    int x = Round(curr.x + (y - curr.y) * k);
                    _beg[y] = Simd::Min(_beg[y], Simd::Max(x, 0));
                    _end[y] = Simd::Max(_end[y], Simd::Min(x + 1, w));
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
                return new Base::WarpAffineByteBilinear(param);
            else
                return NULL;
        }
    }
#endif
}
