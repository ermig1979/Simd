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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdCopyPixel.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        ResizerByteBicubic::ResizerByteBicubic(const ResParam& param)
            : Base::ResizerByteBicubic(param)
        {
        }

        void ResizerByteBicubic::EstimateIndexAlphaY()
        {
            size_t sizeD = _param.dstH, sizeS = _param.srcH;
            _iy.Resize(sizeD);
            _ay.Resize(sizeD * 4);
            float scale = float(sizeS) / float(sizeD);
            size_t i = 0, sizeDF = AlignLo(sizeD, F);
            int32_t* ay = _ay.data;
            if (sizeDF)
            {
                __m128i _i = _mm_setr_epi32(0, 1, 2, 3);
                __m128 _scale = _mm_set1_ps(scale);
                __m128 _0 = _mm_set1_ps(0.0f);
                __m128 _05 = _mm_set1_ps(0.5f);
                __m128 _1 = _mm_set1_ps(1.0f);
                __m128 _2 = _mm_set1_ps(2.0f);
                __m128 _1_6 = _mm_set1_ps(1.0f / 6.0f);
                __m128 _max = _mm_set1_ps(float(sizeS - 2));
                __m128 _range = _mm_set1_ps(float(Base::BICUBIC_RANGE));

                for (; i < sizeDF; i += F, ay += 4 * F)
                {
                    __m128 _pos = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_cvtepi32_ps(_i), _05), _scale), _05);
                    __m128 idx = _mm_round_ps(_pos, _MM_FROUND_FLOOR);
                    __m128 d = _mm_sub_ps(_pos, idx);

                    __m128 minMask = _mm_cmplt_ps(idx, _0);
                    idx = _mm_blendv_ps(idx, _0, minMask);
                    d = _mm_blendv_ps(d, _0, minMask);

                    __m128 maxMask = _mm_cmpgt_ps(idx, _max);
                    idx = _mm_blendv_ps(idx, _max, maxMask);
                    d = _mm_blendv_ps(d, _1, maxMask);

                    _mm_storeu_si128((__m128i*)(_iy.data + i), _mm_cvtps_epi32(idx));

                    __m128i a0 = _mm_cvtps_epi32(_mm_mul_ps(_range, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_2, d), _mm_sub_ps(_1, d)), _mm_mul_ps(d, _1_6))));
                    __m128i a1 = _mm_cvtps_epi32(_mm_mul_ps(_range, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(d, _2), _mm_add_ps(_1, d)), _mm_mul_ps(_mm_sub_ps(_1, d), _05))));
                    __m128i a2 = _mm_cvtps_epi32(_mm_mul_ps(_range, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(d, _2), _mm_add_ps(_1, d)), _mm_mul_ps(d, _05))));
                    __m128i a3 = _mm_cvtps_epi32(_mm_mul_ps(_range, _mm_mul_ps(_mm_mul_ps(_mm_add_ps(_1, d), _mm_sub_ps(_1, d)), _mm_mul_ps(d, _1_6))));
                    __m128i a00 = _mm_unpacklo_epi32(a0, a2);
                    __m128i a01 = _mm_unpacklo_epi32(a1, a3);
                    __m128i a10 = _mm_unpackhi_epi32(a0, a2);
                    __m128i a11 = _mm_unpackhi_epi32(a1, a3);
                    _mm_storeu_si128((__m128i*)ay + 0, _mm_unpacklo_epi32(a00, a01));
                    _mm_storeu_si128((__m128i*)ay + 1, _mm_unpackhi_epi32(a00, a01));
                    _mm_storeu_si128((__m128i*)ay + 2, _mm_unpacklo_epi32(a10, a11));
                    _mm_storeu_si128((__m128i*)ay + 3, _mm_unpackhi_epi32(a10, a11));

                    _i = _mm_add_epi32(_i, K32_00000004);
                }
            }
            for (; i < sizeD; ++i, ay += 4)
            {
                float pos = (float)((i + 0.5f) * scale - 0.5f);
                int idx = (int)::floor(pos);
                float d = pos - idx;
                if (idx < 0)
                {
                    idx = 0;
                    d = 0.0f;
                }
                if (idx > (int)sizeS - 2)
                {
                    idx = (int)sizeS - 2;
                    d = 1.0f;
                }
                _iy[i] = idx;
                ay[0] = Round(Base::BICUBIC_RANGE * (2.0f - d) * (1.0f - d) * d / 6.0f);
                ay[1] = Round(Base::BICUBIC_RANGE * (d - 2.0f) * (d + 1.0f) * (1.0f - d) / 2.0f);
                ay[2] = Round(Base::BICUBIC_RANGE * (d - 2.0f) * (d + 1.0f) * d / 2.0f);
                ay[3] = Round(Base::BICUBIC_RANGE * (1.0f + d) * (1.0f - d) * d / 6.0f);
            }
        }

        void ResizerByteBicubic::EstimateIndexAlphaX()
        {
            size_t sizeD = _param.dstW, sizeS = _param.srcW;
            _ix.Resize(sizeD);
            _ax.Resize(sizeD * 4);
            float scale = float(sizeS) / float(sizeD);
            size_t i = 0, sizeDF = AlignLo(sizeD, F);
            int8_t* ax = _ax.data;
            if (sizeDF)
            {
                static const __m128i _SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
                __m128i _i = _mm_setr_epi32(0, 1, 2, 3);
                __m128 _scale = _mm_set1_ps(scale);
                __m128 _0 = _mm_set1_ps(0.0f);
                __m128 _05 = _mm_set1_ps(0.5f);
                __m128 _1 = _mm_set1_ps(1.0f);
                __m128 _2 = _mm_set1_ps(2.0f);
                __m128 _1_6 = _mm_set1_ps(1.0f / 6.0f);
                __m128 _max = _mm_set1_ps(float(sizeS - 2));
                __m128 _range = _mm_set1_ps(float(Base::BICUBIC_RANGE));
                __m128i _channels = _mm_set1_epi32((int)_param.channels);

                for (; i < sizeDF; i += F, ax += 4 * F)
                {
                    __m128 _pos = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_cvtepi32_ps(_i), _05), _scale), _05);
                    __m128 idx = _mm_round_ps(_pos, _MM_FROUND_FLOOR);
                    __m128 d = _mm_sub_ps(_pos, idx);

                    __m128 minMask = _mm_cmplt_ps(idx, _0);
                    idx = _mm_blendv_ps(idx, _0, minMask);
                    d = _mm_blendv_ps(d, _0, minMask);

                    __m128 maxMask = _mm_cmpgt_ps(idx, _max);
                    idx = _mm_blendv_ps(idx, _max, maxMask);
                    d = _mm_blendv_ps(d, _1, maxMask);

                    _mm_storeu_si128((__m128i*)(_ix.data + i), _mm_mullo_epi32(_mm_cvtps_epi32(idx), _channels));

                    __m128i a0 = _mm_cvtps_epi32(_mm_mul_ps(_range, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_2, d), _mm_sub_ps(_1, d)), _mm_mul_ps(d, _1_6))));
                    __m128i a1 = _mm_cvtps_epi32(_mm_mul_ps(_range, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(d, _2), _mm_add_ps(_1, d)), _mm_mul_ps(_mm_sub_ps(_1, d), _05))));
                    __m128i a2 = _mm_cvtps_epi32(_mm_mul_ps(_range, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(d, _2), _mm_add_ps(_1, d)), _mm_mul_ps(d, _05))));
                    __m128i a3 = _mm_cvtps_epi32(_mm_mul_ps(_range, _mm_mul_ps(_mm_mul_ps(_mm_add_ps(_1, d), _mm_sub_ps(_1, d)), _mm_mul_ps(d, _1_6))));
                    _mm_storeu_si128((__m128i*)ax, _mm_shuffle_epi8(_mm_packs_epi16(_mm_packs_epi32(a0, a1), _mm_packs_epi32(a2, a3)), _SHUFFLE));

                    _i = _mm_add_epi32(_i, K32_00000004);
                }
            }
            for (; i < sizeD; ++i, ax += 4)
            {
                float pos = (float)((i + 0.5f) * scale - 0.5f);
                int idx = (int)::floor(pos);
                float d = pos - idx;
                if (idx < 0)
                {
                    idx = 0;
                    d = 0.0f;
                }
                if (idx > (int)sizeS - 2)
                {
                    idx = (int)sizeS - 2;
                    d = 1.0f;
                }
                _ix[i] = idx * (int)_param.channels;
                ax[0] = (int8_t)Round(Base::BICUBIC_RANGE * (2.0f - d) * (1.0f - d) * d / 6.0f);
                ax[1] = (int8_t)Round(Base::BICUBIC_RANGE * (d - 2.0f) * (d + 1.0f) * (1.0f - d) / 2.0f);
                ax[2] = (int8_t)Round(Base::BICUBIC_RANGE * (d - 2.0f) * (d + 1.0f) * d / 2.0f);
                ax[3] = (int8_t)Round(Base::BICUBIC_RANGE * (1.0f + d) * (1.0f - d) * d / 6.0f);
            }
        }

        void ResizerByteBicubic::ResizerByteBicubic::Init(bool sparse)
        {
            if (_iy.data)
                return;
            EstimateIndexAlphaY();
            EstimateIndexAlphaX();
            if (!sparse)
            {
                for (int i = 0; i < 4; ++i)
                    _bx[i].Resize(_param.dstW * _param.channels);
            }
            _sxl = (_param.srcW - 2) * _param.channels;
            for (_xn = 0; _ix[_xn] == 0; _xn++);
            for (_xt = _param.dstW; _ix[_xt - 1] == _sxl; _xt--);
        }

        template<int N> __m128i LoadAx(const int8_t* ax);

        template<> SIMD_INLINE __m128i LoadAx<1>(const int8_t* ax)
        {
            return _mm_loadu_si128((__m128i*)ax);
        }

        template<> SIMD_INLINE __m128i LoadAx<2>(const int8_t* ax)
        {
            return _mm_shuffle_epi32(_mm_loadl_epi64((__m128i*)ax), 0x50);
        }

        template<> SIMD_INLINE __m128i LoadAx<3>(const int8_t* ax)
        {
            return _mm_set1_epi32(*(int32_t*)ax);
        }

        template<> SIMD_INLINE __m128i LoadAx<4>(const int8_t* ax)
        {
            return _mm_set1_epi32(*(int32_t*)ax);
        }

        template<int N> __m128i CubicSumX(const uint8_t* src, const int32_t* ix, __m128i ax, __m128i ay);

        template<> SIMD_INLINE __m128i CubicSumX<1>(const uint8_t* src, const int32_t* ix, __m128i ax, __m128i ay)
        {
            __m128i _src = _mm_setr_epi32(*(int32_t*)(src + ix[0]), *(int32_t*)(src + ix[1]), *(int32_t*)(src + ix[2]), *(int32_t*)(src + ix[3]));
            return _mm_madd_epi16(_mm_maddubs_epi16(_src, ax), ay);
        }

        template<> SIMD_INLINE __m128i CubicSumX<2>(const uint8_t* src, const int32_t* ix, __m128i ax, __m128i ay)
        {
            static const __m128i SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF);
            __m128i _src = _mm_shuffle_epi8(Load((__m128i*)(src + ix[0]), (__m128i*)(src + ix[1])), SHUFFLE);
            return _mm_madd_epi16(_mm_maddubs_epi16(_src, ax), ay);
        }

        template<> SIMD_INLINE __m128i CubicSumX<3>(const uint8_t* src, const int32_t* ix, __m128i ax, __m128i ay)
        {
            static const __m128i SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1);
            __m128i _src = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(src + ix[0])), SHUFFLE);
            return _mm_madd_epi16(_mm_maddubs_epi16(_src, ax), ay);
        }

        template<> SIMD_INLINE __m128i CubicSumX<4>(const uint8_t* src, const int32_t* ix, __m128i ax, __m128i ay)
        {
            static const __m128i SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
            __m128i _src = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(src + ix[0])), SHUFFLE);
            return _mm_madd_epi16(_mm_maddubs_epi16(_src, ax), ay);
        }

        template <int N> SIMD_INLINE void BicubicInt(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, const int32_t* ix, const int8_t* ax, const __m128i* ay, uint8_t* dst)
        {
            static const __m128i ROUND = SIMD_MM_SET1_EPI32(Base::BICUBIC_ROUND);
            __m128i _ax = LoadAx<N>(ax);
            __m128i say0 = CubicSumX<N>(src0 - N, ix, _ax, ay[0]);
            __m128i say1 = CubicSumX<N>(src1 - N, ix, _ax, ay[1]);
            __m128i say2 = CubicSumX<N>(src2 - N, ix, _ax, ay[2]);
            __m128i say3 = CubicSumX<N>(src3 - N, ix, _ax, ay[3]);
            __m128i sum = _mm_add_epi32(_mm_add_epi32(say0, say1), _mm_add_epi32(say2, say3));
            __m128i dst0 = _mm_srai_epi32(_mm_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
            *((int32_t*)(dst)) = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(dst0, K_ZERO), K_ZERO));
        }

        template<int N> void ResizerByteBicubic::RunS(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_xn == 0 && _xt == _param.dstW);
            size_t step = 4 / N;
            size_t body = AlignLoAny(_param.dstW, step);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                const int32_t* ay = _ay.data + dy * 4;
                __m128i ays[4];
                ays[0] = _mm_set1_epi16(ay[0]);
                ays[1] = _mm_set1_epi16(ay[1]);
                ays[2] = _mm_set1_epi16(ay[2]);
                ays[3] = _mm_set1_epi16(ay[3]);
                size_t dx = 0;
                for (; dx < body; dx += step)
                    BicubicInt<N>(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx * N);
                for (; dx < _param.dstW; dx++)
                    Base::BicubicInt<N, -1, 2>(src0, src1, src2, src3, _ix[dx], _ax.data + dx * 4, ay, dst + dx * N);
            }
        }

        template<int F> SIMD_INLINE void PixelCubicSumX(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst);

        template<> SIMD_INLINE void PixelCubicSumX<1>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            __m128i _src = _mm_setr_epi32(*(int32_t*)(src + ix[0]), *(int32_t*)(src + ix[1]), *(int32_t*)(src + ix[2]), *(int32_t*)(src + ix[3]));
            __m128i _ax = _mm_loadu_si128((__m128i*)ax);
            _mm_storeu_si128((__m128i*)dst, _mm_madd_epi16(_mm_maddubs_epi16(_src, _ax), Sse2::K16_0001));
        }

        template<> SIMD_INLINE void PixelCubicSumX<2>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            static const __m128i SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF);
            __m128i _src = _mm_shuffle_epi8(Load((__m128i*)(src + ix[0]), (__m128i*)(src + ix[1])), SHUFFLE);
            __m128i _ax = _mm_shuffle_epi32(_mm_loadl_epi64((__m128i*)ax), 0x50);
            _mm_storeu_si128((__m128i*)dst, _mm_madd_epi16(_mm_maddubs_epi16(_src, _ax), Sse2::K16_0001));
        }

        template<> SIMD_INLINE void PixelCubicSumX<3>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            static const __m128i SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1);
            __m128i _src = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(src + ix[0])), SHUFFLE);
            __m128i _ax = _mm_set1_epi32(*(int32_t*)ax);
            _mm_storeu_si128((__m128i*)dst, _mm_madd_epi16(_mm_maddubs_epi16(_src, _ax), Sse2::K16_0001));
        }

        template<> SIMD_INLINE void PixelCubicSumX<4>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            static const __m128i SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
            __m128i _src = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(src + ix[0])), SHUFFLE);
            __m128i _ax = _mm_set1_epi32(*(int32_t*)ax);
            _mm_storeu_si128((__m128i*)dst, _mm_madd_epi16(_mm_maddubs_epi16(_src, _ax), Sse2::K16_0001));
        }

        template<int N> SIMD_INLINE void RowCubicSumX(const uint8_t* src, size_t nose, size_t body, size_t tail, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            size_t step = 4 / N;
            size_t bodyS = nose + AlignLoAny(body - nose, step);

            size_t dx = 0;
            for (; dx < nose; dx++, ax += 4, dst += N)
                Base::PixelCubicSumX<N, 0, 2>(src + ix[dx], ax, dst);
            for (; dx < bodyS; dx += step, ax += 4 * step, dst += N * step)
                PixelCubicSumX<N>(src - N, ix + dx, ax, dst);
            for (; dx < body; dx++, ax += 4, dst += N)
                Base::PixelCubicSumX<N, -1, 2>(src + ix[dx], ax, dst);
            for (; dx < tail; dx++, ax += 4, dst += N)
                Base::PixelCubicSumX<N, -1, 1>(src + ix[dx], ax, dst);
        }

        SIMD_INLINE void BicubicRowInt(const int32_t* src0, const int32_t* src1, const int32_t* src2, const int32_t* src3, size_t n, const int32_t* ay, uint8_t* dst)
        {
            size_t nF = AlignLo(n, F);
            size_t i = 0;
            if (nF)
            {
                static const __m128i ROUND = SIMD_MM_SET1_EPI32(Base::BICUBIC_ROUND);
                __m128i ay0 = _mm_set1_epi32(ay[0]);
                __m128i ay1 = _mm_set1_epi32(ay[1]);
                __m128i ay2 = _mm_set1_epi32(ay[2]);
                __m128i ay3 = _mm_set1_epi32(ay[3]);
                for (; i < nF; i += F)
                {
                    __m128i say0 = _mm_mullo_epi32(_mm_loadu_si128((__m128i*)(src0 + i)), ay0);
                    __m128i say1 = _mm_mullo_epi32(_mm_loadu_si128((__m128i*)(src1 + i)), ay1);
                    __m128i say2 = _mm_mullo_epi32(_mm_loadu_si128((__m128i*)(src2 + i)), ay2);
                    __m128i say3 = _mm_mullo_epi32(_mm_loadu_si128((__m128i*)(src3 + i)), ay3);
                    __m128i sum = _mm_add_epi32(_mm_add_epi32(say0, say1), _mm_add_epi32(say2, say3));
                    __m128i dst0 = _mm_srai_epi32(_mm_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
                   *((int32_t*)(dst + i)) = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(dst0, K_ZERO), K_ZERO));
                }
            }
            for (; i < n; ++i)
            {
                int32_t sum = ay[0] * src0[i] + ay[1] * src1[i] + ay[2] * src2[i] + ay[3] * src3[i];
                dst[i] = Base::RestrictRange((sum + Base::BICUBIC_ROUND) >> Base::BICUBIC_SHIFT, 0, 255);
            }
        }

        template<int N> void ResizerByteBicubic::RunB(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            int32_t prev = -1;
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                int32_t sy = _iy[dy], next = prev;
                for (int32_t curr = sy - 1, end = sy + 3; curr < end; ++curr)
                {
                    if (curr < prev)
                        continue;
                    const uint8_t* ps = src + RestrictRange(curr, 0, (int)_param.srcH - 1) * srcStride;
                    int32_t* pb = _bx[(curr + 1) & 3].data;
                    RowCubicSumX<N>(ps, _xn, _xt, _param.dstW, _ix.data, _ax.data, pb);
                    next++;
                }
                prev = next;

                const int32_t* ay = _ay.data + dy * 4;
                int32_t* pb0 = _bx[(sy + 0) & 3].data;
                int32_t* pb1 = _bx[(sy + 1) & 3].data;
                int32_t* pb2 = _bx[(sy + 2) & 3].data;
                int32_t* pb3 = _bx[(sy + 3) & 3].data;
                BicubicRowInt(pb0, pb1, pb2, pb3, _bx[0].size, ay, dst);
            }
        }

        void ResizerByteBicubic::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            bool sparse = _param.dstH * 3.0 <= _param.srcH;
            Init(sparse);
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
    }
#endif//SIMD_SSE41_ENABLE
}

