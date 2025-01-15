/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        ResizerByteBilinear::ResizerByteBilinear(const ResParam& param)
            : Base::ResizerByteBilinear(param)
            , _blocks(0)
        {
        }

        size_t ResizerByteBilinear::BlockCountMax(size_t align)
        {
            return (size_t)Simd::Max(::ceil(float(_param.srcW) / (align - 1)), ::ceil(float(_param.dstW) * 2.0f / align));
        }

        void ResizerByteBilinear::EstimateParams()
        {
            if (_ax.data)
                return;
            if (_param.channels == 1 && _param.srcW < 4 * _param.dstW)
                _blocks = BlockCountMax(A);
            float scale = (float)_param.srcW / _param.dstW;
            _ax.Resize(AlignHi(_param.dstW, A) * _param.channels * 2, false, _param.align);
            uint8_t* alphas = _ax.data;
            if (_blocks)
            {
                _ixg.Resize(_blocks);
                int block = 0;
                _ixg[0].src = 0;
                _ixg[0].dst = 0;
                for (int dstIndex = 0; dstIndex < (int)_param.dstW; ++dstIndex)
                {
                    float alpha = (float)((dstIndex + 0.5) * scale - 0.5);
                    int srcIndex = (int)::floor(alpha);
                    alpha -= srcIndex;

                    if (srcIndex < 0)
                    {
                        srcIndex = 0;
                        alpha = 0;
                    }

                    if (srcIndex > (int)_param.srcW - 2)
                    {
                        srcIndex = (int)_param.srcW - 2;
                        alpha = 1;
                    }

                    int dst = 2 * dstIndex - _ixg[block].dst;
                    int src = srcIndex - _ixg[block].src;
                    if (src >= A - 1 || dst >= A)
                    {
                        block++;
                        _ixg[block].src = Simd::Min(srcIndex, int(_param.srcW - A));
                        _ixg[block].dst = 2 * dstIndex;
                        dst = 0;
                        src = srcIndex - _ixg[block].src;
                    }
                    _ixg[block].shuffle[dst] = src;
                    _ixg[block].shuffle[dst + 1] = src + 1;

                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    alphas += 2;
                }
                _blocks = block + 1;
            }
            else
            {
                _ix.Resize(_param.dstW);
                for (size_t i = 0; i < _param.dstW; ++i)
                {
                    float alpha = (float)((i + 0.5) * scale - 0.5);
                    ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                    alpha -= index;

                    if (index < 0)
                    {
                        index = 0;
                        alpha = 0;
                    }

                    if (index > (ptrdiff_t)_param.srcW - 2)
                    {
                        index = _param.srcW - 2;
                        alpha = 1;
                    }

                    _ix[i] = (int)index;
                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    for (size_t channel = 1; channel < _param.channels; channel++)
                        ((uint16_t*)alphas)[channel] = *(uint16_t*)alphas;
                    alphas += 2 * _param.channels;
                }
            }
            size_t size = AlignHi(_param.dstW, _param.align) * _param.channels * 2 + SIMD_ALIGN;
            _bx[0].Resize(size, false, _param.align);
            _bx[1].Resize(size, false, _param.align);
        }

        template <size_t N> void ResizerByteBilinearInterpolateX(const __m128i* alpha, __m128i* buffer);

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<1>(const __m128i* alpha, __m128i* buffer)
        {
            _mm_store_si128(buffer, _mm_maddubs_epi16(_mm_load_si128(buffer), _mm_load_si128(alpha)));
        }

        const __m128i K8_SHUFFLE_X2 = SIMD_MM_SETR_EPI8(0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

        SIMD_INLINE void ResizerByteBilinearInterpolateX2(const __m128i* alpha, __m128i* buffer)
        {
            __m128i src = _mm_shuffle_epi8(_mm_load_si128(buffer), K8_SHUFFLE_X2);
            _mm_store_si128(buffer, _mm_maddubs_epi16(src, _mm_load_si128(alpha)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<2>(const __m128i* alpha, __m128i* buffer)
        {
            ResizerByteBilinearInterpolateX2(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX2(alpha + 1, buffer + 1);
        }

        const __m128i K8_SHUFFLE_X3_00 = SIMD_MM_SETR_EPI8(0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1);
        const __m128i K8_SHUFFLE_X3_01 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0);
        const __m128i K8_SHUFFLE_X3_10 = SIMD_MM_SETR_EPI8(0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_X3_11 = SIMD_MM_SETR_EPI8(-1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1);
        const __m128i K8_SHUFFLE_X3_12 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1);
        const __m128i K8_SHUFFLE_X3_21 = SIMD_MM_SETR_EPI8(0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_SHUFFLE_X3_22 = SIMD_MM_SETR_EPI8(-1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF);

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<3>(const __m128i* alpha, __m128i* buffer)
        {
            __m128i src[3], shuffled[3];
            src[0] = _mm_load_si128(buffer + 0);
            src[1] = _mm_load_si128(buffer + 1);
            src[2] = _mm_load_si128(buffer + 2);
            shuffled[0] = _mm_shuffle_epi8(src[0], K8_SHUFFLE_X3_00);
            shuffled[0] = _mm_or_si128(shuffled[0], _mm_shuffle_epi8(src[1], K8_SHUFFLE_X3_01));
            _mm_store_si128(buffer + 0, _mm_maddubs_epi16(shuffled[0], _mm_load_si128(alpha + 0)));
            shuffled[1] = _mm_shuffle_epi8(src[0], K8_SHUFFLE_X3_10);
            shuffled[1] = _mm_or_si128(shuffled[1], _mm_shuffle_epi8(src[1], K8_SHUFFLE_X3_11));
            shuffled[1] = _mm_or_si128(shuffled[1], _mm_shuffle_epi8(src[2], K8_SHUFFLE_X3_12));
            _mm_store_si128(buffer + 1, _mm_maddubs_epi16(shuffled[1], _mm_load_si128(alpha + 1)));
            shuffled[2] = _mm_shuffle_epi8(src[1], K8_SHUFFLE_X3_21);
            shuffled[2] = _mm_or_si128(shuffled[2], _mm_shuffle_epi8(src[2], K8_SHUFFLE_X3_22));
            _mm_store_si128(buffer + 2, _mm_maddubs_epi16(shuffled[2], _mm_load_si128(alpha + 2)));
        }

        const __m128i K8_SHUFFLE_X4 = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

        SIMD_INLINE void ResizerByteBilinearInterpolateX4(const __m128i* alpha, __m128i* buffer)
        {
            __m128i src = _mm_shuffle_epi8(_mm_load_si128(buffer), K8_SHUFFLE_X4);
            _mm_store_si128(buffer, _mm_maddubs_epi16(src, _mm_load_si128(alpha)));
        }

        template <> SIMD_INLINE void ResizerByteBilinearInterpolateX<4>(const __m128i* alpha, __m128i* buffer)
        {
            ResizerByteBilinearInterpolateX4(alpha + 0, buffer + 0);
            ResizerByteBilinearInterpolateX4(alpha + 1, buffer + 1);
            ResizerByteBilinearInterpolateX4(alpha + 2, buffer + 2);
            ResizerByteBilinearInterpolateX4(alpha + 3, buffer + 3);
        }

        const __m128i K16_FRACTION_ROUND_TERM = SIMD_MM_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        template<bool align> SIMD_INLINE __m128i ResizerByteBilinearInterpolateY(const __m128i* pbx0, const __m128i* pbx1, __m128i alpha[2])
        {
            __m128i sum = _mm_add_epi16(_mm_mullo_epi16(Load<align>(pbx0), alpha[0]), _mm_mullo_epi16(Load<align>(pbx1), alpha[1]));
            return _mm_srli_epi16(_mm_add_epi16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool align> SIMD_INLINE void ResizerByteBilinearInterpolateY(const uint8_t* bx0, const uint8_t* bx1, __m128i alpha[2], uint8_t* dst)
        {
            __m128i lo = ResizerByteBilinearInterpolateY<align>((__m128i*)bx0 + 0, (__m128i*)bx1 + 0, alpha);
            __m128i hi = ResizerByteBilinearInterpolateY<align>((__m128i*)bx0 + 1, (__m128i*)bx1 + 1, alpha);
            Store<false>((__m128i*)dst, _mm_packus_epi16(lo, hi));
        }

        template<size_t N> void ResizerByteBilinear::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            struct One { uint8_t val[N * 1]; };
            struct Two { uint8_t val[N * 2]; };

            size_t size = 2 * _param.dstW * N;
            size_t aligned = AlignHi(size, DA) - DA;
            const size_t step = A * N;
            ptrdiff_t previous = -2;
            __m128i a[2];
            uint8_t* bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t* ax = _ax.data;
            const int32_t* ix = _ix.data;
            size_t dstW = _param.dstW;

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
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    Two* pb = (Two*)bx[k];
                    const One* psrc = (const One*)(src + (sy + k) * srcStride);
                    for (size_t x = 0; x < dstW; x++)
                        pb[x] = *(Two*)(psrc + ix[x]);

                    uint8_t* pbx = bx[k];
                    for (size_t i = 0; i < size; i += step)
                        ResizerByteBilinearInterpolateX<N>((__m128i*)(ax + i), (__m128i*)(pbx + i));
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(bx[0] + ib, bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(bx[0] + i, bx[1] + i, a, dst + i / 2);
            }
        }

        template <class Idx> SIMD_INLINE void ResizerByteBilinearLoadGrayInterpolated(const uint8_t* src, const Idx& index, const uint8_t* alpha, uint8_t* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)(src + index.src));
            __m128i _shuffle = _mm_loadu_si128((__m128i*) & index.shuffle);
            __m128i _alpha = _mm_loadu_si128((__m128i*)(alpha + index.dst));
            _mm_storeu_si128((__m128i*)(dst + index.dst), _mm_maddubs_epi16(_mm_shuffle_epi8(_src, _shuffle), _alpha));
        }

        void ResizerByteBilinear::RunG(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t bufW = AlignHi(_param.dstW, A) * 2;
            size_t size = 2 * _param.dstW;
            size_t aligned = AlignHi(size, DA) - DA;
            size_t blocks = _blocks;
            ptrdiff_t previous = -2;
            __m128i a[2];
            uint8_t* bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t* ax = _ax.data;
            const Idx* ixg = _ixg.data;

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
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    const uint8_t* psrc = src + (sy + k) * srcStride;
                    uint8_t* pdst = bx[k];
                    for (size_t i = 0; i < blocks; ++i)
                        ResizerByteBilinearLoadGrayInterpolated(psrc, ixg[i], ax, pdst);
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY<true>(bx[0] + ib, bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY<false>(bx[0] + i, bx[1] + i, a, dst + i / 2);
            }
        }

        void ResizerByteBilinear::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
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

        //-------------------------------------------------------------------------------------------------

        ResizerShortBilinear::ResizerShortBilinear(const ResParam& param)
            : Base::ResizerShortBilinear(param)
        {
        }

        template<size_t N> void ResizerShortBilinear::RunB(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N;
            float* pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rs3 = AlignLoAny(rs - 1, 3);
            size_t rs4 = AlignLo(rs, 4);
            size_t rs8 = AlignLo(rs, 8);
            __m128 _1 = _mm_set1_ps(1.0f);
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
                    if (N == 1)
                    {
                        for (; dx < rs4; dx += 4)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            _mm_storeu_ps(pb + dx, BilColS1(ps, _ix.data + dx, fx0, fx1));
                        }
                    }
                    if (N == 2)
                    {
                        for (; dx < rs4; dx += 4)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            _mm_storeu_ps(pb + dx, BilColS2(ps, _ix.data + dx, fx0, fx1));
                        }
                    }
                    if (N == 3)
                    {
                        for (; dx < rs3; dx += 3)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            _mm_storeu_ps(pb + dx, BilColS3(ps + _ix[dx], fx0, fx1));
                        }
                    }
                    if (N == 4)
                    {
                        for (; dx < rs4; dx += 4)
                        {
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            _mm_storeu_ps(pb + dx, BilColS4(ps + _ix[dx], fx0, fx1));
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
            size_t rs3 = AlignLoAny(rs - 1, 3);
            size_t rs6 = AlignLoAny(rs - 1, 6);
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
                if (N == 1)
                {
                    for (; dx < rs8; dx += 8)
                    {
                        __m128 fx01 = _mm_loadu_ps(_ax.data + dx + 0);
                        __m128 fx00 = _mm_sub_ps(_1, fx01);
                        __m128 m00 = _mm_mul_ps(BilColS1(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        __m128 m01 = _mm_mul_ps(BilColS1(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m00, m01));
                        __m128 fx11 = _mm_loadu_ps(_ax.data + dx + 4);
                        __m128 fx10 = _mm_sub_ps(_1, fx11);
                        __m128 m10 = _mm_mul_ps(BilColS1(ps0, _ix.data + dx + 4, fx10, fx11), _fy0);
                        __m128 m11 = _mm_mul_ps(BilColS1(ps1, _ix.data + dx + 4, fx10, fx11), _fy1);
                        __m128i i1 = _mm_cvttps_epi32(_mm_add_ps(m10, m11));
                        _mm_storeu_si128((__m128i*)(dst + dx), _mm_packus_epi32(i0, i1));
                    }
                    for (; dx < rs4; dx += 4)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_1, fx1);
                        __m128 m0 = _mm_mul_ps(BilColS1(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        __m128 m1 = _mm_mul_ps(BilColS1(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, K_ZERO));
                    }
                }
                if (N == 2)
                {
                    for (; dx < rs8; dx += 8)
                    {
                        __m128 fx01 = _mm_loadu_ps(_ax.data + dx + 0);
                        __m128 fx00 = _mm_sub_ps(_1, fx01);
                        __m128 m00 = _mm_mul_ps(BilColS2(ps0, _ix.data + dx + 0, fx00, fx01), _fy0);
                        __m128 m01 = _mm_mul_ps(BilColS2(ps1, _ix.data + dx + 0, fx00, fx01), _fy1);
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m00, m01));
                        __m128 fx11 = _mm_loadu_ps(_ax.data + dx + 4);
                        __m128 fx10 = _mm_sub_ps(_1, fx11);
                        __m128 m10 = _mm_mul_ps(BilColS2(ps0, _ix.data + dx + 4, fx10, fx11), _fy0);
                        __m128 m11 = _mm_mul_ps(BilColS2(ps1, _ix.data + dx + 4, fx10, fx11), _fy1);
                        __m128i i1 = _mm_cvttps_epi32(_mm_add_ps(m10, m11));
                        _mm_storeu_si128((__m128i*)(dst + dx), _mm_packus_epi32(i0, i1));
                    }
                    for (; dx < rs4; dx += 4)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_1, fx1);
                        __m128 m0 = _mm_mul_ps(BilColS2(ps0, _ix.data + dx, fx0, fx1), _fy0);
                        __m128 m1 = _mm_mul_ps(BilColS2(ps1, _ix.data + dx, fx0, fx1), _fy1);
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, K_ZERO));
                    }
                }
                if (N == 3)
                {
                    for (; dx < rs6; dx += 6)
                    {
                        __m128 fx01 = _mm_loadu_ps(_ax.data + dx + 0);
                        __m128 fx00 = _mm_sub_ps(_1, fx01);
                        __m128 m00 = _mm_mul_ps(BilColS3(ps0 + _ix[dx + 0], fx00, fx01), _fy0);
                        __m128 m01 = _mm_mul_ps(BilColS3(ps1 + _ix[dx + 0], fx00, fx01), _fy1);
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m00, m01));
                        __m128 fx11 = _mm_loadu_ps(_ax.data + dx + 3);
                        __m128 fx10 = _mm_sub_ps(_1, fx11);
                        __m128 m10 = _mm_mul_ps(BilColS3(ps0 + _ix[dx + 3], fx10, fx11), _fy0);
                        __m128 m11 = _mm_mul_ps(BilColS3(ps1 + _ix[dx + 3], fx10, fx11), _fy1);
                        __m128i i1 = _mm_cvttps_epi32(_mm_add_ps(m10, m11));
                        _mm_storeu_si128((__m128i*)(dst + dx), _mm_shuffle_epi8(_mm_packus_epi32(i0, i1), RSB_3_P));
                    }
                    for (; dx < rs3; dx += 3)
                    {
                        __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                        __m128 fx0 = _mm_sub_ps(_1, fx1);
                        __m128 m0 = _mm_mul_ps(BilColS3(ps0 + _ix[dx], fx0, fx1), _fy0);
                        __m128 m1 = _mm_mul_ps(BilColS3(ps1 + _ix[dx], fx0, fx1), _fy1);
                        __m128i i0 = _mm_cvttps_epi32(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(i0, K_ZERO));
                    }
                }
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
            case 1: sparse ? RunS<1>(src, srcStride, dst, dstStride) : RunB<1>(src, srcStride, dst, dstStride); return;
            case 2: sparse ? RunS<2>(src, srcStride, dst, dstStride) : RunB<2>(src, srcStride, dst, dstStride); return;
            case 3: sparse ? RunS<3>(src, srcStride, dst, dstStride) : RunB<3>(src, srcStride, dst, dstStride); return;
            case 4: sparse ? RunS<4>(src, srcStride, dst, dstStride) : RunB<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        ResizerFloatBilinear::ResizerFloatBilinear(const ResParam& param)
            : Base::ResizerFloatBilinear(param)
        {
        }

        void ResizerFloatBilinear::Run(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            size_t cn = _param.channels, cnF = AlignLo(cn, F), cnT = cn - cnF, cnL = cnT - F;
            size_t dw = _param.dstW, dw4 = AlignLo(dw, 4);
            __m128 _1 = _mm_set1_ps(1.0f);
            if (_rowBuf)
            {
                size_t rs = _param.dstW * cn;
                float* pbx[2] = { _bx[0].data, _bx[1].data };
                int32_t prev = -2;
                size_t rsF = AlignLo(rs, F);
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
                        const float* ps = src + (sy + k) * srcStride;
                        size_t dx = 0;
                        if (cn == 1)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m128 s0 = Load(ps + _ix[dx + 0], ps + _ix[dx + 1]);
                                __m128 s1 = Load(ps + _ix[dx + 2], ps + _ix[dx + 3]);
                                __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                __m128 m0 = _mm_mul_ps(fx0, _mm_shuffle_ps(s0, s1, 0x88));
                                __m128 m1 = _mm_mul_ps(fx1, _mm_shuffle_ps(s0, s1, 0xDD));
                                _mm_store_ps(pb + dx, _mm_add_ps(m0, m1));
                            }
                        }
                        else if (cn == 2)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                __m128 s0 = _mm_loadu_ps(ps + _ix[dx + 0]);
                                __m128 s1 = _mm_loadu_ps(ps + _ix[dx + 2]);
                                __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                __m128 m0 = _mm_mul_ps(fx0, _mm_shuffle_ps(s0, s1, 0x44));
                                __m128 m1 = _mm_mul_ps(fx1, _mm_shuffle_ps(s0, s1, 0xEE));
                                _mm_store_ps(pb + dx, _mm_add_ps(m0, m1));
                            }
                        }
                        else if (cn == 3 && rs > 3)
                        {
                            size_t rs3 = rs - 3;
                            for (; dx < rs3; dx += 3)
                            {
                                const float * ps0 = ps + _ix[dx];
                                __m128 fx1 = _mm_set1_ps(_ax.data[dx]);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, _mm_loadu_ps(ps0 + 0)), _mm_mul_ps(fx1, _mm_loadu_ps(ps0 + 3))));
                            }
                        }
                        else if (cn == 4)
                        {
                            for (; dx < rsF; dx += F)
                            {
                                const float* ps0 = ps + _ix[dx];
                                __m128 fx1 = _mm_set1_ps(_ax.data[dx]);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, _mm_loadu_ps(ps0 + 0)), _mm_mul_ps(fx1, _mm_loadu_ps(ps0 + 4))));
                            }
                        }
                        else
                        {
                            for (; dx < rs;)
                            {
                                const float* ps0 = ps + _ix[dx];
                                size_t eF = dx + cnF;
                                __m128 fx1 = _mm_set1_ps(_ax[dx]);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                for (; dx < eF; dx += F, ps0 += F)
                                    _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, _mm_loadu_ps(ps0)), _mm_mul_ps(fx1, _mm_loadu_ps(ps0 + cn))));
                                if (cnT)
                                    _mm_storeu_ps(pb + dx + cnL, _mm_add_ps(_mm_mul_ps(fx0, _mm_loadu_ps(ps0 + cnL)), _mm_mul_ps(fx1, _mm_loadu_ps(ps0 + cnL + cn)))), dx += cnT;
                            }
                        }
                        for (; dx < rs; dx++)
                        {
                            int32_t sx = _ix[dx];
                            float fx = _ax[dx];
                            pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + cn] * fx;
                        }
                    }

                    size_t dx = 0;
                    __m128 _fy0 = _mm_set1_ps(fy0);
                    __m128 _fy1 = _mm_set1_ps(fy1);
                    for (; dx < rsF; dx += F)
                    {
                        __m128 m0 = _mm_mul_ps(_mm_load_ps(pbx[0] + dx), _fy0);
                        __m128 m1 = _mm_mul_ps(_mm_load_ps(pbx[1] + dx), _fy1);
                        _mm_storeu_ps(dst + dx, _mm_add_ps(m0, m1));
                    }
                    for (; dx < rs; dx++)
                        dst[dx] = pbx[0][dx] * fy0 + pbx[1][dx] * fy1;
                }
            }
            else
            {
                for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                {
                    __m128 fy1 = _mm_set1_ps(_ay[dy]);
                    __m128 fy0 = _mm_sub_ps(_1, fy1);
                    const float* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                    if (cn == 1)
                    {
                        size_t dx = 0;
                        for (; dx < dw4; dx += 4)
                        {
                            size_t os = _ix[dx];
                            __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            __m128 s00 = Load(src0 + _ix[dx + 0], src0 + _ix[dx + 1]);
                            __m128 s01 = Load(src0 + _ix[dx + 2], src0 + _ix[dx + 3]);
                            __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(s00, s01, 0x88), fx0), _mm_mul_ps(_mm_shuffle_ps(s00, s01, 0xDD), fx1));
                            __m128 s10 = Load(src1 + _ix[dx + 0], src1 + _ix[dx + 1]);
                            __m128 s11 = Load(src1 + _ix[dx + 2], src1 + _ix[dx + 3]);
                            __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_shuffle_ps(s10, s11, 0x88), fx0), _mm_mul_ps(_mm_shuffle_ps(s10, s11, 0xDD), fx1));
                            _mm_storeu_ps(dst + dx, _mm_add_ps(_mm_mul_ps(r0, fy0), _mm_mul_ps(r1, fy1)));
                        }
                        for (; dx < dw; dx++)
                        {
                            size_t os = _ix[dx];
                            __m128 fx1 = _mm_set1_ps(_ax[dx]);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_load_ss(src0 + os), fx0), _mm_mul_ps(_mm_load_ss(src0 + os + 1), fx1));
                            __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_load_ss(src1 + os), fx0), _mm_mul_ps(_mm_load_ss(src1 + os + 1), fx1));
                            _mm_store_ss(dst + dx, _mm_add_ps(_mm_mul_ps(r0, fy0), _mm_mul_ps(r1, fy1)));
                        }
                    }
                    else
                    {
                        for (size_t dx = 0; dx < dw; dx++)
                        {
                            size_t os = _ix[dx], eF = os + cnF, od = dx * cn;
                            __m128 fx1 = _mm_set1_ps(_ax[dx]);
                            __m128 fx0 = _mm_sub_ps(_1, fx1);
                            for (; os < eF; os += F, od += F)
                            {
                                __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src0 + os + cn), fx1));
                                __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src1 + os + cn), fx1));
                                _mm_storeu_ps(dst + od, _mm_add_ps(_mm_mul_ps(r0, fy0), _mm_mul_ps(r1, fy1)));
                            }
                            if (cnT)
                            {
                                os += cnL;
                                od += cnL;
                                __m128 r0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src0 + os + cn), fx1));
                                __m128 r1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + os), fx0), _mm_mul_ps(_mm_loadu_ps(src1 + os + cn), fx1));
                                _mm_storeu_ps(dst + od, _mm_add_ps(_mm_mul_ps(r0, fy0), _mm_mul_ps(r1, fy1)));
                            }
                        }
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        ResizerBf16Bilinear::ResizerBf16Bilinear(const ResParam& param)
            : Base::ResizerBf16Bilinear(param)
        {
        }

        __m128i K8_IDX_20 = SIMD_MM_SETR_EPI8(-1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB);
        __m128i K8_IDX_21 = SIMD_MM_SETR_EPI8(-1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF);

        __m128i K8_IDX_30 = SIMD_MM_SETR_EPI8(-1, -1, 0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1);
        __m128i K8_IDX_31 = SIMD_MM_SETR_EPI8(-1, -1, 0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1);

        SIMD_INLINE __m128 BilinearRowSumBf16(const uint16_t* src, size_t channels, __m128 fx0, __m128 fx1)
        {
            __m128 s0 = BFloat16ToFloat32(UnpackU16<0>(_mm_loadl_epi64((__m128i*)src)));
            __m128 s1 = BFloat16ToFloat32(UnpackU16<0>(_mm_loadl_epi64((__m128i*)(src + channels))));
            return _mm_add_ps(_mm_mul_ps(fx0, s0), _mm_mul_ps(fx1, s1));
        }

        void ResizerBf16Bilinear::Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t cn = _param.channels, cnD = AlignLo(cn, DF), cnF = AlignLo(cn, F), cnT = cn - cnF, cnL = cnT - F;
            __m128 _1 = _mm_set1_ps(1.0f);
            if (_rowBuf)
            {
                size_t rs = _param.dstW * cn, rsF = AlignLo(rs, F), rsD = AlignLo(rs, DF);
                float* pbx[2] = { _bx[0].data, _bx[1].data };
                int32_t prev = -2;
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
                        if (cn == 1)
                        {
                            for (; dx < rsF; dx += Sse41::F)
                            {
                                SIMD_ALIGNED(16) uint32_t buf[4];
                                buf[0] = *(uint32_t*)(ps + _ix[dx + 0]);
                                buf[1] = *(uint32_t*)(ps + _ix[dx + 1]);
                                buf[2] = *(uint32_t*)(ps + _ix[dx + 2]);
                                buf[3] = *(uint32_t*)(ps + _ix[dx + 3]);
                                __m128i _src = _mm_loadu_si128((__m128i*)buf);
                                __m128 s0 = BFloat16ToFloat32Even(_src);
                                __m128 s1 = BFloat16ToFloat32Odd(_src);
                                __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, s0), _mm_mul_ps(fx1, s1)));
                            }
                        }
                        else if (cn == 2)
                        {
                            for (; dx < rsF; dx += Sse41::F)
                            {
                                __m128i _src = Load((__m128i*)(ps + _ix[dx + 0]), (__m128i*)(ps + _ix[dx + 2]));
                                __m128 s0 = _mm_castsi128_ps(_mm_shuffle_epi8(_src, K8_IDX_20));
                                __m128 s1 = _mm_castsi128_ps(_mm_shuffle_epi8(_src, K8_IDX_21));
                                __m128 fx1 = _mm_loadu_ps(_ax.data + dx);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, s0), _mm_mul_ps(fx1, s1)));
                            }
                        }
                        else if (cn == 3 && rs > 3)
                        {
                            size_t rs3 = rs - 3;
                            for (; dx < rs3; dx += 3)
                            {
                                __m128 fx1 = _mm_set1_ps(_ax[dx]);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                __m128i _src = _mm_loadu_si128((__m128i*)(ps + _ix[dx]));
                                __m128 s0 = _mm_castsi128_ps(_mm_shuffle_epi8(_src, K8_IDX_30));
                                __m128 s1 = _mm_castsi128_ps(_mm_shuffle_epi8(_src, K8_IDX_31));
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, s0), _mm_mul_ps(fx1, s1)));
                            }
                        }
                        else if (cn == 4)
                        {
                            for (; dx < rs; dx += 4)
                            {
                                __m128 fx1 = _mm_set1_ps(_ax[dx]);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                __m128i _src = _mm_loadu_si128((__m128i*)(ps + _ix[dx]));
                                _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, BFloat16ToFloat32<0>(_src)), _mm_mul_ps(fx1, BFloat16ToFloat32<1>(_src))));
                            }
                        }
                        else
                        {
                            for (; dx < rs;)
                            {
                                const uint16_t * ps0 = ps + _ix[dx];
                                size_t eD = dx + cnD, eF = dx + cnF;
                                __m128 fx1 = _mm_set1_ps(_ax[dx]);
                                __m128 fx0 = _mm_sub_ps(_1, fx1);
                                for (; dx < eD; dx += DF, ps0 += DF)
                                {
                                    __m128i s0 = _mm_loadu_si128((__m128i*)ps0);
                                    __m128i s1 = _mm_loadu_si128((__m128i*)(ps0 + cn));
                                    _mm_storeu_ps(pb + dx + 0, _mm_add_ps(_mm_mul_ps(fx0, BFloat16ToFloat32<0>(s0)), _mm_mul_ps(fx1, BFloat16ToFloat32<0>(s1))));
                                    _mm_storeu_ps(pb + dx + F, _mm_add_ps(_mm_mul_ps(fx0, BFloat16ToFloat32<1>(s0)), _mm_mul_ps(fx1, BFloat16ToFloat32<1>(s1))));
                                }
                                for (; dx < eF; dx += F, ps0 += F)
                                    _mm_storeu_ps(pb + dx, BilinearRowSumBf16(ps0, cn, fx0, fx1));
                                if (cnT)
                                    _mm_storeu_ps(pb + dx + cnL, BilinearRowSumBf16(ps0 + cnL, cn, fx0, fx1)), dx += cnT;
                            }
                        }
                        for (; dx < rs; dx++)
                        {
                            int32_t sx = _ix[dx];
                            float fx = _ax[dx];
                            pb[dx] = Base::BFloat16ToFloat32(ps[sx]) * (1.0f - fx) + Base::BFloat16ToFloat32(ps[sx + cn]) * fx;
                        }
                    }

                    size_t dx = 0;
                    __m128 _fy0 = _mm_set1_ps(fy0);
                    __m128 _fy1 = _mm_set1_ps(fy1);
                    for (; dx < rsD; dx += DF)
                    {
                        __m128 m00 = _mm_mul_ps(_mm_load_ps(pbx[0] + dx + 0), _fy0);
                        __m128 m10 = _mm_mul_ps(_mm_load_ps(pbx[1] + dx + 0), _fy1);
                        __m128i d0 = Float32ToBFloat16(_mm_add_ps(m00, m10));
                        __m128 m01 = _mm_mul_ps(_mm_load_ps(pbx[0] + dx + F), _fy0);
                        __m128 m11 = _mm_mul_ps(_mm_load_ps(pbx[1] + dx + F), _fy1);
                        __m128i d1 = Float32ToBFloat16(_mm_add_ps(m01, m11));
                        _mm_storeu_si128((__m128i*)(dst + dx), _mm_packus_epi32(d0, d1));
                    }
                    for (; dx < rsF; dx += F)
                    {
                        __m128 m0 = _mm_mul_ps(_mm_load_ps(pbx[0] + dx), _fy0);
                        __m128 m1 = _mm_mul_ps(_mm_load_ps(pbx[1] + dx), _fy1);
                        __m128i d0 = Float32ToBFloat16(_mm_add_ps(m0, m1));
                        _mm_storel_epi64((__m128i*)(dst + dx), _mm_packus_epi32(d0, K_ZERO));
                    }
                    for (; dx < rs; dx++)
                        dst[dx] = Base::Float32ToBFloat16(pbx[0][dx] * fy0 + pbx[1][dx] * fy1);
                }
            }
            else
            {
                for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                {
                    __m128 fy1 = _mm_set1_ps(_ay[dy]);
                    __m128 fy0 = _mm_sub_ps(_1, fy1);
                    const uint16_t* src0 = src + _iy[dy] * srcStride, * src1 = src0 + srcStride;
                    for (size_t dx = 0; dx < _param.dstW; dx++)
                    {
                        size_t os = _ix[dx], eD = os + cnD, eF = os + cnF, od = dx * cn;
                        __m128 fx1 = _mm_set1_ps(_ax[dx]);
                        __m128 fx0 = _mm_sub_ps(_1, fx1);
                        for (; os < eD; os += DF, od += DF)
                        {
                            __m128i s00 = _mm_loadu_si128((__m128i*)(src0 + os));
                            __m128i s01 = _mm_loadu_si128((__m128i*)(src0 + os + cn));
                            __m128 r00 = _mm_add_ps(_mm_mul_ps(fx0, BFloat16ToFloat32<0>(s00)), _mm_mul_ps(fx1, BFloat16ToFloat32<0>(s01)));
                            __m128 r01 = _mm_add_ps(_mm_mul_ps(fx0, BFloat16ToFloat32<1>(s00)), _mm_mul_ps(fx1, BFloat16ToFloat32<1>(s01)));                            
                            __m128i s10 = _mm_loadu_si128((__m128i*)(src1 + os));
                            __m128i s11 = _mm_loadu_si128((__m128i*)(src1 + os + cn));
                            __m128 r10 = _mm_add_ps(_mm_mul_ps(fx0, BFloat16ToFloat32<0>(s10)), _mm_mul_ps(fx1, BFloat16ToFloat32<0>(s11)));
                            __m128 r11 = _mm_add_ps(_mm_mul_ps(fx0, BFloat16ToFloat32<1>(s10)), _mm_mul_ps(fx1, BFloat16ToFloat32<1>(s11)));
                            __m128i d0 = Float32ToBFloat16(_mm_add_ps(_mm_mul_ps(r00, fy0), _mm_mul_ps(r10, fy1)));
                            __m128i d1 = Float32ToBFloat16(_mm_add_ps(_mm_mul_ps(r01, fy0), _mm_mul_ps(r11, fy1)));
                            _mm_storeu_si128((__m128i*)(dst + od), _mm_packus_epi32(d0, d1));
                        }
                        for (; os < eF; os += F, od += F)
                        {
                            __m128 r0 = BilinearRowSumBf16(src0 + os, cn, fx0, fx1);
                            __m128 r1 = BilinearRowSumBf16(src1 + os, cn, fx0, fx1);
                            __m128i d0 = Float32ToBFloat16(_mm_add_ps(_mm_mul_ps(r0, fy0), _mm_mul_ps(r1, fy1)));
                            _mm_storel_epi64((__m128i*)(dst + od), _mm_packus_epi32(d0, K_ZERO));
                        }
                        if (cnT)
                        {
                            os += cnL;
                            od += cnL;
                            __m128 r0 = BilinearRowSumBf16(src0 + os, cn, fx0, fx1);
                            __m128 r1 = BilinearRowSumBf16(src1 + os, cn, fx0, fx1);
                            __m128i d0 = Float32ToBFloat16(_mm_add_ps(_mm_mul_ps(r0, fy0), _mm_mul_ps(r1, fy1)));
                            _mm_storel_epi64((__m128i*)(dst + od), _mm_packus_epi32(d0, K_ZERO));
                        }
                    }
                }
            }
        }
    }
#endif
}

