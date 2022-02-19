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
#include "Simd/SimdResizer.h"
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE 
    namespace Avx2
    {
        ResizerByteBicubic::ResizerByteBicubic(const ResParam& param)
            : Sse41::ResizerByteBicubic(param)
        {
        }

        template<int N> __m256i LoadAx(const int8_t* ax);

        template<> SIMD_INLINE __m256i LoadAx<1>(const int8_t* ax)
        {
            return _mm256_loadu_si256((__m256i*)ax);
        }

        template<> SIMD_INLINE __m256i LoadAx<2>(const int8_t* ax)
        {
            static const __m256i PERMUTE = SIMD_MM256_SETR_EPI32(0, 0, 1, 1, 2, 2, 3, 3);
            return _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)ax)), PERMUTE);
        }

        template<> SIMD_INLINE __m256i LoadAx<3>(const int8_t* ax)
        {
            static const __m256i PERMUTE = SIMD_MM256_SETR_EPI32(0, 0, 0, 0, 1, 1, 1, 1);
            return  _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)ax)), PERMUTE);
        }

        template<> SIMD_INLINE __m256i LoadAx<4>(const int8_t* ax)
        {
            static const __m256i PERMUTE = SIMD_MM256_SETR_EPI32(0, 0, 0, 0, 1, 1, 1, 1);
            return  _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)ax)), PERMUTE);
        }

        template<int N> __m256i CubicSumX(const uint8_t* src, const int32_t* ix, __m256i ax, __m256i ay);

        template<> SIMD_INLINE __m256i CubicSumX<1>(const uint8_t* src, const int32_t* ix, __m256i ax, __m256i ay)
        {
            __m256i _src = _mm256_i32gather_epi32((int32_t*)src, _mm256_loadu_si256((__m256i*)ix), 1);
            return _mm256_madd_epi16(_mm256_maddubs_epi16(_src, ax), ay);
        }

        template<> SIMD_INLINE __m256i CubicSumX<2>(const uint8_t* src, const int32_t* ix, __m256i ax, __m256i ay)
        {
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF);
            __m256i _src = _mm256_shuffle_epi8(_mm256_i32gather_epi64((long long*)src, _mm_loadu_si128((__m128i*)ix), 1), SHUFFLE);
            return _mm256_madd_epi16(_mm256_maddubs_epi16(_src, ax), ay);
        }

        template<> SIMD_INLINE __m256i CubicSumX<3>(const uint8_t* src, const int32_t* ix, __m256i ax, __m256i ay)
        {
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1);
            __m256i _src = _mm256_shuffle_epi8(Load<false>((__m128i*)(src + ix[0]), (__m128i*)(src + ix[1])), SHUFFLE);
            return _mm256_madd_epi16(_mm256_maddubs_epi16(_src, ax), ay);
        }

        template<> SIMD_INLINE __m256i CubicSumX<4>(const uint8_t* src, const int32_t* ix, __m256i ax, __m256i ay)
        {
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
            __m256i _src = _mm256_shuffle_epi8(Load<false>((__m128i*)(src + ix[0]), (__m128i*)(src + ix[1])), SHUFFLE);
            return _mm256_madd_epi16(_mm256_maddubs_epi16(_src, ax), ay);
        }

        template <int N> SIMD_INLINE void StoreBicubicInt(__m256i val, uint8_t* dst)
        {
            *((int64_t*)dst) = Extract64i<0>(PackI16ToU8(PackI32ToI16(val, K_ZERO), K_ZERO));
        }

        template <> SIMD_INLINE void StoreBicubicInt<3>(__m256i val, uint8_t* dst)
        {
            static const __m128i SHUFFLE = SIMD_MM_SETR_EPI8(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m128i u8 = _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(val, K_ZERO), K_ZERO));
            *((int64_t*)dst) = Sse2::ExtractInt64<0>(_mm_shuffle_epi8(u8, SHUFFLE));
        }

        template <int N> SIMD_INLINE void BicubicInt(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, const int32_t* ix, const int8_t* ax, const __m256i* ay, uint8_t* dst)
        {
            static const __m256i ROUND = SIMD_MM256_SET1_EPI32(Base::BICUBIC_ROUND);
            __m256i _ax = LoadAx<N>(ax);
            __m256i say0 = CubicSumX<N>(src0 - N, ix, _ax, ay[0]);
            __m256i say1 = CubicSumX<N>(src1 - N, ix, _ax, ay[1]);
            __m256i say2 = CubicSumX<N>(src2 - N, ix, _ax, ay[2]);
            __m256i say3 = CubicSumX<N>(src3 - N, ix, _ax, ay[3]);
            __m256i sum = _mm256_add_epi32(_mm256_add_epi32(say0, say1), _mm256_add_epi32(say2, say3));
            __m256i dst0 = _mm256_srai_epi32(_mm256_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
            StoreBicubicInt<N>(dst0, dst);
        }

        SIMD_INLINE __m256i CubicSumX1(const uint8_t* src, __m256i ix, __m256i ax, __m256i ay)
        {
            __m256i _src = _mm256_i32gather_epi32((int32_t*)(src - 1), ix, 1);
            return _mm256_madd_epi16(_mm256_maddubs_epi16(_src, ax), ay);
        }

        template <> SIMD_INLINE void BicubicInt<1>(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, const int32_t* ix, const int8_t* ax, const __m256i* ay, uint8_t* dst)
        {
            static const __m256i ROUND = SIMD_MM256_SET1_EPI32(Base::BICUBIC_ROUND);
            __m256i _ix = _mm256_loadu_si256((__m256i*)ix);
            __m256i _ax = LoadAx<1>(ax);
            __m256i say0 = CubicSumX1(src0, _ix, _ax, ay[0]);
            __m256i say1 = CubicSumX1(src1, _ix, _ax, ay[1]);
            __m256i say2 = CubicSumX1(src2, _ix, _ax, ay[2]);
            __m256i say3 = CubicSumX1(src3, _ix, _ax, ay[3]);
            __m256i sum = _mm256_add_epi32(_mm256_add_epi32(say0, say1), _mm256_add_epi32(say2, say3));
            __m256i dst0 = _mm256_srai_epi32(_mm256_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
            StoreBicubicInt<1>(dst0, dst);
        }

        template<int N> void ResizerByteBicubic::RunS(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_xn == 0 && _xt == _param.dstW);
            size_t step = 4 / N * 2;
            size_t body = AlignLoAny(_param.dstW, step);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                const int32_t* ay = _ay.data + dy * 4;
                __m256i ays[4];
                ays[0] = _mm256_set1_epi16(ay[0]);
                ays[1] = _mm256_set1_epi16(ay[1]);
                ays[2] = _mm256_set1_epi16(ay[2]);
                ays[3] = _mm256_set1_epi16(ay[3]);
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
#if 1
            __m128i src0 = _mm_setr_epi32(*(int32_t*)(src + ix[0]), *(int32_t*)(src + ix[1]), *(int32_t*)(src + ix[2]), *(int32_t*)(src + ix[3]));
            __m128i src1 = _mm_setr_epi32(*(int32_t*)(src + ix[4]), *(int32_t*)(src + ix[5]), *(int32_t*)(src + ix[6]), *(int32_t*)(src + ix[7]));
            __m256i _src = Set(src0, src1);
#else
            __m256i _src = _mm256_i32gather_epi32((int32_t*)src, _mm256_loadu_si256((__m256i*)ix), 1);
#endif
            __m256i _ax = _mm256_loadu_si256((__m256i*)ax);
            _mm256_storeu_si256((__m256i*)dst, _mm256_madd_epi16(_mm256_maddubs_epi16(_src, _ax), K16_0001));
        }

        template<> SIMD_INLINE void PixelCubicSumX<2>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            static const __m256i PERMUTE = SIMD_MM256_SETR_EPI32(0, 0, 1, 1, 2, 2, 3, 3);
            __m256i _ax = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)ax)), PERMUTE);
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF);
            __m256i _src = _mm256_shuffle_epi8(_mm256_i32gather_epi64((long long*)src, _mm_loadu_si128((__m128i*)ix), 1), SHUFFLE);
            _mm256_storeu_si256((__m256i*)dst, _mm256_madd_epi16(_mm256_maddubs_epi16(_src, _ax), K16_0001));
        }

        template<> SIMD_INLINE void PixelCubicSumX<3>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            static const __m256i PERM_A = SIMD_MM256_SETR_EPI32(0, 0, 0, 1, 1, 1, 0, 0);
            __m256i _ax = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)ax)), PERM_A);
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1);
            static const __m256i PERM_B = SIMD_MM256_SETR_EPI32(0, 1, 2, 4, 5, 6, 0, 0);
            __m256i _src = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(Load<false>((__m128i*)(src + ix[0]), (__m128i*)(src + ix[1])), SHUFFLE), PERM_B);
            _mm256_storeu_si256((__m256i*)dst, _mm256_madd_epi16(_mm256_maddubs_epi16(_src, _ax), K16_0001));
        }

        template<> SIMD_INLINE void PixelCubicSumX<4>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            static const __m256i PERMUTE = SIMD_MM256_SETR_EPI32(0, 0, 0, 0, 1, 1, 1, 1);
            __m256i _ax = _mm256_permutevar8x32_epi32(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)ax)), PERMUTE);
            static const __m256i SHUFFLE = SIMD_MM256_SETR_EPI8(
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
            __m256i _src = _mm256_shuffle_epi8(Load<false>((__m128i*)(src + ix[0]), (__m128i*)(src + ix[1])), SHUFFLE);
            _mm256_storeu_si256((__m256i*)dst, _mm256_madd_epi16(_mm256_maddubs_epi16(_src, _ax), K16_0001));
        }

        template<int N> SIMD_INLINE void RowCubicSumX(const uint8_t* src, size_t nose, size_t body, size_t tail, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            size_t step = 4 / N * 2;
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
                static const __m256i ROUND = SIMD_MM256_SET1_EPI32(Base::BICUBIC_ROUND);
                __m256i ay0 = _mm256_set1_epi32(ay[0]);
                __m256i ay1 = _mm256_set1_epi32(ay[1]);
                __m256i ay2 = _mm256_set1_epi32(ay[2]);
                __m256i ay3 = _mm256_set1_epi32(ay[3]);
                for (; i < nF; i += F)
                {
                    __m256i say0 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(src0 + i)), ay0);
                    __m256i say1 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(src1 + i)), ay1);
                    __m256i say2 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(src2 + i)), ay2);
                    __m256i say3 = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)(src3 + i)), ay3);
                    __m256i sum = _mm256_add_epi32(_mm256_add_epi32(say0, say1), _mm256_add_epi32(say2, say3));
                    __m256i dst0 = _mm256_srai_epi32(_mm256_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
                    *((int64_t*)(dst + i)) = Extract64i<0>(PackI16ToU8(PackI32ToI16(dst0, K_ZERO), K_ZERO));
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
            case 1: sparse ? Sse41::ResizerByteBicubic::RunS<1>(src, srcStride, dst, dstStride) : RunB<1>(src, srcStride, dst, dstStride); return;
            case 2: sparse ? RunS<2>(src, srcStride, dst, dstStride) : RunB<2>(src, srcStride, dst, dstStride); return;
            case 3: sparse ? RunS<3>(src, srcStride, dst, dstStride) : RunB<3>(src, srcStride, dst, dstStride); return;
            case 4: sparse ? RunS<4>(src, srcStride, dst, dstStride) : RunB<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }
    }
#endif //SIMD_AVX2_ENABLE 
}

