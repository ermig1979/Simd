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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template<int bits> void UnpackDataA(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t size8 = size - 8;
            for (size_t i = 0; i < count; i++)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = dst + i * size;
                size_t j = 0;
                for (; j < size8; j += 8, ps += bits, pd += 8)
                    Store<false>(pd, CvtTo8<bits>(LoadHalf<false>(ps)));
                for (; j < size; j += 8, ps += bits, pd += 8)
                    Store<false>(pd, CvtTo8<bits>(LoadLast8<bits>(ps)));
            }
        }

        template<> void UnpackDataA<8>(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t size16 = AlignLo(size, 16);
            for (size_t i = 0, j; i < count; i++)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = dst + i * size;
                for (j = 0; j < size16; j += 16, ps += 16, pd += 16)
                    Store<false>(pd, Load<false>(ps));
                for (; j < size; j += 8, ps += 8, pd += 8)
                    Store<false>(pd, LoadHalf<false>(ps));
            }
        }  

        //-------------------------------------------------------------------------------------------------

        template<int bits, int step> SIMD_INLINE void UnpackDataBX_4x8(const uint8_t* const* src, size_t offset, uint8_t* dst)
        {
            uint16x4x2_t a0, a1, b0, b1;
            a0.val[0] = (uint16x4_t)CvtTo8<bits>(LoadHalf<false>(src[0] + offset));
            a0.val[1] = (uint16x4_t)CvtTo8<bits>(LoadHalf<false>(src[1] + offset));
            a1.val[0] = (uint16x4_t)CvtTo8<bits>(LoadHalf<false>(src[2] + offset));
            a1.val[1] = (uint16x4_t)CvtTo8<bits>(LoadHalf<false>(src[3] + offset));
            b0 = vzip_u16(a0.val[0], a1.val[0]);
            b1 = vzip_u16(a0.val[1], a1.val[1]);
            a0 = vzip_u16(b0.val[0], b1.val[0]);
            a1 = vzip_u16(b0.val[1], b1.val[1]);
            Store<false>(dst + 0 * step, (uint8x8_t)a0.val[0]);
            Store<false>(dst + 1 * step, (uint8x8_t)a0.val[1]);
            Store<false>(dst + 2 * step, (uint8x8_t)a1.val[0]);
            Store<false>(dst + 3 * step, (uint8x8_t)a1.val[1]);
        }

#if defined(SIMD_ARM64_ENABLE)
        template<int bits> void UnpackDataB(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t count16 = AlignLo(count, 16), i, j, o;
            for (i = 0; i < count16; i += 16, src += 16)
            {
                for (j = 0, o = 16; j < size; j += 8, o += bits, dst += 4 * 32)
                {
                    UnpackDataBX_4x8<bits, 32>(src + 0 * 4, o, dst + 0 * 8);
                    UnpackDataBX_4x8<bits, 32>(src + 1 * 4, o, dst + 1 * 8);
                    UnpackDataBX_4x8<bits, 32>(src + 2 * 4, o, dst + 2 * 8);
                    UnpackDataBX_4x8<bits, 32>(src + 3 * 4, o, dst + 3 * 8);
                }
            }
            if (i < count)
            {
                const uint8_t* _src[16];
                for (size_t j = 0; j < 16; i++, j++)
                    _src[j] = i < count ? *src++ : src[-1];
                for (j = 0, o = 16; j < size; j += 8, o += bits, dst += 4 * 32)
                {
                    UnpackDataBX_4x8<bits, 32>(_src + 0 * 4, o, dst + 0 * 8);
                    UnpackDataBX_4x8<bits, 32>(_src + 1 * 4, o, dst + 1 * 8);
                    UnpackDataBX_4x8<bits, 32>(_src + 2 * 4, o, dst + 2 * 8);
                    UnpackDataBX_4x8<bits, 32>(_src + 3 * 4, o, dst + 3 * 8);
                }
            }
        }
#else
        template<int bits> void UnpackDataB(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t count8 = AlignLo(count, 8), i, j, o;
            for (i = 0; i < count8; i += 8, src += 8)
            {
                for (j = 0, o = 16; j < size; j += 8, o += bits, dst += 4 * 16)
                {
                    UnpackDataBX_4x8<bits, 16>(src + 0, o, dst + 0);
                    UnpackDataBX_4x8<bits, 16>(src + 4, o, dst + 8);
                }
            }
            if (i < count)
            {
                const uint8_t* _src[8];
                for (size_t j = 0; j < 8; i++, j++)
                    _src[j] = i < count ? *src++ : src[-1];
                for (j = 0, o = 16; j < size; j += 8, o += bits, dst += 4 * 16)
                {
                    UnpackDataBX_4x8<bits, 16>(_src + 0, o, dst + 0);
                    UnpackDataBX_4x8<bits, 16>(_src + 4, o, dst + 8);
                }
            }
        }
#endif
        
        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE uint8x8_t Set2(const uint8_t* src)
        {
            return (uint8x8_t)vdup_n_u16(*(uint16_t*)src);
        }

        SIMD_INLINE void Madd2(uint32x4_t & ab, uint8x8_t a, uint8x8_t b)
        {
            ab = vpadalq_u16(ab, vmull_u8(a, b));
        }

        SIMD_INLINE void Madd2(uint32x4_t& ab0, uint32x4_t& ab1, uint8x8_t a, uint8x16_t b)
        {
            ab0 = vpadalq_u16(ab0, vmull_u8(a, Half<0>(b)));
            ab1 = vpadalq_u16(ab1, vmull_u8(a, Half<1>(b)));
        }

        //-------------------------------------------------------------------------------------------------

#if defined(SIMD_ARM64_ENABLE)
        template<int M> void Correlation_4xM(size_t N, size_t K, const uint8_t* ad0, const uint8_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            uint32x4_t ab00, ab01, ab02, ab03, ab10, ab11, ab12, ab13, ab20, ab21, ab22, ab23, ab30, ab31, ab32, ab33, ab40, ab41, ab42, ab43, ab50, ab51, ab52, ab53;
            uint8x8_t a0;
            const uint8_t* ad1 = ad0 + 1 * K;
            const uint8_t* ad2 = ad0 + 2 * K;
            const uint8_t* ad3 = ad0 + 3 * K;
            const uint8_t* ad4 = ad0 + 4 * K;
            const uint8_t* ad5 = ad0 + 5 * K;
            if (N > 12)
            {
                if (M > 0) ab00 = K32_00000000, ab01 = K32_00000000, ab02 = K32_00000000, ab03 = K32_00000000;
                if (M > 1) ab10 = K32_00000000, ab11 = K32_00000000, ab12 = K32_00000000, ab13 = K32_00000000;
                if (M > 2) ab20 = K32_00000000, ab21 = K32_00000000, ab22 = K32_00000000, ab23= K32_00000000;
                if (M > 3) ab30 = K32_00000000, ab31 = K32_00000000, ab32 = K32_00000000, ab33 = K32_00000000;
                if (M > 4) ab40 = K32_00000000, ab41 = K32_00000000, ab42 = K32_00000000, ab43 = K32_00000000;
                if (M > 5) ab50 = K32_00000000, ab51 = K32_00000000, ab52 = K32_00000000, ab53 = K32_00000000;
                for (size_t k = 0; k < K; k += 2)
                {
                    uint8x16_t b0 = Load<false>(bd + 0);
                    uint8x16_t b1 = Load<false>(bd + A);
                    if (M > 0) a0 = Set2(ad0 + k), Madd2(ab00, ab01, a0, b0), Madd2(ab02, ab03, a0, b1);
                    if (M > 1) a0 = Set2(ad1 + k), Madd2(ab10, ab11, a0, b0), Madd2(ab12, ab13, a0, b1);
                    if (M > 2) a0 = Set2(ad2 + k), Madd2(ab20, ab21, a0, b0), Madd2(ab22, ab23, a0, b1);
                    if (M > 3) a0 = Set2(ad3 + k), Madd2(ab30, ab31, a0, b0), Madd2(ab32, ab33, a0, b1);
                    if (M > 4) a0 = Set2(ad4 + k), Madd2(ab40, ab41, a0, b0), Madd2(ab42, ab43, a0, b1);
                    if (M > 5) a0 = Set2(ad5 + k), Madd2(ab50, ab51, a0, b0), Madd2(ab52, ab53, a0, b1);
                    bd += 32;
                }
                if (N == 16)
                {
                    if (M > 0) DecodeCosineDistances1x16(an, bn, bnStride, ab00, ab01, ab02, ab03, distances), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x16(an, bn, bnStride, ab10, ab11, ab12, ab13, distances), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x16(an, bn, bnStride, ab20, ab21, ab22, ab23, distances), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x16(an, bn, bnStride, ab30, ab31, ab32, ab33, distances), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x16(an, bn, bnStride, ab40, ab41, ab42, ab43, distances), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x16(an, bn, bnStride, ab50, ab51, ab52, ab53, distances), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1x16(an, bn, bnStride, ab00, ab01, ab02, ab03, distances, N), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x16(an, bn, bnStride, ab10, ab11, ab12, ab13, distances, N), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x16(an, bn, bnStride, ab20, ab21, ab22, ab23, distances, N), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x16(an, bn, bnStride, ab30, ab31, ab32, ab33, distances, N), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x16(an, bn, bnStride, ab40, ab41, ab42, ab43, distances, N), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x16(an, bn, bnStride, ab50, ab51, ab52, ab53, distances, N), an += 4, distances += stride;
                }
            }
            else if (N > 8)
            {
                if (M > 0) ab00 = K32_00000000, ab01 = K32_00000000, ab02 = K32_00000000;
                if (M > 1) ab10 = K32_00000000, ab11 = K32_00000000, ab12 = K32_00000000;
                if (M > 2) ab20 = K32_00000000, ab21 = K32_00000000, ab22 = K32_00000000;
                if (M > 3) ab30 = K32_00000000, ab31 = K32_00000000, ab32 = K32_00000000;
                if (M > 4) ab40 = K32_00000000, ab41 = K32_00000000, ab42 = K32_00000000;
                if (M > 5) ab50 = K32_00000000, ab51 = K32_00000000, ab52 = K32_00000000;
                for (size_t k = 0; k < K; k += 2)
                {
                    uint8x16_t b0 = Load<false>(bd);
                    uint8x8_t b1 = LoadHalf<false>(bd + A);
                    if (M > 0) a0 = Set2(ad0 + k), Madd2(ab00, ab01, a0, b0), Madd2(ab02, a0, b1);
                    if (M > 1) a0 = Set2(ad1 + k), Madd2(ab10, ab11, a0, b0), Madd2(ab12, a0, b1);
                    if (M > 2) a0 = Set2(ad2 + k), Madd2(ab20, ab21, a0, b0), Madd2(ab22, a0, b1);
                    if (M > 3) a0 = Set2(ad3 + k), Madd2(ab30, ab31, a0, b0), Madd2(ab32, a0, b1);
                    if (M > 4) a0 = Set2(ad4 + k), Madd2(ab40, ab41, a0, b0), Madd2(ab42, a0, b1);
                    if (M > 5) a0 = Set2(ad5 + k), Madd2(ab50, ab51, a0, b0), Madd2(ab52, a0, b1);
                    bd += 32;
                }
                if (N == 12)
                {
                    if (M > 0) DecodeCosineDistances1x12(an, bn, bnStride, ab00, ab01, ab02, distances), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x12(an, bn, bnStride, ab10, ab11, ab12, distances), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x12(an, bn, bnStride, ab20, ab21, ab22, distances), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x12(an, bn, bnStride, ab30, ab31, ab32, distances), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x12(an, bn, bnStride, ab40, ab41, ab42, distances), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x12(an, bn, bnStride, ab50, ab51, ab52, distances), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1x12(an, bn, bnStride, ab00, ab01, ab02, distances, N), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x12(an, bn, bnStride, ab10, ab11, ab12, distances, N), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x12(an, bn, bnStride, ab20, ab21, ab22, distances, N), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x12(an, bn, bnStride, ab30, ab31, ab32, distances, N), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x12(an, bn, bnStride, ab40, ab41, ab42, distances, N), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x12(an, bn, bnStride, ab50, ab51, ab52, distances, N), an += 4, distances += stride;
                }
            }
            else if (N > 4)
            {
                if (M > 0) ab00 = K32_00000000, ab01 = K32_00000000;
                if (M > 1) ab10 = K32_00000000, ab11 = K32_00000000;
                if (M > 2) ab20 = K32_00000000, ab21 = K32_00000000;
                if (M > 3) ab30 = K32_00000000, ab31 = K32_00000000;
                if (M > 4) ab40 = K32_00000000, ab41 = K32_00000000;
                if (M > 5) ab50 = K32_00000000, ab51 = K32_00000000;
                for (size_t k = 0; k < K; k += 2)
                {
                    uint8x16_t b0 = Load<false>(bd);
                    if (M > 0) a0 = Set2(ad0 + k), Madd2(ab00, ab01, a0, b0);
                    if (M > 1) a0 = Set2(ad1 + k), Madd2(ab10, ab11, a0, b0);
                    if (M > 2) a0 = Set2(ad2 + k), Madd2(ab20, ab21, a0, b0);
                    if (M > 3) a0 = Set2(ad3 + k), Madd2(ab30, ab31, a0, b0);
                    if (M > 4) a0 = Set2(ad4 + k), Madd2(ab40, ab41, a0, b0);
                    if (M > 5) a0 = Set2(ad5 + k), Madd2(ab50, ab51, a0, b0);
                    bd += 32;
                }
                if (N == 8)
                {
                    if (M > 0) DecodeCosineDistances1x8(an, bn, bnStride, ab00, ab01, distances), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x8(an, bn, bnStride, ab10, ab11, distances), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x8(an, bn, bnStride, ab20, ab21, distances), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x8(an, bn, bnStride, ab30, ab31, distances), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x8(an, bn, bnStride, ab40, ab41, distances), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x8(an, bn, bnStride, ab50, ab51, distances), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1x8(an, bn, bnStride, ab00, ab01, distances, N), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x8(an, bn, bnStride, ab10, ab11, distances, N), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x8(an, bn, bnStride, ab20, ab21, distances, N), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x8(an, bn, bnStride, ab30, ab31, distances, N), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x8(an, bn, bnStride, ab40, ab41, distances, N), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x8(an, bn, bnStride, ab50, ab51, distances, N), an += 4, distances += stride;
                }
            }
            else
            {
                if (M > 0) ab00 = K32_00000000;
                if (M > 1) ab10 = K32_00000000;
                if (M > 2) ab20 = K32_00000000;
                if (M > 3) ab30 = K32_00000000;
                if (M > 4) ab40 = K32_00000000;
                if (M > 5) ab50 = K32_00000000;
                for (size_t k = 0; k < K; k += 2)
                {
                    uint8x8_t b0 = LoadHalf<false>(bd);
                    if (M > 0) a0 = Set2(ad0 + k), Madd2(ab00, a0, b0);
                    if (M > 1) a0 = Set2(ad1 + k), Madd2(ab10, a0, b0);
                    if (M > 2) a0 = Set2(ad2 + k), Madd2(ab20, a0, b0);
                    if (M > 3) a0 = Set2(ad3 + k), Madd2(ab30, a0, b0);
                    if (M > 4) a0 = Set2(ad4 + k), Madd2(ab40, a0, b0);
                    if (M > 5) a0 = Set2(ad5 + k), Madd2(ab50, a0, b0);
                    bd += 32;
                }
                if (N == 4)
                {
                    if (M > 0) DecodeCosineDistances1x4(an, bn, bnStride, ab00, distances), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x4(an, bn, bnStride, ab10, distances), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x4(an, bn, bnStride, ab20, distances), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x4(an, bn, bnStride, ab30, distances), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x4(an, bn, bnStride, ab40, distances), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x4(an, bn, bnStride, ab50, distances), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1x4(an, bn, bnStride, ab00, distances, N), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x4(an, bn, bnStride, ab10, distances, N), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x4(an, bn, bnStride, ab20, distances, N), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x4(an, bn, bnStride, ab30, distances, N), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x4(an, bn, bnStride, ab40, distances, N), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x4(an, bn, bnStride, ab50, distances, N), an += 4, distances += stride;
                }
            }
        }

        typedef void(*Correlation_NxM_Ptr)(size_t N, size_t K, const uint8_t* ad0, const uint8_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride);

        SIMD_INLINE Correlation_NxM_Ptr GetCorrelation_4xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Correlation_4xM<1>;
            case 2: return Correlation_4xM<2>;
            case 3: return Correlation_4xM<3>;
            case 4: return Correlation_4xM<4>;
            case 5: return Correlation_4xM<5>;
            case 6: return Correlation_4xM<6>;
            }
            assert(0);
            return NULL;
        }

        void MacroCorrelation(size_t M, size_t N, size_t K, const uint8_t* ad, const float* an, const uint8_t* bd, const float* bn, float* distances, size_t stride)
        {
            size_t M6 = AlignLoAny(M, 6), j = 0;
            Correlation_NxM_Ptr correlation_4x6 = GetCorrelation_4xM(6);
            Correlation_NxM_Ptr correlation_4xT = GetCorrelation_4xM(M - M6);
            for (; j < N; j += 16)
            {
                size_t dN = Simd::Min<size_t>(16, N - j);
                size_t i = 0;
                for (; i < M6; i += 6)
                    correlation_4x6(dN, K, ad + i * K, bd, an + i * 4, bn, N, distances + i * stride, stride);
                if (i < M)
                    correlation_4xT(dN, K, ad + i * K, bd, an + i * 4, bn, N, distances + i * stride, stride);
                bd += K * 16;
                bn += 16;
                distances += 16;
            }
        }
#else
        template<int M> void Correlation_2xM(size_t N, size_t K, const uint8_t* ad0, const uint8_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            uint32x4_t ab00, ab01, ab10, ab11, ab20, ab21, ab30, ab31, ab40, ab41, ab50, ab51;
            uint8x8_t a0;
            const uint8_t* ad1 = ad0 + 1 * K;
            const uint8_t* ad2 = ad0 + 2 * K;
            const uint8_t* ad3 = ad0 + 3 * K;
            const uint8_t* ad4 = ad0 + 4 * K;
            const uint8_t* ad5 = ad0 + 5 * K;
            if (N > 4)
            {
                if (M > 0) ab00 = K32_00000000, ab01 = K32_00000000;
                if (M > 1) ab10 = K32_00000000, ab11 = K32_00000000;
                if (M > 2) ab20 = K32_00000000, ab21 = K32_00000000;
                if (M > 3) ab30 = K32_00000000, ab31 = K32_00000000;
                if (M > 4) ab40 = K32_00000000, ab41 = K32_00000000;
                if (M > 5) ab50 = K32_00000000, ab51 = K32_00000000;
                for (size_t k = 0; k < K; k += 2)
                {
                    uint8x16_t b0 = Load<false>(bd);
                    if (M > 0) a0 = Set2(ad0 + k), Madd2(ab00, ab01, a0, b0);
                    if (M > 1) a0 = Set2(ad1 + k), Madd2(ab10, ab11, a0, b0);
                    if (M > 2) a0 = Set2(ad2 + k), Madd2(ab20, ab21, a0, b0);
                    if (M > 3) a0 = Set2(ad3 + k), Madd2(ab30, ab31, a0, b0);
                    if (M > 4) a0 = Set2(ad4 + k), Madd2(ab40, ab41, a0, b0);
                    if (M > 5) a0 = Set2(ad5 + k), Madd2(ab50, ab51, a0, b0);
                    bd += 16;
                }
                if (N == 8)
                {
                    if (M > 0) DecodeCosineDistances1x8(an, bn, bnStride, ab00, ab01, distances), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x8(an, bn, bnStride, ab10, ab11, distances), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x8(an, bn, bnStride, ab20, ab21, distances), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x8(an, bn, bnStride, ab30, ab31, distances), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x8(an, bn, bnStride, ab40, ab41, distances), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x8(an, bn, bnStride, ab50, ab51, distances), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1x8(an, bn, bnStride, ab00, ab01, distances, N), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x8(an, bn, bnStride, ab10, ab11, distances, N), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x8(an, bn, bnStride, ab20, ab21, distances, N), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x8(an, bn, bnStride, ab30, ab31, distances, N), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x8(an, bn, bnStride, ab40, ab41, distances, N), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x8(an, bn, bnStride, ab50, ab51, distances, N), an += 4, distances += stride;
                }
            }
            else
            {
                if (M > 0) ab00 = K32_00000000;
                if (M > 1) ab10 = K32_00000000;
                if (M > 2) ab20 = K32_00000000;
                if (M > 3) ab30 = K32_00000000;
                if (M > 4) ab40 = K32_00000000;
                if (M > 5) ab50 = K32_00000000;
                for (size_t k = 0; k < K; k += 2)
                {
                    uint8x8_t b0 = LoadHalf<false>(bd);
                    if (M > 0) a0 = Set2(ad0 + k), Madd2(ab00, a0, b0);
                    if (M > 1) a0 = Set2(ad1 + k), Madd2(ab10, a0, b0);
                    if (M > 2) a0 = Set2(ad2 + k), Madd2(ab20, a0, b0);
                    if (M > 3) a0 = Set2(ad3 + k), Madd2(ab30, a0, b0);
                    if (M > 4) a0 = Set2(ad4 + k), Madd2(ab40, a0, b0);
                    if (M > 5) a0 = Set2(ad5 + k), Madd2(ab50, a0, b0);
                    bd += 16;
                }
                if (N == 4)
                {
                    if (M > 0) DecodeCosineDistances1x4(an, bn, bnStride, ab00, distances), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x4(an, bn, bnStride, ab10, distances), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x4(an, bn, bnStride, ab20, distances), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x4(an, bn, bnStride, ab30, distances), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x4(an, bn, bnStride, ab40, distances), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x4(an, bn, bnStride, ab50, distances), an += 4, distances += stride;
                }
                else
                {
                    if (M > 0) DecodeCosineDistances1x4(an, bn, bnStride, ab00, distances, N), an += 4, distances += stride;
                    if (M > 1) DecodeCosineDistances1x4(an, bn, bnStride, ab10, distances, N), an += 4, distances += stride;
                    if (M > 2) DecodeCosineDistances1x4(an, bn, bnStride, ab20, distances, N), an += 4, distances += stride;
                    if (M > 3) DecodeCosineDistances1x4(an, bn, bnStride, ab30, distances, N), an += 4, distances += stride;
                    if (M > 4) DecodeCosineDistances1x4(an, bn, bnStride, ab40, distances, N), an += 4, distances += stride;
                    if (M > 5) DecodeCosineDistances1x4(an, bn, bnStride, ab50, distances, N), an += 4, distances += stride;
                }
            }
        }

        typedef void(*Correlation_NxM_Ptr)(size_t N, size_t K, const uint8_t* ad0, const uint8_t* bd, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride);

        SIMD_INLINE Correlation_NxM_Ptr GetCorrelation_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Correlation_2xM<1>;
            case 2: return Correlation_2xM<2>;
            case 3: return Correlation_2xM<3>;
            case 4: return Correlation_2xM<4>;
            case 5: return Correlation_2xM<5>;
            case 6: return Correlation_2xM<6>;
            }
            assert(0);
            return NULL;
        }

        void MacroCorrelation(size_t M, size_t N, size_t K, const uint8_t* ad, const float* an, const uint8_t* bd, const float* bn, float* distances, size_t stride)
        {
            size_t M6 = AlignLoAny(M, 6), j = 0;
            Correlation_NxM_Ptr correlation_2x6 = GetCorrelation_2xM(6);
            Correlation_NxM_Ptr correlation_2xT = GetCorrelation_2xM(M - M6);
            for (; j < N; j += 8)
            {
                size_t dN = Simd::Min<size_t>(8, N - j);
                size_t i = 0;
                for (; i < M6; i += 6)
                    correlation_2x6(dN, K, ad + i * K, bd, an + i * 4, bn, N, distances + i * stride, stride);
                if (i < M)
                    correlation_2xT(dN, K, ad + i * K, bd, an + i * 4, bn, N, distances + i * stride, stride);
                bd += K * 8;
                bn += 8;
                distances += 8;
            }
        }
#endif

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::UnpackDataPtr GetUnpackData(size_t depth, bool transpose)
        {
            switch (depth)
            {
            case 4: return transpose ? UnpackDataB<4> : UnpackDataA<4>;
            case 5: return transpose ? UnpackDataB<5> : UnpackDataA<5>;
            case 6: return transpose ? UnpackDataB<6> : UnpackDataA<6>;
            case 7: return transpose ? UnpackDataB<7> : UnpackDataA<7>;
            case 8: return transpose ? UnpackDataB<8> : UnpackDataA<8>;
            default: return NULL;
            }
        }

        Base::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth)
        {
            return MacroCorrelation;
        }
    }
#endif// SIMD_NEON_ENABLE
}
