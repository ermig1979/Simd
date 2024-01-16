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
#include "Simd/SimdExtract.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0 && size >= 8);
            uint32x4_t _ab = K32_00000000;
            size_t size1 = size - 8, i = 0;
            for (; i < size1; i += 8)
            {
                uint16x8_t a16 = CvtTo16<bits>(LoadHalf<false>(a));
                uint16x8_t b16 = CvtTo16<bits>(LoadHalf<false>(b));
                _ab = vpadalq_u16(_ab, vmulq_u16(a16, b16));
                a += bits;
                b += bits;
            }
            for (; i < size; i += 8)
            {
                uint16x8_t a16 = CvtTo16<bits>(LoadLast8<bits>(a));
                uint16x8_t b16 = CvtTo16<bits>(LoadLast8<bits>(b));
                _ab = vpadalq_u16(_ab, vmulq_u16(a16, b16));
                a += bits;
                b += bits;
            }
            return ExtractSum32u(_ab);
        }

        template<> int32_t Correlation<4>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0 && size >= 8);
            uint32x4_t _ab = K32_00000000;
            size_t size16 = AlignLo(size - 8, 16), i = 0;
            for (; i < size16; i += 16)
            {
                uint8x16_t a8 = Cvt4To8(LoadHalf<false>(a));
                uint8x16_t b8 = Cvt4To8(LoadHalf<false>(b));
                _ab = vpadalq_u16(_ab, vmull_u8(Half<0>(a8), Half<0>(b8)));
                _ab = vpadalq_u16(_ab, vmull_u8(Half<1>(a8), Half<1>(b8)));
                a += 8;
                b += 8;
            }
            for (; i < size; i += 8)
            {
                uint8x8_t a8 = Half<0>(Cvt4To8(LoadLast8<4>(a)));
                uint8x8_t b8 = Half<0>(Cvt4To8(LoadLast8<4>(b)));
                _ab = vpadalq_u16(_ab, vmull_u8(a8, b8));
                a += 4;
                b += 4;
            }
            return ExtractSum32u(_ab);
        }

        template<bool align> SIMD_INLINE void Correlation8(const uint8_t* a, const uint8_t* b, uint32x4_t & ab)
        {
            uint8x16_t _a = Load<align>(a);
            uint8x16_t _b = Load<align>(b);
            ab = vpadalq_u16(ab, vmull_u8(Half<0>(_a), Half<0>(_b)));
            ab = vpadalq_u16(ab, vmull_u8(Half<1>(_a), Half<1>(_b)));
        }

        template<> int32_t Correlation<8>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            uint32x4_t _ab = K32_00000000;
            size_t i = 0, size16 = AlignLo(size, 16);
            if (Aligned(a) && Aligned(b))
            {
                for (; i < size16; i += 16)
                    Correlation8<true>(a + i, b + i, _ab);
            }
            else
            {
                for (; i < size16; i += 16)
                    Correlation8<false>(a + i, b + i, _ab);
            }
            for (; i < size; i += 8)
            {
                uint8x8_t _a = LoadHalf<false>(a + i);
                uint8x8_t _b = LoadHalf<false>(b + i);
                _ab = vpadalq_u16(_ab, vmull_u8(_a, _b));
            }
            return ExtractSum32u(_ab);
        }

        template<int bits> void CosineDistance(const uint8_t* a, const uint8_t* b, size_t size, float* distance)
        {
            float abSum = (float)Correlation<bits>(a + 16, b + 16, size);
            Base::DecodeCosineDistance(a, b, abSum, distance);
        }

        //-------------------------------------------------------------------------------------------------

#if defined(SIMD_ARM64_ENABLE)
        template<int bits> void MicroCosineDistancesDirect4x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size1 = size - 8, o = 16;
            uint32x4_t ab00 = K32_00000000;
            uint32x4_t ab01 = K32_00000000;
            uint32x4_t ab02 = K32_00000000;
            uint32x4_t ab03 = K32_00000000;
            uint32x4_t ab10 = K32_00000000;
            uint32x4_t ab11 = K32_00000000;
            uint32x4_t ab12 = K32_00000000;
            uint32x4_t ab13 = K32_00000000;
            uint32x4_t ab20 = K32_00000000;
            uint32x4_t ab21 = K32_00000000;
            uint32x4_t ab22 = K32_00000000;
            uint32x4_t ab23 = K32_00000000;
            uint32x4_t ab30 = K32_00000000;
            uint32x4_t ab31 = K32_00000000;
            uint32x4_t ab32 = K32_00000000;
            uint32x4_t ab33 = K32_00000000;
            for (; i < size1; i += 8, o += bits)
            {
                uint16x8_t a0, a1, a2, a3, b0;
                a0 = CvtTo16<bits>(LoadHalf<false>(A[0] + o));
                a1 = CvtTo16<bits>(LoadHalf<false>(A[1] + o));
                a2 = CvtTo16<bits>(LoadHalf<false>(A[2] + o));
                a3 = CvtTo16<bits>(LoadHalf<false>(A[3] + o));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[0] + o));
                ab00 = vpadalq_u16(ab00, vmulq_u16(a0, b0));
                ab10 = vpadalq_u16(ab10, vmulq_u16(a1, b0));
                ab20 = vpadalq_u16(ab20, vmulq_u16(a2, b0));
                ab30 = vpadalq_u16(ab30, vmulq_u16(a3, b0));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[1] + o));
                ab01 = vpadalq_u16(ab01, vmulq_u16(a0, b0));
                ab11 = vpadalq_u16(ab11, vmulq_u16(a1, b0));
                ab21 = vpadalq_u16(ab21, vmulq_u16(a2, b0));
                ab31 = vpadalq_u16(ab31, vmulq_u16(a3, b0));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[2] + o));
                ab02 = vpadalq_u16(ab02, vmulq_u16(a0, b0));
                ab12 = vpadalq_u16(ab12, vmulq_u16(a1, b0));
                ab22 = vpadalq_u16(ab22, vmulq_u16(a2, b0));
                ab32 = vpadalq_u16(ab32, vmulq_u16(a3, b0));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[3] + o));
                ab03 = vpadalq_u16(ab03, vmulq_u16(a0, b0));
                ab13 = vpadalq_u16(ab13, vmulq_u16(a1, b0));
                ab23 = vpadalq_u16(ab23, vmulq_u16(a2, b0));
                ab33 = vpadalq_u16(ab33, vmulq_u16(a3, b0));
            }
            for (; i < size; i += 8, o += bits)
            {
                uint16x8_t a0, a1, a2, a3, b0;
                a0 = CvtTo16<bits>(LoadLast8<bits>(A[0] + o));
                a1 = CvtTo16<bits>(LoadLast8<bits>(A[1] + o));
                a2 = CvtTo16<bits>(LoadLast8<bits>(A[2] + o));
                a3 = CvtTo16<bits>(LoadLast8<bits>(A[3] + o));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[0] + o));
                ab00 = vpadalq_u16(ab00, vmulq_u16(a0, b0));
                ab10 = vpadalq_u16(ab10, vmulq_u16(a1, b0));
                ab20 = vpadalq_u16(ab20, vmulq_u16(a2, b0));
                ab30 = vpadalq_u16(ab30, vmulq_u16(a3, b0));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[1] + o));
                ab01 = vpadalq_u16(ab01, vmulq_u16(a0, b0));
                ab11 = vpadalq_u16(ab11, vmulq_u16(a1, b0));
                ab21 = vpadalq_u16(ab21, vmulq_u16(a2, b0));
                ab31 = vpadalq_u16(ab31, vmulq_u16(a3, b0));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[2] + o));
                ab02 = vpadalq_u16(ab02, vmulq_u16(a0, b0));
                ab12 = vpadalq_u16(ab12, vmulq_u16(a1, b0));
                ab22 = vpadalq_u16(ab22, vmulq_u16(a2, b0));
                ab32 = vpadalq_u16(ab32, vmulq_u16(a3, b0));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[3] + o));
                ab03 = vpadalq_u16(ab03, vmulq_u16(a0, b0));
                ab13 = vpadalq_u16(ab13, vmulq_u16(a1, b0));
                ab23 = vpadalq_u16(ab23, vmulq_u16(a2, b0));
                ab33 = vpadalq_u16(ab33, vmulq_u16(a3, b0));
            }
            float32x4_t ab0 = vcvtq_f32_u32(Extract4Sums32u(ab00, ab01, ab02, ab03));
            float32x4_t ab1 = vcvtq_f32_u32(Extract4Sums32u(ab10, ab11, ab12, ab13));
            float32x4_t ab2 = vcvtq_f32_u32(Extract4Sums32u(ab20, ab21, ab22, ab23));
            float32x4_t ab3 = vcvtq_f32_u32(Extract4Sums32u(ab30, ab31, ab32, ab33));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
            DecodeCosineDistances1x4(A[2], B, ab2, distances + 2 * stride);
            DecodeCosineDistances1x4(A[3], B, ab3, distances + 3 * stride);
        }

        template<> void MicroCosineDistancesDirect4x4<4>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size - 8, 16), o = 16;
            uint32x4_t ab00 = K32_00000000;
            uint32x4_t ab01 = K32_00000000;
            uint32x4_t ab02 = K32_00000000;
            uint32x4_t ab03 = K32_00000000;
            uint32x4_t ab10 = K32_00000000;
            uint32x4_t ab11 = K32_00000000;
            uint32x4_t ab12 = K32_00000000;
            uint32x4_t ab13 = K32_00000000;
            uint32x4_t ab20 = K32_00000000;
            uint32x4_t ab21 = K32_00000000;
            uint32x4_t ab22 = K32_00000000;
            uint32x4_t ab23 = K32_00000000;
            uint32x4_t ab30 = K32_00000000;
            uint32x4_t ab31 = K32_00000000;
            uint32x4_t ab32 = K32_00000000;
            uint32x4_t ab33 = K32_00000000;
            for (; i < size16; i += 16, o += 8)
            {
                uint8x16_t a0, a1, a2, a3, b0;
                a0 = Cvt4To8(LoadHalf<false>(A[0] + o));
                a1 = Cvt4To8(LoadHalf<false>(A[1] + o));
                a2 = Cvt4To8(LoadHalf<false>(A[2] + o));
                a3 = Cvt4To8(LoadHalf<false>(A[3] + o));

                b0 = Cvt4To8(LoadHalf<false>(B[0] + o));
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab10 = vpadalq_u16(ab10, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab20 = vpadalq_u16(ab20, vmull_u8(Half<0>(a2), Half<0>(b0)));
                ab30 = vpadalq_u16(ab30, vmull_u8(Half<0>(a3), Half<0>(b0)));
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab10 = vpadalq_u16(ab10, vmull_u8(Half<1>(a1), Half<1>(b0)));
                ab20 = vpadalq_u16(ab20, vmull_u8(Half<1>(a2), Half<1>(b0)));
                ab30 = vpadalq_u16(ab30, vmull_u8(Half<1>(a3), Half<1>(b0)));

                b0 = Cvt4To8(LoadHalf<false>(B[1] + o));
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab11 = vpadalq_u16(ab11, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab21 = vpadalq_u16(ab21, vmull_u8(Half<0>(a2), Half<0>(b0)));
                ab31 = vpadalq_u16(ab31, vmull_u8(Half<0>(a3), Half<0>(b0)));
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab11 = vpadalq_u16(ab11, vmull_u8(Half<1>(a1), Half<1>(b0)));
                ab21 = vpadalq_u16(ab21, vmull_u8(Half<1>(a2), Half<1>(b0)));
                ab31 = vpadalq_u16(ab31, vmull_u8(Half<1>(a3), Half<1>(b0)));

                b0 = Cvt4To8(LoadHalf<false>(B[2] + o));
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab12 = vpadalq_u16(ab12, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab22 = vpadalq_u16(ab22, vmull_u8(Half<0>(a2), Half<0>(b0)));
                ab32 = vpadalq_u16(ab32, vmull_u8(Half<0>(a3), Half<0>(b0)));
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab12 = vpadalq_u16(ab12, vmull_u8(Half<1>(a1), Half<1>(b0)));
                ab22 = vpadalq_u16(ab22, vmull_u8(Half<1>(a2), Half<1>(b0)));
                ab32 = vpadalq_u16(ab32, vmull_u8(Half<1>(a3), Half<1>(b0)));

                b0 = Cvt4To8(LoadHalf<false>(B[3] + o));
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab13 = vpadalq_u16(ab13, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab23 = vpadalq_u16(ab23, vmull_u8(Half<0>(a2), Half<0>(b0)));
                ab33 = vpadalq_u16(ab33, vmull_u8(Half<0>(a3), Half<0>(b0)));
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab13 = vpadalq_u16(ab13, vmull_u8(Half<1>(a1), Half<1>(b0)));
                ab23 = vpadalq_u16(ab23, vmull_u8(Half<1>(a2), Half<1>(b0)));
                ab33 = vpadalq_u16(ab33, vmull_u8(Half<1>(a3), Half<1>(b0)));
            }
            for (; i < size; i += 8, o += 4)
            {
                uint8x8_t a0, a1, a2, a3, b0;
                a0 = Half<0>(Cvt4To8(LoadLast8<4>(A[0] + o)));
                a1 = Half<0>(Cvt4To8(LoadLast8<4>(A[1] + o)));
                a2 = Half<0>(Cvt4To8(LoadLast8<4>(A[2] + o)));
                a3 = Half<0>(Cvt4To8(LoadLast8<4>(A[3] + o)));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[0] + o)));
                ab00 = vpadalq_u16(ab00, vmull_u8(a0, b0));
                ab10 = vpadalq_u16(ab10, vmull_u8(a1, b0));
                ab20 = vpadalq_u16(ab20, vmull_u8(a2, b0));
                ab30 = vpadalq_u16(ab30, vmull_u8(a3, b0));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[1] + o)));
                ab01 = vpadalq_u16(ab01, vmull_u8(a0, b0));
                ab11 = vpadalq_u16(ab11, vmull_u8(a1, b0));
                ab21 = vpadalq_u16(ab21, vmull_u8(a2, b0));
                ab31 = vpadalq_u16(ab31, vmull_u8(a3, b0));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[2] + o)));
                ab02 = vpadalq_u16(ab02, vmull_u8(a0, b0));
                ab12 = vpadalq_u16(ab12, vmull_u8(a1, b0));
                ab22 = vpadalq_u16(ab22, vmull_u8(a2, b0));
                ab32 = vpadalq_u16(ab32, vmull_u8(a3, b0));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[3] + o)));
                ab03 = vpadalq_u16(ab03, vmull_u8(a0, b0));
                ab13 = vpadalq_u16(ab13, vmull_u8(a1, b0));
                ab23 = vpadalq_u16(ab23, vmull_u8(a2, b0));
                ab33 = vpadalq_u16(ab33, vmull_u8(a3, b0));
            }
            float32x4_t ab0 = vcvtq_f32_u32(Extract4Sums32u(ab00, ab01, ab02, ab03));
            float32x4_t ab1 = vcvtq_f32_u32(Extract4Sums32u(ab10, ab11, ab12, ab13));
            float32x4_t ab2 = vcvtq_f32_u32(Extract4Sums32u(ab20, ab21, ab22, ab23));
            float32x4_t ab3 = vcvtq_f32_u32(Extract4Sums32u(ab30, ab31, ab32, ab33));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
            DecodeCosineDistances1x4(A[2], B, ab2, distances + 2 * stride);
            DecodeCosineDistances1x4(A[3], B, ab3, distances + 3 * stride);
        }

        template<> void MicroCosineDistancesDirect4x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            uint32x4_t ab00 = K32_00000000;
            uint32x4_t ab01 = K32_00000000;
            uint32x4_t ab02 = K32_00000000;
            uint32x4_t ab03 = K32_00000000;
            uint32x4_t ab10 = K32_00000000;
            uint32x4_t ab11 = K32_00000000;
            uint32x4_t ab12 = K32_00000000;
            uint32x4_t ab13 = K32_00000000;
            uint32x4_t ab20 = K32_00000000;
            uint32x4_t ab21 = K32_00000000;
            uint32x4_t ab22 = K32_00000000;
            uint32x4_t ab23 = K32_00000000;
            uint32x4_t ab30 = K32_00000000;
            uint32x4_t ab31 = K32_00000000;
            uint32x4_t ab32 = K32_00000000;
            uint32x4_t ab33 = K32_00000000;
            for (; i < size16; i += 16, o += 16)
            {
                uint8x16_t a0, a1, a2, a3, b0;
                a0 = Load<false>(A[0] + o);
                a1 = Load<false>(A[1] + o);
                a2 = Load<false>(A[2] + o);
                a3 = Load<false>(A[3] + o);

                b0 = Load<false>(B[0] + o);
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab10 = vpadalq_u16(ab10, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab20 = vpadalq_u16(ab20, vmull_u8(Half<0>(a2), Half<0>(b0)));
                ab30 = vpadalq_u16(ab30, vmull_u8(Half<0>(a3), Half<0>(b0)));
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab10 = vpadalq_u16(ab10, vmull_u8(Half<1>(a1), Half<1>(b0)));
                ab20 = vpadalq_u16(ab20, vmull_u8(Half<1>(a2), Half<1>(b0)));
                ab30 = vpadalq_u16(ab30, vmull_u8(Half<1>(a3), Half<1>(b0)));

                b0 = Load<false>(B[1] + o);
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab11 = vpadalq_u16(ab11, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab21 = vpadalq_u16(ab21, vmull_u8(Half<0>(a2), Half<0>(b0)));
                ab31 = vpadalq_u16(ab31, vmull_u8(Half<0>(a3), Half<0>(b0)));
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab11 = vpadalq_u16(ab11, vmull_u8(Half<1>(a1), Half<1>(b0)));
                ab21 = vpadalq_u16(ab21, vmull_u8(Half<1>(a2), Half<1>(b0)));
                ab31 = vpadalq_u16(ab31, vmull_u8(Half<1>(a3), Half<1>(b0)));

                b0 = Load<false>(B[2] + o);
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab12 = vpadalq_u16(ab12, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab22 = vpadalq_u16(ab22, vmull_u8(Half<0>(a2), Half<0>(b0)));
                ab32 = vpadalq_u16(ab32, vmull_u8(Half<0>(a3), Half<0>(b0)));
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab12 = vpadalq_u16(ab12, vmull_u8(Half<1>(a1), Half<1>(b0)));
                ab22 = vpadalq_u16(ab22, vmull_u8(Half<1>(a2), Half<1>(b0)));
                ab32 = vpadalq_u16(ab32, vmull_u8(Half<1>(a3), Half<1>(b0)));

                b0 = Load<false>(B[3] + o);
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab13 = vpadalq_u16(ab13, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab23 = vpadalq_u16(ab23, vmull_u8(Half<0>(a2), Half<0>(b0)));
                ab33 = vpadalq_u16(ab33, vmull_u8(Half<0>(a3), Half<0>(b0)));
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab13 = vpadalq_u16(ab13, vmull_u8(Half<1>(a1), Half<1>(b0)));
                ab23 = vpadalq_u16(ab23, vmull_u8(Half<1>(a2), Half<1>(b0)));
                ab33 = vpadalq_u16(ab33, vmull_u8(Half<1>(a3), Half<1>(b0)));
            }
            for (; i < size; i += 8, o += 8)
            {
                uint8x8_t a0, a1, a2, a3, b0;
                a0 = LoadHalf<false>(A[0] + o);
                a1 = LoadHalf<false>(A[1] + o);
                a2 = LoadHalf<false>(A[2] + o);
                a3 = LoadHalf<false>(A[3] + o);

                b0 = LoadHalf<false>(B[0] + o);
                ab00 = vpadalq_u16(ab00, vmull_u8(a0, b0));
                ab10 = vpadalq_u16(ab10, vmull_u8(a1, b0));
                ab20 = vpadalq_u16(ab20, vmull_u8(a2, b0));
                ab30 = vpadalq_u16(ab30, vmull_u8(a3, b0));

                b0 = LoadHalf<false>(B[1] + o);
                ab01 = vpadalq_u16(ab01, vmull_u8(a0, b0));
                ab11 = vpadalq_u16(ab11, vmull_u8(a1, b0));
                ab21 = vpadalq_u16(ab21, vmull_u8(a2, b0));
                ab31 = vpadalq_u16(ab31, vmull_u8(a3, b0));

                b0 = LoadHalf<false>(B[2] + o);
                ab02 = vpadalq_u16(ab02, vmull_u8(a0, b0));
                ab12 = vpadalq_u16(ab12, vmull_u8(a1, b0));
                ab22 = vpadalq_u16(ab22, vmull_u8(a2, b0));
                ab32 = vpadalq_u16(ab32, vmull_u8(a3, b0));

                b0 = LoadHalf<false>(B[3] + o);
                ab03 = vpadalq_u16(ab03, vmull_u8(a0, b0));
                ab13 = vpadalq_u16(ab13, vmull_u8(a1, b0));
                ab23 = vpadalq_u16(ab23, vmull_u8(a2, b0));
                ab33 = vpadalq_u16(ab33, vmull_u8(a3, b0));
            }
            float32x4_t ab0 = vcvtq_f32_u32(Extract4Sums32u(ab00, ab01, ab02, ab03));
            float32x4_t ab1 = vcvtq_f32_u32(Extract4Sums32u(ab10, ab11, ab12, ab13));
            float32x4_t ab2 = vcvtq_f32_u32(Extract4Sums32u(ab20, ab21, ab22, ab23));
            float32x4_t ab3 = vcvtq_f32_u32(Extract4Sums32u(ab30, ab31, ab32, ab33));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
            DecodeCosineDistances1x4(A[2], B, ab2, distances + 2 * stride);
            DecodeCosineDistances1x4(A[3], B, ab3, distances + 3 * stride);
        }
#endif

        template<int bits> void MicroCosineDistancesDirect2x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size1 = size - 8, o = 16;
            uint32x4_t ab00 = K32_00000000;
            uint32x4_t ab01 = K32_00000000;
            uint32x4_t ab02 = K32_00000000;
            uint32x4_t ab03 = K32_00000000;
            uint32x4_t ab10 = K32_00000000;
            uint32x4_t ab11 = K32_00000000;
            uint32x4_t ab12 = K32_00000000;
            uint32x4_t ab13 = K32_00000000;
            for (; i < size1; i += 8, o += bits)
            {
                uint16x8_t a0, a1, b0;
                a0 = CvtTo16<bits>(LoadHalf<false>(A[0] + o));
                a1 = CvtTo16<bits>(LoadHalf<false>(A[1] + o));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[0] + o));
                ab00 = vpadalq_u16(ab00, vmulq_u16(a0, b0));
                ab10 = vpadalq_u16(ab10, vmulq_u16(a1, b0));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[1] + o));
                ab01 = vpadalq_u16(ab01, vmulq_u16(a0, b0));
                ab11 = vpadalq_u16(ab11, vmulq_u16(a1, b0));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[2] + o));
                ab02 = vpadalq_u16(ab02, vmulq_u16(a0, b0));
                ab12 = vpadalq_u16(ab12, vmulq_u16(a1, b0));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[3] + o));
                ab03 = vpadalq_u16(ab03, vmulq_u16(a0, b0));
                ab13 = vpadalq_u16(ab13, vmulq_u16(a1, b0));
            }
            for (; i < size; i += 8, o += bits)
            {
                uint16x8_t a0, a1, b0;
                a0 = CvtTo16<bits>(LoadLast8<bits>(A[0] + o));
                a1 = CvtTo16<bits>(LoadLast8<bits>(A[1] + o));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[0] + o));
                ab00 = vpadalq_u16(ab00, vmulq_u16(a0, b0));
                ab10 = vpadalq_u16(ab10, vmulq_u16(a1, b0));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[1] + o));
                ab01 = vpadalq_u16(ab01, vmulq_u16(a0, b0));
                ab11 = vpadalq_u16(ab11, vmulq_u16(a1, b0));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[2] + o));
                ab02 = vpadalq_u16(ab02, vmulq_u16(a0, b0));
                ab12 = vpadalq_u16(ab12, vmulq_u16(a1, b0));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[3] + o));
                ab03 = vpadalq_u16(ab03, vmulq_u16(a0, b0));
                ab13 = vpadalq_u16(ab13, vmulq_u16(a1, b0));
            }
            float32x4_t ab0 = vcvtq_f32_u32(Extract4Sums32u(ab00, ab01, ab02, ab03));
            float32x4_t ab1 = vcvtq_f32_u32(Extract4Sums32u(ab10, ab11, ab12, ab13));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
        }

        template<> void MicroCosineDistancesDirect2x4<4>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size - 8, 16), o = 16;
            uint32x4_t ab00 = K32_00000000;
            uint32x4_t ab01 = K32_00000000;
            uint32x4_t ab02 = K32_00000000;
            uint32x4_t ab03 = K32_00000000;
            uint32x4_t ab10 = K32_00000000;
            uint32x4_t ab11 = K32_00000000;
            uint32x4_t ab12 = K32_00000000;
            uint32x4_t ab13 = K32_00000000;
            for (; i < size16; i += 16, o += 8)
            {
                uint8x16_t a0, a1, b0;
                a0 = Cvt4To8(LoadHalf<false>(A[0] + o));
                a1 = Cvt4To8(LoadHalf<false>(A[1] + o));

                b0 = Cvt4To8(LoadHalf<false>(B[0] + o));
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab10 = vpadalq_u16(ab10, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab10 = vpadalq_u16(ab10, vmull_u8(Half<1>(a1), Half<1>(b0)));

                b0 = Cvt4To8(LoadHalf<false>(B[1] + o));
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab11 = vpadalq_u16(ab11, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab11 = vpadalq_u16(ab11, vmull_u8(Half<1>(a1), Half<1>(b0)));

                b0 = Cvt4To8(LoadHalf<false>(B[2] + o));
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab12 = vpadalq_u16(ab12, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab12 = vpadalq_u16(ab12, vmull_u8(Half<1>(a1), Half<1>(b0)));

                b0 = Cvt4To8(LoadHalf<false>(B[3] + o));
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab13 = vpadalq_u16(ab13, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab13 = vpadalq_u16(ab13, vmull_u8(Half<1>(a1), Half<1>(b0)));
            }
            for (; i < size; i += 8, o += 4)
            {
                uint8x8_t a0, a1, b0;
                a0 = Half<0>(Cvt4To8(LoadLast8<4>(A[0] + o)));
                a1 = Half<0>(Cvt4To8(LoadLast8<4>(A[1] + o)));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[0] + o)));
                ab00 = vpadalq_u16(ab00, vmull_u8(a0, b0));
                ab10 = vpadalq_u16(ab10, vmull_u8(a1, b0));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[1] + o)));
                ab01 = vpadalq_u16(ab01, vmull_u8(a0, b0));
                ab11 = vpadalq_u16(ab11, vmull_u8(a1, b0));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[2] + o)));
                ab02 = vpadalq_u16(ab02, vmull_u8(a0, b0));
                ab12 = vpadalq_u16(ab12, vmull_u8(a1, b0));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[3] + o)));
                ab03 = vpadalq_u16(ab03, vmull_u8(a0, b0));
                ab13 = vpadalq_u16(ab13, vmull_u8(a1, b0));
            }
            float32x4_t ab0 = vcvtq_f32_u32(Extract4Sums32u(ab00, ab01, ab02, ab03));
            float32x4_t ab1 = vcvtq_f32_u32(Extract4Sums32u(ab10, ab11, ab12, ab13));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
        }

        template<> void MicroCosineDistancesDirect2x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            uint32x4_t ab00 = K32_00000000;
            uint32x4_t ab01 = K32_00000000;
            uint32x4_t ab02 = K32_00000000;
            uint32x4_t ab03 = K32_00000000;
            uint32x4_t ab10 = K32_00000000;
            uint32x4_t ab11 = K32_00000000;
            uint32x4_t ab12 = K32_00000000;
            uint32x4_t ab13 = K32_00000000;
            for (; i < size16; i += 16, o += 16)
            {
                uint8x16_t a0, a1, b0;
                a0 = Load<false>(A[0] + o);
                a1 = Load<false>(A[1] + o);

                b0 = Load<false>(B[0] + o);
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab10 = vpadalq_u16(ab10, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab10 = vpadalq_u16(ab10, vmull_u8(Half<1>(a1), Half<1>(b0)));

                b0 = Load<false>(B[1] + o);
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab11 = vpadalq_u16(ab11, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab11 = vpadalq_u16(ab11, vmull_u8(Half<1>(a1), Half<1>(b0)));

                b0 = Load<false>(B[2] + o);
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab12 = vpadalq_u16(ab12, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab12 = vpadalq_u16(ab12, vmull_u8(Half<1>(a1), Half<1>(b0)));

                b0 = Load<false>(B[3] + o);
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<1>(a0), Half<1>(b0)));
                ab13 = vpadalq_u16(ab13, vmull_u8(Half<0>(a1), Half<0>(b0)));
                ab13 = vpadalq_u16(ab13, vmull_u8(Half<1>(a1), Half<1>(b0)));
            }
            for (; i < size; i += 8, o += 8)
            {
                uint8x8_t a0, a1, b0;
                a0 = LoadHalf<false>(A[0] + o);
                a1 = LoadHalf<false>(A[1] + o);

                b0 = LoadHalf<false>(B[0] + o);
                ab00 = vpadalq_u16(ab00, vmull_u8(a0, b0));
                ab10 = vpadalq_u16(ab10, vmull_u8(a1, b0));

                b0 = LoadHalf<false>(B[1] + o);
                ab01 = vpadalq_u16(ab01, vmull_u8(a0, b0));
                ab11 = vpadalq_u16(ab11, vmull_u8(a1, b0));

                b0 = LoadHalf<false>(B[2] + o);
                ab02 = vpadalq_u16(ab02, vmull_u8(a0, b0));
                ab12 = vpadalq_u16(ab12, vmull_u8(a1, b0));

                b0 = LoadHalf<false>(B[3] + o);
                ab03 = vpadalq_u16(ab03, vmull_u8(a0, b0));
                ab13 = vpadalq_u16(ab13, vmull_u8(a1, b0));
            }
            float32x4_t ab0 = vcvtq_f32_u32(Extract4Sums32u(ab00, ab01, ab02, ab03));
            float32x4_t ab1 = vcvtq_f32_u32(Extract4Sums32u(ab10, ab11, ab12, ab13));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
        }

        template<int bits> void MicroCosineDistancesDirect1x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size1 = size - 8, o = 16;
            uint32x4_t ab00 = K32_00000000;
            uint32x4_t ab01 = K32_00000000;
            uint32x4_t ab02 = K32_00000000;
            uint32x4_t ab03 = K32_00000000;
            for (; i < size1; i += 8, o += bits)
            {
                uint16x8_t a0, b0;
                a0 = CvtTo16<bits>(LoadHalf<false>(A[0] + o));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[0] + o));
                ab00 = vpadalq_u16(ab00, vmulq_u16(a0, b0));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[1] + o));
                ab01 = vpadalq_u16(ab01, vmulq_u16(a0, b0));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[2] + o));
                ab02 = vpadalq_u16(ab02, vmulq_u16(a0, b0));

                b0 = CvtTo16<bits>(LoadHalf<false>(B[3] + o));
                ab03 = vpadalq_u16(ab03, vmulq_u16(a0, b0));
            }
            for (; i < size; i += 8, o += bits)
            {
                uint16x8_t a0, b0;
                a0 = CvtTo16<bits>(LoadLast8<bits>(A[0] + o));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[0] + o));
                ab00 = vpadalq_u16(ab00, vmulq_u16(a0, b0));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[1] + o));
                ab01 = vpadalq_u16(ab01, vmulq_u16(a0, b0));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[2] + o));
                ab02 = vpadalq_u16(ab02, vmulq_u16(a0, b0));

                b0 = CvtTo16<bits>(LoadLast8<bits>(B[3] + o));
                ab03 = vpadalq_u16(ab03, vmulq_u16(a0, b0));
            }
            float32x4_t ab0 = vcvtq_f32_u32(Extract4Sums32u(ab00, ab01, ab02, ab03));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<4>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size - 8, 16), o = 16;
            uint32x4_t ab00 = K32_00000000;
            uint32x4_t ab01 = K32_00000000;
            uint32x4_t ab02 = K32_00000000;
            uint32x4_t ab03 = K32_00000000;
            for (; i < size16; i += 16, o += 8)
            {
                uint8x16_t a0, b0;
                a0 = Cvt4To8(LoadHalf<false>(A[0] + o));

                b0 = Cvt4To8(LoadHalf<false>(B[0] + o));
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<1>(a0), Half<1>(b0)));

                b0 = Cvt4To8(LoadHalf<false>(B[1] + o));
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<1>(a0), Half<1>(b0)));

                b0 = Cvt4To8(LoadHalf<false>(B[2] + o));
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<1>(a0), Half<1>(b0)));

                b0 = Cvt4To8(LoadHalf<false>(B[3] + o));
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<1>(a0), Half<1>(b0)));
            }
            for (; i < size; i += 8, o += 4)
            {
                uint8x8_t a0, b0;
                a0 = Half<0>(Cvt4To8(LoadLast8<4>(A[0] + o)));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[0] + o)));
                ab00 = vpadalq_u16(ab00, vmull_u8(a0, b0));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[1] + o)));
                ab01 = vpadalq_u16(ab01, vmull_u8(a0, b0));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[2] + o)));
                ab02 = vpadalq_u16(ab02, vmull_u8(a0, b0));

                b0 = Half<0>(Cvt4To8(LoadLast8<4>(B[3] + o)));
                ab03 = vpadalq_u16(ab03, vmull_u8(a0, b0));
            }
            float32x4_t ab0 = vcvtq_f32_u32(Extract4Sums32u(ab00, ab01, ab02, ab03));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            uint32x4_t ab00 = K32_00000000;
            uint32x4_t ab01 = K32_00000000;
            uint32x4_t ab02 = K32_00000000;
            uint32x4_t ab03 = K32_00000000;
            for (; i < size16; i += 16, o += 16)
            {
                uint8x16_t a0, b0;
                a0 = Load<false>(A[0] + o);

                b0 = Load<false>(B[0] + o);
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab00 = vpadalq_u16(ab00, vmull_u8(Half<1>(a0), Half<1>(b0)));

                b0 = Load<false>(B[1] + o);
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab01 = vpadalq_u16(ab01, vmull_u8(Half<1>(a0), Half<1>(b0)));

                b0 = Load<false>(B[2] + o);
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab02 = vpadalq_u16(ab02, vmull_u8(Half<1>(a0), Half<1>(b0)));

                b0 = Load<false>(B[3] + o);
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<0>(a0), Half<0>(b0)));
                ab03 = vpadalq_u16(ab03, vmull_u8(Half<1>(a0), Half<1>(b0)));
            }
            for (; i < size; i += 8, o += 8)
            {
                uint8x8_t a0, b0;
                a0 = LoadHalf<false>(A[0] + o);

                b0 = LoadHalf<false>(B[0] + o);
                ab00 = vpadalq_u16(ab00, vmull_u8(a0, b0));

                b0 = LoadHalf<false>(B[1] + o);
                ab01 = vpadalq_u16(ab01, vmull_u8(a0, b0));
                
                b0 = LoadHalf<false>(B[2] + o);
                ab02 = vpadalq_u16(ab02, vmull_u8(a0, b0));
                
                b0 = LoadHalf<false>(B[3] + o);
                ab03 = vpadalq_u16(ab03, vmull_u8(a0, b0));
            }
            float32x4_t ab0 = vcvtq_f32_u32(Extract4Sums32u(ab00, ab01, ab02, ab03));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M2 = AlignLoAny(M, 2), N4 = AlignLo(N, 4), i = 0;
#if defined(SIMD_ARM64_ENABLE)
            size_t M4 = AlignLoAny(M, 4);
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistancesDirect4x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                {
                    CosineDistance<bits>(A[i + 0], B[j], size, distances + j + 0 * stride);
                    CosineDistance<bits>(A[i + 1], B[j], size, distances + j + 1 * stride);
                    CosineDistance<bits>(A[i + 2], B[j], size, distances + j + 2 * stride);
                    CosineDistance<bits>(A[i + 3], B[j], size, distances + j + 3 * stride);
                }
                distances += 4 * stride;
            }
#endif
            for (; i < M2; i += 2)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistancesDirect2x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                {
                    CosineDistance<bits>(A[i + 0], B[j], size, distances + j + 0 * stride);
                    CosineDistance<bits>(A[i + 1], B[j], size, distances + j + 1 * stride);
                }
                distances += 2 * stride;
            }
            for (; i < M; i++)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistancesDirect1x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                    CosineDistance<bits>(A[i], B[j], size, distances + j);
                distances += 1 * stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::CosineDistancePtr GetCosineDistance(size_t depth)
        {
            switch (depth)
            {
            case 4: return CosineDistance<4>;
            //case 5: return CosineDistance<5>;
            //case 6: return CosineDistance<6>;
            //case 7: return CosineDistance<7>;
            case 8: return CosineDistance<8>;
            default: return Base::GetCosineDistance(depth);
            }
        }

        Base::DescrInt::MacroCosineDistancesDirectPtr GetMacroCosineDistancesDirect(size_t depth)
        {
            switch (depth)
            {
            case 4: return MacroCosineDistancesDirect<4>;
            case 5: return MacroCosineDistancesDirect<5>;
            case 6: return MacroCosineDistancesDirect<6>;
            case 7: return MacroCosineDistancesDirect<7>;
            case 8: return MacroCosineDistancesDirect<8>;
            default: return NULL;
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
