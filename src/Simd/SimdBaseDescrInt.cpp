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

#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdFloat16.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
    namespace Base
    {
        static void MinMax32f(const float* src, size_t size, float& min, float& max)
        {
            min = FLT_MAX;
            max = -FLT_MAX;
            for (size_t i = 0; i < size; ++i)
            {
                float val = src[i];
                min = Simd::Min(val, min);
                max = Simd::Max(val, max);
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void MinMax16f(const uint16_t* src, size_t size, float& min, float& max)
        {
            min = FLT_MAX;
            max = -FLT_MAX;
            for (size_t i = 0; i < size; ++i)
            {
                float val = Float16ToFloat32(src[i]);
                min = Simd::Min(val, min);
                max = Simd::Max(val, max);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int32_t Encode32f(float src, float scale, float min, int32_t& sum, int32_t& sqsum)
        {
            int32_t value = Round((src - min) * scale);
            sum += value;
            sqsum += value * value;
            return value;
        }

        static void Encode32f4(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 2 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 2)
            {
                uint32_t v0 = Encode32f(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode32f(src[1], scale, min, sum, sqsum);
                dst[0] = v0 | v1 << 4;
                src += 2;
                dst += 1;
            }
        }

        static void Encode32f5(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t v0 = Encode32f(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode32f(src[1], scale, min, sum, sqsum);
                uint32_t v2 = Encode32f(src[2], scale, min, sum, sqsum);
                uint32_t v3 = Encode32f(src[3], scale, min, sum, sqsum);
                uint32_t v4 = Encode32f(src[4], scale, min, sum, sqsum);
                uint32_t v5 = Encode32f(src[5], scale, min, sum, sqsum);
                uint32_t v6 = Encode32f(src[6], scale, min, sum, sqsum);
                uint32_t v7 = Encode32f(src[7], scale, min, sum, sqsum);
                dst[0] = v0 | v1 << 5;
                dst[1] = v1 >> 3 | v2 << 2 | v3 << 7;
                dst[2] = v3 >> 1 | v4 << 4;
                dst[3] = v4 >> 4 | v5 << 1 | v6 << 6;
                dst[4] = v6 >> 2 | v7 << 3;
                src += 8;
                dst += 5;
            }
        }

        static void Encode32f6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 4 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 4)
            {
                uint32_t v0 = Encode32f(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode32f(src[1], scale, min, sum, sqsum);
                uint32_t v2 = Encode32f(src[2], scale, min, sum, sqsum);
                uint32_t v3 = Encode32f(src[3], scale, min, sum, sqsum);
                dst[0] = v0 | v1 << 6;
                dst[1] = v1 >> 2 | v2 << 4;
                dst[2] = v2 >> 4 | v3 << 2;
                src += 4;
                dst += 3;
            }
        }

        static void Encode32f7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t v0 = Encode32f(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode32f(src[1], scale, min, sum, sqsum);
                uint32_t v2 = Encode32f(src[2], scale, min, sum, sqsum);
                uint32_t v3 = Encode32f(src[3], scale, min, sum, sqsum);
                uint32_t v4 = Encode32f(src[4], scale, min, sum, sqsum);
                uint32_t v5 = Encode32f(src[5], scale, min, sum, sqsum);
                uint32_t v6 = Encode32f(src[6], scale, min, sum, sqsum);
                uint32_t v7 = Encode32f(src[7], scale, min, sum, sqsum);
                dst[0] = v0 | v1 << 7;
                dst[1] = v1 >> 1 | v2 << 6;
                dst[2] = v2 >> 2 | v3 << 5;
                dst[3] = v3 >> 3 | v4 << 4;
                dst[4] = v4 >> 4 | v5 << 3;
                dst[5] = v5 >> 5 | v6 << 2;
                dst[6] = v6 >> 6 | v7 << 1;
                src += 8;
                dst += 7;
            }
        }

        static void Encode32f8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; ++i)
                dst[i] = (uint8_t)Encode32f(src[i], scale, min, sum, sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int32_t Encode16f(uint16_t src, float scale, float min, int32_t& sum, int32_t& sqsum)
        {
            float val = Float16ToFloat32(src);
            int32_t value = Round((val - min) * scale);
            sum += value;
            sqsum += value * value;
            return value;
        }

        static void Encode16f4(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 2 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 2)
            {
                uint32_t v0 = Encode16f(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode16f(src[1], scale, min, sum, sqsum);
                dst[0] = v0 | v1 << 4;
                src += 2;
                dst += 1;
            }
        }

        static void Encode16f5(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t v0 = Encode16f(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode16f(src[1], scale, min, sum, sqsum);
                uint32_t v2 = Encode16f(src[2], scale, min, sum, sqsum);
                uint32_t v3 = Encode16f(src[3], scale, min, sum, sqsum);
                uint32_t v4 = Encode16f(src[4], scale, min, sum, sqsum);
                uint32_t v5 = Encode16f(src[5], scale, min, sum, sqsum);
                uint32_t v6 = Encode16f(src[6], scale, min, sum, sqsum);
                uint32_t v7 = Encode16f(src[7], scale, min, sum, sqsum);
                dst[0] = v0 | v1 << 5;
                dst[1] = v1 >> 3 | v2 << 2 | v3 << 7;
                dst[2] = v3 >> 1 | v4 << 4;
                dst[3] = v4 >> 4 | v5 << 1 | v6 << 6;
                dst[4] = v6 >> 2 | v7 << 3;
                src += 8;
                dst += 5;
            }
        }

        static void Encode16f6(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 4 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 4)
            {
                uint32_t v0 = Encode16f(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode16f(src[1], scale, min, sum, sqsum);
                uint32_t v2 = Encode16f(src[2], scale, min, sum, sqsum);
                uint32_t v3 = Encode16f(src[3], scale, min, sum, sqsum);
                dst[0] = v0 | v1 << 6;
                dst[1] = v1 >> 2 | v2 << 4;
                dst[2] = v2 >> 4 | v3 << 2;
                src += 4;
                dst += 3;
            }
        }

        static void Encode16f7(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t v0 = Encode16f(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode16f(src[1], scale, min, sum, sqsum);
                uint32_t v2 = Encode16f(src[2], scale, min, sum, sqsum);
                uint32_t v3 = Encode16f(src[3], scale, min, sum, sqsum);
                uint32_t v4 = Encode16f(src[4], scale, min, sum, sqsum);
                uint32_t v5 = Encode16f(src[5], scale, min, sum, sqsum);
                uint32_t v6 = Encode16f(src[6], scale, min, sum, sqsum);
                uint32_t v7 = Encode16f(src[7], scale, min, sum, sqsum);
                dst[0] = v0 | v1 << 7;
                dst[1] = v1 >> 1 | v2 << 6;
                dst[2] = v2 >> 2 | v3 << 5;
                dst[3] = v3 >> 3 | v4 << 4;
                dst[4] = v4 >> 4 | v5 << 3;
                dst[5] = v5 >> 5 | v6 << 2;
                dst[6] = v6 >> 6 | v7 << 1;
                src += 8;
                dst += 7;
            }
        }

        static void Encode16f8(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; ++i)
                dst[i] = (uint8_t)Encode16f(src[i], scale, min, sum, sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        static void Decode32f4(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t val = *(uint32_t*)(src + 0);
                dst[0] = ((val >> 0) & 0xF) * scale + shift;
                dst[1] = ((val >> 4) & 0xF) * scale + shift;
                dst[2] = ((val >> 8) & 0xF) * scale + shift;
                dst[3] = ((val >> 12) & 0xF) * scale + shift;
                dst[4] = ((val >> 16) & 0xF) * scale + shift;
                dst[5] = ((val >> 20) & 0xF) * scale + shift;
                dst[6] = ((val >> 24) & 0xF) * scale + shift;
                dst[7] = ((val >> 28) & 0xF) * scale + shift;
                src += 4;
                dst += 8;
            }
        }

        static void Decode32f5(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t lo = *(uint32_t*)(src + 0);
                dst[0] = ((lo >> 0) & 0x1F) * scale + shift;
                dst[1] = ((lo >> 5) & 0x1F) * scale + shift;
                dst[2] = ((lo >> 10) & 0x1F) * scale + shift;
                dst[3] = ((lo >> 15) & 0x1F) * scale + shift;
                uint32_t hi = *(uint32_t*)(src + 1);
                dst[4] = ((hi >> 12) & 0x1F) * scale + shift;
                dst[5] = ((hi >> 17) & 0x1F) * scale + shift;
                dst[6] = ((hi >> 22) & 0x1F) * scale + shift;
                dst[7] = ((hi >> 27) & 0x1F) * scale + shift;
                src += 5;
                dst += 8;
            }
        }

        static void Decode32f6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t lo = *(uint32_t*)(src + 0);
                dst[0] = ((lo >> 0) & 0x3F) * scale + shift;
                dst[1] = ((lo >> 6) & 0x3F) * scale + shift;
                dst[2] = ((lo >> 12) & 0x3F) * scale + shift;
                dst[3] = ((lo >> 18) & 0x3F) * scale + shift;
                uint32_t hi = *(uint32_t*)(src + 2);
                dst[4] = ((hi >> 8) & 0x3F) * scale + shift;
                dst[5] = ((hi >> 14) & 0x3F) * scale + shift;
                dst[6] = ((hi >> 20) & 0x3F) * scale + shift;
                dst[7] = ((hi >> 26) & 0x3F) * scale + shift;
                src += 6;
                dst += 8;
            }
        }

        static void Decode32f7(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t lo = *(uint32_t*)(src + 0);
                dst[0] = ((lo >> 0) & 0x7F) * scale + shift;
                dst[1] = ((lo >> 7) & 0x7F) * scale + shift;
                dst[2] = ((lo >> 14) & 0x7F) * scale + shift;
                dst[3] = ((lo >> 21) & 0x7F) * scale + shift;
                uint32_t hi = *(uint32_t*)(src + 3);
                dst[4] = ((hi >> 4) & 0x7F) * scale + shift;
                dst[5] = ((hi >> 11) & 0x7F) * scale + shift;
                dst[6] = ((hi >> 18) & 0x7F) * scale + shift;
                dst[7] = ((hi >> 25) & 0x7F) * scale + shift;
                src += 7;
                dst += 8;
            }
        }

        static void Decode32f8(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = src[i] * scale + shift;
        }

        //-------------------------------------------------------------------------------------------------

        static void Decode16f4(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t val = *(uint32_t*)(src + 0);
                dst[0] = Float32ToFloat16(((val >> 0) & 0xF) * scale + shift);
                dst[1] = Float32ToFloat16(((val >> 4) & 0xF) * scale + shift);
                dst[2] = Float32ToFloat16(((val >> 8) & 0xF) * scale + shift);
                dst[3] = Float32ToFloat16(((val >> 12) & 0xF) * scale + shift);
                dst[4] = Float32ToFloat16(((val >> 16) & 0xF) * scale + shift);
                dst[5] = Float32ToFloat16(((val >> 20) & 0xF) * scale + shift);
                dst[6] = Float32ToFloat16(((val >> 24) & 0xF) * scale + shift);
                dst[7] = Float32ToFloat16(((val >> 28) & 0xF) * scale + shift);
                src += 4;
                dst += 8;
            }
        }

        static void Decode16f5(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t lo = *(uint32_t*)(src + 0);
                dst[0] = Float32ToFloat16(((lo >> 0) & 0x1F) * scale + shift);
                dst[1] = Float32ToFloat16(((lo >> 5) & 0x1F) * scale + shift);
                dst[2] = Float32ToFloat16(((lo >> 10) & 0x1F) * scale + shift);
                dst[3] = Float32ToFloat16(((lo >> 15) & 0x1F) * scale + shift);
                uint32_t hi = *(uint32_t*)(src + 1);
                dst[4] = Float32ToFloat16(((hi >> 12) & 0x1F) * scale + shift);
                dst[5] = Float32ToFloat16(((hi >> 17) & 0x1F) * scale + shift);
                dst[6] = Float32ToFloat16(((hi >> 22) & 0x1F) * scale + shift);
                dst[7] = Float32ToFloat16(((hi >> 27) & 0x1F) * scale + shift);
                src += 5;
                dst += 8;
            }
        }

        static void Decode16f6(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t lo = *(uint32_t*)(src + 0);
                dst[0] = Float32ToFloat16(((lo >> 0) & 0x3F) * scale + shift);
                dst[1] = Float32ToFloat16(((lo >> 6) & 0x3F) * scale + shift);
                dst[2] = Float32ToFloat16(((lo >> 12) & 0x3F) * scale + shift);
                dst[3] = Float32ToFloat16(((lo >> 18) & 0x3F) * scale + shift);
                uint32_t hi = *(uint32_t*)(src + 2);
                dst[4] = Float32ToFloat16(((hi >> 8) & 0x3F) * scale + shift);
                dst[5] = Float32ToFloat16(((hi >> 14) & 0x3F) * scale + shift);
                dst[6] = Float32ToFloat16(((hi >> 20) & 0x3F) * scale + shift);
                dst[7] = Float32ToFloat16(((hi >> 26) & 0x3F) * scale + shift);
                src += 6;
                dst += 8;
            }
        }

        static void Decode16f7(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t lo = *(uint32_t*)(src + 0);
                dst[0] = Float32ToFloat16(((lo >> 0) & 0x7F) * scale + shift);
                dst[1] = Float32ToFloat16(((lo >> 7) & 0x7F) * scale + shift);
                dst[2] = Float32ToFloat16(((lo >> 14) & 0x7F) * scale + shift);
                dst[3] = Float32ToFloat16(((lo >> 21) & 0x7F) * scale + shift);
                uint32_t hi = *(uint32_t*)(src + 3);
                dst[4] = Float32ToFloat16(((hi >> 4) & 0x7F) * scale + shift);
                dst[5] = Float32ToFloat16(((hi >> 11) & 0x7F) * scale + shift);
                dst[6] = Float32ToFloat16(((hi >> 18) & 0x7F) * scale + shift);
                dst[7] = Float32ToFloat16(((hi >> 25) & 0x7F) * scale + shift);
                src += 7;
                dst += 8;
            }
        }

        static void Decode16f8(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = Float32ToFloat16(src[i] * scale + shift);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size);

        SIMD_INLINE int32_t Mul(int32_t a, int32_t b)
        {
            return a * b;
        }

        template<> int32_t Correlation<4>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            int32_t ab = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t a0 = *(uint32_t*)(a + 0);
                uint32_t b0 = *(uint32_t*)(b + 0);
                ab += Mul((a0 >> 0) & 0xF, (b0 >> 0) & 0xF);
                ab += Mul((a0 >> 4) & 0xF, (b0 >> 4) & 0xF);
                ab += Mul((a0 >> 8) & 0xF, (b0 >> 8) & 0xF);
                ab += Mul((a0 >> 12) & 0xF, (b0 >> 12) & 0xF);
                ab += Mul((a0 >> 16) & 0xF, (b0 >> 16) & 0xF);
                ab += Mul((a0 >> 20) & 0xF, (b0 >> 20) & 0xF);
                ab += Mul((a0 >> 24) & 0xF, (b0 >> 24) & 0xF);
                ab += Mul((a0 >> 28) & 0xF, (b0 >> 28) & 0xF);
                a += 4;
                b += 4;
            }
            return ab;
        }

        template<> int32_t Correlation<5>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            int32_t ab = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t a0 = *(uint32_t*)(a + 0);
                uint32_t b0 = *(uint32_t*)(b + 0);
                ab += Mul((a0 >> 0) & 0x1F, (b0 >> 0) & 0x1F);
                ab += Mul((a0 >> 5) & 0x1F, (b0 >> 5) & 0x1F);
                ab += Mul((a0 >> 10) & 0x1F, (b0 >> 10) & 0x1F);
                ab += Mul((a0 >> 15) & 0x1F, (b0 >> 15) & 0x1F);
                uint32_t a1 = *(uint32_t*)(a + 1);
                uint32_t b1 = *(uint32_t*)(b + 1);
                ab += Mul((a1 >> 12) & 0x1F, (b1 >> 12) & 0x1F);
                ab += Mul((a1 >> 17) & 0x1F, (b1 >> 17) & 0x1F);
                ab += Mul((a1 >> 22) & 0x1F, (b1 >> 22) & 0x1F);
                ab += Mul((a1 >> 27) & 0x1F, (b1 >> 27) & 0x1F);
                a += 5;
                b += 5;
            }
            return ab;
        }

        template<> int32_t Correlation<6>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            int32_t ab = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t a0 = *(uint32_t*)(a + 0);
                uint32_t b0 = *(uint32_t*)(b + 0);
                ab += Mul((a0 >> 0) & 0x3F, (b0 >> 0) & 0x3F);
                ab += Mul((a0 >> 6) & 0x3F, (b0 >> 6) & 0x3F);
                ab += Mul((a0 >> 12) & 0x3F, (b0 >> 12) & 0x3F);
                ab += Mul((a0 >> 18) & 0x3F, (b0 >> 18) & 0x3F);
                uint32_t a2 = *(uint32_t*)(a + 2);
                uint32_t b2 = *(uint32_t*)(b + 2);
                ab += Mul((a2 >> 8) & 0x3F, (b2 >> 8) & 0x3F);
                ab += Mul((a2 >> 14) & 0x3F, (b2 >> 14) & 0x3F);
                ab += Mul((a2 >> 20) & 0x3F, (b2 >> 20) & 0x3F);
                ab += Mul((a2 >> 26) & 0x3F, (b2 >> 26) & 0x3F);
                a += 6;
                b += 6;
            }
            return ab;
        }

        template<> int32_t Correlation<7>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            int32_t ab = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t a0 = *(uint32_t*)(a + 0);
                uint32_t b0 = *(uint32_t*)(b + 0);
                ab += Mul((a0 >> 0) & 0x7F, (b0 >> 0) & 0x7F);
                ab += Mul((a0 >> 7) & 0x7F, (b0 >> 7) & 0x7F);
                ab += Mul((a0 >> 14) & 0x7F, (b0 >> 14) & 0x7F);
                ab += Mul((a0 >> 21) & 0x7F, (b0 >> 21) & 0x7F);
                uint32_t a3 = *(uint32_t*)(a + 3);
                uint32_t b3 = *(uint32_t*)(b + 3);
                ab += Mul((a3 >> 4) & 0x7F, (b3 >> 4) & 0x7F);
                ab += Mul((a3 >> 11) & 0x7F, (b3 >> 11) & 0x7F);
                ab += Mul((a3 >> 18) & 0x7F, (b3 >> 18) & 0x7F);
                ab += Mul((a3 >> 25) & 0x7F, (b3 >> 25) & 0x7F);
                a += 7;
                b += 7;
            }
            return ab;
        }

        template<> int32_t Correlation<8>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            int32_t ab = 0;
            for (size_t i = 0; i < size; ++i)
                ab += a[i] * b[i];
            return ab;
        }

        template<int bits> void CosineDistance(const uint8_t* a, const uint8_t* b, size_t size, float* distance)
        {
            float abSum = (float)Correlation<bits>(a + 16, b + 16, size);
            Base::DecodeCosineDistance(a, b, abSum, distance);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            for (size_t i = 0; i < M; ++i)
            {
                const uint8_t* a = A[i];
                for (size_t j = 0; j < N; ++j)
                {
                    const uint8_t* b = B[j];
                    CosineDistance<bits>(a, b, size, distances++);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::Encode32fPtr GetEncode32f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Encode32f4;
            case 5: return Encode32f5;
            case 6: return Encode32f6;
            case 7: return Encode32f7;
            case 8: return Encode32f8;
            default: assert(0); return NULL;
            }
        }

        Base::DescrInt::Encode16fPtr GetEncode16f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Encode16f4;
            case 5: return Encode16f5;
            case 6: return Encode16f6;
            case 7: return Encode16f7;
            case 8: return Encode16f8;
            default: assert(0); return NULL;
            }
        }

        Base::DescrInt::Decode32fPtr GetDecode32f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Decode32f4;
            case 5: return Decode32f5;
            case 6: return Decode32f6;
            case 7: return Decode32f7;
            case 8: return Decode32f8;
            default: assert(0); return NULL;
            }
        }

        Base::DescrInt::Decode16fPtr GetDecode16f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Decode16f4;
            case 5: return Decode16f5;
            case 6: return Decode16f6;
            case 7: return Decode16f7;
            case 8: return Decode16f8;
            default: assert(0); return NULL;
            }
        }

        Base::DescrInt::CosineDistancePtr GetCosineDistance(size_t depth)
        {
            switch (depth)
            {
            case 4: return CosineDistance<4>;
            case 5: return CosineDistance<5>;
            case 6: return CosineDistance<6>;
            case 7: return CosineDistance<7>;
            case 8: return CosineDistance<8>;
            default: assert(0); return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        bool DescrInt::Valid(size_t size, size_t depth)
        {
            if (depth < 4 || depth > 8)
                return false;
            if (size == 0 || size % 8 != 0 || size > 128 * 256)
                return false;
            return true;
        }

        DescrInt::DescrInt(size_t size, size_t depth)
            : _size(size)
            , _depth(depth)
        {
            _encSize = 16 + DivHi(size * depth, 8);
            _range = float((1 << _depth) - 1);
            _minMax32f = MinMax32f;
            _minMax16f = MinMax16f;

            _encode32f = GetEncode32f(_depth);
            _encode16f = GetEncode16f(_depth);

            _decode32f = GetDecode32f(_depth);
            _decode16f = GetDecode16f(_depth);

            _cosineDistance = GetCosineDistance(_depth);

            _macroCosineDistancesDirect = NULL;
            _microMd = 0;
            _microNd = 0;

            _unpackNormA = NULL;
            _unpackNormB = NULL;
            _unpackDataA = NULL;
            _unpackDataB = NULL;
            _macroCosineDistancesUnpack = NULL;
            _unpSize = 0;
            _microMu = 0;
            _microNu = 0;
        }

        void DescrInt::Encode32f(const float* src, uint8_t* dst) const
        {
            float min, max;
            _minMax32f(src, _size, min, max);
            max = min + Simd::Max(max - min, SIMD_DESCR_INT_EPS);
            float scale = _range / (max - min), invScale = 1.0f / scale;
            ((float*)dst)[0] = invScale;
            ((float*)dst)[1] = min;
            int sum, sqsum;
            _encode32f(src, scale, min, _size, sum, sqsum, dst + 16);
            ((float*)dst)[2] = float(sum) * invScale + 0.5f * float(_size) * min;
            ((float*)dst)[3] = ::sqrt(float(sqsum) * invScale * invScale + 2.0f * sum * invScale * min  + float(_size) * min * min);
        }

        void DescrInt::Encode16f(const uint16_t* src, uint8_t* dst) const
        {
            float min, max;
            _minMax16f(src, _size, min, max);
            max = min + Simd::Max(max - min, SIMD_DESCR_INT_EPS);
            float scale = _range / (max - min), invScale = 1.0f / scale;
            ((float*)dst)[0] = invScale;
            ((float*)dst)[1] = min;
            int sum, sqsum;
            _encode16f(src, scale, min, _size, sum, sqsum, dst + 16);
            ((float*)dst)[2] = float(sum) * invScale + 0.5f * float(_size) * min;
            ((float*)dst)[3] = ::sqrt(float(sqsum) * invScale * invScale + 2.0f * sum * invScale * min + float(_size) * min * min);
        }

        void DescrInt::Decode32f(const uint8_t* src, float* dst) const
        {
            _decode32f(src + 16, ((float*)src)[0], ((float*)src)[1], _size, dst);
        }

        void DescrInt::Decode16f(const uint8_t* src, uint16_t* dst) const
        {
            _decode16f(src + 16, ((float*)src)[0], ((float*)src)[1], _size, dst);
        }

        void DescrInt::CosineDistance(const uint8_t* a, const uint8_t* b, float* distance) const
        {
            _cosineDistance(a, b, _size, distance);
        }

        void DescrInt::CosineDistancesMxNa(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const
        {
            if (_macroCosineDistancesDirect)
            {
                if (_unpSize * _microNu > Base::AlgCacheL1() || N * 2 < _microNu || M * 2 < _microMu  || _macroCosineDistancesUnpack == NULL)
                    CosineDistancesDirect(M, N, A, B, distances);
                else
                    CosineDistancesUnpack(M, N, A, B, distances);
            }
            else
            {
                for (size_t i = 0; i < M; ++i)
                {
                    const uint8_t* a = A[i];
                    for (size_t j = 0; j < N; ++j)
                    {
                        const uint8_t* b = B[j];
                        _cosineDistance(a, b, _size, distances++);
                    }
                }
            }
        }

        void DescrInt::CosineDistancesMxNp(size_t M, size_t N, const uint8_t* A, const uint8_t* B, float* distances) const
        {
            Array8ucp a(M);
            for (size_t i = 0; i < M; ++i)
                a[i] = A + i * _encSize;
            Array8ucp b(N);
            for (size_t j = 0; j < N; ++j)
                b[j] = B + j * _encSize;
            CosineDistancesMxNa(M, N, a.data, b.data, distances);
        }

        void DescrInt::VectorNorm(const uint8_t* a, float* norm) const
        {
            *norm = ((float*)a)[3];
        }

        void DescrInt::CosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const
        {
            const size_t L2 = Base::AlgCacheL2();
            size_t mN = AlignLoAny(L2 / _encSize, _microNd);
            size_t mM = AlignLoAny(L2 / _encSize, _microMd);
            for (size_t i = 0; i < M; i += mM)
            {
                size_t dM = Simd::Min(M, i + mM) - i;
                for (size_t j = 0; j < N; j += mN)
                {
                    size_t dN = Simd::Min(N, j + mN) - j;
                    _macroCosineDistancesDirect(dM, dN, A + i, B + j, _size, distances + i * N + j, N);
                }
            }
        }

        void DescrInt::CosineDistancesUnpack(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const
        {
            size_t macroM = AlignLoAny(Base::AlgCacheL2() / _unpSize, _microMu);
            size_t macroN = AlignLoAny(Base::AlgCacheL3() / _unpSize, _microNu);
            size_t sizeA = Min(macroM, M), sizeB = AlignHi(Min(macroN, N), _microNu);
            Array8u dA(sizeA * _unpSize), dB(sizeB * _unpSize);
            Array32f nA(sizeA * 4), nB(sizeB * 4);
            for (size_t i = 0; i < M; i += macroM)
            {
                size_t dM = Simd::Min(M, i + macroM) - i;
                _unpackNormA(dM, A + i, nA.data, 1);
                _unpackDataA(dM, A + i, _size, dA.data, _unpSize);
                for (size_t j = 0; j < N; j += macroN)
                {
                    size_t dN = Simd::Min(N, j + macroN) - j;
                    _unpackNormB(dN, B + j, nB.data, dN);
                    _unpackDataB(dN, B + j, _size, dB.data, 1);
                    _macroCosineDistancesUnpack(dM, dN, _size, dA.data, nA.data, dB.data, nB.data, distances + i * N + j, N);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth)
        {
            if(!Base::DescrInt::Valid(size, depth))
                return NULL;
            return new Base::DescrInt(size, depth);
        }
    }
}
