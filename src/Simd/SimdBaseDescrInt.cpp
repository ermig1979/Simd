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

#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"

namespace Simd
{
    namespace Base
    {
        static void MinMax(const float* src, size_t size, float& min, float& max)
        {
            min = FLT_MAX;
            max = -FLT_MAX;
            for (size_t i = 0; i < size; ++i)
            {
                min = Simd::Min(src[i], min);
                max = Simd::Max(src[i], max);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int32_t Encode(float src, float scale, float min, int32_t& sum, int32_t& sqsum)
        {
            int32_t value = Round((src - min) * scale);
            sum += value;
            sqsum += value * value;
            return value;
        }

        static void Encode6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 4 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 4)
            {
                uint32_t v0 = Encode(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode(src[1], scale, min, sum, sqsum);
                uint32_t v2 = Encode(src[2], scale, min, sum, sqsum);
                uint32_t v3 = Encode(src[3], scale, min, sum, sqsum);
                dst[0] = v0 | v1 << 6;
                dst[1] = v1 >> 2 | v2 << 4;
                dst[2] = v2 >> 4 | v3 << 2;
                src += 4;
                dst += 3;
            }
        }

        static void Encode7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t v0 = Encode(src[0], scale, min, sum, sqsum);
                uint32_t v1 = Encode(src[1], scale, min, sum, sqsum);
                uint32_t v2 = Encode(src[2], scale, min, sum, sqsum);
                uint32_t v3 = Encode(src[3], scale, min, sum, sqsum);
                uint32_t v4 = Encode(src[4], scale, min, sum, sqsum);
                uint32_t v5 = Encode(src[5], scale, min, sum, sqsum);
                uint32_t v6 = Encode(src[6], scale, min, sum, sqsum);
                uint32_t v7 = Encode(src[7], scale, min, sum, sqsum);
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

        static void Encode8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            sum = 0, sqsum = 0;
            for (size_t i = 0; i < size; ++i)
                dst[i] = (uint8_t)Encode(src[i], scale, min, sum, sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        static void Decode6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 4 == 0);
            for (size_t i = 0; i < size; i += 4)
            {
                uint32_t val = *(uint32_t*)(src + 0);
                dst[0] = (val & 0x3F) * scale + shift;
                dst[1] = ((val >> 6) & 0x3F) * scale + shift;
                dst[2] = ((val >> 12) & 0x3F) * scale + shift;
                dst[3] = ((val >> 18) & 0x3F) * scale + shift;
                src += 3;
                dst += 4;
            }
        }

        static void Decode7(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t lo = *(uint32_t*)(src + 0);
                dst[0] = (lo & 0x7F) * scale + shift;
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

        static void Decode8(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = src[i] * scale + shift;
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size);

        SIMD_INLINE int32_t Mul(int32_t a, int32_t b)
        {
            return a * b;
        }

        template<> int32_t Correlation<6>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 4 == 0);
            int32_t ab = 0;
            for (size_t i = 0; i < size; i += 4)
            {
                uint32_t a0 = *(uint32_t*)(a + 0);
                uint32_t b0 = *(uint32_t*)(b + 0);
                ab += Mul(a0 & 0x3F, b0 & 0x3F);
                ab += Mul((a0 >> 6) & 0x3F, (b0 >> 6) & 0x3F);
                ab += Mul((a0 >> 12) & 0x3F, (b0 >> 12) & 0x3F);
                ab += Mul((a0 >> 18) & 0x3F, (b0 >> 18) & 0x3F);
                a += 3;
                b += 3;
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
                ab += Mul(a0 & 0x7F, b0 & 0x7F);
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
            Base::DecodeCosineDistance(a, b, abSum, float(size), distance);
        }

        //-------------------------------------------------------------------------------------------------

        bool DescrInt::Valid(size_t size, size_t depth)
        {
            if (depth < 6 || depth > 8)
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
            _minMax = MinMax;
            switch (depth)
            {
            case 6:
            {
                _encode = Encode6;
                _decode = Decode6;
                _cosineDistance = Base::CosineDistance<6>;
                break;
            }
            case 7:
            {
                _encode = Encode7;
                _decode = Decode7;
                _cosineDistance = Base::CosineDistance<7>;
                break;
            }
            case 8: 
            {
                _encode = Encode8; 
                _decode = Decode8;
                _cosineDistance = Base::CosineDistance<8>;
                break;
            }
            default:
                assert(0);
            }
        }

        void DescrInt::Encode(const float* src, uint8_t* dst) const
        {
            float min, max;
            _minMax(src, _size, min, max);
            max = min + Simd::Max(max - min, SIMD_DESCR_INT_EPS);
            float scale = _range / (max - min), invScale = 1.0f / scale;
            ((float*)dst)[0] = invScale;
            ((float*)dst)[1] = min;
            int sum, sqsum;
            _encode(src, scale, min, _size, sum, sqsum, dst + 16);
#if SIMD_DESCR_INT_VER  == 1
            ((float*)dst)[2] = float(sum) * invScale + 0.5f * float(_size) * min;
            ((float*)dst)[3] = ::sqrt(float(sqsum) * invScale * invScale + 2.0f * sum * invScale * min  + float(_size) * min * min);

#else
            ((float*)dst)[2] = (float)sum;
            ((float*)dst)[3] = (float)sqsum;
#endif
        }

        void DescrInt::Decode(const uint8_t* src, float* dst) const
        {
            _decode(src + 16, ((float*)src)[0], ((float*)src)[1], _size, dst);
        }

        void DescrInt::CosineDistance(const uint8_t* a, const uint8_t* b, float* distance) const
        {
            _cosineDistance(a, b, _size, distance);
        }

        void DescrInt::CosineDistancesMxNa(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const
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

        void DescrInt::CosineDistancesMxNp(size_t M, size_t N, const uint8_t* A, const uint8_t* B, float* distances) const
        {
            for (size_t i = 0; i < M; ++i)
            {
                const uint8_t* a = A + i * _encSize;
                for (size_t j = 0; j < N; ++j)
                {
                    const uint8_t* b = B + j * _encSize;
                    _cosineDistance(a, b, _size, distances++);
                }
            }
        }

        void DescrInt::VectorNorm(const uint8_t* a, float* norm) const
        {
#if SIMD_DESCR_INT_VER  == 1
            *norm = ((float*)a)[3];
#else
            float scale = ((float*)a)[0];
            float shift = ((float*)a)[1];
            float sum = ((float*)a)[2];
            float sqsum = ((float*)a)[3];
            *norm = sqrt(sqsum * scale * scale + sum * scale * shift * 2.0f + float(_size) * shift * shift);
#endif
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
