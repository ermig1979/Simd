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

        static void Encode7(const float* src, float scale, float min, size_t size, uint8_t* dst)
        {
            assert(size % 8 == 0);
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t v0 = Round((src[0] - min) * scale);
                uint32_t v1 = Round((src[1] - min) * scale);
                uint32_t v2 = Round((src[2] - min) * scale);
                uint32_t v3 = Round((src[3] - min) * scale);
                uint32_t v4 = Round((src[4] - min) * scale);
                uint32_t v5 = Round((src[5] - min) * scale);
                uint32_t v6 = Round((src[6] - min) * scale);
                uint32_t v7 = Round((src[7] - min) * scale);
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

        static void Encode8(const float* src, float scale, float min, size_t size, uint8_t* dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = Round((src[i] - min) * scale);
        }

        //-------------------------------------------------------------------------------------------------

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

        static SIMD_INLINE void Add(float a, float b, float & aa, float& ab, float& bb)
        {
            aa += a * a; 
            ab += a * b;
            bb += b * b;
        }

        static void CosineDistance7(const uint8_t* a, float aScale, float aShift, const uint8_t* b, float bScale, float bShift, size_t size, float* distance)
        {
            assert(size % 8 == 0);
            float aa = 0, ab = 0, bb = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t a0 = *(uint32_t*)(a + 0);
                uint32_t b0 = *(uint32_t*)(b + 0);
                Add((a0 & 0x7F) * aScale + aShift, (b0 & 0x7F) * bScale + bShift, aa, ab, bb);
                Add(((a0 >> 7) & 0x7F) * aScale + aShift, ((b0 >> 7) & 0x7F) * bScale + bShift, aa, ab, bb);
                Add(((a0 >> 14) & 0x7F) * aScale + aShift, ((b0 >> 14) & 0x7F) * bScale + bShift, aa, ab, bb);
                Add(((a0 >> 21) & 0x7F) * aScale + aShift, ((b0 >> 21) & 0x7F) * bScale + bShift, aa, ab, bb);
                uint32_t a3 = *(uint32_t*)(a + 3);
                uint32_t b3 = *(uint32_t*)(b + 3);
                Add(((a3 >> 4) & 0x7F) * aScale + aShift, ((b3 >> 4) & 0x7F) * bScale + bShift, aa, ab, bb);
                Add(((a3 >> 11) & 0x7F) * aScale + aShift, ((b3 >> 11) & 0x7F) * bScale + bShift, aa, ab, bb);
                Add(((a3 >> 18) & 0x7F) * aScale + aShift, ((b3 >> 18) & 0x7F) * bScale + bShift, aa, ab, bb);
                Add(((a3 >> 25) & 0x7F) * aScale + aShift, ((b3 >> 25) & 0x7F) * bScale + bShift, aa, ab, bb);
                a += 7;
                b += 7;
            }
            *distance = 1.0f - ab / ::sqrt(aa * bb);
        }

        static void CosineDistance8(const uint8_t* a, float aScale, float aShift, const uint8_t* b, float bScale, float bShift, size_t size, float* distance)
        {
            float aa = 0, ab = 0, bb = 0;
            for (size_t i = 0; i < size; ++i)
            {
                float _a = a[i] * aScale + aShift;
                float _b = b[i] * bScale + bShift;
                Add(_a, _b, aa, ab, bb);
            }
            *distance = 1.0f - ab / ::sqrt(aa * bb);
        }

        //-------------------------------------------------------------------------------------------------

        static void VectorNorm7(const uint8_t* src, float scale, float shift, size_t size, float* norm)
        {
            assert(size % 8 == 0);
            float sqsum = 0;
            for (size_t i = 0; i < size; i += 8)
            {
                uint32_t lo = *(uint32_t*)(src + 0);
                sqsum += Simd::Square((lo & 0x7F) * scale + shift);
                sqsum += Simd::Square(((lo >> 7) & 0x7F) * scale + shift);
                sqsum += Simd::Square(((lo >> 14) & 0x7F) * scale + shift);
                sqsum += Simd::Square(((lo >> 21) & 0x7F) * scale + shift);
                uint32_t hi = *(uint32_t*)(src + 3);
                sqsum += Simd::Square(((hi >> 4) & 0x7F) * scale + shift);
                sqsum += Simd::Square(((hi >> 11) & 0x7F) * scale + shift);
                sqsum += Simd::Square(((hi >> 18) & 0x7F) * scale + shift);
                sqsum += Simd::Square(((hi >> 25) & 0x7F) * scale + shift);
                src += 7;
            }
            *norm = ::sqrt(sqsum);
        }

        static void VectorNorm8(const uint8_t* src, float scale, float shift, size_t size, float* norm)
        {
            float sqsum = 0;
            for (size_t i = 0; i < size; ++i)
            {
                float val = src[i] * scale + shift;
                sqsum += val * val;
            }
            *norm = ::sqrt(sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        bool DescrInt::Valid(size_t size, size_t depth)
        {
            if (depth < 7 || depth > 8)
                return false;
            if (size == 0 || size % 8 != 0)
                return false;
            return true;
        }

        DescrInt::DescrInt(size_t size, size_t depth)
            : _size(size)
            , _depth(depth)
        {
            _encSize = 8 + DivHi(size * depth, 8);
            _range = float((1 << _depth) - 1);
            _minMax = MinMax;
            switch (depth)
            {
            case 7:
            {
                _encode = Encode7;
                _decode = Decode7;
                _cosineDistance = CosineDistance7;
                _vectorNorm = VectorNorm7;
                break;
            }
            case 8: 
            {
                _encode = Encode8; 
                _decode = Decode8;
                _cosineDistance = CosineDistance8;
                _vectorNorm = VectorNorm8;
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
            float scale = _range / (max - min);
            ((float*)dst)[0] = 1.0f / scale;
            ((float*)dst)[1] = min;
            _encode(src, scale, min, _size, dst + 8);
        }

        void DescrInt::Decode(const uint8_t* src, float* dst) const
        {
            _decode(src + 8, ((float*)src)[0], ((float*)src)[1], _size, dst);
        }

        void DescrInt::CosineDistance(const uint8_t* a, const uint8_t* b, float* distance) const
        {
            _cosineDistance(a + 8, ((float*)a)[0], ((float*)a)[1], b + 8, ((float*)b)[0], ((float*)b)[1], _size, distance);
        }

        void DescrInt::CosineDistancesMxNa(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const
        {
            for (size_t i = 0; i < M; ++i)
            {
                const uint8_t* a = A[i];
                for (size_t j = 0; j < N; ++j)
                {
                    const uint8_t* b = B[j];
                    _cosineDistance(a + 8, ((float*)a)[0], ((float*)a)[1], b + 8, ((float*)b)[0], ((float*)b)[1], _size, distances++);
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
                    _cosineDistance(a + 8, ((float*)a)[0], ((float*)a)[1], b + 8, ((float*)b)[0], ((float*)b)[1], _size, distances++);
                }
            }
        }

        void DescrInt::VectorNorm(const uint8_t* a, float* norm) const
        {
            _vectorNorm(a + 8, ((float*)a)[0], ((float*)a)[1], _size, norm);
        }

        void DescrInt::VectorNormsNa(size_t N, const uint8_t* const* A, float* norms) const
        {
            for (size_t i = 0; i < N; ++i)
            {
                const uint8_t* a = A[i];
                _vectorNorm(a + 8, ((float*)a)[0], ((float*)a)[1], _size, norms++);
            }
        }

        void DescrInt::VectorNormsNp(size_t N, const uint8_t* A, float* norms) const
        {
            for (size_t i = 0; i < N; ++i)
            {
                const uint8_t* a = A + i * _encSize;
                _vectorNorm(a + 8, ((float*)a)[0], ((float*)a)[1], _size, norms++);
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
