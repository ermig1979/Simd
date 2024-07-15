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
#include "Simd/SimdSynetAdd16b.h"
#include "Simd/SimdSynetAdd16bCommon.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse41
    {
        template <typename A, typename B, typename D> void Add16bDF(const A* a, const B* b, D* dst);

        template <> SIMD_INLINE void Add16bDF(const float* a, const float* b, float* dst)
        {
            _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_loadu_ps(a + 0), _mm_loadu_ps(b + 0)));
            _mm_storeu_ps(dst + F, _mm_add_ps(_mm_loadu_ps(a + F), _mm_loadu_ps(b + F)));
        }

        template <> SIMD_INLINE void Add16bDF(const float* a, const uint16_t* b, float* dst)
        {
            __m128i _b = _mm_loadu_si128((__m128i*)b);
            _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_loadu_ps(a + 0), BFloat16ToFloat32<0>(_b)));
            _mm_storeu_ps(dst + F, _mm_add_ps(_mm_loadu_ps(a + F), BFloat16ToFloat32<1>(_b)));
        }

        template <> SIMD_INLINE void Add16bDF(const uint16_t* a, const float* b, float* dst)
        {
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            _mm_storeu_ps(dst + 0, _mm_add_ps(BFloat16ToFloat32<0>(_a), _mm_loadu_ps(b + 0)));
            _mm_storeu_ps(dst + F, _mm_add_ps(BFloat16ToFloat32<1>(_a), _mm_loadu_ps(b + F)));
        }

        template <> SIMD_INLINE void Add16bDF(const uint16_t* a, const uint16_t* b, float* dst)
        {
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            __m128i _b = _mm_loadu_si128((__m128i*)b);
            _mm_storeu_ps(dst + 0, _mm_add_ps(BFloat16ToFloat32<0>(_a), BFloat16ToFloat32<0>(_b)));
            _mm_storeu_ps(dst + F, _mm_add_ps(BFloat16ToFloat32<1>(_a), BFloat16ToFloat32<1>(_b)));
        }

        template <> SIMD_INLINE void Add16bDF(const float* a, const float* b, uint16_t* dst)
        {
            __m128 dst0 = _mm_add_ps(_mm_loadu_ps(a + 0), _mm_loadu_ps(b + 0));
            __m128 dst1 = _mm_add_ps(_mm_loadu_ps(a + F), _mm_loadu_ps(b + F));
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void Add16bDF(const float* a, const uint16_t* b, uint16_t* dst)
        {
            __m128i _b = _mm_loadu_si128((__m128i*)b);
            __m128 dst0 = _mm_add_ps(_mm_loadu_ps(a + 0), BFloat16ToFloat32<0>(_b));
            __m128 dst1 = _mm_add_ps(_mm_loadu_ps(a + F), BFloat16ToFloat32<1>(_b));
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void Add16bDF(const uint16_t* a, const float* b, uint16_t* dst)
        {
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            __m128 dst0 = _mm_add_ps(BFloat16ToFloat32<0>(_a), _mm_loadu_ps(b + 0));
            __m128 dst1 = _mm_add_ps(BFloat16ToFloat32<1>(_a), _mm_loadu_ps(b + F));
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16(dst0, dst1));
        }

        template <> SIMD_INLINE void Add16bDF(const uint16_t* a, const uint16_t* b, uint16_t* dst)
        {
            __m128i _a = _mm_loadu_si128((__m128i*)a);
            __m128i _b = _mm_loadu_si128((__m128i*)b);
            __m128 even = _mm_add_ps(BFloat16ToFloat32Even(_a), BFloat16ToFloat32Even(_b));
            __m128 odd = _mm_add_ps(BFloat16ToFloat32Odd(_a), BFloat16ToFloat32Odd(_b));
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16Interlived(even, odd));
        }

        template <typename A, typename B, typename D> static void Add16bUniform(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
        {
            const A* a = (const A*)a8;
            const B* b = (const B*)b8;
            D* dst = (D*)dst8;
            size_t sizeDF = AlignLo(size, DF), i = 0;

            for (; i < sizeDF; i += DF)
                Add16bDF(a + i, b + i, dst + i);
            for (; i < size; ++i)
                Base::Add16b(a[i], b[i], dst[i]);
        }

        template<class A, class B> static SynetAdd16bUniform::UniformPtr GetAdd16bUniform(SimdTensorDataType dType)
        {
            switch (dType)
            {
            case SimdTensorData32f: return Add16bUniform<A, B, float>;
            case SimdTensorData16b: return Add16bUniform<A, B, uint16_t>;
            default:
                return NULL;
            }
        }

        template<class A> static SynetAdd16bUniform::UniformPtr GetAdd16bUniform(SimdTensorDataType bType, SimdTensorDataType dType)
        {
            switch (bType)
            {
            case SimdTensorData32f: return GetAdd16bUniform<A, float>(dType);
            case SimdTensorData16b: return GetAdd16bUniform<A, uint16_t>(dType);
            default:
                return NULL;
            }
        }

        static SynetAdd16bUniform::UniformPtr GetAdd16bUniform(SimdTensorDataType aType, SimdTensorDataType bType, SimdTensorDataType dType)
        {
            switch (aType)
            {
            case SimdTensorData32f: return GetAdd16bUniform<float>(bType, dType);
            case SimdTensorData16b: return GetAdd16bUniform<uint16_t>(bType, dType);
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetAdd16bUniform::SynetAdd16bUniform(const Add16bParam& p)
            : Base::SynetAdd16bUniform(p)
        {
             _uniform = GetAdd16bUniform(p.aType, p.bType, p.dType);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetAdd16bInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const size_t* bShape, size_t bCount, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format)
        {
            Add16bParam param(aShape, aCount, aType, bShape, bCount, bType, dstType, format);
            if (!param.Valid())
                return NULL;
            if (Base::SynetAdd16bUniform::Preferable(param))
                return new SynetAdd16bUniform(param);
            return NULL;
        }
    }
#endif
}
