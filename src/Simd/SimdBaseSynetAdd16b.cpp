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
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

    SynetAdd16b::SynetAdd16b(const Add16bParam& p)
        : _param(p)
    {

    }

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        template <class S, class D> SIMD_INLINE D Convert16b(const S& src)
        {
            return (D)src;
        }

        template <> SIMD_INLINE float Convert16b(const uint16_t& src)
        {
            return BFloat16ToFloat32(src);
        }

        template <> SIMD_INLINE uint16_t Convert16b(const float& src)
        {
            return Float32ToBFloat16(src);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename A, typename B, typename D> void Add16b(const A& a, const B& b, D& dst)
        {
            float _a = Convert16b<A, float>(a);
            float _b = Convert16b<B, float>(b);
            dst = Convert16b<float, D>(_a + _b);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename A, typename B, typename D> static void Add16bUniform(const uint8_t* a8, const uint8_t* b8, size_t size, uint8_t* dst8)
        {
            const A* a = (const A*)a8;
            const B* b = (const B*)b8;
            D* dst = (D*)dst8;
            for (size_t i = 0; i < size; ++i)
                Add16b(a[i], b[i], dst[i]);
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
            : SynetAdd16b(p)
            , _size(0)
            , _uniform(0)
        {
            assert(p.aShape == p.bShape);
            _size = 1;
            for(size_t i = 0; i < p.aShape.size(); ++i)
                _size *= p.aShape[i];
            _uniform = GetAdd16bUniform(p.aType, p.bType, p.dType);
        }

        bool SynetAdd16bUniform::Preferable(const Add16bParam& p)
        {
            if (p.aShape == p.bShape)
                return true;
            return false;
        }

        void SynetAdd16bUniform::Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst)
        {
            _uniform(a, b, _size, dst);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetAdd16bInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const size_t* bShape, size_t bCount, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format)
        {
            Add16bParam param(aShape, aCount, aType, bShape, bCount, bType, dstType, format);
            if (!param.Valid())
                return NULL;
            if (SynetAdd16bUniform::Preferable(param))
                return new SynetAdd16bUniform(param);
            return NULL;
        }
    }
#endif
}
