/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSve2.h"
#include "Simd/SimdSynetPermute.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        template<class T> SIMD_INLINE svuint32_t Load(const T* src, const svuint32_t& offset, const svbool_t& mask);

        template<> SIMD_INLINE svuint32_t Load<uint8_t>(const uint8_t* src, const svuint32_t& offset, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src, offset);
        }

        template<> SIMD_INLINE svuint32_t Load<uint16_t>(const uint16_t* src, const svuint32_t& offset, const svbool_t& mask)
        {
            return svld1uh_gather_u32offset_u32(mask, src, offset);
        }

        template<> SIMD_INLINE svuint32_t Load<uint32_t>(const uint32_t* src, const svuint32_t& offset, const svbool_t& mask)
        {
            return svld1_gather_u32offset_u32(mask, src, offset);
        }

        template<class T> SIMD_INLINE void Store(T* dst, const svuint32_t& value, const svbool_t& mask);

        template<> SIMD_INLINE void Store<uint8_t>(uint8_t* dst, const svuint32_t& value, const svbool_t& mask)
        {
            svst1b_u32(mask, dst, value);
        }

        template<> SIMD_INLINE void Store<uint16_t>(uint16_t* dst, const svuint32_t& value, const svbool_t& mask)
        {
            svst1h_u32(mask, dst, value);
        }

        template<> SIMD_INLINE void Store<uint32_t>(uint32_t* dst, const svuint32_t& value, const svbool_t& mask)
        {
            svst1_u32(mask, dst, value);
        }

        template<class T> void Permute2(const uint8_t* src_, const Base::Shape& shape, const Base::Shape& stride, uint8_t* dst_)
        {
            const T* src = (const T*)src_;
            T* dst = (T*)dst_;
            const size_t width = shape[0], height = shape[1], F = svcntw(), scale = stride[1] * sizeof(T);
            const svuint32_t offset = svmul_n_u32_x(svptrue_b32(), svindex_u32(0, 1), (uint32_t)scale);
            for (size_t i = 0; i < width; ++i)
            {
                const T* ps = src + i * stride[0];
                T* pd = dst + i * height;
                size_t j = 0;
                for (; j < height; j += F)
                {
                    svbool_t mask = svwhilelt_b32(j, height);
                    svuint32_t offs = svadd_n_u32_x(mask, offset, (uint32_t)(j * scale));
                    Store<T>(pd + j, Load<T>(ps, offs, mask), mask);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetPermute::SynetPermute(const Base::PermuteParam& param)
#ifdef SIMD_NEON_ENABLE
            : Neon::SynetPermute(param)
#else
            : Base::SynetPermute(param)
#endif
        {
            if (_count == 2)
            {
                switch (_param.type)
                {
                case SimdTensorData32f:
                case SimdTensorData32i:
                    _permute = Permute2<uint32_t>;
                    break;
                case SimdTensorData8i:
                case SimdTensorData8u:
                    _permute = Permute2<uint8_t>;
                    break;
                case SimdTensorData16b:
                case SimdTensorData16f:
                    _permute = Permute2<uint16_t>;
                    break;
                default:
                    assert(0);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetPermuteInit(const size_t* shape, const size_t* order, size_t count, SimdTensorDataType type)
        {
            Base::PermuteParam param(shape, order, count, type,
#ifdef SIMD_NEON_ENABLE
                Neon::A
#else
                1
#endif
            );
            if (!param.Valid())
                return NULL;
            return new SynetPermute(param);
        }
    }
#endif
}
