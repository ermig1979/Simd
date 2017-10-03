/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t count)
                {
                    size_t size = sizeof(float)*count;
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    sums = (float*)_p;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                float * sums;
            private:
                void *_p;
            };
        }

        SIMD_INLINE void AddMul(const float * svs, const v128_f32 & _v, size_t offset, float * sums)
        {
            v128_f32 _sums = Load<true>(sums + offset);
            v128_f32 _svs = Load<false>(svs + offset);
            Store<true>(sums + offset, vec_add(_sums, vec_mul(_v, _svs)));
        }

        SIMD_INLINE void AddMul(const float * sums, const float * weights, size_t offset, v128_f32 & sum)
        {
            v128_f32 _sums = Load<true>(sums + offset);
            v128_f32 _weights = Load<false>(weights + offset);
            sum = vec_add(sum, vec_mul(_sums, _weights));
        }

        void SvmSumLinear(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum)
        {
            Buffer buffer(count);
            size_t partialAlignedCount = AlignLo(count, 4);
            size_t fullAlignedCount = AlignLo(count, 16);

            for (size_t j = 0; j < length; ++j)
            {
                size_t i = 0;
                float v = x[j];
                v128_f32 _v = SetF32(v);
                for (; i < fullAlignedCount; i += 16)
                {
                    AddMul(svs, _v, i, buffer.sums);
                    AddMul(svs, _v, i + 4, buffer.sums);
                    AddMul(svs, _v, i + 8, buffer.sums);
                    AddMul(svs, _v, i + 12, buffer.sums);
                }
                for (; i < partialAlignedCount; i += 4)
                    AddMul(svs, _v, i, buffer.sums);
                for (; i < count; ++i)
                    buffer.sums[i] += v*svs[i];
                svs += count;
            }

            size_t i = 0;
            v128_f32 _sum = K_0_0f;
            for (; i < fullAlignedCount; i += 16)
            {
                AddMul(buffer.sums, weights, i, _sum);
                AddMul(buffer.sums, weights, i + 4, _sum);
                AddMul(buffer.sums, weights, i + 8, _sum);
                AddMul(buffer.sums, weights, i + 12, _sum);
            }
            for (; i < partialAlignedCount; i += 4)
                AddMul(buffer.sums, weights, i, _sum);
            *sum = ExtractSum(_sum);
            for (; i < count; ++i)
                *sum += buffer.sums[i] * weights[i];
        }
    }
#endif// SIMD_VSX_ENABLE
}
