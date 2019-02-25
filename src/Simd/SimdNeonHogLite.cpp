/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        const uint32x4_t K32_0123 = SIMD_VEC_SETR_EPI32(0, 1, 2, 3);

        void HogLiteFindMax7x7(const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * pValue, size_t * pCol, size_t * pRow)
        {
            float32x4_t max = vdupq_n_f32(-FLT_MAX), val;
            uint32x4_t idx = vdupq_n_u32(0);
            uint32x4_t cur = K32_0123;
            uint32x4_t _3 = vdupq_n_u32(3), _5 = vdupq_n_u32(5);
            for (size_t row = 0; row < height; ++row)
            {
                val = vaddq_f32(Load<false>(a + 0), Load<false>(b + 0));
                max = vmaxq_f32(max, val);
                idx = vbslq_u32(vceqq_f32(max, val), cur, idx);
                cur = vaddq_u32(cur, _3);
                val = vaddq_f32(Load<false>(a + 3), Load<false>(b + 3));
                max = vmaxq_f32(max, val);
                idx = vbslq_u32(vceqq_f32(max, val), cur, idx);
                cur = vaddq_u32(cur, _5);
                a += aStride;
                b += bStride;
            }

            uint32_t _idx[F];
            float _max[F];
            Store<false>(_max, max);
            Store<false>(_idx, idx);
            *pValue = -FLT_MAX;
            for (size_t i = 0; i < F; ++i)
            {
                if (_max[i] > *pValue)
                {
                    *pValue = _max[i];
                    *pCol = _idx[i]&7;
                    *pRow = _idx[i]/8;
                }
            }
        }
    }
#endif// SIMD_NEON_ENABLE
}
