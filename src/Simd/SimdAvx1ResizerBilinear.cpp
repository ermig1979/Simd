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
#include "Simd/SimdMemory.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_AVX_ENABLE 
    namespace Avx
    {
        ResizerFloatBilinear::ResizerFloatBilinear(const ResParam & param)
            : Base::ResizerFloatBilinear(param)
        {
        }

        void ResizerFloatBilinear::Run(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            size_t cn = _param.channels;
            size_t rs = _param.dstW * cn;
            float * pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            size_t rsa = AlignLo(rs, Avx::F);
            size_t rsh = AlignLo(rs, Sse2::F);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    float * pb = pbx[k];
                    const float * ps = src + (sy + k)*srcStride;
                    size_t dx = 0;
                    if (cn == 1)
                    {
                        __m256 _1 = _mm256_set1_ps(1.0f);
                        for (; dx < rsa; dx += Avx::F)
                        {
                            __m256 s0145 = Avx::Load(ps + _ix[dx + 0], ps + _ix[dx + 1], ps + _ix[dx + 4], ps + _ix[dx + 5]);
                            __m256 s2367 = Avx::Load(ps + _ix[dx + 2], ps + _ix[dx + 3], ps + _ix[dx + 6], ps + _ix[dx + 7]);
                            __m256 fx1 = _mm256_load_ps(_ax.data + dx);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            __m256 m0 = _mm256_mul_ps(fx0, _mm256_shuffle_ps(s0145, s2367, 0x88));
                            __m256 m1 = _mm256_mul_ps(fx1, _mm256_shuffle_ps(s0145, s2367, 0xDD));
                            _mm256_store_ps(pb + dx, _mm256_add_ps(m0, m1));
                        }
                        for (; dx < rsh; dx += Sse2::F)
                        {
                            __m128 s01 = Sse2::Load(ps + _ix[dx + 0], ps + _ix[dx + 1]);
                            __m128 s23 = Sse2::Load(ps + _ix[dx + 2], ps + _ix[dx + 3]);
                            __m128 fx1 = _mm_load_ps(_ax.data + dx);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            __m128 m0 = _mm_mul_ps(fx0, _mm_shuffle_ps(s01, s23, 0x88));
                            __m128 m1 = _mm_mul_ps(fx1, _mm_shuffle_ps(s01, s23, 0xDD));
                            _mm_store_ps(pb + dx, _mm_add_ps(m0, m1));
                        }
                    }
                    if (cn == 3 && rs > 3)
                    {
                        __m256 _1 = _mm256_set1_ps(1.0f);
                        size_t rs3 = rs - 3;
                        size_t rs6 = AlignLoAny(rs3, 6);
                        for (; dx < rs6; dx += 6)
                        {
                            __m256 s0 = Load<false>(ps + _ix[dx + 0] + 0, ps + _ix[dx + 3] + 0);
                            __m256 s1 = Load<false>(ps + _ix[dx + 0] + 3, ps + _ix[dx + 3] + 3);
                            __m256 fx1 = Load<false>(_ax.data + dx + 0, _ax.data + dx + 3);
                            __m256 fx0 = _mm256_sub_ps(_1, fx1);
                            Store<false>(pb + dx + 0, pb + dx + 3, _mm256_add_ps(_mm256_mul_ps(fx0, s0), _mm256_mul_ps(fx1, s1)));
                        }
                        for (; dx < rs3; dx += 3)
                        {
                            __m128 s0 = _mm_loadu_ps(ps + _ix[dx] + 0);
                            __m128 s1 = _mm_loadu_ps(ps + _ix[dx] + 3);
                            __m128 fx1 = _mm_set1_ps(_ax.data[dx]);
                            __m128 fx0 = _mm_sub_ps(_mm256_castps256_ps128(_1), fx1);
                            _mm_storeu_ps(pb + dx, _mm_add_ps(_mm_mul_ps(fx0, s0), _mm_mul_ps(fx1, s1)));
                        }
                    }
                    for (; dx < rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        float fx = _ax[dx];
                        pb[dx] = ps[sx] * (1.0f - fx) + ps[sx + cn] * fx;
                    }
                }  

                size_t dx = 0;
                __m256 _fy0 = _mm256_set1_ps(fy0);
                __m256 _fy1 = _mm256_set1_ps(fy1);
                for (; dx < rsa; dx += Avx::F)
                {
                    __m256 m0 = _mm256_mul_ps(_mm256_load_ps(pbx[0] + dx), _fy0);
                    __m256 m1 = _mm256_mul_ps(_mm256_load_ps(pbx[1] + dx), _fy1);
                    _mm256_storeu_ps(dst + dx, _mm256_add_ps(m0, m1));
                }
                for (; dx < rsh; dx += Sse2::F)
                {
                    __m128 m0 = _mm_mul_ps(_mm_load_ps(pbx[0] + dx), _mm256_castps256_ps128(_fy0));
                    __m128 m1 = _mm_mul_ps(_mm_load_ps(pbx[1] + dx), _mm256_castps256_ps128(_fy1));
                    _mm_storeu_ps(dst + dx, _mm_add_ps(m0, m1));
                }
                for (; dx < rs; dx++)
                    dst[dx] = pbx[0][dx] * fy0 + pbx[1][dx] * fy1;
            }
        }
    }
#endif //SIMD_AVX_ENABLE 
}

