/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#ifdef SIMD_AVX512F_ENABLE 
    namespace Avx512f
    {
        const __m512i K64_PERMUTE_FOR_PACK = SIMD_MM512_SETR_EPI64(0, 2, 4, 6, 1, 3, 5, 7);

        ResizerFloatBilinear::ResizerFloatBilinear(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, bool caffeInterp)
            : Base::ResizerFloatBilinear(srcX, srcY, dstX, dstY, channels, sizeof(__m512), caffeInterp)
        {
        }

        void ResizerFloatBilinear::Run(const float * src, size_t srcStride, float * dst, size_t dstStride) const
        {
            Array32f bx[2];
            bx[0].Resize(_rs);
            bx[1].Resize(_rs);
            float * pbx[2] = { bx[0].data, bx[1].data };
            int32_t prev = -2;
            size_t rsa = AlignLo(_rs, Avx512f::F);
            __mmask16 tail = TailMask16(_rs - rsa);
            for (size_t dy = 0; dy < _dy; dy++, dst += dstStride)
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
                    if (_cn == 1)
                    {
                        __m512 _1 = _mm512_set1_ps(1.0f);
                        for (; dx < rsa; dx += Avx512f::F)
                        {
                            __m512i idx = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_load_si512(_ix.data + dx));
                            __m512 sp0 = _mm512_castpd_ps(_mm512_i32gather_pd(_mm512_extracti64x4_epi64(idx, 0), (double*)ps, 4));
                            __m512 sp1 = _mm512_castpd_ps(_mm512_i32gather_pd(_mm512_extracti64x4_epi64(idx, 1), (double*)ps, 4));
                            __m512 fx1 = _mm512_load_ps(_ax.data + dx);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            __m512 s0 = _mm512_shuffle_ps(sp0, sp1, 0x88);
                            __m512 s1 = _mm512_shuffle_ps(sp0, sp1, 0xDD);
                            _mm512_store_ps(pb + dx, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                        }
                        if (dx < _rs)
                        {
                            __m512i idx = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_maskz_loadu_epi32(tail, _ix.data + dx));
                            __m512 sp0 = _mm512_castpd_ps(_mm512_i32gather_pd(_mm512_extracti64x4_epi64(idx, 0), (double*)ps, 4));
                            __m512 sp1 = _mm512_castpd_ps(_mm512_i32gather_pd(_mm512_extracti64x4_epi64(idx, 1), (double*)ps, 4));
                            __m512 fx1 = _mm512_maskz_loadu_ps(tail, _ax.data + dx);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            __m512 s0 = _mm512_shuffle_ps(sp0, sp1, 0x88);
                            __m512 s1 = _mm512_shuffle_ps(sp0, sp1, 0xDD);
                            _mm512_mask_store_ps(pb + dx, tail, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                        }
                    }
                    else
                    {
                        __m512 _1 = _mm512_set1_ps(1.0f);
                        __m512i cn = _mm512_set1_epi32((int)_cn);
                        for (; dx < rsa; dx += Avx512f::F)
                        {
                            __m512i i0 = _mm512_load_si512(_ix.data + dx);
                            __m512i i1 = _mm512_add_epi32(i0, cn);
                            __m512 s0 = _mm512_i32gather_ps(i0, ps, 4);
                            __m512 s1 = _mm512_i32gather_ps(i1, ps, 4);
                            __m512 fx1 = _mm512_load_ps(_ax.data + dx);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            _mm512_store_ps(pb + dx, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                        }
                        if (dx < _rs)
                        {
                            __m512i i0 = _mm512_maskz_loadu_epi32(tail, _ix.data + dx);
                            __m512i i1 = _mm512_add_epi32(i0, cn);
                            __m512 s0 = _mm512_i32gather_ps(i0, ps, 4);
                            __m512 s1 = _mm512_i32gather_ps(i1, ps, 4);
                            __m512 fx1 = _mm512_maskz_loadu_ps(tail, _ax.data + dx);
                            __m512 fx0 = _mm512_sub_ps(_1, fx1);
                            _mm512_mask_store_ps(pb + dx, tail, _mm512_fmadd_ps(s0, fx0, _mm512_mul_ps(s1, fx1)));
                        }
                    }
                }  

                size_t dx = 0;
                __m512 _fy0 = _mm512_set1_ps(fy0);
                __m512 _fy1 = _mm512_set1_ps(fy1);
                for (; dx < rsa; dx += Avx512f::F)
                {
                    __m512 b0 = Load<true>(pbx[0] + dx);
                    __m512 b1 = Load<true>(pbx[1] + dx);
                    Store<false>(dst + dx, _mm512_fmadd_ps(b0, _fy0, _mm512_mul_ps(b1, _fy1)));
                }
                if (dx < _rs)
                {
                    __m512 b0 = Load<true, true>(pbx[0] + dx, tail);
                    __m512 b1 = Load<true, true>(pbx[1] + dx, tail);
                    Store<false, true>(dst + dx, _mm512_fmadd_ps(b0, _fy0, _mm512_mul_ps(b1, _fy1)), tail);
                }
            }
        }

        //---------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
        {
            if (type == SimdResizeChannelFloat && method == SimdResizeMethodBilinear)
                return new ResizerFloatBilinear(srcX, srcY, dstX, dstY, channels, false);
            else if (type == SimdResizeChannelFloat && method == SimdResizeMethodCaffeInterp)
                return new ResizerFloatBilinear(srcX, srcY, dstX, dstY, channels, true);
            else
                return Base::ResizerInit(srcX, srcY, dstX, dstY, channels, type, method);
        }
    }
#endif //SIMD_AVX512f_ENABLE 
}

