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
#include "Simd/SimdSynetDeconvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512bw
    {
        typedef Base::SynetDeconvolution16bNhwcGemm::AlgParam AlgParam;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcGemm(const uint8_t* src8, const DeconvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8 + yBeg * p.srcW * p.srcC;
            size_t size = p.srcC, gap = a.bufK - size;
            size_t size32 = AlignLo(size, 32);
            __mmask16 srcMask[2];
            __mmask32 dstMask[1];
            if (size32 < size)
            {
                srcMask[0] = TailMask16(size - size32 - F * 0);
                srcMask[1] = TailMask16(size - size32 - F * 1);
                dstMask[0] = TailMask32(size - size32);
            }
            for (size_t sy = yBeg; sy < yEnd; ++sy)
            {
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < size32; sc += 32)
                        Avx512bw::Float32ToBFloat16<false, false>(src + sc, dst + sc, srcMask, dstMask);
                    if (size32 < size)
                        Avx512bw::Float32ToBFloat16<false, true>(src + sc, dst + sc, srcMask, dstMask);
                    src += size;
                    dst += size;
                    for (size_t g = 0; g < gap; ++g)
                        *(dst++) = 0;
                }
            }
        }

        static void Reorder16bNhwcGemm(const uint8_t* src8, const DeconvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            size_t size = a.K, gap = a.bufK - size;
            const uint16_t* src = (uint16_t*)src8 + yBeg * p.srcW * p.srcC;
            for (size_t sy = yBeg; sy < yEnd; ++sy)
            {
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    memcpy(dst, src, size * 2);
                    src += size;
                    dst += size;                    
                    for (size_t g = 0; g < gap; ++g)
                        *(dst++) = 0;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Load1(__m512 & val0, const float* src, const __mmask16 tail)
        {
            val0 = _mm512_maskz_loadu_ps(tail, src + 0);
        }

        SIMD_INLINE void Load2(__m512& val0, __m512& val1, const float* src, const __mmask16 tail)
        {
            val0 = _mm512_loadu_ps(src + 0);
            val1 = _mm512_maskz_loadu_ps(tail, src + F);
        }

        SIMD_INLINE void Save1(__m512 val0, float* dst, const __mmask16 tail)
        {
            _mm512_mask_storeu_ps(dst, tail, val0);
        }

        SIMD_INLINE void Save2(__m512 val0, __m512 val1, float* dst, const __mmask16 tail)
        {
            _mm512_storeu_ps(dst + 0, val0);
            _mm512_mask_storeu_ps(dst + F, tail, val1);
        }

        template<int M> void Deconvolution16bNhwcGemm_2xM(const uint16_t* src0, const DeconvParam& p, const AlgParam& a, 
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, float* dst, const __mmask16 tails[2])
        {
            __m512 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51,
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1,
                s0, w00, w01, w10, w11, m = _mm512_castsi512_ps(Bf16::MASK);
            size_t dD = a.bufN, dS = a.bufK;
            const uint16_t* weight1 = weight0 + a.bufK * F;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            const uint16_t* src5 = src0 + 5 * dS;
            if (tails[1])
            {
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps(), d01 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps(), d11 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps(), d21 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps(), d31 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps(), d41 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps(), d51 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps(), d61 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps(), d71 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps(), d81 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps(), d91 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps(), da1 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps(), db1 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) Load2(d00, d01, dst + 0x0 * dD, tails[1]);
                    if (M > 0x1) Load2(d10, d11, dst + 0x1 * dD, tails[1]);
                    if (M > 0x2) Load2(d20, d21, dst + 0x2 * dD, tails[1]);
                    if (M > 0x3) Load2(d30, d31, dst + 0x3 * dD, tails[1]);
                    if (M > 0x4) Load2(d40, d41, dst + 0x4 * dD, tails[1]);
                    if (M > 0x5) Load2(d50, d51, dst + 0x5 * dD, tails[1]);
                    if (M > 0x6) Load2(d60, d61, dst + 0x6 * dD, tails[1]);
                    if (M > 0x7) Load2(d70, d71, dst + 0x7 * dD, tails[1]);
                    if (M > 0x8) Load2(d80, d81, dst + 0x8 * dD, tails[1]);
                    if (M > 0x9) Load2(d90, d91, dst + 0x9 * dD, tails[1]);
                    if (M > 0xa) Load2(da0, da1, dst + 0xa * dD, tails[1]);
                    if (M > 0xb) Load2(db0, db1, dst + 0xb * dD, tails[1]);
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 2, offs6 += 2)
                {
                    w01 = _mm512_loadu_ps((float*)weight0);
                    w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                    w01 = _mm512_and_ps(w01, m);
                    w11 = _mm512_loadu_ps((float*)weight1);
                    w10 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w11), Base::Bf16::SHIFT));
                    w11 = _mm512_and_ps(w11, m);
                    if (M > 0x0)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                        d00 = _mm512_fmadd_ps(s0, w00, d00); d01 = _mm512_fmadd_ps(s0, w10, d01);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                        d00 = _mm512_fmadd_ps(s0, w01, d00); d01 = _mm512_fmadd_ps(s0, w11, d01);
                    }
                    if (M > 0x1)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                        d10 = _mm512_fmadd_ps(s0, w00, d10); d11 = _mm512_fmadd_ps(s0, w10, d11);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                        d10 = _mm512_fmadd_ps(s0, w01, d10); d11 = _mm512_fmadd_ps(s0, w11, d11);
                    }
                    if (M > 0x2)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                        d20 = _mm512_fmadd_ps(s0, w00, d20); d21 = _mm512_fmadd_ps(s0, w10, d21);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                        d20 = _mm512_fmadd_ps(s0, w01, d20); d21 = _mm512_fmadd_ps(s0, w11, d21);
                    }
                    if (M > 0x3)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                        d30 = _mm512_fmadd_ps(s0, w00, d30); d31 = _mm512_fmadd_ps(s0, w10, d31);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                        d30 = _mm512_fmadd_ps(s0, w01, d30); d31 = _mm512_fmadd_ps(s0, w11, d31);
                    }
                    if (M > 0x4)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                        d40 = _mm512_fmadd_ps(s0, w00, d40); d41 = _mm512_fmadd_ps(s0, w10, d41);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                        d40 = _mm512_fmadd_ps(s0, w01, d40); d41 = _mm512_fmadd_ps(s0, w11, d41);
                    }
                    if (M > 0x5)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                        d50 = _mm512_fmadd_ps(s0, w00, d50); d51 = _mm512_fmadd_ps(s0, w10, d51);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                        d50 = _mm512_fmadd_ps(s0, w01, d50); d51 = _mm512_fmadd_ps(s0, w11, d51);
                    }
                    if (M > 0x6)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                        d60 = _mm512_fmadd_ps(s0, w00, d60); d61 = _mm512_fmadd_ps(s0, w10, d61);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                        d60 = _mm512_fmadd_ps(s0, w01, d60); d61 = _mm512_fmadd_ps(s0, w11, d61);
                    }
                    if (M > 0x7)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                        d70 = _mm512_fmadd_ps(s0, w00, d70); d71 = _mm512_fmadd_ps(s0, w10, d71);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                        d70 = _mm512_fmadd_ps(s0, w01, d70); d71 = _mm512_fmadd_ps(s0, w11, d71);
                    }
                    if (M > 0x8)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                        d80 = _mm512_fmadd_ps(s0, w00, d80); d81 = _mm512_fmadd_ps(s0, w10, d81);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                        d80 = _mm512_fmadd_ps(s0, w01, d80); d81 = _mm512_fmadd_ps(s0, w11, d81);
                    }
                    if (M > 0x9)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                        d90 = _mm512_fmadd_ps(s0, w00, d90); d91 = _mm512_fmadd_ps(s0, w10, d91);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                        d90 = _mm512_fmadd_ps(s0, w01, d90); d91 = _mm512_fmadd_ps(s0, w11, d91);
                    }
                    if (M > 0xa)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                        da0 = _mm512_fmadd_ps(s0, w00, da0); da1 = _mm512_fmadd_ps(s0, w10, da1);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                        da0 = _mm512_fmadd_ps(s0, w01, da0); da1 = _mm512_fmadd_ps(s0, w11, da1);
                    }
                    if (M > 0xb)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                        db0 = _mm512_fmadd_ps(s0, w00, db0); db1 = _mm512_fmadd_ps(s0, w10, db1);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                        db0 = _mm512_fmadd_ps(s0, w01, db0); db1 = _mm512_fmadd_ps(s0, w11, db1);
                    }
                    weight0 += DF;
                    weight1 += DF;
                }
                if (M > 0x0) Save2(d00, d01, dst + 0x0 * dD, tails[1]);
                if (M > 0x1) Save2(d10, d11, dst + 0x1 * dD, tails[1]);
                if (M > 0x2) Save2(d20, d21, dst + 0x2 * dD, tails[1]);
                if (M > 0x3) Save2(d30, d31, dst + 0x3 * dD, tails[1]);
                if (M > 0x4) Save2(d40, d41, dst + 0x4 * dD, tails[1]);
                if (M > 0x5) Save2(d50, d51, dst + 0x5 * dD, tails[1]);
                if (M > 0x6) Save2(d60, d61, dst + 0x6 * dD, tails[1]);
                if (M > 0x7) Save2(d70, d71, dst + 0x7 * dD, tails[1]);
                if (M > 0x8) Save2(d80, d81, dst + 0x8 * dD, tails[1]);
                if (M > 0x9) Save2(d90, d91, dst + 0x9 * dD, tails[1]);
                if (M > 0xa) Save2(da0, da1, dst + 0xa * dD, tails[1]);
                if (M > 0xb) Save2(db0, db1, dst + 0xb * dD, tails[1]);
            }
            else
            {
                if (zero)
                {
                    if (M > 0x0) d00 = _mm512_setzero_ps();
                    if (M > 0x1) d10 = _mm512_setzero_ps();
                    if (M > 0x2) d20 = _mm512_setzero_ps();
                    if (M > 0x3) d30 = _mm512_setzero_ps();
                    if (M > 0x4) d40 = _mm512_setzero_ps();
                    if (M > 0x5) d50 = _mm512_setzero_ps();
                    if (M > 0x6) d60 = _mm512_setzero_ps();
                    if (M > 0x7) d70 = _mm512_setzero_ps();
                    if (M > 0x8) d80 = _mm512_setzero_ps();
                    if (M > 0x9) d90 = _mm512_setzero_ps();
                    if (M > 0xa) da0 = _mm512_setzero_ps();
                    if (M > 0xb) db0 = _mm512_setzero_ps();
                }
                else
                {
                    if (M > 0x0) Load1(d00, dst + 0x0 * dD, tails[0]);
                    if (M > 0x1) Load1(d10, dst + 0x1 * dD, tails[0]);
                    if (M > 0x2) Load1(d20, dst + 0x2 * dD, tails[0]);
                    if (M > 0x3) Load1(d30, dst + 0x3 * dD, tails[0]);
                    if (M > 0x4) Load1(d40, dst + 0x4 * dD, tails[0]);
                    if (M > 0x5) Load1(d50, dst + 0x5 * dD, tails[0]);
                    if (M > 0x6) Load1(d60, dst + 0x6 * dD, tails[0]);
                    if (M > 0x7) Load1(d70, dst + 0x7 * dD, tails[0]);
                    if (M > 0x8) Load1(d80, dst + 0x8 * dD, tails[0]);
                    if (M > 0x9) Load1(d90, dst + 0x9 * dD, tails[0]);
                    if (M > 0xa) Load1(da0, dst + 0xa * dD, tails[0]);
                    if (M > 0xb) Load1(db0, dst + 0xb * dD, tails[0]);
                }
                for (size_t offs0 = 0, offs6 = offs0 + 6 * dS; offs0 < srcC; offs0 += 2, offs6 += 2)
                {
                    w01 = _mm512_loadu_ps((float*)weight0);
                    w00 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(w01), Base::Bf16::SHIFT));
                    w01 = _mm512_and_ps(w01, m);
                    if (M > 0x0)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs0]));
                        d00 = _mm512_fmadd_ps(s0, w00, d00);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs0)), m);
                        d00 = _mm512_fmadd_ps(s0, w01, d00);
                    }
                    if (M > 0x1)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs0]));
                        d10 = _mm512_fmadd_ps(s0, w00, d10);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs0)), m);
                        d10 = _mm512_fmadd_ps(s0, w01, d10);
                    }
                    if (M > 0x2)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs0]));
                        d20 = _mm512_fmadd_ps(s0, w00, d20);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs0)), m);
                        d20 = _mm512_fmadd_ps(s0, w01, d20);
                    }
                    if (M > 0x3)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs0]));
                        d30 = _mm512_fmadd_ps(s0, w00, d30);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs0)), m);
                        d30 = _mm512_fmadd_ps(s0, w01, d30);
                    }
                    if (M > 0x4)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs0]));
                        d40 = _mm512_fmadd_ps(s0, w00, d40);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs0)), m);
                        d40 = _mm512_fmadd_ps(s0, w01, d40);
                    }
                    if (M > 0x5)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs0]));
                        d50 = _mm512_fmadd_ps(s0, w00, d50);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs0)), m);
                        d50 = _mm512_fmadd_ps(s0, w01, d50);
                    }
                    if (M > 0x6)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src0[offs6]));
                        d60 = _mm512_fmadd_ps(s0, w00, d60);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src0 + offs6)), m);
                        d60 = _mm512_fmadd_ps(s0, w01, d60);
                    }
                    if (M > 0x7)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src1[offs6]));
                        d70 = _mm512_fmadd_ps(s0, w00, d70);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src1 + offs6)), m);
                        d70 = _mm512_fmadd_ps(s0, w01, d70);
                    }
                    if (M > 0x8)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src2[offs6]));
                        d80 = _mm512_fmadd_ps(s0, w00, d80);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src2 + offs6)), m);
                        d80 = _mm512_fmadd_ps(s0, w01, d80);
                    }
                    if (M > 0x9)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src3[offs6]));
                        d90 = _mm512_fmadd_ps(s0, w00, d90);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src3 + offs6)), m);
                        d90 = _mm512_fmadd_ps(s0, w01, d90);
                    }
                    if (M > 0xa)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src4[offs6]));
                        da0 = _mm512_fmadd_ps(s0, w00, da0);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src4 + offs6)), m);
                        da0 = _mm512_fmadd_ps(s0, w01, da0);
                    }
                    if (M > 0xb)
                    {
                        s0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, src5[offs6]));
                        db0 = _mm512_fmadd_ps(s0, w00, db0);
                        s0 = _mm512_and_ps(_mm512_set1_ps(*(float*)(src5 + offs6)), m);
                        db0 = _mm512_fmadd_ps(s0, w01, db0);
                    }
                    weight0 += DF;
                }
                if (M > 0x0) Save1(d00, dst + 0x0 * dD, tails[0]);
                if (M > 0x1) Save1(d10, dst + 0x1 * dD, tails[0]);
                if (M > 0x2) Save1(d20, dst + 0x2 * dD, tails[0]);
                if (M > 0x3) Save1(d30, dst + 0x3 * dD, tails[0]);
                if (M > 0x4) Save1(d40, dst + 0x4 * dD, tails[0]);
                if (M > 0x5) Save1(d50, dst + 0x5 * dD, tails[0]);
                if (M > 0x6) Save1(d60, dst + 0x6 * dD, tails[0]);
                if (M > 0x7) Save1(d70, dst + 0x7 * dD, tails[0]);
                if (M > 0x8) Save1(d80, dst + 0x8 * dD, tails[0]);
                if (M > 0x9) Save1(d90, dst + 0x9 * dD, tails[0]);
                if (M > 0xa) Save1(da0, dst + 0xa * dD, tails[0]);
                if (M > 0xb) Save1(db0, dst + 0xb * dD, tails[0]);
            }
        }

        typedef void(*Deconvolution16bNhwcGemm_2xM_Ptr)(const uint16_t* src0, const DeconvParam& p, const AlgParam& a, 
            size_t srcC, size_t dstC, int zero, const uint16_t* weight, float* dst, const __mmask16 tails[2]);

        Deconvolution16bNhwcGemm_2xM_Ptr GetDeconvolution16bNhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return Deconvolution16bNhwcGemm_2xM<0x1>;
            case 0x2: return Deconvolution16bNhwcGemm_2xM<0x2>;
            case 0x3: return Deconvolution16bNhwcGemm_2xM<0x3>;
            case 0x4: return Deconvolution16bNhwcGemm_2xM<0x4>;
            case 0x5: return Deconvolution16bNhwcGemm_2xM<0x5>;
            case 0x6: return Deconvolution16bNhwcGemm_2xM<0x6>;
            case 0x7: return Deconvolution16bNhwcGemm_2xM<0x7>;
            case 0x8: return Deconvolution16bNhwcGemm_2xM<0x8>;
            case 0x9: return Deconvolution16bNhwcGemm_2xM<0x9>;
            case 0xa: return Deconvolution16bNhwcGemm_2xM<0xa>;
            case 0xb: return Deconvolution16bNhwcGemm_2xM<0xb>;
            case 0xc: return Deconvolution16bNhwcGemm_2xM<0xc>;
            }
            assert(0);
            return NULL;
        }

        void Deconvolution16bNhwcGemm_2(const uint16_t* src, const DeconvParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int zero, const uint16_t* wgt, float* dst)
        {
            size_t m1 = M, m = 12, mm = AlignLoAny(m1, m), t = m1 - mm;
            size_t dS = a.bufK, dW = a.bufK * DF, dD = a.bufN;
            Deconvolution16bNhwcGemm_2xM_Ptr deconvolution_2xM = GetDeconvolution16bNhwcGemm_2xM(m);
            Deconvolution16bNhwcGemm_2xM_Ptr deconvolution_2xT = GetDeconvolution16bNhwcGemm_2xM(t);

            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                __mmask16 tails[2] = { TailMask16(dN), TailMask16(dN - F) };
                size_t i = 0;
                for (; i < mm; i += m)
                    deconvolution_2xM(src + i * dS, p, a, K, dN, zero, wgt, dst + i * dD, tails);
                for (; i < m1; i += t)
                    deconvolution_2xT(src + i * dS, p, a, K, dN, zero, wgt, dst + i * dD, tails);
                wgt += dW;
                dst += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void RowToImgCommon(const float* src, const DeconvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, float* dst)
        {
            size_t dstCF = AlignLo(p.dstC, F);
            __mmask16 tail = TailMask16(p.dstC - dstCF);
            for (size_t dy = 0; dy < p.dstH; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    memset(dst + (dy * p.dstW + dx) * p.dstC, 0, p.dstC * sizeof(float));
            size_t gap = a.bufN - a.N;
            for (size_t sy = 0; sy < p.srcH; ++sy)
            {
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t dy = sy * p.strideY - p.padY;
                    for (size_t ky = 0; ky < p.kernelY; ky++, dy += p.dilationY)
                    {
                        if (dy < p.dstH)
                        {
                            size_t dx = sx * p.strideX - p.padX;
                            for (size_t kx = 0; kx < p.kernelX; kx++, dx += p.dilationX)
                            {
                                if (dx < p.dstW)
                                {
                                    float* d = dst + (dy * p.dstW + dx) * p.dstC;
                                    size_t dc = 0;
                                    for (; dc < dstCF; dc += F)
                                        _mm512_storeu_ps(d + dc, _mm512_add_ps(_mm512_loadu_ps(d + dc), _mm512_loadu_ps(src + dc)));
                                    if(tail)
                                        _mm512_mask_storeu_ps(d + dc, tail, _mm512_add_ps(_mm512_maskz_loadu_ps(tail, d + dc), _mm512_maskz_loadu_ps(tail, src + dc)));
                                }
                                src += p.dstC;
                            }
                        }
                        else
                            src += p.kernelX * p.dstC;
                    }
                    src += gap;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term, SimdConvolutionActivationType type> void BiasActivationCommon(const float* src, const DeconvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, const float* bias, const float* params, uint8_t* dst)
        {
            size_t body = AlignLo(p.dstC, F);
            __mmask16 tail = TailMask16(p.dstC - body);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < body; dc += F)
                        Postprocess<term, type>(src, bias, params, dc, dst);
                    if(tail)
                        Postprocess<term, type>(src, bias, params, dc, dst, tail);
                    src += p.dstC;
                    dst += p.dstC * a.elem;
                }
            }
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetBiasAct(const DeconvParam& p, const AlgParam & a, Base::SynetDeconvolution16bNhwcGemm::BiasActPtr& biasAct)
        {
            if(p.dstT == SimdTensorData16b)
                biasAct = BiasActivationCommon<Term16bLast16b, type>;
            else
                biasAct = BiasActivationCommon<Term16bLast32f, type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetDeconvolution16bNhwcGemm::SynetDeconvolution16bNhwcGemm(const DeconvParam & p)
            : Avx2::SynetDeconvolution16bNhwcGemm(p)
        {
            SetAlgParam(F, F * 2, 12, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
            {
                AlgParam& a = _alg;
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                    _convert = Reorder16bNhwcGemm;
            }
            else
                _convert = Convert16bNhwcGemm;
            _gemm = Deconvolution16bNhwcGemm_2;
            _toImg = RowToImgCommon;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetBiasAct<SimdConvolutionActivationRestrictRange>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationRelu: SetBiasAct<SimdConvolutionActivationRestrictRange>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationLeakyRelu: SetBiasAct<SimdConvolutionActivationPrelu>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationRestrictRange: SetBiasAct<SimdConvolutionActivationRestrictRange>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationPrelu: SetBiasAct<SimdConvolutionActivationPrelu>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationElu: SetBiasAct<SimdConvolutionActivationElu>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationHswish: SetBiasAct<SimdConvolutionActivationHswish>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationMish: SetBiasAct<SimdConvolutionActivationMish>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationHardSigmoid: SetBiasAct<SimdConvolutionActivationHardSigmoid>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationSwish: SetBiasAct<SimdConvolutionActivationSwish>(p, _alg, _biasAct); break;
            case SimdConvolutionActivationGelu: SetBiasAct<SimdConvolutionActivationGelu>(p, _alg, _biasAct); break;
            default: assert(0);
            }
        }
    }
#endif
}
