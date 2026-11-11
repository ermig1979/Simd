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
#include "Simd/SimdSynetDeconvolution16b.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BF16_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Avx512bf16
    {
        typedef Base::SynetDeconvolution16bNhwcGemm::AlgParam AlgParam;

        SIMD_INLINE __m512bh LoadBf16(const uint16_t* p)
        {
            return (__m512bh)_mm512_loadu_si512((const void*)p);
        }

        SIMD_INLINE __m512bh PairBf16(const uint16_t* p)
        {
            return (__m512bh)_mm512_set1_epi32(*(const uint32_t*)p);
        }

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
                        Avx512bf16::Float32ToBFloat16<false, false>(src + sc, dst + sc, srcMask, dstMask);
                    if (size32 < size)
                        Avx512bf16::Float32ToBFloat16<false, true>(src + sc, dst + sc, srcMask, dstMask);
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

        SIMD_INLINE void Load1(__m512& val0, const float* src, const __mmask16 tail)
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
                d60, d61, d70, d71, d80, d81, d90, d91, da0, da1, db0, db1;
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
                    __m512bh w0 = LoadBf16(weight0);
                    __m512bh w1 = LoadBf16(weight1);
                    if (M > 0x0) { __m512bh s0 = PairBf16(src0 + offs0); d00 = _mm512_dpbf16_ps(d00, s0, w0); d01 = _mm512_dpbf16_ps(d01, s0, w1); }
                    if (M > 0x1) { __m512bh s0 = PairBf16(src1 + offs0); d10 = _mm512_dpbf16_ps(d10, s0, w0); d11 = _mm512_dpbf16_ps(d11, s0, w1); }
                    if (M > 0x2) { __m512bh s0 = PairBf16(src2 + offs0); d20 = _mm512_dpbf16_ps(d20, s0, w0); d21 = _mm512_dpbf16_ps(d21, s0, w1); }
                    if (M > 0x3) { __m512bh s0 = PairBf16(src3 + offs0); d30 = _mm512_dpbf16_ps(d30, s0, w0); d31 = _mm512_dpbf16_ps(d31, s0, w1); }
                    if (M > 0x4) { __m512bh s0 = PairBf16(src4 + offs0); d40 = _mm512_dpbf16_ps(d40, s0, w0); d41 = _mm512_dpbf16_ps(d41, s0, w1); }
                    if (M > 0x5) { __m512bh s0 = PairBf16(src5 + offs0); d50 = _mm512_dpbf16_ps(d50, s0, w0); d51 = _mm512_dpbf16_ps(d51, s0, w1); }
                    if (M > 0x6) { __m512bh s0 = PairBf16(src0 + offs6); d60 = _mm512_dpbf16_ps(d60, s0, w0); d61 = _mm512_dpbf16_ps(d61, s0, w1); }
                    if (M > 0x7) { __m512bh s0 = PairBf16(src1 + offs6); d70 = _mm512_dpbf16_ps(d70, s0, w0); d71 = _mm512_dpbf16_ps(d71, s0, w1); }
                    if (M > 0x8) { __m512bh s0 = PairBf16(src2 + offs6); d80 = _mm512_dpbf16_ps(d80, s0, w0); d81 = _mm512_dpbf16_ps(d81, s0, w1); }
                    if (M > 0x9) { __m512bh s0 = PairBf16(src3 + offs6); d90 = _mm512_dpbf16_ps(d90, s0, w0); d91 = _mm512_dpbf16_ps(d91, s0, w1); }
                    if (M > 0xa) { __m512bh s0 = PairBf16(src4 + offs6); da0 = _mm512_dpbf16_ps(da0, s0, w0); da1 = _mm512_dpbf16_ps(da1, s0, w1); }
                    if (M > 0xb) { __m512bh s0 = PairBf16(src5 + offs6); db0 = _mm512_dpbf16_ps(db0, s0, w0); db1 = _mm512_dpbf16_ps(db1, s0, w1); }
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
                    __m512bh w0 = LoadBf16(weight0);
                    if (M > 0x0) { __m512bh s0 = PairBf16(src0 + offs0); d00 = _mm512_dpbf16_ps(d00, s0, w0); }
                    if (M > 0x1) { __m512bh s0 = PairBf16(src1 + offs0); d10 = _mm512_dpbf16_ps(d10, s0, w0); }
                    if (M > 0x2) { __m512bh s0 = PairBf16(src2 + offs0); d20 = _mm512_dpbf16_ps(d20, s0, w0); }
                    if (M > 0x3) { __m512bh s0 = PairBf16(src3 + offs0); d30 = _mm512_dpbf16_ps(d30, s0, w0); }
                    if (M > 0x4) { __m512bh s0 = PairBf16(src4 + offs0); d40 = _mm512_dpbf16_ps(d40, s0, w0); }
                    if (M > 0x5) { __m512bh s0 = PairBf16(src5 + offs0); d50 = _mm512_dpbf16_ps(d50, s0, w0); }
                    if (M > 0x6) { __m512bh s0 = PairBf16(src0 + offs6); d60 = _mm512_dpbf16_ps(d60, s0, w0); }
                    if (M > 0x7) { __m512bh s0 = PairBf16(src1 + offs6); d70 = _mm512_dpbf16_ps(d70, s0, w0); }
                    if (M > 0x8) { __m512bh s0 = PairBf16(src2 + offs6); d80 = _mm512_dpbf16_ps(d80, s0, w0); }
                    if (M > 0x9) { __m512bh s0 = PairBf16(src3 + offs6); d90 = _mm512_dpbf16_ps(d90, s0, w0); }
                    if (M > 0xa) { __m512bh s0 = PairBf16(src4 + offs6); da0 = _mm512_dpbf16_ps(da0, s0, w0); }
                    if (M > 0xb) { __m512bh s0 = PairBf16(src5 + offs6); db0 = _mm512_dpbf16_ps(db0, s0, w0); }
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

        SynetDeconvolution16bNhwcGemm::SynetDeconvolution16bNhwcGemm(const DeconvParam& p)
            : Avx512bw::SynetDeconvolution16bNhwcGemm(p)
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
        }
    }
#endif
}
