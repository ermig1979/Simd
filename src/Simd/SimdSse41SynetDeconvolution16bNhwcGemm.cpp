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
#include "Simd/SimdSse41.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse41
    {
        typedef Base::SynetDeconvolution16bNhwcGemm::AlgParam AlgParam;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcGemm(const uint8_t* src8, const DeconvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8 + yBeg * p.srcW * p.srcC;
            size_t size = p.srcC, gap = a.bufK - size;
            size_t size8 = Simd::AlignLo(size, 8);
            size_t size4 = Simd::AlignLo(size, 4);
            for (size_t sy = yBeg; sy < yEnd; ++sy)
            {
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < size8; sc += 8)
                    {
                        __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + sc + 0));
                        __m128i d1 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + sc + 4));
                        _mm_storeu_si128((__m128i*)(dst + sc), _mm_packus_epi32(d0, d1));
                    }
                    for (; sc < size4; sc += 4)
                    {
                        __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + sc + 0));
                        _mm_storel_epi64((__m128i*)(dst + sc), _mm_packus_epi32(d0, Sse41::K_ZERO));
                    }
                    for (; sc < p.srcC; ++sc)
                        dst[sc] = Base::Float32ToBFloat16(src[sc]);
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

        SIMD_INLINE void Save1(float* dst, __m128 val0)
        {
            _mm_storeu_ps(dst, val0);
        }

        SIMD_INLINE void Save1(float* dst, __m128 val0, size_t tail)
        {
            float tmp[F];
            _mm_storeu_ps(tmp, val0);
            for (size_t i = 0; i < tail; ++i)
                dst[i] = tmp[i];
        }

        SIMD_INLINE void Save2(float* dst, __m128 val0, __m128 val1)
        {
            _mm_storeu_ps(dst + 0, val0);
            _mm_storeu_ps(dst + F, val1);
        }

        SIMD_INLINE void Save2(float* dst, __m128 val0, __m128 val1, size_t tail)
        {
            _mm_storeu_ps(dst + 0, val0);
            Save1(dst + F, val1, tail);
        }

        template<int M> void Deconvolution16bNhwcGemm_2xM(const uint16_t* src0, const DeconvParam& p, const AlgParam& a, 
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, float* dst)
        {
            __m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w00, w01, w10, w11, m = _mm_castsi128_ps(Bf16::MASK);
            size_t dD = a.bufN, dS = a.bufK;
            const uint16_t* weight1 = weight0 + a.bufK * F;
            const uint16_t* src1 = src0 + 1 * dS;
            const uint16_t* src2 = src0 + 2 * dS;
            const uint16_t* src3 = src0 + 3 * dS;
            const uint16_t* src4 = src0 + 4 * dS;
            if (dstC > F)
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
                    if (M > 1) d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
                    if (M > 2) d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
                    if (M > 3) d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
                    if (M > 4) d40 = _mm_setzero_ps(), d41 = _mm_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_ps(dst + 0 * dD + 0), d01 = _mm_loadu_ps(dst + 0 * dD + F);
                    if (M > 1) d10 = _mm_loadu_ps(dst + 1 * dD + 0), d11 = _mm_loadu_ps(dst + 1 * dD + F);
                    if (M > 2) d20 = _mm_loadu_ps(dst + 2 * dD + 0), d21 = _mm_loadu_ps(dst + 2 * dD + F);
                    if (M > 3) d30 = _mm_loadu_ps(dst + 3 * dD + 0), d31 = _mm_loadu_ps(dst + 3 * dD + F);
                    if (M > 4) d40 = _mm_loadu_ps(dst + 4 * dD + 0), d41 = _mm_loadu_ps(dst + 4 * dD + F);
                }
                for (size_t offs = 0; offs < srcC; offs += 2)
                {
                    w01 = _mm_loadu_ps((float*)weight0);
                    w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                    w01 = _mm_and_ps(w01, m);
                    w11 = _mm_loadu_ps((float*)weight1);
                    w10 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w11), Base::Bf16::SHIFT));
                    w11 = _mm_and_ps(w11, m);
                    if (M > 0)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(s0, w00), d00);
                        d01 = _mm_add_ps(_mm_mul_ps(s0, w10), d01);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(s0, w01), d00);
                        d01 = _mm_add_ps(_mm_mul_ps(s0, w11), d01);
                    }
                    if (M > 1)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 1)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(s0, w00), d10);
                        d11 = _mm_add_ps(_mm_mul_ps(s0, w10), d11);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 0)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(s0, w01), d10);
                        d11 = _mm_add_ps(_mm_mul_ps(s0, w11), d11);
                    }
                    if (M > 2)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 1)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(s0, w00), d20);
                        d21 = _mm_add_ps(_mm_mul_ps(s0, w10), d21);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 0)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(s0, w01), d20);
                        d21 = _mm_add_ps(_mm_mul_ps(s0, w11), d21);
                    }
                    if (M > 3)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 1)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(s0, w00), d30);
                        d31 = _mm_add_ps(_mm_mul_ps(s0, w10), d31);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 0)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(s0, w01), d30);
                        d31 = _mm_add_ps(_mm_mul_ps(s0, w11), d31);
                    }
                    if (M > 4)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 1)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(s0, w00), d40);
                        d41 = _mm_add_ps(_mm_mul_ps(s0, w10), d41);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 0)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(s0, w01), d40);
                        d41 = _mm_add_ps(_mm_mul_ps(s0, w11), d41);
                    }
                    weight0 += DF;
                    weight1 += DF;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2(dst, d00, d01), dst += dD;
                    if (M > 1) Save2(dst, d10, d11), dst += dD;
                    if (M > 2) Save2(dst, d20, d21), dst += dD;
                    if (M > 3) Save2(dst, d30, d31), dst += dD;
                    if (M > 4) Save2(dst, d40, d41), dst += dD;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2(dst, d00, d01, dstC), dst += dD;
                    if (M > 1) Save2(dst, d10, d11, dstC), dst += dD;
                    if (M > 2) Save2(dst, d20, d21, dstC), dst += dD;
                    if (M > 3) Save2(dst, d30, d31, dstC), dst += dD;
                    if (M > 4) Save2(dst, d40, d41, dstC), dst += dD;
                }
            }
            else
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm_setzero_ps();
                    if (M > 1) d10 = _mm_setzero_ps();
                    if (M > 2) d20 = _mm_setzero_ps();
                    if (M > 3) d30 = _mm_setzero_ps();
                    if (M > 4) d40 = _mm_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_ps(dst + 0 * dD + 0);
                    if (M > 1) d10 = _mm_loadu_ps(dst + 1 * dD + 0);
                    if (M > 2) d20 = _mm_loadu_ps(dst + 2 * dD + 0);
                    if (M > 3) d30 = _mm_loadu_ps(dst + 3 * dD + 0);
                    if (M > 4) d40 = _mm_loadu_ps(dst + 4 * dD + 0);
                }
                for (size_t offs = 0; offs < srcC; offs += 2)
                {
                    w01 = _mm_loadu_ps((float*)weight0);
                    w00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(w01), Base::Bf16::SHIFT));
                    w01 = _mm_and_ps(w01, m);
                    if (M > 0)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 1)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(s0, w00), d00);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src0 + offs - 0)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(s0, w01), d00);
                    }
                    if (M > 1)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 1)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(s0, w00), d10);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src1 + offs - 0)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(s0, w01), d10);
                    }
                    if (M > 2)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 1)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(s0, w00), d20);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src2 + offs - 0)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(s0, w01), d20);
                    }
                    if (M > 3)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 1)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(s0, w00), d30);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src3 + offs - 0)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(s0, w01), d30);
                    }
                    if (M > 4)
                    {
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 1)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(s0, w00), d40);
                        s0 = _mm_and_ps(_mm_set1_ps(*(float*)(src4 + offs - 0)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(s0, w01), d40);
                    }
                    weight0 += DF;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1(dst, d00), dst += dD;
                    if (M > 1) Save1(dst, d10), dst += dD;
                    if (M > 2) Save1(dst, d20), dst += dD;
                    if (M > 3) Save1(dst, d30), dst += dD;
                    if (M > 4) Save1(dst, d40), dst += dD;
                }
                else
                {
                    if (M > 0) Save1(dst, d00, dstC), dst += dD;
                    if (M > 1) Save1(dst, d10, dstC), dst += dD;
                    if (M > 2) Save1(dst, d20, dstC), dst += dD;
                    if (M > 3) Save1(dst, d30, dstC), dst += dD;
                    if (M > 4) Save1(dst, d40, dstC), dst += dD;
                }
            }
        }

        typedef void(*Deconvolution16bNhwcGemm_2xM_Ptr)(const uint16_t* src0, const DeconvParam& p, const AlgParam& a, 
            size_t srcC, size_t dstC, int zero, const uint16_t* weight, float* dst);

        Deconvolution16bNhwcGemm_2xM_Ptr GetDeconvolution16bNhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Deconvolution16bNhwcGemm_2xM<1>;
            case 2: return Deconvolution16bNhwcGemm_2xM<2>;
            case 3: return Deconvolution16bNhwcGemm_2xM<3>;
            case 4: return Deconvolution16bNhwcGemm_2xM<4>;
            case 5: return Deconvolution16bNhwcGemm_2xM<5>;
            }
            assert(0);
            return NULL;
        }

        void Deconvolution16bNhwcGemm_2(const uint16_t* src, const DeconvParam& p, const AlgParam& a, size_t M, size_t N, size_t K, int zero, const uint16_t* wgt, float* dst)
        {
            size_t m1 = M, m = 5, mm = AlignLoAny(m1, m), t = m1 - mm;
            size_t dS = a.bufK, dW = a.bufK * DF, dD = a.bufN;
            Deconvolution16bNhwcGemm_2xM_Ptr deconvolution_2xM = GetDeconvolution16bNhwcGemm_2xM(m);
            Deconvolution16bNhwcGemm_2xM_Ptr deconvolution_2xT = GetDeconvolution16bNhwcGemm_2xM(t);

            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                for (; i < mm; i += m)
                    deconvolution_2xM(src + i * dS, p, a, K, dN, zero, wgt, dst + i * dD);
                for (; i < m1; i += t)
                    deconvolution_2xT(src + i * dS, p, a, K, dN, zero, wgt, dst + i * dD);
                wgt += dW;
                dst += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void RowToImgCommon(const float* src, const DeconvParam& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, float* dst)
        {
            size_t dstCF = AlignLo(p.dstC, F);
            size_t rowSize = p.dstW * p.dstC, gap = a.bufN - a.N;
            size_t dyBeg = yBeg ? yBeg * p.strideY + a.preH : 0;
            size_t dyEnd = Simd::Min(yEnd * p.strideY + a.preH, p.dstH);
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
                memset(dst + dy * rowSize, 0, rowSize * sizeof(float));
            for (size_t sy = yBeg; sy < yEnd; ++sy)
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
                                        _mm_storeu_ps(d + dc, _mm_add_ps(_mm_loadu_ps(d + dc), _mm_loadu_ps(src + dc)));
                                    for (; dc < p.dstC; ++dc)
                                        d[dc] += src[dc];
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
            size_t body = AlignLo(p.dstC, F), tail = p.dstC - body;
            src += yBeg * p.dstW * p.dstC;
            dst += yBeg * p.dstW * p.dstC * a.elem;
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
            : Base::SynetDeconvolution16bNhwcGemm(p)
        {
            SetAlgParam(F, F * 2, 5, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
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

        bool SynetDeconvolution16bNhwcGemm::Preferable(const DeconvParam& p)
        {
            return p.trans && p.group == 1;
        }
    }
#endif
}
