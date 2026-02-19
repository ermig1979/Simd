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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
    {
        typedef Base::SynetConvolution16bNhwcSpecV0::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcSpecV0::PostprocessPtr PostprocessPtr;

        //-------------------------------------------------------------------------------------------------

        static void Convert16bNhwcSpecV0(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            assert(a.microC == DF);
            const float* src = (float*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            __mmask32 tailC = TailMask32(p.srcC - srcCDF);
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += a.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        AmxBf16::Float32ToBFloat16(src + sc, dst + sc * cD);
                    if (tailC)
                        AmxBf16::Float32ToBFloat16(src + sc, dst + sc * cD, tailC);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
        }

        static void Reorder16bNhwcSpecV0(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            assert(a.microC == DF);
            const uint16_t* src = (uint16_t*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            __mmask32 tailC = TailMask32(p.srcC - srcCDF);
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += a.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        Avx512bw::Copy(src + sc, dst + sc * cD);
                    if (tailC)
                        Avx512bw::Copy(src + sc, dst + sc * cD, tailC);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int stub> void Convolution16bNhwcSpecV0_Direct32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* weight1 = weight0 + a.K * F;
            const uint16_t* src1 = src0 + 16 * dX;
            float* dst1 = dst0 + 16 * dD;

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(1, dst0 + F, strideD);
                _tile_stream_loadd(2, dst1 + 0, strideD);
                _tile_stream_loadd(3, dst1 + F, strideD);
            }

            int n1 = (int)nK - 1, o = offs[0];
            _tile_stream_loadd(4, src0 + o, strideS);
            _tile_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(5, src1 + o, strideS);
                _tile_loadd(7, weight1, strideW);                        
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                _tile_dpbf16ps(2, 5, 6);
                weight0 += dW;
                _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(3, 5, 7);
                weight1 += dW;
            }
            _tile_loadd(7, weight1, strideW);
            _tile_stream_loadd(5, src1 + offs[n1], strideS);

            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);

            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, dst0 + F, strideD);
            TileMoveToMemory(dst0 + F, dD);

            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, dst1 + 0, strideD);
            TileMoveToMemory(dst1 + 0, dD);

            _tile_dpbf16ps(3, 5, 7);
            _tile_stored(3, dst1 + F, strideD);
            TileMoveToMemory(dst1 + F, dD);
        }

        template<int stub> void Convolution16bNhwcSpecV0_Direct32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* src1 = src0 + 16 * dX;
            float* dst1 = dst0 + 16 * dD;

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(2, dst1 + 0, strideD);
            }

            int n1 = (int)nK - 1, o = offs[0];
            _tile_loadd(4, src0 + o, strideS);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(6, weight0, strideW);
                _tile_loadd(5, src1 + o, strideS);
                _tile_dpbf16ps(0, 4, 6);
                o = offs[i + 1];
                _tile_loadd(4, src0 + o, strideS);
                _tile_dpbf16ps(2, 5, 6);
                weight0 += dW;
            }
            _tile_stream_loadd(6, weight0, strideW);
            _tile_loadd(5, src1 + offs[n1], strideS);

            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);

            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, dst1 + 0, strideD);
            TileMoveToMemory(dst1 + 0, dD);
        }

        template<int stub> void Convolution16bNhwcSpecV0_Direct16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* weight1 = weight0 + a.K * F;

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(1, dst0 + F, strideD);
            }

            int n1 = (int)nK - 1;
            _tile_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(4, src0 + offs[i], strideS);
                _tile_loadd(7, weight1, strideW);
                _tile_dpbf16ps(0, 4, 6);
                weight0 += dW;
                _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(1, 4, 7);
                weight1 += dW;
            }
            _tile_stream_loadd(4, src0 + offs[n1], strideS);
            _tile_loadd(7, weight1, strideW);

            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);

            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, dst0 + F, strideD);
            TileMoveToMemory(dst0 + F, dD);
        }

        template<int stub> void Convolution16bNhwcSpecV0_Direct16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;

            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
            }

            int n = (int)nK;
            for (int i = 0; i < n; ++i)
            {
                _tile_stream_loadd(4, src0 + offs[i], strideS);
                _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(0, 4, 6);
                weight0 += dW;
            }

            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);
        }

        typedef void (*Convolution16bNhwcSpecV0DirectPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, int zero, const uint16_t* weight0, float* dst0);

        static void Convolution16bNhwcSpecV0_Direct(const uint16_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t nK, int zero, const uint16_t* weight, float* dst)
        {
            size_t n1 = dstH * a.srcW - a.gapH, n = 32;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.K * DF;
            size_t dD = a.macroD, dS = a.microC;
            Convolution16bNhwcSpecV0DirectPtr body_2 = Convolution16bNhwcSpecV0_Direct32x32<0>;
            Convolution16bNhwcSpecV0DirectPtr tail_2 = m > 16 ? Convolution16bNhwcSpecV0_Direct32x32<0> : Convolution16bNhwcSpecV0_Direct16x32<0>;
            Convolution16bNhwcSpecV0DirectPtr body_1 = Convolution16bNhwcSpecV0_Direct32x16<0>;
            Convolution16bNhwcSpecV0DirectPtr tail_1 = m > 16 ? Convolution16bNhwcSpecV0_Direct32x16<0> : Convolution16bNhwcSpecV0_Direct16x16<0>;

            SetTileConfFull();
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n)
                        body_2(src + i * dS, p, a, offs, nK, zero, weight, dst + i * dD);
                    if (m)
                        tail_2(src + i * dS, p, a, offs, nK, zero, weight, dst + i * dD);
                }
                else
                {
                    for (; i < nn; i += n)
                        body_1(src + i * dS, p, a, offs, nK, zero, weight, dst + i * dD);
                    if (m)
                        tail_1(src + i * dS, p, a, offs, nK, zero, weight, dst + i * dD);
                }
                weight += dW;
                dst += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

#ifdef _MSC_VER
        template<int hint> SIMD_INLINE void PrefetchTile(const uint16_t* ptr)
#else
        template<_mm_hint hint> SIMD_INLINE void PrefetchTile(const uint16_t* ptr)
#endif
        {
            _mm_prefetch((const char*)(ptr + 0x0 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0x1 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0x2 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0x3 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0x4 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0x5 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0x6 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0x7 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0x8 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0x9 * 64), hint);
            _mm_prefetch((const char*)(ptr + 0xA * 64), hint);
            _mm_prefetch((const char*)(ptr + 0xB * 64), hint);
            _mm_prefetch((const char*)(ptr + 0xC * 64), hint);
            _mm_prefetch((const char*)(ptr + 0xD * 64), hint);
            _mm_prefetch((const char*)(ptr + 0xE * 64), hint);
            _mm_prefetch((const char*)(ptr + 0xF * 64), hint);
        }

        template<int first> void Convolution16bNhwcSpecV0_Inverse32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* weight1 = weight0 + a.K * F;
            const uint16_t* src1 = src0 + 16 * dX;
            float* dst1 = dst0 + 16 * dD;

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(1, dst0 + F, strideD);
                _tile_stream_loadd(2, dst1 + 0, strideD);
                _tile_stream_loadd(3, dst1 + F, strideD);
            }

            int n1 = (int)nK - 1, o = offs[0];
            _tile_loadd(4, src0 + o, strideS);
            if(first)
                _tile_loadd(6, weight0, strideW);
            else
                _tile_stream_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i)
            {
                _tile_loadd(5, src1 + o, strideS);
                if (first)
                    _tile_loadd(7, weight1, strideW);
                else
                    _tile_stream_loadd(7, weight1, strideW);
                _tile_dpbf16ps(0, 4, 6);
                //PrefetchTile<_MM_HINT_T1>(weight0 + 512);
                _tile_dpbf16ps(1, 4, 7);
                //PrefetchTile<_MM_HINT_T1>(weight1 + 512);
                o = offs[i + 1];
                _tile_loadd(4, src0 + o, strideS);
                _tile_dpbf16ps(2, 5, 6);
                weight0 += dW;
                if (first)
                    _tile_loadd(6, weight0, strideW);
                else
                    _tile_stream_loadd(6, weight0, strideW);
                _tile_dpbf16ps(3, 5, 7);
                weight1 += dW;
            }
            if (first)
                _tile_loadd(7, weight1, strideW);
            else
                _tile_stream_loadd(7, weight1, strideW);
            _tile_loadd(5, src1 + offs[n1], strideS);

            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);

            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, dst0 + F, strideD);
            TileMoveToMemory(dst0 + F, dD);

            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, dst1 + 0, strideD);
            TileMoveToMemory(dst1 + 0, dD);

            _tile_dpbf16ps(3, 5, 7);
            _tile_stored(3, dst1 + F, strideD);
            TileMoveToMemory(dst1 + F, dD);
        }

        template<int inv> void Convolution16bNhwcSpecV0_Inverse32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* src1 = src0 + 16 * dX;
            float* dst1 = dst0 + 16 * dD;

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(2, dst1 + 0, strideD);
            }

            int n1 = (int)nK - 1, o = offs[0];
            if (inv)
            {
                _tile_loadd(4, src0 + o, strideS);
                for (int i = 0; i < n1; ++i)
                {
                    _tile_stream_loadd(6, weight0, strideW);
                    _tile_loadd(5, src1 + o, strideS);
                    _tile_dpbf16ps(0, 4, 6);
                    o = offs[i + 1];
                    _tile_loadd(4, src0 + o, strideS);
                    _tile_dpbf16ps(2, 5, 6);
                    weight0 += dW;
                }
                _tile_stream_loadd(6, weight0, strideW);
                _tile_loadd(5, src1 + offs[n1], strideS);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(2, 5, 6);
            }
            else
            {
                _tile_stream_loadd(4, src0 + o, strideS);
                for (int i = 0; i < n1; ++i)
                {
                    _tile_loadd(6, weight0, strideW);
                    _tile_stream_loadd(5, src1 + o, strideS);
                    _tile_dpbf16ps(0, 4, 6);
                    o = offs[i + 1];
                    _tile_stream_loadd(4, src0 + o, strideS);
                    _tile_dpbf16ps(2, 5, 6);
                    weight0 += dW;
                }
                _tile_loadd(6, weight0, strideW);
                _tile_stream_loadd(5, src1 + offs[n1], strideS);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(2, 5, 6);
            }

            _tile_stored(0, dst0 + 0, strideD);
            _tile_stored(2, dst1 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);
            TileMoveToMemory(dst1 + 0, dD);
        }

        template<int inv> void Convolution16bNhwcSpecV0_Inverse16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* weight1 = weight0 + a.K * F;

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(1, dst0 + F, strideD);
            }

            int n1 = (int)nK - 1;
            if (inv)
            {
                _tile_stream_loadd(6, weight0, strideW);
                for (int i = 0; i < n1; ++i)
                {
                    _tile_loadd(4, src0 + offs[i], strideS);
                    _tile_stream_loadd(7, weight1, strideW);
                    _tile_dpbf16ps(0, 4, 6);
                    weight0 += dW;
                    _tile_stream_loadd(6, weight0, strideW);
                    _tile_dpbf16ps(1, 4, 7);
                    weight1 += dW;
                }
                _tile_loadd(4, src0 + offs[n1], strideS);
                _tile_stream_loadd(7, weight1, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
            }
            else
            {
                _tile_loadd(6, weight0, strideW);
                for (int i = 0; i < n1; ++i)
                {
                    _tile_stream_loadd(4, src0 + offs[i], strideS);
                    _tile_loadd(7, weight1, strideW);
                    _tile_dpbf16ps(0, 4, 6);
                    weight0 += dW;
                    _tile_loadd(6, weight0, strideW);
                    _tile_dpbf16ps(1, 4, 7);
                    weight1 += dW;
                }
                _tile_stream_loadd(4, src0 + offs[n1], strideS);
                _tile_loadd(7, weight1, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
            }

            _tile_stored(0, dst0 + 0, strideD);
            _tile_stored(1, dst0 + F, strideD);
            TileMoveToMemory(dst0 + 0, dD);
            TileMoveToMemory(dst0 + F, dD);
        }

        template<int inv> void Convolution16bNhwcSpecV0_Inverse16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;

            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
            }

            int n = (int)nK;
            if (inv)
            {
                for (int i = 0; i < n; ++i)
                {
                    _tile_loadd(4, src0 + offs[i], strideS);
                    _tile_stream_loadd(6, weight0, strideW);
                    _tile_dpbf16ps(0, 4, 6);
                    weight0 += dW;
                }
            }
            else
            {
                for (int i = 0; i < n; ++i)
                {
                    _tile_stream_loadd(4, src0 + offs[i], strideS);
                    _tile_loadd(6, weight0, strideW);
                    _tile_dpbf16ps(0, 4, 6);
                    weight0 += dW;
                }
            }

            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);
        }

        typedef void (*Convolution16bNhwcSpecV0_InversePtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, int zero, const uint16_t* weight0, float* dst0);

        static void Convolution16bNhwcSpecV0_Inverse(const uint16_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t nK, int zero, const uint16_t* weight, float* dst)
        {
            size_t n1 = dstH * a.srcW - a.gapH, n = 32, n4 = n * 4;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, nn4 = AlignLoAny(nn, n4) * 0, dW = a.K * DF;
            size_t dstCDF = AlignLo(dstC, DF), dC = dstC - dstCDF;
            size_t dD = a.macroD, dS = a.microC;
            Convolution16bNhwcSpecV0_InversePtr body = Convolution16bNhwcSpecV0_Direct32x32<0>;
            Convolution16bNhwcSpecV0_InversePtr tail = dC > 16 ? Convolution16bNhwcSpecV0_Inverse32x32<1> : Convolution16bNhwcSpecV0_Inverse32x16<1>;

            SetTileConfFull();
            size_t i = 0;
            for (; i < nn4; i += n4)
            {
                size_t dc = 0;
                for (; dc < dstCDF; dc += DF)
                {
                    Convolution16bNhwcSpecV0_Direct32x32<1>(src + 0 * 32 * dS, p, a, offs, nK, zero, weight + a.K * dc, dst + dc + 0 * 32 * dD);
                    body(src + 1 * 32 * dS, p, a, offs, nK, zero, weight + a.K * dc, dst + dc + 1 * 32 * dD);
                    body(src + 2 * 32 * dS, p, a, offs, nK, zero, weight + a.K * dc, dst + dc + 2 * 32 * dD);
                    body(src + 3 * 32 * dS, p, a, offs, nK, zero, weight + a.K * dc, dst + dc + 3 * 32 * dD);
                }
                if (dc < dstC)
                {
                    tail(src + 0 * 32 * dS, p, a, offs, nK, zero, weight + a.K * dc, dst + dc + 0 * 32 * dD);
                    tail(src + 1 * 32 * dS, p, a, offs, nK, zero, weight + a.K * dc, dst + dc + 1 * 32 * dD);
                    tail(src + 2 * 32 * dS, p, a, offs, nK, zero, weight + a.K * dc, dst + dc + 2 * 32 * dD);
                    tail(src + 3 * 32 * dS, p, a, offs, nK, zero, weight + a.K * dc, dst + dc + 3 * 32 * dD);
                }
                src += n4 * dS;
                dst += n4 * dD;
            }
            for (; i < nn; i += n)
            {
                size_t dc = 0;
                if (i == 0)
                {
                    for (; dc < dstCDF; dc += DF)
                        Convolution16bNhwcSpecV0_Inverse32x32<1>(src, p, a, offs, nK, zero, weight + a.K * dc, dst + dc);
                }
                else
                {
                    for (; dc < dstCDF; dc += DF)
                        Convolution16bNhwcSpecV0_Inverse32x32<0>(src, p, a, offs, nK, zero, weight + a.K * dc, dst + dc);
                }
                if (dc < dstC)
                    tail(src, p, a, offs, nK, zero, weight + a.K * dc, dst + dc);
                src += n * dS;
                dst += n * dD;
            }
            if (m)
            {
                if (m <= F)
                {
                    body = Convolution16bNhwcSpecV0_Inverse16x32<1>;
                    tail = dC > 16 ? Convolution16bNhwcSpecV0_Inverse16x32<1> : Convolution16bNhwcSpecV0_Inverse16x16<1>;
                }
                size_t dc = 0;
                for (; dc < dstCDF; dc += DF)
                    body(src, p, a, offs, nK, zero, weight + a.K * dc, dst + dc);
                if (dc < dstC)
                    tail(src, p, a, offs, nK, zero, weight + a.K * dc, dst + dc);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type>  void Postprocess16bNhwcSpecV0(const float* src, const ConvParam& p,
            const AlgParam& a, size_t dstC, size_t dyBeg, size_t dyEnd, const float* bias, const float* params, uint8_t* dst)
        {
            size_t dstCF = AlignLo(dstC, F);
            __mmask16 tailD = TailMask16(dstC - dstCF);
            size_t rowGap = a.gapH * a.macroD;
            src += dyBeg * a.srcW * a.macroD;
            dst += dyBeg * p.dstW * p.dstC * a.elem;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < dstCF; dc += F)
                        AmxBf16::Postprocess<term, type>(src, bias, params, dc, dst);
                    if (tailD)
                        AmxBf16::Postprocess<term, type>(src, bias, params, dc, dst, tailD);
                    src += a.macroD;
                    dst += p.dstC * a.elem;
                }
                src += rowGap;
            }
        }

        template<SimdConvolutionActivationType type> void SetPostprocess(const ConvParam& p, const AlgParam& a, PostprocessPtr & postprocess)
        {
            if (p.dstT == SimdTensorData16b)
                postprocess = Postprocess16bNhwcSpecV0<Term16bLast16b, type>;
            else
                postprocess = Postprocess16bNhwcSpecV0<Term16bLast32f, type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution16bNhwcSpecV0::SynetConvolution16bNhwcSpecV0(const ConvParam & p)
            : Avx512bw::SynetConvolution16bNhwcSpecV0(p)
        {
            SetAlgParam(F, F * 2, 32, F * 2, int(Base::AlgCacheL1() * 1.05), int(Base::AlgCacheL2() * 0.5), Base::AlgCacheL3());
            if (_src16b)
                _preprocess = Reorder16bNhwcSpecV0;
            else
                _preprocess = Convert16bNhwcSpecV0;
            _convolution = _alg.inv ? Convolution16bNhwcSpecV0_Inverse : Convolution16bNhwcSpecV0_Direct;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationRelu: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationLeakyRelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationRestrictRange: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationPrelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationElu: SetPostprocess<SimdConvolutionActivationElu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationHswish: SetPostprocess<SimdConvolutionActivationHswish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationMish: SetPostprocess<SimdConvolutionActivationMish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationHardSigmoid: SetPostprocess<SimdConvolutionActivationHardSigmoid>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationSwish: SetPostprocess<SimdConvolutionActivationSwish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationGelu: SetPostprocess<SimdConvolutionActivationGelu>(p, _alg, _postprocess); break;
            default: assert(0);
            }
        }
    }
#endif
}
