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
#include "Simd/SimdSet.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
    {
        typedef Base::SynetConvolution16bNhwcSpecV4::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcSpecV4::ConvolutionPtr ConvolutionPtr;

        //-------------------------------------------------------------------------------------------------

        static void SetTileConfSpecV4(int rows, bool twoM)
        {
            if (rows == 16)
            {
                SetTileConfFull();
                return;
            }
            TileConf conf(false);
            uint8_t r = (uint8_t)rows;
            if (twoM)
            {
                conf.rows[0] = conf.rows[1] = conf.rows[2] = conf.rows[3] = r;
                conf.rows[4] = conf.rows[5] = r;
            }
            else
            {
                conf.rows[0] = r;
                conf.rows[1] = r;
                conf.rows[4] = r;
            }
            _tile_loadconfig(&conf);
            TileConfChanged(0, 1);
        }

        //-------------------------------------------------------------------------------------------------

        static void Convert16bNhwcSpecV4(const uint8_t* src8, const ConvParam& p, const AlgParam& a, uint16_t* dst)
        {
            const float* src = (float*)src8;
            if (p.padY || p.padX || p.padH || p.padW)
                memset(dst, 0, a.bufS * sizeof(uint16_t));
            size_t row = p.srcW * p.srcC;
            for (size_t y = 0; y < p.srcH; ++y)
            {
                const float* s = src + y * row;
                uint16_t* d = dst + ((y + p.padY) * a.srcW + p.padX) * p.srcC;
                for (size_t c = 0; c < row; c += DF)
                    Float32ToBFloat16(s + c, d + c);
            }
        }

        static void Reorder16bNhwcSpecV4(const uint8_t* src8, const ConvParam& p, const AlgParam& a, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            if (p.padY || p.padX || p.padH || p.padW)
                memset(dst, 0, a.bufS * sizeof(uint16_t));
            size_t row = p.srcW * p.srcC;
            for (size_t y = 0; y < p.srcH; ++y)
                memcpy(dst + ((y + p.padY) * a.srcW + p.padX) * p.srcC, src + y * row, row * sizeof(uint16_t));
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> static SIMD_INLINE void StoreActivation(
            const float* acc, int rows, int oc0, const float* bias, const float* params, uint8_t* dst, int dstC)
        {
            const int elem = term == Term16bLast16b ? 2 : 4;
            __m512 _bias[2], _params[2];
            _bias[0] = _mm512_loadu_ps(bias + oc0);
            _bias[1] = _mm512_loadu_ps(bias + oc0 + F);
            if (type == SimdConvolutionActivationPrelu)
            {
                _params[0] = _mm512_loadu_ps(params + oc0);
                _params[1] = _mm512_loadu_ps(params + oc0 + F);
            }
            else
            {
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }
            for (int r = 0; r < rows; ++r)
            {
                const float* row = acc + r * 32;
                uint8_t* out = dst + (r * dstC + oc0) * elem;
                __m512 v0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(row + 0), _bias[0]), _params, 0);
                __m512 v1 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(row + F), _bias[1]), _params, 1);
                if (term == Term16bLast16b)
                    _mm512_storeu_si512(out, (__m512i)_mm512_cvtne2ps_pbh(v1, v0));
                else
                {
                    _mm512_storeu_ps((float*)out + 0, v0);
                    _mm512_storeu_ps((float*)out + F, v1);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void GemmRows(const uint16_t* src, int oh0, int oh1, int ow, int wp, int ic, int kh, int kw, const uint16_t* weight, float* acc)
        {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            const int strideS = ic * (int)sizeof(uint16_t);
            const int nK = kh * kw * (ic / 32);
            const uint16_t* b = weight;
            const uint16_t* a1 = src + ((size_t)oh1 * wp + ow) * ic;
            _tile_loadd(4, src + ((size_t)oh0 * wp + ow) * ic, strideS);
            _tile_stream_loadd(6, b, 64);
            int step = 0;
            for (int y = 0; y < kh; ++y)
            {
                for (int x = 0; x < kw; ++x)
                {
                    const uint16_t* row0 = src + ((size_t)(oh0 + y) * wp + (ow + x)) * ic;
                    const uint16_t* row1 = src + ((size_t)(oh1 + y) * wp + (ow + x)) * ic;
                    for (int c = 0; c < ic; c += 32, ++step)
                    {
                        if (step == 0)
                            continue;
                        const uint16_t* next1 = row1 + c;
                        const uint16_t* nextB = weight + (size_t)step * 1024;
                        _tile_loadd(5, a1, strideS);
                        _tile_stream_loadd(7, b + 512, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(4, row0 + c, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_stream_loadd(6, nextB, 64);
                        _tile_dpbf16ps(3, 5, 7);
                        a1 = next1;
                        b = nextB;
                    }
                }
            }
            if (nK > 0)
            {
                _tile_loadd(5, a1, strideS);
                _tile_stream_loadd(7, b + 512, 64);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stored(0, acc, 128);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stored(1, acc + 16, 128);
                _tile_dpbf16ps(2, 5, 6);
                _tile_stored(2, acc + 16 * 32, 128);
                _tile_dpbf16ps(3, 5, 7);
                _tile_stored(3, acc + 16 * 32 + 16, 128);
            }
        }

        static void Gemm16x32(const uint16_t* src, int oh, int ow, int wp, int ic, int kh, int kw, const uint16_t* weight, float* acc)
        {
            _tile_zero(0);
            _tile_zero(1);
            const int strideS = ic * (int)sizeof(uint16_t);
            const int nK = kh * kw * (ic / 32);
            const uint16_t* b = weight;
            _tile_loadd(4, src + ((size_t)oh * wp + ow) * ic, strideS);
            _tile_stream_loadd(6, b, 64);
            int step = 0;
            for (int y = 0; y < kh; ++y)
            {
                for (int x = 0; x < kw; ++x)
                {
                    const uint16_t* row = src + ((size_t)(oh + y) * wp + (ow + x)) * ic;
                    for (int c = 0; c < ic; c += 32, ++step)
                    {
                        const uint16_t* nextA = row + c;
                        const uint16_t* nextB = weight + (size_t)step * 1024;
                        if (step == 0)
                            continue;
                        _tile_stream_loadd(7, b + 512, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(4, nextA, strideS);
                        _tile_stream_loadd(6, nextB, 64);
                        b = nextB;
                    }
                }
            }
            if (nK > 0)
            {
                _tile_stream_loadd(7, b + 512, 64);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stored(0, acc, 128);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stored(1, acc + 16, 128);
            }
        }

        static void Gemm32x32(const uint16_t* src, int oh, int ow, int wp, int ic, int kh, int kw, const uint16_t* weight, float* acc)
        {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            const int strideS = ic * (int)sizeof(uint16_t);
            const int nK = kh * kw * (ic / 32);
            const uint16_t* b = weight;
            const uint16_t* a0 = src + ((size_t)oh * wp + ow) * ic;
            _tile_loadd(4, a0, strideS);
            _tile_stream_loadd(6, b, 64);
            int step = 0;
            for (int y = 0; y < kh; ++y)
            {
                for (int x = 0; x < kw; ++x)
                {
                    const uint16_t* row = src + ((size_t)(oh + y) * wp + (ow + x)) * ic;
                    for (int c = 0; c < ic; c += 32, ++step)
                    {
                        if (step == 0)
                            continue;
                        const uint16_t* nextA = row + c;
                        const uint16_t* nextB = weight + (size_t)step * 1024;
                        _tile_loadd(5, a0 + 16 * ic, strideS);
                        _tile_stream_loadd(7, b + 512, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(4, nextA, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_stream_loadd(6, nextB, 64);
                        _tile_dpbf16ps(3, 5, 7);
                        a0 = nextA;
                        b = nextB;
                    }
                }
            }
            if (nK > 0)
            {
                _tile_loadd(5, a0 + 16 * ic, strideS);
                _tile_stream_loadd(7, b + 512, 64);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stored(0, acc, 128);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stored(1, acc + 16, 128);
                _tile_dpbf16ps(2, 5, 6);
                _tile_stored(2, acc + 16 * 32, 128);
                _tile_dpbf16ps(3, 5, 7);
                _tile_stored(3, acc + 16 * 32 + 16, 128);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcSpecV4(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            const uint16_t* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const int ic = (int)p.srcC, oc = (int)p.dstC, kh = (int)p.kernelY, kw = (int)p.kernelX;
            const int wp = (int)a.srcW, hp = (int)a.srcH, dstH = (int)p.dstH, dstW = (int)p.dstW;
            const int elem = term == Term16bLast16b ? 2 : 4;
            const size_t weightBlock = a.weightStep;
            SIMD_ALIGNED(64) float acc[32 * 32];

            if (dstW < 32)
            {
                const size_t panelBytes = weightBlock * sizeof(uint16_t);
                const size_t budget = 1536u * 1024u;
                const int pairs = dstH / 2;
                int chunkPairs = pairs > 0 ? pairs : 1;
                const size_t imageBytes = (size_t)hp * wp * ic * sizeof(uint16_t);
                if (pairs > 1 && imageBytes + panelBytes > budget)
                {
                    chunkPairs = 1;
                    for (int np = 1; np <= pairs; ++np)
                    {
                        int h = Simd::Min(hp, 2 * np + kh - 1);
                        size_t bytes = (size_t)h * wp * ic * sizeof(uint16_t) + panelBytes;
                        if (bytes <= budget)
                            chunkPairs = np;
                    }
                }
                const int full = dstW / 16;
                const int tail = dstW % 16;
                for (int pc = 0; pc < pairs; pc += chunkPairs)
                {
                    int np = Simd::Min(chunkPairs, pairs - pc);
                    if (full > 0)
                    {
                        SetTileConfFull();
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* w = weight + (size_t)(oc0 / 32) * weightBlock;
                            for (int pi = 0; pi < np; ++pi)
                            {
                                int oh0 = (pc + pi) * 2;
                                int oh1 = oh0 + 1;
                                for (int b = 0; b < full; ++b)
                                {
                                    int ow = b * 16;
                                    GemmRows(src, oh0, oh1, ow, wp, ic, kh, kw, w, acc);
                                    StoreActivation<term, type>(acc, 16, oc0, bias, params, dst + ((size_t)oh0 * dstW + ow) * oc * elem, oc);
                                    StoreActivation<term, type>(acc + 16 * 32, 16, oc0, bias, params, dst + ((size_t)oh1 * dstW + ow) * oc * elem, oc);
                                }
                            }
                        }
                    }
                    if (tail > 0)
                    {
                        SetTileConfSpecV4(tail, true);
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* w = weight + (size_t)(oc0 / 32) * weightBlock;
                            for (int pi = 0; pi < np; ++pi)
                            {
                                int oh0 = (pc + pi) * 2;
                                int oh1 = oh0 + 1;
                                int ow = full * 16;
                                GemmRows(src, oh0, oh1, ow, wp, ic, kh, kw, w, acc);
                                StoreActivation<term, type>(acc, tail, oc0, bias, params, dst + ((size_t)oh0 * dstW + ow) * oc * elem, oc);
                                StoreActivation<term, type>(acc + 16 * 32, tail, oc0, bias, params, dst + ((size_t)oh1 * dstW + ow) * oc * elem, oc);
                            }
                        }
                    }
                }
                if ((dstH % 2) != 0)
                {
                    int oh = dstH - 1;
                    int ow = 0;
                    for (; ow + 16 <= dstW; ow += 16)
                    {
                        SetTileConfFull();
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* w = weight + (size_t)(oc0 / 32) * weightBlock;
                            Gemm16x32(src, oh, ow, wp, ic, kh, kw, w, acc);
                            StoreActivation<term, type>(acc, 16, oc0, bias, params, dst + ((size_t)oh * dstW + ow) * oc * elem, oc);
                        }
                    }
                    if (ow < dstW)
                    {
                        int rows = dstW - ow;
                        SetTileConfSpecV4(rows, false);
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* w = weight + (size_t)(oc0 / 32) * weightBlock;
                            Gemm16x32(src, oh, ow, wp, ic, kh, kw, w, acc);
                            StoreActivation<term, type>(acc, rows, oc0, bias, params, dst + ((size_t)oh * dstW + ow) * oc * elem, oc);
                        }
                    }
                }
            }
            else
            {
                SetTileConfFull();
                for (int oh = 0; oh < dstH; ++oh)
                {
                    int ow = 0;
                    for (; ow + 32 <= dstW; ow += 32)
                    {
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* w = weight + (size_t)(oc0 / 32) * weightBlock;
                            Gemm32x32(src, oh, ow, wp, ic, kh, kw, w, acc);
                            StoreActivation<term, type>(acc, 32, oc0, bias, params, dst + ((size_t)oh * dstW + ow) * oc * elem, oc);
                        }
                    }
                    while (ow < dstW)
                    {
                        int rows = Simd::Min(16, dstW - ow);
                        SetTileConfSpecV4(rows, false);
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* w = weight + (size_t)(oc0 / 32) * weightBlock;
                            Gemm16x32(src, oh, ow, wp, ic, kh, kw, w, acc);
                            StoreActivation<term, type>(acc, rows, oc0, bias, params, dst + ((size_t)oh * dstW + ow) * oc * elem, oc);
                        }
                        ow += rows;
                        SetTileConfFull();
                    }
                }
            }
            SetTileConfFull(true);
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void SetConvolution(ConvolutionPtr& convolution)
        {
            convolution = Convolution16bNhwcSpecV4<term, type>;
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void SetConvolution(const ConvParam& p, ConvolutionPtr& convolution)
        {
            if (p.dstT == SimdTensorData16b)
                SetConvolution<Term16bLast16b, type>(convolution);
            else
                SetConvolution<Term16bLast32f, type>(convolution);
        }

        SynetConvolution16bNhwcSpecV4::SynetConvolution16bNhwcSpecV4(const ConvParam& p)
            : Base::SynetConvolution16bNhwcSpecV4(p)
        {
            SetAlgParam();
            if (_src16b)
                _preprocess = Reorder16bNhwcSpecV4;
            else
                _preprocess = Convert16bNhwcSpecV4;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationRelu: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationLeakyRelu: SetConvolution<SimdConvolutionActivationPrelu>(p, _convolution); break;
            case SimdConvolutionActivationRestrictRange: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationPrelu: SetConvolution<SimdConvolutionActivationPrelu>(p, _convolution); break;
            case SimdConvolutionActivationElu: SetConvolution<SimdConvolutionActivationElu>(p, _convolution); break;
            case SimdConvolutionActivationHswish: SetConvolution<SimdConvolutionActivationHswish>(p, _convolution); break;
            case SimdConvolutionActivationMish: SetConvolution<SimdConvolutionActivationMish>(p, _convolution); break;
            case SimdConvolutionActivationHardSigmoid: SetConvolution<SimdConvolutionActivationHardSigmoid>(p, _convolution); break;
            case SimdConvolutionActivationSwish: SetConvolution<SimdConvolutionActivationSwish>(p, _convolution); break;
            case SimdConvolutionActivationGelu: SetConvolution<SimdConvolutionActivationGelu>(p, _convolution); break;
            default: assert(0);
            }
        }
    }
#endif
}
