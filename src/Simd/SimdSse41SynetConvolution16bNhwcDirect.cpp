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
#include "Simd/SimdSynetConvolution16b.h"
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
        typedef Base::SynetConvolution16bNhwcDirect::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcDirect::PostprocessPtr Postprocess;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcDirect(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t srcC8 = Simd::AlignLo(p.srcC, 8);
            size_t srcC4 = Simd::AlignLo(p.srcC, 4);
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? dyEnd : dyEnd + syPad);
            if (dyBeg == 0)
            {
                memset(dst, 0, p.padY * a.srcW * a.srcC * 2);
                dst += p.padY * a.srcW * a.srcC;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * a.srcC;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (p.padX)
                {
                    memset(dst, 0, p.padX * a.srcC * 2);
                    dst += p.padX * a.srcC;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcC8; sc += 8)
                    {
                        __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + sc + 0));
                        __m128i d1 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + sc + 4));
                        _mm_storeu_si128((__m128i*)(dst + sc), _mm_packus_epi32(d0, d1));
                    }
                    for (; sc < srcC4; sc += 4)
                    {
                        __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(src + sc + 0));
                        _mm_storel_epi64((__m128i*)(dst + sc), _mm_packus_epi32(d0, Sse41::K_ZERO));
                    }
                    for (; sc < p.srcC; ++sc)
                        dst[sc] = Base::Float32ToBFloat16(src[sc]);
                    for (; sc < a.srcC; ++sc)
                        dst[sc] = 0;
                    src += p.srcC;
                    dst += a.srcC;
                }
                if (p.padX)
                {
                    memset(dst, 0, p.padW * a.srcC * 2);
                    dst += p.padW * a.srcC;
                }
            }
            if (dyEnd == p.dstH)
            {
                memset(dst, 0, p.padH * a.srcW * a.srcC * 2);
                dst += p.padH * a.srcW * a.srcC * 2;
            }
        }

        static void Reorder16bNhwcDirect(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? dyEnd : dyEnd + syPad);
            if (dyBeg == 0)
            {
                memset(dst, 0, p.padY * a.srcW * a.srcC * 2);
                dst += p.padY * a.srcW * a.srcC;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * a.srcC;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (p.padX)
                {
                    memset(dst, 0, p.padX * a.srcC * 2);
                    dst += p.padX * a.srcC;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    memcpy(dst, src, p.srcC * 2);
                    for (size_t sc = p.srcC; sc < a.srcC; ++sc)
                        dst[sc] = 0;
                    src += p.srcC;
                    dst += a.srcC;
                }
                if (p.padX)
                {
                    memset(dst, 0, p.padW * a.srcC * 2);
                    dst += p.padW * a.srcC;
                }
            }
            if (dyEnd == p.dstH)
            {
                memset(dst, 0, p.padH * a.srcW * a.srcC * 2);
                dst += p.padH * a.srcW * a.srcC * 2;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetConvolution16bNhwcDirect::SynetConvolution16bNhwcDirect(const ConvParam & p)
            : Base::SynetConvolution16bNhwcDirect(p)
        {
            SetAlgParam(F, F * 2, 4, F * 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
                _preprocess = Reorder16bNhwcDirect;
            else
                _preprocess = Convert16bNhwcDirect;
        }
    }
#endif
}
