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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE) 
	namespace Avx2
	{
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;

        //-----------------------------------------------------------------------------------------

        static void ConvertFp32ToBf16(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t srcC = AlignHi(p.srcC, a.miK);
            if (srcC == p.srcC)
            {
                size_t size = p.srcW * p.srcC;
                Float32ToBFloat16(src + yBeg * size, (yEnd - yBeg) * size, dst);
            }
            else
            {
                size_t srcC16 = Simd::AlignLo(p.srcC, 16);
                size_t srcC8 = Simd::AlignLo(p.srcC, 8);
                size_t srcC4 = Simd::AlignLo(p.srcC, 4);
                for (size_t y = yBeg; y < yEnd; ++y)
                {
                    const float* ps = src + y * p.srcW * p.srcC;
                    uint16_t* pd = dst + (y - yBeg) * p.srcW * srcC;
                    for (size_t x = 0; x < p.srcW; ++x)
                    {
                        size_t c = 0;
                        for (; c < srcC16; c += 16)
                        {
                            __m256i d0 = Float32ToBFloat16(_mm256_loadu_ps(ps + c + 0));
                            __m256i d1 = Float32ToBFloat16(_mm256_loadu_ps(ps + c + 8));
                            _mm256_storeu_si256((__m256i*)(pd + c), _mm256_permute4x64_epi64(_mm256_packus_epi32(d0, d1), 0xD8));
                        }
                        for (; c < srcC8; c += 8)
                        {
                            __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(ps + c + 0));
                            __m128i d1 = Sse41::Float32ToBFloat16(_mm_loadu_ps(ps + c + 4));
                            _mm_storeu_si128((__m128i*)(pd + c), _mm_packus_epi32(d0, d1));
                        }
                        for (; c < srcC4; c += 4)
                        {
                            __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(ps + c + 0));
                            _mm_storel_epi64((__m128i*)(pd + c), _mm_packus_epi32(d0, Sse41::K_ZERO));
                        }
                        for (; c < p.srcC; ++c)
                            pd[c] = Base::Float32ToBFloat16(ps[c]);
                        for (; c < srcC; ++c)
                            pd[c] = 0;
                        ps += p.srcC;
                        pd += srcC;
                    }
                }
            }
        }

        static void ReorderBf16(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t srcC = AlignHi(p.srcC, a.miK);
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            for (size_t y = yBeg; y < yEnd; ++y)
            {
                const uint16_t* ps = src + y * p.srcW * p.srcC;
                uint16_t* pd = dst + (y - yBeg) * p.srcW * srcC;
                for (size_t x = 0; x < p.srcW; ++x)
                {
                    size_t c = 0;
                    for (; c < srcCDF; c += DF)
                        Copy(ps + c, pd + c);
                    for (; c < p.srcC; ++c)
                        pd[c] = ps[c];
                    for (; c < srcC; ++c)
                        pd[c] = 0;
                    ps += p.srcC;
                    pd += srcC;
                }
            }
        }

        static void ConvertBf16ToFp32(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, float* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t rowSize = p.srcW * p.srcC;
            for (size_t y = yBeg; y < yEnd; ++y)
                BFloat16ToFloat32(src + y * rowSize, rowSize, dst + y * rowSize);
        }

        //-----------------------------------------------------------------------------------------

        SynetMergedConvolution16bCdc::SynetMergedConvolution16bCdc(const MergConvParam& p)
            : Sse41::SynetMergedConvolution16bCdc(p)
        {
            SetSize(F, 2);
            if (!_src16b)
                _toBf16 = ConvertFp32ToBf16;
            else if (!Aligned(p.conv[0].srcC, 2))
                _toBf16 = ReorderBf16;
            else
                _toBf16 = NULL;
            if (_src16b)
                _toFp32 = ConvertBf16ToFp32;
            SetInput(_param.conv[0], _input);
            SetDepthwise(_param.conv[1], _depthwise);
            SetOutput(_param.conv[2], _output);
        }

        //-----------------------------------------------------------------------------------------

        SynetMergedConvolution16bCd::SynetMergedConvolution16bCd(const MergConvParam& p)
            : Sse41::SynetMergedConvolution16bCd(p)
        {
            SetSize(F, 2);
            if (!_src16b)
                _toBf16 = ConvertFp32ToBf16;
            else if (!Aligned(p.conv[0].srcC, 2))
                _toBf16 = ReorderBf16;
            else
                _toBf16 = NULL;
            SetInput(_param.conv[0], _input);
            SetDepthwise(_param.conv[1], _depthwise);
        }

        //-----------------------------------------------------------------------------------------

        SynetMergedConvolution16bDc::SynetMergedConvolution16bDc(const MergConvParam& p)
            : Sse41::SynetMergedConvolution16bDc(p)
        {
            SetSize(F, 2);
            SetDepthwise(_param.conv[0], _depthwise);
            SetOutput(_param.conv[1], _output);
        }

        //-----------------------------------------------------------------------------------------

        void* SynetMergedConvolution16bInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add)
        {
            MergConvParam param(batch, convs, count, add);
            if (!param.Valid(SimdTensorData32f, SimdTensorData16b))
                return NULL;
            if (SynetMergedConvolution16bCdc::Preferable(param))
                return new Avx2::SynetMergedConvolution16bCdc(param);
            else if (SynetMergedConvolution16bCd::Preferable(param))
                return new Avx2::SynetMergedConvolution16bCd(param);
            else if (SynetMergedConvolution16bDc::Preferable(param))
                return new Avx2::SynetMergedConvolution16bDc(param);
            else
                return new Base::SynetMergedConvolution16b(param);
        }
	}
#endif
}
