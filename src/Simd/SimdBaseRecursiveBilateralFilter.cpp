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
#include "Simd/SimdRecursiveBilateralFilter.h"

namespace Simd
{
    RbfParam::RbfParam(size_t w, size_t h, size_t c, const float* s, const float* r, SimdRecursiveBilateralFilterFlags f, size_t a)
        : width(w)
        , height(h)
        , channels(c)
        , spatial(*s)
        , range(*r)
        , flags(f)
        , align(a)
    {
    }

    bool RbfParam::Valid() const
    {
        return
            height > 0 &&
            width > 0 &&
            channels > 0 && channels <= 4 &&
            align >= sizeof(float);
    }

    void RbfParam::Init()
    {
        float a = ::exp(-sqrt(2.0f) / (spatial * 255.0f));
        alpha = 1.0f - a;

        float r = 1.0f / (range * 255.0f), f = 0.0f;
        for (int i = 0; i < 256; i++, f -= 1.0f)
            ranges[i] = a * exp(f * r);
    }

    //---------------------------------------------------------------------------------------------

    RecursiveBilateralFilter::RecursiveBilateralFilter(const RbfParam& param)
        : _param(param)
        , _hFilter(NULL)
        , _vFilter(NULL)
    {
        _param.Init();
    }

    //---------------------------------------------------------------------------------------------

    namespace Base
    {
        namespace Prec
        {
            template<size_t channels> void RowRanges(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
            {
                for (size_t x = 0, o = 0; x < width; x += 1, o += channels)
                    dst[x] = ranges[Diff<channels>(src0 + o, src1 + o)];
            }

            template<size_t channels> SIMD_INLINE void SetOut(const float* bc, const float* bf, const float* ec, const float* ef, size_t width, uint8_t* dst)
            {
                for (size_t x = 0; x < width; x++)
                {
                    float factor = 1.f / (bf[x] + ef[x]);
                    for (size_t c = 0; c < channels; c++)
                        dst[c] = uint8_t(factor * (bc[c] + ec[c]));
                    bc += channels;
                    ec += channels;
                    dst += channels;
                }
            }

            template<size_t channels> void HorFilter(const RbfParam& p, float * buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
            {
                size_t size = p.width * channels, cLast = size - 1, fLast = p.width - 1;
                float* cb0 = buf, * cb1 = cb0 + size, * fb0 = cb1 + size, * fb1 = fb0 + p.width, * rb0 = fb1 + p.width;
                for (size_t y = 0; y < p.height; y++)
                {
                    const uint8_t* sl = src, * sr = src + cLast;
                    float* lc = cb0, * rc = cb1 + cLast;
                    float* lf = fb0, * rf = fb1 + fLast;
                    *lf++ = 1.f;
                    *rf-- = 1.f;
                    for (int c = 0; c < channels; c++)
                    {
                        *lc++ = *sl++;
                        *rc-- = *sr--;
                    }
                    RowRanges<channels>(src, src + channels, p.width - 1, p.ranges, rb0 + 1);
                    for (size_t x = 1; x < p.width; x++)
                    {
                        float la = rb0[x];
                        float ra = rb0[p.width - x];
                        *lf++ = p.alpha + la * lf[-1];
                        *rf-- = p.alpha + ra * rf[+1];
                        for (int c = 0; c < channels; c++)
                        {
                            *lc++ = (p.alpha * (*sl++) + la * lc[-channels]);
                            *rc-- = (p.alpha * (*sr--) + ra * rc[+channels]);
                        }
                    }
                    SetOut<channels>(cb0, fb0, cb1, fb1, p.width, dst);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template<size_t channels> void VerSetEdge(const uint8_t* src, size_t width, float* factor, float* colors)
            {
                for (size_t x = 0; x < width; x++)
                    factor[x] = 1.0f;
                for (size_t i = 0, n = width * channels; i < n; i++)
                    colors[i] = src[i];
            }

            template<size_t channels> void VerSetMain(const uint8_t* hor, size_t width, float alpha, 
                const float* ranges, const float* pf, const float* pc, float* cf, float* cc)
            {
                for (size_t x = 0, o = 0; x < width; x++)
                {
                    float ua = ranges[x];
                    cf[x] = alpha + ua * pf[x];
                    for (size_t e = o + channels; o < e; o++)
                        cc[o] = alpha * hor[o] + ua * pc[o];
                }
            }

            template<size_t channels> void VerFilter(const RbfParam& p, float * buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
            {
                size_t size = p.width * channels, srcTail = srcStride - size, dstTail = dstStride - size;
                float *rb0 = buf, *dcb = rb0 + p.width, * dfb = dcb + size * 2, * ucb = dfb + p.width * 2, * ufb = ucb + size * p.height;

                const uint8_t* suc = src + srcStride * (p.height - 1);
                const uint8_t* duc = dst + dstStride * (p.height - 1);
                float* uf = ufb + p.width * (p.height - 1);
                float* uc = ucb + size * (p.height - 1);
                VerSetEdge<channels>(duc, p.width, uf, uc);
                for (size_t y = 1; y < p.height; y++)
                {
                    duc -= dstStride;
                    suc -= srcStride;
                    uf -= p.width;
                    uc -= size;
                    RowRanges<channels>(suc, suc + srcStride, p.width, p.ranges, rb0);
                    VerSetMain<channels>(duc, p.width, p.alpha, rb0, uf + p.width, uc + size, uf, uc);
                }

                VerSetEdge<channels>(dst, p.width, dfb, dcb);
                SetOut<channels>(dcb, dfb, ucb, ufb, p.width, dst);
                for (size_t y = 1; y < p.height; y++)
                {
                    src += srcStride;
                    dst += dstStride;
                    float* dc = dcb + (y & 1) * size;
                    float* df = dfb + (y & 1) * p.width;
                    const float* dpc = dcb + ((y - 1) & 1) * size;
                    const float* dpf = dfb + ((y - 1) & 1) * p.width;
                    RowRanges<channels>(src, src - srcStride, p.width, p.ranges, rb0);
                    VerSetMain<channels>(dst, p.width, p.alpha, rb0, dpf, dpc, df, dc);
                    SetOut<channels>(dc, df, ucb + y * size, ufb + y * p.width, p.width, dst);
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        RecursiveBilateralFilterPrecize::RecursiveBilateralFilterPrecize(const RbfParam& param)
            : Simd::RecursiveBilateralFilter(param)
        {
            switch (_param.channels)
            {
            case 1: _hFilter = Prec::HorFilter<1>; _vFilter = Prec::VerFilter<1>; break;
            case 2: _hFilter = Prec::HorFilter<2>; _vFilter = Prec::VerFilter<2>; break;
            case 3: _hFilter = Prec::HorFilter<3>; _vFilter = Prec::VerFilter<3>; break;
            case 4: _hFilter = Prec::HorFilter<4>; _vFilter = Prec::VerFilter<4>; break;
            default:
                assert(0);
            }
        }

        float* RecursiveBilateralFilterPrecize::GetBuffer()
        {
            if (_buffer.Empty())
            {
                const RbfParam& p = _param;
                size_t size = 0;
                size += p.height * p.width * (p.channels + 1);
                size += p.width * (p.channels * 2 + 3);
                _buffer.Resize(size * sizeof(float));
            }
            return (float*)_buffer.data;
        }

        void RecursiveBilateralFilterPrecize::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            float* buf = GetBuffer();
            _hFilter(_param, buf, src, srcStride, dst, dstStride);
            _vFilter(_param, buf, src, srcStride, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, 
            const float* sigmaSpatial, const float* sigmaRange, SimdRecursiveBilateralFilterFlags flags)
        {
            RbfParam param(width, height, channels, sigmaSpatial, sigmaRange, flags, sizeof(void*));
            if (!param.Valid())
                return NULL;
            return new RecursiveBilateralFilterPrecize(param);
        }
    }
}

