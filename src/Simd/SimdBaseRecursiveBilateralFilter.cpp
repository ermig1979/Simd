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
    RbfParam::RbfParam(size_t w, size_t h, size_t c, const float* s, const float* r, size_t a)
        : width(w)
        , height(h)
        , channels(c)
        , spatial(*s)
        , range(*r)
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

    //---------------------------------------------------------------------------------------------

    RecursiveBilateralFilter::RecursiveBilateralFilter(const RbfParam& param)
        : _param(param)
    {
    }

    //---------------------------------------------------------------------------------------------

    namespace Base
    {
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

        template<size_t channels> void HorFilter(const RbfParam& p, RbfAlg& a, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t size = p.width * channels, cLast = size - 1, fLast = p.width - 1;
            for (size_t y = 0; y < p.height; y++)
            {
                const uint8_t* sl = src, * sr = src + cLast;
                float* lc = a.cb0.data, * rc = a.cb1.data + cLast;
                float* lf = a.fb0.data, * rf = a.fb1.data + fLast;
                *lf++ = 1.f;
                *rf-- = 1.f;
                for (int c = 0; c < channels; c++)
                {
                    *lc++ = *sl++;
                    *rc-- = *sr--;
                }
                for (size_t x = 1; x < p.width; x++)
                {
                    int ld = Diff<channels>(sl, sl - channels);
                    int rd = Diff<channels>(sr + 1 - channels, sr + 1);
                    float la = a.ranges[ld];
                    float ra = a.ranges[rd];
                    *lf++ = a.alpha + la * lf[-1];
                    *rf-- = a.alpha + ra * rf[+1];
                    for (int c = 0; c < channels; c++)
                    {
                        *lc++ = (a.alpha * (*sl++) + la * lc[-channels]);
                        *rc-- = (a.alpha * (*sr--) + ra * rc[+channels]);
                    }
                }
                SetOut<channels>(a.cb0.data, a.fb0.data, a.cb1.data, a.fb1.data, p.width, dst);
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

        template<size_t channels> void VerSetMain(const uint8_t* hor, const uint8_t* src0, const uint8_t* src1, size_t width,
            float alpha, const float *ranges, const float* pf, const float* pc, float* cf, float* cc)
        {
            for (size_t x = 0, o = 0; x < width; x++)
            {
                float ua = ranges[Diff<channels>(src0 + o, src1 + o)];
                cf[x] = alpha + ua * pf[x];
                for (size_t e = o + channels; o < e; o++)
                    cc[o] = alpha * hor[o] + ua * pc[o];
            }
        }

        template<size_t channels> void VerFilter(const RbfParam& p, RbfAlg& a, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t size = p.width * channels, srcTail = srcStride - size, dstTail = dstStride - size;
            float* dcb = a.cb0.data, * dfb = a.fb0.data, * ucb = a.cb1.data, * ufb = a.fb1.data;

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
                VerSetMain<channels>(duc, suc, suc + srcStride, p.width, a.alpha, a.ranges.data, uf + p.width, uc + size, uf, uc);
            }

            VerSetEdge<channels>(dst, p.width, dfb, dcb);
            SetOut<channels>(dcb, dfb, ucb, ufb, p.width, dst);
            for (size_t y = 1; y < p.height; y++)
            {
                src += srcStride;
                dst += dstStride;
                float* dc = dcb + (y & 1) * size;
                float* df = dfb + (y & 1) * p.width;
                const float* dpc = dcb + ((y - 1)&1) * size;
                const float* dpf = dfb + ((y - 1) & 1) * p.width;
                VerSetMain<channels>(dst, src, src - srcStride, p.width, a.alpha, a.ranges.data, dpf, dpc, df, dc);
                SetOut<channels>(dc, df, ucb + y * size, ufb + y * p.width, p.width, dst);
            }
        }

        //-----------------------------------------------------------------------------------------

        RecursiveBilateralFilterDefault::RecursiveBilateralFilterDefault(const RbfParam& param)
            :Simd::RecursiveBilateralFilter(param)
        {
            InitAlg();
            switch (_param.channels)
            {
            case 1: _hFilter = HorFilter<1>; _vFilter = VerFilter<1>; break;
            case 2: _hFilter = HorFilter<2>; _vFilter = VerFilter<2>; break;
            case 3: _hFilter = HorFilter<3>; _vFilter = VerFilter<3>; break;
            case 4: _hFilter = HorFilter<4>; _vFilter = VerFilter<4>; break;
            default:
                assert(0);
            }
        }

        void RecursiveBilateralFilterDefault::InitAlg()
        {
            const RbfParam& p = _param;
            RbfAlg& a = _alg;

            float alpha = ::exp(-sqrt(2.0f) / (p.spatial * 255.0f));
            a.alpha = 1.0f - alpha;
            a.ranges.Resize(256);

            float range = 1.0f / (p.range * 255.0f), f = 0.0f;
            for (int i = 0; i <= 255; i++, f -= 1.0f)
                a.ranges[i] = alpha * exp(f * range);
        }

        void RecursiveBilateralFilterDefault::InitBuf()
        {
            if (_alg.fb0.data)
                return;
            const RbfParam& p = _param;
            _alg.cb0.Resize(p.width * 2 * p.channels);
            _alg.fb0.Resize(p.width * 2);
            _alg.cb1.Resize(p.width * p.channels * p.height);
            _alg.fb1.Resize(p.width * p.height);
            _alg.rb0.Resize(p.width);
        }

        void RecursiveBilateralFilterDefault::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            InitBuf();
            _hFilter(_param, _alg, src, srcStride, dst, dstStride);
            _vFilter(_param, _alg, src, srcStride, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, const float* sigmaSpatial, const float* sigmaRange)
        {
            RbfParam param(width, height, channels, sigmaSpatial, sigmaRange, sizeof(void*));
            if (!param.Valid())
                return NULL;
            return new RecursiveBilateralFilterDefault(param);
        }
    }
}

