/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynetGridSample.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        template <int align> SIMD_INLINE float Denormalize32f(float pos, int dim)
        {
            if (align)
                return float((pos + 1) / 2.0f * (dim - 1));
            else
                return float(((pos + 1) * dim - 1) / 2.0f);
        }

        template<int align, int range>  void IndexCoeffs32fBlZ(const float* grd, size_t dstS, int srcH, int srcW, int padW, uint32_t* idx, float* dy, float* dx, int& yMin, int& yMax)
        {
            for (size_t d = 0; d < dstS; ++d)
            {
                float x = Denormalize32f<align>(grd[0], srcW);
                float y = Denormalize32f<align>(grd[1], srcH);
                int x0 = int(std::floor(x));
                int y0 = int(std::floor(y));
                dy[d] = y - float(y0);
                dx[d] = x - float(x0);
                x0 = Simd::RestrictRange(x0, -2, srcW) + 2;
                y0 = Simd::RestrictRange(y0, -2, srcH) + 2;
                idx[d] = padW * y0 + x0;
                if (range)
                {
                    yMin = Min(yMin, y0);
                    yMax = Max(yMax, y0);
                }
                grd += 2;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void BilinearInterp32fBlZ(const float* pad0, size_t dstS, int padW, uint32_t* idx, float* dy, float* dx, float* dst)
        {
            const float* pad1 = pad0 + padW;
            for (size_t d = 0; d < dstS; ++d)
            {
                int offs = idx[d];
                float p00 = pad0[offs + 0];
                float p01 = pad0[offs + 1];
                float p10 = pad1[offs + 0];
                float p11 = pad1[offs + 1];
                float dy1 = dy[d];
                float dy0 = 1.0f - dy1;
                float dx1 = dx[d];
                float dx0 = 1.0f - dx1;
                dst[d] = dy0 * (dx0 * p00 + dx1 * p01) + dy1 * (dx0 * p10 + dx1 * p11);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetGridSample2d32fBlZ::SynetGridSample2d32fBlZ(const GridSample2dParam& param)
            : Simd::SynetGridSample2d(param)
        {
            _srcS = _param.srcH * _param.srcW;
            _dstS = _param.dstH * _param.dstW;
            _padH = _param.srcH + 4;
            _padW = _param.srcW + 4;
            _padded.Resize(_padH * _padW, true);
            _index.Resize(_dstS);
            _coeffs.Resize(_dstS * 2);
            _sparse = _param.dstH * 3 < _param.srcH;
            if(_sparse)
                _indexCoeffs = _param.align ? IndexCoeffs32fBlZ<1, 1> : IndexCoeffs32fBlZ<0, 1>;
            else
                _indexCoeffs = _param.align ? IndexCoeffs32fBlZ<1, 0> : IndexCoeffs32fBlZ<0, 0>;
            _bilinearInterp = BilinearInterp32fBlZ;
        }

        size_t SynetGridSample2d32fBlZ::InternalBufferSize() const
        {
            return _padded.RawSize() + _index.RawSize() + _coeffs.RawSize();
        }

        void SynetGridSample2d32fBlZ::Forward(const uint8_t* src8, const uint8_t* grd8, uint8_t* dst8)
        {
            const float* src = (const float*)src8;
            const float* grd = (const float*)grd8;
            float* dst = (float*)dst8;
            float* pad = _padded.data + 2 * _padW + 2;
            uint32_t* idx = _index.data;
            float * dy = _coeffs.data;
            float* dx = _coeffs.data + _dstS;
            for (size_t b = 0; b < _param.batch; ++b)
            {
                int yMin = (int)_padH - 2, yMax = 0;
                _indexCoeffs(grd, _dstS, (int)_param.srcH, (int)_param.srcW, (int)_padW, idx, dy, dx, yMin, yMax);
                yMin = _sparse ? Max(0, yMin - 2) : 0;
                yMax = _sparse ? Min((int)_param.srcH, yMax) : (int)_param.srcH;
                for (size_t c = 0; c < _param.channels; ++c)
                {
                    for (int h = yMin; h < yMax; ++h)
                        memcpy(pad + h * _padW, src + h * _param.srcW, _param.srcW * sizeof(float));
                    _bilinearInterp(_padded.data, _dstS, (int)_padW, idx, dy, dx, dst);
                    src += _srcS;
                    dst += _dstS;
                }
                grd += 2 * _dstS;
            }
        }
    }
#endif
}
