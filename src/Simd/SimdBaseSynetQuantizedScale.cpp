/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdFmadd.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SIMD_INLINE void QuantizedScale(const uint8_t& src, int sBias, float sNorm, float scale, float bias, uint8_t& dst, float dNorm, int dZero)
        {
            float _src = DequantizeLinear(src, sBias, sNorm);
            float _dst = Fmadd<false>(_src, scale, bias);
            dst = (uint8_t)QuantizeLinear(_dst, dNorm, dZero, 0, 255);
        }

        void SynetQuantizedScaleLayerForward(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* scale, const float* bias, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format)
        {
            Array32f defaultBias;
            if (bias == NULL)
            {
                defaultBias.Resize(channels, true);
                bias = defaultBias.data;
            }
            float sBias = -srcZero;
            float sNorm = srcScale[0], dNorm = 1.0f / dstScale[0];
            if (format == SimdTensorFormatNhwc)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channels; ++c)
                        QuantizedScale(src[c], sBias, sNorm, scale[c], bias[c], dst[c], dNorm, dstZero);
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    float _scale = scale[c];
                    float _bias = bias[c];
                    size_t s = 0;
                    for (; s < spatial; ++s)
                        QuantizedScale(src[s], sBias, sNorm, _scale, _bias, dst[s], dNorm, dstZero);
                    src += spatial;
                    dst += spatial;
                }
            }
        }
    }
#endif
}
