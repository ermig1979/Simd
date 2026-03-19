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
#include "Simd/SimdArray.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetNormalizeLayerForward16bV2(const uint16_t* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, uint16_t* dst)
        {
            float k = 1.0f / float(channels), e = *eps;
            if (format == SimdTensorFormatNchw)
            {
                assert(0);
            }
            else if (format == SimdTensorFormatNhwc)
            {
                Array32f _buf;
                if (buf == NULL)
                {
                    _buf.Resize(channels);
                    buf = _buf.data;
                }
                for (size_t b = 0; b < batch; ++b)
                {
                    for (size_t i = 0; i < spatial; ++i)
                    {
                        for (size_t c = 0; c < channels; ++c)
                            buf[c] = BFloat16ToFloat32(src[c]);

                        float sum = 0;
                        for (size_t c = 0; c < channels; ++c)
                            sum += buf[c];
                        float mean = sum * k;
                        for (size_t c = 0; c < channels; ++c)
                            buf[c] = buf[c] - mean;

                        float sqsum = 0;
                        for (size_t c = 0; c < channels; ++c)
                            sqsum += Simd::Square(buf[c]);
                        float norm = 1.0f / ::sqrt(sqsum * k + e);
                        for (size_t c = 0; c < channels; ++c)
                            dst[c] = Float32ToBFloat16(buf[c] * norm * scale[c] + shift[c]);

                        dst += channels;
                        src += channels;
                    }
                }
            }
            else
                assert(0);
        }
    }
#endif
}
