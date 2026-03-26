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
#include "Simd/SimdSynet.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdAvx512bw.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        void NormalizeNhwc16bV2(const uint16_t* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, uint16_t* dst)
        {
            float k = 1.0f / float(channels);
            size_t channelsF = AlignLo(channels, F), c;
            __mmask16 channelsM = TailMask16(channels - channelsF);
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(channels);
                buf = _buf.data;
            }
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    BFloat16ToFloat32(src, channels, buf);

                    __m512 _sum = _mm512_setzero_ps();
                    for (c = 0; c < channelsF; c += F)
                        _sum = _mm512_add_ps(_mm512_loadu_ps(buf + c), _sum);
                    if(c < channels)
                        _sum = _mm512_add_ps(_mm512_maskz_loadu_ps(channelsM, buf + c), _sum);
                    __m512 mean = _mm512_set1_ps(ExtractSum(_sum) * k);
                    for (c = 0; c < channelsF; c += F)
                        _mm512_storeu_ps(buf + c, _mm512_sub_ps(_mm512_loadu_ps(buf + c), mean));
                    if(c < channels)
                        _mm512_mask_storeu_ps(buf + c, channelsM, _mm512_sub_ps(_mm512_maskz_loadu_ps(channelsM, buf + c), mean));

                    __m512 _sqsum = _mm512_setzero_ps();
                    for (c = 0; c < channelsF; c += F)
                    {
                        __m512 _buf = _mm512_loadu_ps(buf + c);
                        _sqsum = _mm512_fmadd_ps(_buf, _buf, _sqsum);
                    }
                    if(c < channels)
                    {
                        __m512 _buf = _mm512_maskz_loadu_ps(channelsM, buf + c);
                        _sqsum = _mm512_fmadd_ps(_buf, _buf, _sqsum);
                    }
                    __m512 norm = _mm512_set1_ps(1.0f / ::sqrt(ExtractSum(_sqsum) * k + eps));
                    for (c = 0; c < channelsF; c += F)
                        _mm512_storeu_ps(buf + c, _mm512_fmadd_ps(_mm512_mul_ps(_mm512_loadu_ps(buf + c), norm), _mm512_loadu_ps(scale + c), _mm512_loadu_ps(shift + c)));
                    if(c < channels)
                        _mm512_mask_storeu_ps(buf + c, channelsM, _mm512_fmadd_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(channelsM, buf + c), norm), 
                            _mm512_maskz_loadu_ps(channelsM, scale + c), _mm512_maskz_loadu_ps(channelsM, shift + c)));

                    Float32ToBFloat16(buf, channels, dst);

                    dst += channels;
                    src += channels;
                }
            }
        }

        void SynetNormalizeLayerForward16bV2(const uint16_t* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, uint16_t* dst)
        {
            if (format == SimdTensorFormatNhwc)
                NormalizeNhwc16bV2(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else
                assert(0);
        }
    }
#endif
}
