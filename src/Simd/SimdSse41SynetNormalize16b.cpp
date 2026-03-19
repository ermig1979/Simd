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
#include "Simd/SimdSynet.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSse41.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        void NormalizeNhwc16bV2(const uint16_t* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, float eps, float* buf, uint16_t* dst)
        {
            float k = 1.0f / float(channels);
            Array32f _buf;
            if (buf == NULL)
            {
                _buf.Resize(channels);
                buf = _buf.data;
            }
            size_t channelsF = AlignLo(channels, F), c;
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    BFloat16ToFloat32(src, channels, buf);

                    __m128 _sum = _mm_setzero_ps();
                    for (c = 0; c < channelsF; c += F)
                        _sum = _mm_add_ps(_mm_loadu_ps(buf + c), _sum);
                    float sum = ExtractSum(_sum);
                    for (; c < channels; ++c)
                        sum += buf[c];
                    __m128 mean = _mm_set1_ps(sum * k);
                    for (c = 0; c < channelsF; c += F)
                        _mm_storeu_ps(buf + c, _mm_sub_ps(_mm_loadu_ps(buf + c), mean));
                    for (; c < channels; ++c)
                        _mm_store_ss(buf + c, _mm_sub_ss(_mm_load_ss(buf + c), mean));

                    __m128 _sqsum = _mm_setzero_ps();
                    for (c = 0; c < channelsF; c += F)
                        _sqsum = _mm_add_ps(Square(_mm_loadu_ps(buf + c)), _sqsum);
                    float sqsum = ExtractSum(_sqsum);
                    for (; c < channels; ++c)
                        sqsum += Simd::Square(buf[c]);
                    __m128 norm = _mm_set1_ps(1.0f / ::sqrt(sqsum * k + eps));
                    for (c = 0; c < channelsF; c += F)
                        _mm_storeu_ps(buf + c, _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(buf + c), norm), _mm_loadu_ps(scale + c)), _mm_loadu_ps(shift + c)));
                    for (; c < channels; ++c)
                        _mm_store_ss(buf + c, _mm_add_ss(_mm_mul_ss(_mm_mul_ss(_mm_load_ss(buf + c), norm), _mm_load_ss(scale + c)), _mm_load_ss(shift + c)));

                    Float32ToBFloat16(buf, channels, dst);

                    dst += channels;
                    src += channels;
                }
            }
        }

        void SynetNormalizeLayerForward16bV2(const uint16_t* src, size_t batch, size_t channels, size_t spatial,
            const float* scale, const float* shift, const float* eps, SimdTensorFormatType format, float* buf, uint16_t* dst)
        {
            if(format == SimdTensorFormatNhwc)
                NormalizeNhwc16bV2(src, batch, channels, spatial, scale, shift, *eps, buf, dst);
            else
                assert(0);
        }
    }
#endif
}
