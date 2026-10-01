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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAlignment.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetAddBiasNchw(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float value = bias[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    dst[s + 0] += value;
                    dst[s + 1] += value;
                    dst[s + 2] += value;
                    dst[s + 3] += value;
                }
                for (; s < spatial; ++s)
                    dst[s] += value;
                dst += spatial;
            }
        }

        void SynetAddBiasNhwc(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    dst[c + 0] += bias[c + 0];
                    dst[c + 1] += bias[c + 1];
                    dst[c + 2] += bias[c + 2];
                    dst[c + 3] += bias[c + 3];
                }
                for (; c < channels; ++c)
                    dst[c] += bias[c];
                dst += channels;
            }
        }

        void SynetAddBias(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetAddBiasNchw(bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetAddBiasNhwc(bias, channels, spatial, dst);
            else
                assert(0);
        }
    }
#endif
}
