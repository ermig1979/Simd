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
        namespace Rbf
        {
            template<size_t channels> int DiffFactor(const uint8_t* color1, const uint8_t* color2)
            {
                int final_diff, component_diff[4];
                for (int i = 0; i < channels; i++)
                    component_diff[i] = abs(color1[i] - color2[i]);
                switch (channels)
                {
                case 1:
                    final_diff = component_diff[0];
                    break;
                case 2:
                    final_diff = ((component_diff[0] + component_diff[1]) >> 1);
                    break;
                case 3:
                    final_diff = ((component_diff[0] + component_diff[2]) >> 2) + (component_diff[1] >> 1);
                    break;
                case 4:
                    //final_diff = ((component_diff[0] + component_diff[2]) >> 2) + (component_diff[1] >> 1);
                    final_diff = ((component_diff[0] + component_diff[1] + component_diff[2] + component_diff[3]) >> 2);
                    break;
                default:
                    final_diff = 0;
                }
                assert(final_diff >= 0 && final_diff <= 255);
                return final_diff;
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

            template<size_t channels> void SetOut(const float * bc, const float * bf, const float * ec, const float* ef,
                size_t width, size_t height, uint8_t * dst, size_t dstStride)
            {
                size_t size = width * channels;
                for (size_t y = 0; y < height; ++y)
                {
                    SetOut<channels>(bc, bf, ec, ef, width, dst);
                    bc += size;
                    ec += size;
                    bf += width;
                    ef += width;
                    dst += dstStride;
                }
            }

            template<size_t channels> void HorizontalFilter(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, 
                size_t width, size_t height, float* ranges, float alpha, float* lcb, float* lfb, float* rcb, float* rfb)
            {
                size_t size = width * channels, cLast = size - 1, fLast = width - 1;
                for (size_t y = 0; y < height; y++)
                {
                    const uint8_t* sl = src, * sr = src + cLast;
                    float* lc = lcb, * rc = rcb + cLast;
                    float* lf = lfb, * rf = rfb + fLast;
                    *lf++ = 1.f;
                    *rf-- = 1.f;
                    for (int c = 0; c < channels; c++)
                    {
                        *lc++ = *sl++;
                        *rc-- = *sr--;
                    }
                    for (size_t x = 1; x < width; x++)
                    {
                        int ld = DiffFactor<channels>(sl, sl - channels);
                        int rd = DiffFactor<channels>(sr + 1 - channels, sr + 1);
                        float la = ranges[ld];
                        float ra = ranges[rd];
                        *lf++ = alpha + la * lf[-1];
                        *rf-- = alpha + ra * rf[+1];
                        for (int c = 0; c < channels; c++)
                        {
                            *lc++ = (alpha * (*sl++) + la * lc[-channels]);
                            *rc-- = (alpha * (*sr--) + ra * rc[+channels]);
                        }
                    }
                    SetOut<channels>(lcb, lfb, rcb, rfb, width, dst);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template<size_t channels>
            void VerticalFilter(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, int width, int height, 
                float* ranges, float alpha, float* dcb, float* dfb, float* ucb, float* ufb)
            {
                size_t size = width * channels, srcTail = srcStride - size, dstTail = dstStride - size;

                const uint8_t* src_up_color = src + srcStride * (height - 1);
                const uint8_t* src_color_last_hor = dst + dstStride * (height - 1);
                float* up_factor = ufb + width * (height - 1);
                float* up_color = ucb + size * (height - 1);

                for (int x = 0, o = 0; x < width; x++, o += channels)
                {
                    up_factor[x] = 1.f;
                    for (int c = 0; c < channels; c++)
                        up_color[o + c] = src_color_last_hor[o + c];
                }
                src_color_last_hor -= dstStride;
                src_up_color -= srcStride;
                up_factor -= width;
                up_color -= size;
                for (int y = 1; y < height; y++)
                {
                    for (int x = 0, o = 0; x < width; x++, o += channels)
                    {
                        int ud = DiffFactor<channels>(src_up_color + o, src_up_color + o + srcStride);
                        float ua = ranges[ud];
                        up_factor[x] = alpha + ua * up_factor[x + width];
                        for (int c = 0; c < channels; c++)
                            up_color[o + c] = alpha * src_color_last_hor[o + c] + ua * up_color[o + c + size];
                    }
                    src_color_last_hor -= dstStride;
                    src_up_color -= srcStride;
                    up_factor -= width;
                    up_color -= size;
                }

                const uint8_t* src_color_first_hor = dst;
                const uint8_t* src_down_color = src;
                float* down_color = dcb;
                float* down_factor = dfb;

                const float* down_prev_color = down_color;
                const float* down_prev_factor = down_factor;

                for (int x = 0; x < width; x++)
                {
                    *down_factor++ = 1.f;
                    for (int c = 0; c < channels; c++)
                        *down_color++ = *src_color_first_hor++;
                    src_down_color += channels;
                }
                src_color_first_hor += dstTail;
                src_down_color += srcTail;
                SetOut<channels>(dcb, dfb, ucb, ufb, width, dst);
                for (int y = 1; y < height; y++)
                {
                    float* down_color = dcb + (y & 1) * size;
                    float* down_factor = dfb + (y & 1) * width;
                    const float* down_prev_color = dcb + ((y - 1)&1) * size;
                    const float* down_prev_factor = dfb + ((y - 1) & 1) * width;

                    for (int x = 0; x < width; x++)
                    {
                        int dd = DiffFactor<channels>(src_down_color, src_down_color - srcStride);
                        src_down_color += channels;
                        float da = ranges[dd];

                        *down_factor++ = alpha + da * (*down_prev_factor++);

                        for (int c = 0; c < channels; c++)
                            *down_color++ = alpha * (*src_color_first_hor++) + da * (*down_prev_color++);
                    }
                    src_color_first_hor += dstTail;
                    src_down_color += srcTail;

                    SetOut<channels>(dcb + (y&1) * size, dfb + (y & 1) * width, ucb + y * size,
                        ufb + y*width, width, dst + y * dstStride);
                }
            }

            template<size_t channels>
            void Filter(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, int width, int height, float sigmaSpatial, float sigmaRange)
            {
                Array32f lcb(width * 2 * channels);
                Array32f lfb(width * 2);
                Array32f rcb(height*width * channels);
                Array32f rfb(height*width);

                float alpha_f = static_cast<float>(exp(-sqrt(2.0) / (sigmaSpatial * 255)));
                float inv_alpha_f = 1.f - alpha_f;

                float ranges[255 + 1];
                float inv_sigma_range = 1.0f / (sigmaRange * 255);

                float ii = 0.f;
                for (int i = 0; i <= 255; i++, ii -= 1.f)
                    ranges[i] = alpha_f * exp(ii * inv_sigma_range);

                HorizontalFilter<channels>(src, srcStride, dst, dstStride, width, height,  
                    ranges, inv_alpha_f, lcb.data, lfb.data, rcb.data, rfb.data);

                VerticalFilter<channels>(src, srcStride, dst, dstStride, width, height, 
                    ranges, inv_alpha_f, lcb.data, lfb.data, rcb.data, rfb.data);
            }
        }

        RecursiveBilateralFilterDefault::RecursiveBilateralFilterDefault(const RbfParam& param)
            :Simd::RecursiveBilateralFilter(param)
        {
        }

        void RecursiveBilateralFilterDefault::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            switch (_param.channels)
            {
            case 1: Rbf::Filter<1>(src, srcStride, dst, dstStride, (int)_param.width, (int)_param.height, _param.spatial, _param.range); break;
            case 2: Rbf::Filter<2>(src, srcStride, dst, dstStride, (int)_param.width, (int)_param.height, _param.spatial, _param.range); break;
            case 3: Rbf::Filter<3>(src, srcStride, dst, dstStride, (int)_param.width, (int)_param.height, _param.spatial, _param.range); break;
            case 4: Rbf::Filter<4>(src, srcStride, dst, dstStride, (int)_param.width, (int)_param.height, _param.spatial, _param.range); break;
            default:
                assert(0);
            }
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

