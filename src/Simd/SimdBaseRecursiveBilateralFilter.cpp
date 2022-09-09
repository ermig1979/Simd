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
                    final_diff = ((component_diff[0] + component_diff[1] + component_diff[2] + component_diff[3]) >> 2);
                    break;
                default:
                    final_diff = 0;
                }
                assert(final_diff >= 0 && final_diff <= 255);
                return final_diff;
            }

            template<size_t channels> void SetOut(const float * bc, const float * bf, const float * ec, const float* ef,
                size_t width, size_t height, uint8_t * dst, size_t dstStride)
            {
                size_t tail = dstStride - width * channels;
                for (size_t y = 0; y < height; ++y)
                {
                    for (size_t x = 0; x < width; x++)
                    {
                        float factor = 1.f / (bf[x] + ef[x]);
                        for (size_t c = 0; c < channels; c++)
                        {
                            dst[c] = uint8_t(factor * (bc[c] + ec[c]));
                        }
                        bc += channels;
                        ec += channels;
                        dst += channels;
                    }
                    bf += width;
                    ef += width;
                    dst += tail;
                }
            }

            template<size_t channels>
            void HorizontalFilter(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, size_t width, size_t height, 
                float* ranges, float alpha, float* left_Color_Buffer, float* left_Factor_Buffer, float* right_Color_Buffer, float* right_Factor_Buffer)
            {
                size_t size = width * channels, cLast = size - 1, fLast = width - 1;
                for (size_t y = 0; y < height; y++)
                {
                    const uint8_t* src_left_color = src + y * srcStride;
                    float* left_Color = left_Color_Buffer + y * size;
                    float* left_Factor = left_Factor_Buffer + y * width;

                    const uint8_t* src_right_color = src + y * srcStride + cLast;
                    float* right_Color = right_Color_Buffer + y * size + cLast;
                    float* right_Factor = right_Factor_Buffer + y * width + fLast;

                    const uint8_t* src_left_prev = src_left_color;
                    const float* left_prev_factor = left_Factor;
                    const float* left_prev_color = left_Color;

                    const uint8_t* src_right_prev = src_right_color;
                    const float* right_prev_factor = right_Factor;
                    const float* right_prev_color = right_Color;

                    *left_Factor++ = 1.f;
                    *right_Factor-- = 1.f;
                    for (int c = 0; c < channels; c++)
                    {
                        *left_Color++ = *src_left_color++;
                        *right_Color-- = *src_right_color--;
                    }
                    for (size_t x = 1; x < width; x++)
                    {
                        int left_diff = DiffFactor<channels>(src_left_color, src_left_prev);
                        src_left_prev = src_left_color;

                        int right_diff = DiffFactor<channels> (src_right_color, src_right_prev);
                        src_right_prev = src_right_color;

                        float left_alpha_f = ranges[left_diff];
                        float right_alpha_f = ranges[right_diff];
                        *left_Factor++ = alpha + left_alpha_f * (*left_prev_factor++);
                        *right_Factor-- = alpha + right_alpha_f * (*right_prev_factor--);

                        for (int c = 0; c < channels; c++)
                        {
                            *left_Color++ = (alpha * (*src_left_color++) + left_alpha_f * (*left_prev_color++));
                            *right_Color-- = (alpha * (*src_right_color--) + right_alpha_f * (*right_prev_color--));
                        }
                    }
                }
                SetOut<channels>(left_Color_Buffer, left_Factor_Buffer, right_Color_Buffer, right_Factor_Buffer, width, height, dst, dstStride);
            }

            template<size_t channels>
            void VerticalFilter(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, int width, int height, 
                float* range_table_f, float inv_alpha_f, float* down_Color_Buffer, float* down_Factor_Buffer, float* up_Color_Buffer, float* up_Factor_Buffer)
            {
                size_t size = width * channels, srcTail = srcStride - size, dstTail = dstStride - size;

                const uint8_t* src_color_first_hor = dst;
                const uint8_t* src_down_color = src;
                float* down_color = down_Color_Buffer;
                float* down_factor = down_Factor_Buffer;

                const uint8_t* src_down_prev = src_down_color;
                const float* down_prev_color = down_color;
                const float* down_prev_factor = down_factor;

                int last_index = size * height - 1;
                const uint8_t* src_up_color = src + srcStride * (height - 1) + size - 1;
                const uint8_t* src_color_last_hor = dst + dstStride * (height - 1) + size - 1;
                float* up_color = up_Color_Buffer + last_index;
                float* up_factor = up_Factor_Buffer + (width * height - 1);

                const float* up_prev_color = up_color;
                const float* up_prev_factor = up_factor;

                for (int x = 0; x < width; x++)
                {
                    *down_factor++ = 1.f;
                    *up_factor-- = 1.f;
                    for (int c = 0; c < channels; c++)
                    {
                        *down_color++ = *src_color_first_hor++;
                        *up_color-- = *src_color_last_hor--;
                    }
                    src_down_color += channels;
                    src_up_color -= channels;
                }
                src_color_first_hor += dstTail;
                src_color_last_hor -= dstTail;
                src_down_color += srcTail;
                src_up_color -= srcTail;
                for (int y = 1; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int down_diff = DiffFactor<channels>(src_down_color, src_down_color - srcStride);
                        src_down_prev += channels;
                        src_down_color += channels;
                        src_up_color -= channels;
                        int up_diff = DiffFactor<channels>(src_up_color, src_up_color + srcStride);
                        float down_alpha_f = range_table_f[down_diff];
                        float up_alpha_f = range_table_f[up_diff];

                        *down_factor++ = inv_alpha_f + down_alpha_f * (*down_prev_factor++);
                        *up_factor-- = inv_alpha_f + up_alpha_f * (*up_prev_factor--);

                        for (int c = 0; c < channels; c++)
                        {
                            *down_color++ = inv_alpha_f * (*src_color_first_hor++) + down_alpha_f * (*down_prev_color++);
                            *up_color-- = inv_alpha_f * (*src_color_last_hor--) + up_alpha_f * (*up_prev_color--);
                        }
                    }
                    src_color_first_hor += dstTail;
                    src_color_last_hor -= dstTail;
                    src_down_color += srcTail;
                    src_up_color -= srcTail;
                }

                SetOut<channels>(down_Color_Buffer, down_Factor_Buffer, up_Color_Buffer, up_Factor_Buffer, width, height, dst, dstStride);
            }

            template<size_t channels>
            void Filter(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, int Width, int Height, float sigmaSpatial, float sigmaRange)
            {
                int reserveWidth = Width;
                int reserveHeight = Height;

                assert(reserveWidth >= 10 && reserveWidth < 10000);
                assert(reserveHeight >= 10 && reserveHeight < 10000);
                assert(channels >= 1 && channels <= 4);

                int reservePixels = reserveWidth * reserveHeight;
                int numberOfPixels = reservePixels * channels;

                float* leftColorBuffer = (float*)calloc(sizeof(float) * numberOfPixels, 1);
                float* leftFactorBuffer = (float*)calloc(sizeof(float) * reservePixels, 1);
                float* rightColorBuffer = (float*)calloc(sizeof(float) * numberOfPixels, 1);
                float* rightFactorBuffer = (float*)calloc(sizeof(float) * reservePixels, 1);

                if (leftColorBuffer == NULL || leftFactorBuffer == NULL || rightColorBuffer == NULL || rightFactorBuffer == NULL)
                {
                    if (leftColorBuffer)  free(leftColorBuffer);
                    if (leftFactorBuffer) free(leftFactorBuffer);
                    if (rightColorBuffer) free(rightColorBuffer);
                    if (rightFactorBuffer) free(rightFactorBuffer);

                    return;
                }
                float* downColorBuffer = leftColorBuffer;
                float* downFactorBuffer = leftFactorBuffer;
                float* upColorBuffer = rightColorBuffer;
                float* upFactorBuffer = rightFactorBuffer;

                float alpha_f = static_cast<float>(exp(-sqrt(2.0) / (sigmaSpatial * 255)));
                float inv_alpha_f = 1.f - alpha_f;


                float range_table_f[255 + 1];
                float inv_sigma_range = 1.0f / (sigmaRange * 255);

                float ii = 0.f;
                for (int i = 0; i <= 255; i++, ii -= 1.f)
                {
                    range_table_f[i] = alpha_f * exp(ii * inv_sigma_range);
                }

                HorizontalFilter<channels>(src, srcStride, dst, dstStride, Width, Height,  
                    range_table_f, inv_alpha_f, leftColorBuffer, leftFactorBuffer, rightColorBuffer, rightFactorBuffer);

                VerticalFilter<channels>(src, srcStride, dst, dstStride, Width, Height, 
                    range_table_f, inv_alpha_f, downColorBuffer, downFactorBuffer, upColorBuffer, upFactorBuffer);

                if (leftColorBuffer)
                {
                    free(leftColorBuffer);
                    leftColorBuffer = NULL;
                }

                if (leftFactorBuffer)
                {
                    free(leftFactorBuffer);
                    leftFactorBuffer = NULL;
                }

                if (rightColorBuffer)
                {
                    free(rightColorBuffer);
                    rightColorBuffer = NULL;
                }

                if (rightFactorBuffer)
                {
                    free(rightFactorBuffer);
                    rightFactorBuffer = NULL;
                }
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

