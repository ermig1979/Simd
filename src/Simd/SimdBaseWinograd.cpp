/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdWinograd.h"

namespace Simd
{
    namespace Base
    {
        void Winograd2x3SetFilter(const float * src, size_t srcChannels, size_t dstChannels, float * dst, size_t dstStride)
        {
            size_t size = dstChannels * srcChannels;
            for (size_t i = 0; i < size; i += 1, src += 9, dst += 1)
                Base::Winograd2x3SetFilter1(src, dst, dstStride);
        }

        void Winograd2x3SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, size_t dstStride, int pad)
        {
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstHeightFull = AlignLo(dstHeight, 2);
            size_t dstWidthFull = AlignLo(dstWidth, 2);
            size_t noseW = Simd::Min<size_t>(4, dstWidth + 1);
            size_t noseH = Simd::Min<size_t>(4, dstHeight + 1);
            size_t start = pad ? 2 : 0;
            if (pad)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 2;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 2;
                src -= srcWidth + 1;
            }
            size_t tailW = dstWidth - dstWidthFull + (pad ? 1 : 2);
            size_t tailH = dstHeight - dstHeightFull + (pad ? 1 : 2);
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        Winograd2x3SetInput1p(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 2)
                        Winograd2x3SetInput1p(src + col, srcWidth, 1, noseH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        Winograd2x3SetInput1p(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                }
                for (row = start; row < dstHeightFull; row += 2)
                {
                    if (pad)
                        Winograd2x3SetInput1p(src + row * srcWidth, srcWidth, 0, 4, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 2)
                        Winograd2x3SetInput1(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                    if (col < dstWidth)
                        Winograd2x3SetInput1p(src + row * srcWidth + col, srcWidth, 0, 4, 0, tailW, dst++, dstStride);
                }
                if (row < dstHeight)
                {
                    if (pad)
                        Winograd2x3SetInput1p(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 2)
                        Winograd2x3SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 4, dst++, dstStride);
                    if (col < dstWidth)
                        Winograd2x3SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                }
                src += srcWidth * srcHeight;
            }
        }

        void Winograd2x3SetOutput1(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float c1[16];
            c1[0] = src[0 * srcStride];
            c1[1] = src[1 * srcStride];
            c1[2] = src[2 * srcStride];
            c1[3] = src[3 * srcStride];
            c1[4] = src[4 * srcStride];
            c1[5] = src[5 * srcStride];
            c1[6] = src[6 * srcStride];
            c1[7] = src[7 * srcStride];
            c1[8] = src[8 * srcStride];
            c1[9] = src[9 * srcStride];
            c1[10] = src[10 * srcStride];
            c1[11] = src[11 * srcStride];
            c1[12] = src[12 * srcStride];
            c1[13] = src[13 * srcStride];
            c1[14] = src[14 * srcStride];
            c1[15] = src[15 * srcStride];

            float tmp[8];
            tmp[0] = c1[0] + c1[1] + c1[2];
            tmp[1] = c1[1] - c1[2] - c1[3];
            tmp[2] = c1[4] + c1[5] + c1[6];
            tmp[3] = c1[5] - c1[6] - c1[7];
            tmp[4] = c1[8] + c1[9] + c1[10];
            tmp[5] = c1[9] - c1[10] - c1[11];
            tmp[6] = c1[12] + c1[13] + c1[14];
            tmp[7] = c1[13] - c1[14] - c1[15];

            dst[0 * dstStride + 0] = tmp[0] + tmp[2] + tmp[4];
            dst[0 * dstStride + 1] = tmp[1] + tmp[3] + tmp[5];
            dst[1 * dstStride + 0] = tmp[2] - tmp[4] - tmp[6];
            dst[1 * dstStride + 1] = tmp[3] - tmp[5] - tmp[7];
        }

        void Winograd2x3SetOutput1p(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t rowE, size_t colE)
        {
            float tmp[2 * 2];
            Winograd2x3SetOutput1(src, srcStride, tmp, 2);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row*dstStride + col] = tmp[row * 2 + col];
        }

        void Winograd2x3SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth)
        {
            size_t dstHeightFull = dstHeight / 2 * 2;
            size_t dstWidthFull = dstWidth / 2 * 2;
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 2)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        Winograd2x3SetOutput1(src++, srcStride, dst + row * dstWidth + col, dstWidth);
                    if (col < dstWidth)
                        Winograd2x3SetOutput1p(src++, srcStride, dst + row * dstWidth + col, dstWidth, 2, dstWidth - col);
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        Winograd2x3SetOutput1p(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, 2);
                    if (col < dstWidth)
                        Winograd2x3SetOutput1p(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                }
                dst += dstHeight * dstWidth;
            }
        }

        void Winograd4x3SetFilter(const float * src, size_t srcChannels, size_t dstChannels, float * dst, size_t dstStride)
        {
            size_t size = dstChannels * srcChannels;
            for (size_t i = 0; i < size; i += 1, src += 9, dst += 1)
                Base::Winograd4x3SetFilter1(src, dst, dstStride);
        }

        void Winograd4x3SetInput1(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float tmp1[36];
            tmp1[0] = src[0 * srcStride + 0];
            tmp1[1] = src[0 * srcStride + 1];
            tmp1[2] = src[0 * srcStride + 2];
            tmp1[3] = src[0 * srcStride + 3];
            tmp1[4] = src[0 * srcStride + 4];
            tmp1[5] = src[0 * srcStride + 5];
            tmp1[6] = src[1 * srcStride + 0];
            tmp1[7] = src[1 * srcStride + 1];
            tmp1[8] = src[1 * srcStride + 2];
            tmp1[9] = src[1 * srcStride + 3];
            tmp1[10] = src[1 * srcStride + 4];
            tmp1[11] = src[1 * srcStride + 5];
            tmp1[12] = src[2 * srcStride + 0];
            tmp1[13] = src[2 * srcStride + 1];
            tmp1[14] = src[2 * srcStride + 2];
            tmp1[15] = src[2 * srcStride + 3];
            tmp1[16] = src[2 * srcStride + 4];
            tmp1[17] = src[2 * srcStride + 5];
            tmp1[18] = src[3 * srcStride + 0];
            tmp1[19] = src[3 * srcStride + 1];
            tmp1[20] = src[3 * srcStride + 2];
            tmp1[21] = src[3 * srcStride + 3];
            tmp1[22] = src[3 * srcStride + 4];
            tmp1[23] = src[3 * srcStride + 5];
            tmp1[24] = src[4 * srcStride + 0];
            tmp1[25] = src[4 * srcStride + 1];
            tmp1[26] = src[4 * srcStride + 2];
            tmp1[27] = src[4 * srcStride + 3];
            tmp1[28] = src[4 * srcStride + 4];
            tmp1[29] = src[4 * srcStride + 5];
            tmp1[30] = src[5 * srcStride + 0];
            tmp1[31] = src[5 * srcStride + 1];
            tmp1[32] = src[5 * srcStride + 2];
            tmp1[33] = src[5 * srcStride + 3];
            tmp1[34] = src[5 * srcStride + 4];
            tmp1[35] = src[5 * srcStride + 5];

            float tmp2[36];
            tmp2[0] = 4 * tmp1[0] - 5 * tmp1[12] + tmp1[24];
            tmp2[1] = 4 * tmp1[1] - 5 * tmp1[13] + tmp1[25];
            tmp2[2] = 4 * tmp1[2] - 5 * tmp1[14] + tmp1[26];
            tmp2[3] = 4 * tmp1[3] - 5 * tmp1[15] + tmp1[27];
            tmp2[4] = 4 * tmp1[4] - 5 * tmp1[16] + tmp1[28];
            tmp2[5] = 4 * tmp1[5] - 5 * tmp1[17] + tmp1[29];
            tmp2[6] = -4 * tmp1[6] - 4 * tmp1[12] + tmp1[18] + tmp1[24];
            tmp2[7] = -4 * tmp1[7] - 4 * tmp1[13] + tmp1[19] + tmp1[25];
            tmp2[8] = -4 * tmp1[8] - 4 * tmp1[14] + tmp1[20] + tmp1[26];
            tmp2[9] = -4 * tmp1[9] - 4 * tmp1[15] + tmp1[21] + tmp1[27];
            tmp2[10] = -4 * tmp1[10] - 4 * tmp1[16] + tmp1[22] + tmp1[28];
            tmp2[11] = -4 * tmp1[11] - 4 * tmp1[17] + tmp1[23] + tmp1[29];
            tmp2[12] = 4 * tmp1[6] - 4 * tmp1[12] - tmp1[18] + tmp1[24];
            tmp2[13] = 4 * tmp1[7] - 4 * tmp1[13] - tmp1[19] + tmp1[25];
            tmp2[14] = 4 * tmp1[8] - 4 * tmp1[14] - tmp1[20] + tmp1[26];
            tmp2[15] = 4 * tmp1[9] - 4 * tmp1[15] - tmp1[21] + tmp1[27];
            tmp2[16] = 4 * tmp1[10] - 4 * tmp1[16] - tmp1[22] + tmp1[28];
            tmp2[17] = 4 * tmp1[11] - 4 * tmp1[17] - tmp1[23] + tmp1[29];
            tmp2[18] = -2 * tmp1[6] - tmp1[12] + 2 * tmp1[18] + tmp1[24];
            tmp2[19] = -2 * tmp1[7] - tmp1[13] + 2 * tmp1[19] + tmp1[25];
            tmp2[20] = -2 * tmp1[8] - tmp1[14] + 2 * tmp1[20] + tmp1[26];
            tmp2[21] = -2 * tmp1[9] - tmp1[15] + 2 * tmp1[21] + tmp1[27];
            tmp2[22] = -2 * tmp1[10] - tmp1[16] + 2 * tmp1[22] + tmp1[28];
            tmp2[23] = -2 * tmp1[11] - tmp1[17] + 2 * tmp1[23] + tmp1[29];
            tmp2[24] = 2 * tmp1[6] - tmp1[12] - 2 * tmp1[18] + tmp1[24];
            tmp2[25] = 2 * tmp1[7] - tmp1[13] - 2 * tmp1[19] + tmp1[25];
            tmp2[26] = 2 * tmp1[8] - tmp1[14] - 2 * tmp1[20] + tmp1[26];
            tmp2[27] = 2 * tmp1[9] - tmp1[15] - 2 * tmp1[21] + tmp1[27];
            tmp2[28] = 2 * tmp1[10] - tmp1[16] - 2 * tmp1[22] + tmp1[28];
            tmp2[29] = 2 * tmp1[11] - tmp1[17] - 2 * tmp1[23] + tmp1[29];
            tmp2[30] = 4 * tmp1[6] - 5 * tmp1[18] + tmp1[30];
            tmp2[31] = 4 * tmp1[7] - 5 * tmp1[19] + tmp1[31];
            tmp2[32] = 4 * tmp1[8] - 5 * tmp1[20] + tmp1[32];
            tmp2[33] = 4 * tmp1[9] - 5 * tmp1[21] + tmp1[33];
            tmp2[34] = 4 * tmp1[10] - 5 * tmp1[22] + tmp1[34];
            tmp2[35] = 4 * tmp1[11] - 5 * tmp1[23] + tmp1[35];

            dst[0 * dstStride] = tmp2[0] * 4 - tmp2[2] * 5 + tmp2[4];
            dst[1 * dstStride] = -tmp2[1] * 4 - tmp2[2] * 4 + tmp2[3] + tmp2[4];
            dst[2 * dstStride] = tmp2[1] * 4 - tmp2[2] * 4 - tmp2[3] + tmp2[4];
            dst[3 * dstStride] = -tmp2[1] * 2 - tmp2[2] + tmp2[3] * 2 + tmp2[4];
            dst[4 * dstStride] = tmp2[1] * 2 - tmp2[2] - tmp2[3] * 2 + tmp2[4];
            dst[5 * dstStride] = tmp2[1] * 4 - tmp2[3] * 5 + tmp2[5];
            dst[6 * dstStride] = tmp2[6] * 4 - tmp2[8] * 5 + tmp2[10];
            dst[7 * dstStride] = -tmp2[7] * 4 - tmp2[8] * 4 + tmp2[9] + tmp2[10];
            dst[8 * dstStride] = tmp2[7] * 4 - tmp2[8] * 4 - tmp2[9] + tmp2[10];
            dst[9 * dstStride] = -tmp2[7] * 2 - tmp2[8] + tmp2[9] * 2 + tmp2[10];
            dst[10 * dstStride] = tmp2[7] * 2 - tmp2[8] - tmp2[9] * 2 + tmp2[10];
            dst[11 * dstStride] = tmp2[7] * 4 - tmp2[9] * 5 + tmp2[11];
            dst[12 * dstStride] = tmp2[12] * 4 - tmp2[14] * 5 + tmp2[16];
            dst[13 * dstStride] = -tmp2[13] * 4 - tmp2[14] * 4 + tmp2[15] + tmp2[16];
            dst[14 * dstStride] = tmp2[13] * 4 - tmp2[14] * 4 - tmp2[15] + tmp2[16];
            dst[15 * dstStride] = -tmp2[13] * 2 - tmp2[14] + tmp2[15] * 2 + tmp2[16];
            dst[16 * dstStride] = tmp2[13] * 2 - tmp2[14] - tmp2[15] * 2 + tmp2[16];
            dst[17 * dstStride] = tmp2[13] * 4 - tmp2[15] * 5 + tmp2[17];
            dst[18 * dstStride] = tmp2[18] * 4 - tmp2[20] * 5 + tmp2[22];
            dst[19 * dstStride] = -tmp2[19] * 4 - tmp2[20] * 4 + tmp2[21] + tmp2[22];
            dst[20 * dstStride] = tmp2[19] * 4 - tmp2[20] * 4 - tmp2[21] + tmp2[22];
            dst[21 * dstStride] = -tmp2[19] * 2 - tmp2[20] + tmp2[21] * 2 + tmp2[22];
            dst[22 * dstStride] = tmp2[19] * 2 - tmp2[20] - tmp2[21] * 2 + tmp2[22];
            dst[23 * dstStride] = tmp2[19] * 4 - tmp2[21] * 5 + tmp2[23];
            dst[24 * dstStride] = tmp2[24] * 4 - tmp2[26] * 5 + tmp2[28];
            dst[25 * dstStride] = -tmp2[25] * 4 - tmp2[26] * 4 + tmp2[27] + tmp2[28];
            dst[26 * dstStride] = tmp2[25] * 4 - tmp2[26] * 4 - tmp2[27] + tmp2[28];
            dst[27 * dstStride] = -tmp2[25] * 2 - tmp2[26] + tmp2[27] * 2 + tmp2[28];
            dst[28 * dstStride] = tmp2[25] * 2 - tmp2[26] - tmp2[27] * 2 + tmp2[28];
            dst[29 * dstStride] = tmp2[25] * 4 - tmp2[27] * 5 + tmp2[29];
            dst[30 * dstStride] = tmp2[30] * 4 - tmp2[32] * 5 + tmp2[34];
            dst[31 * dstStride] = -tmp2[31] * 4 - tmp2[32] * 4 + tmp2[33] + tmp2[34];
            dst[32 * dstStride] = tmp2[31] * 4 - tmp2[32] * 4 - tmp2[33] + tmp2[34];
            dst[33 * dstStride] = -tmp2[31] * 2 - tmp2[32] + tmp2[33] * 2 + tmp2[34];
            dst[34 * dstStride] = tmp2[31] * 2 - tmp2[32] - tmp2[33] * 2 + tmp2[34];
            dst[35 * dstStride] = tmp2[31] * 4 - tmp2[33] * 5 + tmp2[35];
        }

        void Winograd4x3SetInput1p(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            float tmp[6 * 6] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 6 + col] = src[row * srcStride + col];
            Winograd4x3SetInput1(tmp, 6, dst, dstStride);
        }

        void Winograd4x3SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, size_t dstStride, int pad)
        {
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstHeightFull = dstHeight / 4 * 4;
            size_t dstWidthFull = dstWidth / 4 * 4;
            size_t noseW = Simd::Min<size_t>(6, dstWidth + 1);
            size_t noseH = Simd::Min<size_t>(6, dstHeight + 1);
            size_t start = pad ? 4 : 0;
            if (pad)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 4;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 4;
                src -= srcWidth + 1;
            }
            size_t tailW = dstWidth - dstWidthFull + (pad ? 1 : 2);
            size_t tailH = dstHeight - dstHeightFull + (pad ? 1 : 2);
            for (size_t c = 0; c < srcChannels; ++c)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        Winograd4x3SetInput1p(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 4)
                        Winograd4x3SetInput1p(src + col, srcWidth, 1, noseH, 0, 6, dst++, dstStride);
                    if (col < dstWidth)
                        Winograd4x3SetInput1p(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                }
                for (row = start; row < dstHeightFull; row += 4)
                {
                    if (pad)
                        Winograd4x3SetInput1p(src + row * srcWidth, srcWidth, 0, 6, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 4)
                        Winograd4x3SetInput1(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                    if (col < dstWidth)
                        Winograd4x3SetInput1p(src + row * srcWidth + col, srcWidth, 0, 6, 0, tailW, dst++, dstStride);
                }
                if (row < dstHeight)
                {
                    if (pad)
                        Winograd4x3SetInput1p(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                    for (col = start; col < dstWidthFull; col += 4)
                        Winograd4x3SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 6, dst++, dstStride);
                    if (col < dstWidth)
                        Winograd4x3SetInput1p(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                }
                src += srcWidth * srcHeight;
            }
        }

        void Winograd4x3SetOutput1(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float c1[36];
            c1[0] = src[0 * srcStride];
            c1[1] = src[1 * srcStride];
            c1[2] = src[2 * srcStride];
            c1[3] = src[3 * srcStride];
            c1[4] = src[4 * srcStride];
            c1[5] = src[5 * srcStride];
            c1[6] = src[6 * srcStride];
            c1[7] = src[7 * srcStride];
            c1[8] = src[8 * srcStride];
            c1[9] = src[9 * srcStride];
            c1[10] = src[10 * srcStride];
            c1[11] = src[11 * srcStride];
            c1[12] = src[12 * srcStride];
            c1[13] = src[13 * srcStride];
            c1[14] = src[14 * srcStride];
            c1[15] = src[15 * srcStride];
            c1[16] = src[16 * srcStride];
            c1[17] = src[17 * srcStride];
            c1[18] = src[18 * srcStride];
            c1[19] = src[19 * srcStride];
            c1[20] = src[20 * srcStride];
            c1[21] = src[21 * srcStride];
            c1[22] = src[22 * srcStride];
            c1[23] = src[23 * srcStride];
            c1[24] = src[24 * srcStride];
            c1[25] = src[25 * srcStride];
            c1[26] = src[26 * srcStride];
            c1[27] = src[27 * srcStride];
            c1[28] = src[28 * srcStride];
            c1[29] = src[29 * srcStride];
            c1[30] = src[30 * srcStride];
            c1[31] = src[31 * srcStride];
            c1[32] = src[32 * srcStride];
            c1[33] = src[33 * srcStride];
            c1[34] = src[34 * srcStride];
            c1[35] = src[35 * srcStride];

            float tmp[24];
            tmp[0] = c1[0] + c1[6] + c1[12] + c1[18] + c1[24];
            tmp[1] = c1[1] + c1[7] + c1[13] + c1[19] + c1[25];
            tmp[2] = c1[2] + c1[8] + c1[14] + c1[20] + c1[26];
            tmp[3] = c1[3] + c1[9] + c1[15] + c1[21] + c1[27];
            tmp[4] = c1[4] + c1[10] + c1[16] + c1[22] + c1[28];
            tmp[5] = c1[5] + c1[11] + c1[17] + c1[23] + c1[29];
            tmp[6] = c1[6] - c1[12] + 2 * c1[18] - 2 * c1[24];
            tmp[7] = c1[7] - c1[13] + 2 * c1[19] - 2 * c1[25];
            tmp[8] = c1[8] - c1[14] + 2 * c1[20] - 2 * c1[26];
            tmp[9] = c1[9] - c1[15] + 2 * c1[21] - 2 * c1[27];
            tmp[10] = c1[10] - c1[16] + 2 * c1[22] - 2 * c1[28];
            tmp[11] = c1[11] - c1[17] + 2 * c1[23] - 2 * c1[29];
            tmp[12] = c1[6] + c1[12] + 4 * c1[18] + 4 * c1[24];
            tmp[13] = c1[7] + c1[13] + 4 * c1[19] + 4 * c1[25];
            tmp[14] = c1[8] + c1[14] + 4 * c1[20] + 4 * c1[26];
            tmp[15] = c1[9] + c1[15] + 4 * c1[21] + 4 * c1[27];
            tmp[16] = c1[10] + c1[16] + 4 * c1[22] + 4 * c1[28];
            tmp[17] = c1[11] + c1[17] + 4 * c1[23] + 4 * c1[29];
            tmp[18] = c1[6] - c1[12] + 8 * c1[18] - 8 * c1[24] + c1[30];
            tmp[19] = c1[7] - c1[13] + 8 * c1[19] - 8 * c1[25] + c1[31];
            tmp[20] = c1[8] - c1[14] + 8 * c1[20] - 8 * c1[26] + c1[32];
            tmp[21] = c1[9] - c1[15] + 8 * c1[21] - 8 * c1[27] + c1[33];
            tmp[22] = c1[10] - c1[16] + 8 * c1[22] - 8 * c1[28] + c1[34];
            tmp[23] = c1[11] - c1[17] + 8 * c1[23] - 8 * c1[29] + c1[35];

            dst[0 * dstStride + 0] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4];
            dst[0 * dstStride + 1] = tmp[1] - tmp[2] + 2 * tmp[3] - 2 * tmp[4];
            dst[0 * dstStride + 2] = tmp[1] + tmp[2] + 4 * tmp[3] + 4 * tmp[4];
            dst[0 * dstStride + 3] = tmp[1] - tmp[2] + 8 * tmp[3] - 8 * tmp[4] + tmp[5];
            dst[1 * dstStride + 0] = tmp[6] + tmp[7] + tmp[8] + tmp[9] + tmp[10];
            dst[1 * dstStride + 1] = tmp[7] - tmp[8] + 2 * tmp[9] - 2 * tmp[10];
            dst[1 * dstStride + 2] = tmp[7] + tmp[8] + 4 * tmp[9] + 4 * tmp[10];
            dst[1 * dstStride + 3] = tmp[7] - tmp[8] + 8 * tmp[9] - 8 * tmp[10] + tmp[11];
            dst[2 * dstStride + 0] = tmp[12] + tmp[13] + tmp[14] + tmp[15] + tmp[16];
            dst[2 * dstStride + 1] = tmp[13] - tmp[14] + 2 * tmp[15] - 2 * tmp[16];
            dst[2 * dstStride + 2] = tmp[13] + tmp[14] + 4 * tmp[15] + 4 * tmp[16];
            dst[2 * dstStride + 3] = tmp[13] - tmp[14] + 8 * tmp[15] - 8 * tmp[16] + tmp[17];
            dst[3 * dstStride + 0] = tmp[18] + tmp[19] + tmp[20] + tmp[21] + tmp[22];
            dst[3 * dstStride + 1] = tmp[19] - tmp[20] + 2 * tmp[21] - 2 * tmp[22];
            dst[3 * dstStride + 2] = tmp[19] + tmp[20] + 4 * tmp[21] + 4 * tmp[22];
            dst[3 * dstStride + 3] = tmp[19] - tmp[20] + 8 * tmp[21] - 8 * tmp[22] + tmp[23];
        }

        void Winograd4x3SetOutput1p(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t rowE, size_t colE)
        {
            float tmp[4 * 4];
            Winograd4x3SetOutput1(src, srcStride, tmp, 4);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row*dstStride + col] = tmp[row * 4 + col];
        }

        void Winograd4x3SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth)
        {
            size_t dstHeightFull = dstHeight / 4 * 4;
            size_t dstWidthFull = dstWidth / 4 * 4;
            for (size_t c = 0; c < dstChannels; ++c)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 4)
                {
                    for (col = 0; col < dstWidthFull; col += 4)
                        Winograd4x3SetOutput1(src++, srcStride, dst + row * dstWidth + col, dstWidth);
                    if (col < dstWidth)
                        Winograd4x3SetOutput1p(src++, srcStride, dst + row * dstWidth + col, dstWidth, 4, dstWidth - col);
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 4)
                        Winograd4x3SetOutput1p(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, 4);
                    if (col < dstWidth)
                        Winograd4x3SetOutput1p(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                }
                dst += dstHeight * dstWidth;
            }
        }
    }
}
