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
#include "Simd/SimdImageSave.h"
#include "Simd/SimdImageSaveJpeg.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        const uint8_t JpegZigZagD[64] = { 
            0, 1, 5, 6, 14, 15, 27, 28, 
            2, 4, 7, 13, 16, 26, 29, 42, 
            3, 8, 12, 17, 25, 30, 41, 43, 
            9, 11, 18, 24, 31, 40, 44, 53, 
            10, 19, 23, 32, 39, 45, 52, 54, 
            20, 22, 33, 38, 46, 51, 55, 60, 
            21, 34, 37, 47, 50, 56, 59, 61, 
            35, 36, 48, 49, 57, 58, 62, 63 };

        const uint8_t JpegZigZagT[64] = { 
            0, 2, 3, 9, 10, 20, 21, 35,
            1, 4, 8, 11, 19, 22, 34, 36,
            5, 7, 12, 18, 23, 33, 37, 48,
            6, 13, 17, 24, 32, 38, 47, 49,
            14, 16, 25, 31, 39, 46, 50, 57,
            15, 26, 30, 40, 45, 51, 56, 58,
            27, 29, 41, 44, 52, 55, 59, 62,
            28, 42, 43, 53, 54, 60, 61, 63 };        

        const uint16_t HuffmanYdc[256][2] = { {0, 2}, {2, 3}, {3, 3}, {4, 3}, {5, 3}, {6, 3}, {14, 4}, {30, 5}, {62, 6}, {126, 7}, {254, 8}, {510, 9} };
        const uint16_t HuffmanUVdc[256][2] = { {0, 2}, {1, 2}, {2, 2}, {6, 3}, {14, 4}, {30, 5}, {62, 6}, {126, 7}, {254, 8}, {510, 9}, {1022, 10}, {2046, 11} };
        const uint16_t HuffmanYac[256][2] = {
           {10, 4}, {0, 2}, {1, 2}, {4, 3}, {11, 4}, {26, 5}, {120, 7}, {248, 8}, {1014, 10}, {65410, 16}, {65411, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {12, 4}, {27, 5}, {121, 7}, {502, 9}, {2038, 11}, {65412, 16}, {65413, 16}, {65414, 16}, {65415, 16}, {65416, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {28, 5}, {249, 8}, {1015, 10}, {4084, 12}, {65417, 16}, {65418, 16}, {65419, 16}, {65420, 16}, {65421, 16}, {65422, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {58, 6}, {503, 9}, {4085, 12}, {65423, 16}, {65424, 16}, {65425, 16}, {65426, 16}, {65427, 16}, {65428, 16}, {65429, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {59, 6}, {1016, 10}, {65430, 16}, {65431, 16}, {65432, 16}, {65433, 16}, {65434, 16}, {65435, 16}, {65436, 16}, {65437, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {122, 7}, {2039, 11}, {65438, 16}, {65439, 16}, {65440, 16}, {65441, 16}, {65442, 16}, {65443, 16}, {65444, 16}, {65445, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {123, 7}, {4086, 12}, {65446, 16}, {65447, 16}, {65448, 16}, {65449, 16}, {65450, 16}, {65451, 16}, {65452, 16}, {65453, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {250, 8}, {4087, 12}, {65454, 16}, {65455, 16}, {65456, 16}, {65457, 16}, {65458, 16}, {65459, 16}, {65460, 16}, {65461, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {504, 9}, {32704, 15}, {65462, 16}, {65463, 16}, {65464, 16}, {65465, 16}, {65466, 16}, {65467, 16}, {65468, 16}, {65469, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {505, 9}, {65470, 16}, {65471, 16}, {65472, 16}, {65473, 16}, {65474, 16}, {65475, 16}, {65476, 16}, {65477, 16}, {65478, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {506, 9}, {65479, 16}, {65480, 16}, {65481, 16}, {65482, 16}, {65483, 16}, {65484, 16}, {65485, 16}, {65486, 16}, {65487, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {1017, 10}, {65488, 16}, {65489, 16}, {65490, 16}, {65491, 16}, {65492, 16}, {65493, 16}, {65494, 16}, {65495, 16}, {65496, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {1018, 10}, {65497, 16}, {65498, 16}, {65499, 16}, {65500, 16}, {65501, 16}, {65502, 16}, {65503, 16}, {65504, 16}, {65505, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {2040, 11}, {65506, 16}, {65507, 16}, {65508, 16}, {65509, 16}, {65510, 16}, {65511, 16}, {65512, 16}, {65513, 16}, {65514, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {65515, 16}, {65516, 16}, {65517, 16}, {65518, 16}, {65519, 16}, {65520, 16}, {65521, 16}, {65522, 16}, {65523, 16}, {65524, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {2041, 11}, {65525, 16}, {65526, 16}, {65527, 16}, {65528, 16}, {65529, 16}, {65530, 16}, {65531, 16}, {65532, 16}, {65533, 16}, {65534, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}
        };
        const uint16_t HuffmanUVac[256][2] = {
           {0, 2}, {1, 2}, {4, 3}, {10, 4}, {24, 5}, {25, 5}, {56, 6}, {120, 7}, {500, 9}, {1014, 10}, {4084, 12}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {11, 4}, {57, 6}, {246, 8}, {501, 9}, {2038, 11}, {4085, 12}, {65416, 16}, {65417, 16}, {65418, 16}, {65419, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {26, 5}, {247, 8}, {1015, 10}, {4086, 12}, {32706, 15}, {65420, 16}, {65421, 16}, {65422, 16}, {65423, 16}, {65424, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {27, 5}, {248, 8}, {1016, 10}, {4087, 12}, {65425, 16}, {65426, 16}, {65427, 16}, {65428, 16}, {65429, 16}, {65430, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {58, 6}, {502, 9}, {65431, 16}, {65432, 16}, {65433, 16}, {65434, 16}, {65435, 16}, {65436, 16}, {65437, 16}, {65438, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {59, 6}, {1017, 10}, {65439, 16}, {65440, 16}, {65441, 16}, {65442, 16}, {65443, 16}, {65444, 16}, {65445, 16}, {65446, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {121, 7}, {2039, 11}, {65447, 16}, {65448, 16}, {65449, 16}, {65450, 16}, {65451, 16}, {65452, 16}, {65453, 16}, {65454, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {122, 7}, {2040, 11}, {65455, 16}, {65456, 16}, {65457, 16}, {65458, 16}, {65459, 16}, {65460, 16}, {65461, 16}, {65462, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {249, 8}, {65463, 16}, {65464, 16}, {65465, 16}, {65466, 16}, {65467, 16}, {65468, 16}, {65469, 16}, {65470, 16}, {65471, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {503, 9}, {65472, 16}, {65473, 16}, {65474, 16}, {65475, 16}, {65476, 16}, {65477, 16}, {65478, 16}, {65479, 16}, {65480, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {504, 9}, {65481, 16}, {65482, 16}, {65483, 16}, {65484, 16}, {65485, 16}, {65486, 16}, {65487, 16}, {65488, 16}, {65489, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {505, 9}, {65490, 16}, {65491, 16}, {65492, 16}, {65493, 16}, {65494, 16}, {65495, 16}, {65496, 16}, {65497, 16}, {65498, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {506, 9}, {65499, 16}, {65500, 16}, {65501, 16}, {65502, 16}, {65503, 16}, {65504, 16}, {65505, 16}, {65506, 16}, {65507, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {2041, 11}, {65508, 16}, {65509, 16}, {65510, 16}, {65511, 16}, {65512, 16}, {65513, 16}, {65514, 16}, {65515, 16}, {65516, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {16352, 14}, {65517, 16}, {65518, 16}, {65519, 16}, {65520, 16}, {65521, 16}, {65522, 16}, {65523, 16}, {65524, 16}, {65525, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, 
           {1018, 10}, {32707, 15}, {65526, 16}, {65527, 16}, {65528, 16}, {65529, 16}, {65530, 16}, {65531, 16}, {65532, 16}, {65533, 16}, {65534, 16}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}
        };

#if defined(SIMD_JPEG_CALC_BITS_TABLE)
        uint16_t JpegCalcBitsTable[JpegCalcBitsRange * 2][2];
        bool JpegCalcBitsTableInit()
        {
            for (int i = 0, n = JpegCalcBitsRange * 2; i < n; ++i)
            {
                int val = i - JpegCalcBitsRange;
                int tmp = val < 0 ? -val : val;
                val = val < 0 ? val - 1 : val;
                int cnt = 1;
                while (tmp >>= 1)
                    ++cnt;
                JpegCalcBitsTable[i][0] = val & ((1 << cnt) - 1);
                JpegCalcBitsTable[i][1] = cnt;
            }
            return true;
        }
        bool JpegCalcBitsTableInited = JpegCalcBitsTableInit();
#endif

        SIMD_INLINE void JpegDct(float* d0p, float* d1p, float* d2p, float* d3p, float* d4p, float* d5p, float* d6p, float* d7p)
        {
            float d0 = *d0p, d1 = *d1p, d2 = *d2p, d3 = *d3p, d4 = *d4p, d5 = *d5p, d6 = *d6p, d7 = *d7p;
            float z1, z2, z3, z4, z5, z11, z13;
            float tmp0 = d0 + d7;
            float tmp7 = d0 - d7;
            float tmp1 = d1 + d6;
            float tmp6 = d1 - d6;
            float tmp2 = d2 + d5;
            float tmp5 = d2 - d5;
            float tmp3 = d3 + d4;
            float tmp4 = d3 - d4;

            float tmp10 = tmp0 + tmp3;
            float tmp13 = tmp0 - tmp3;
            float tmp11 = tmp1 + tmp2;
            float tmp12 = tmp1 - tmp2;

            d0 = tmp10 + tmp11;
            d4 = tmp10 - tmp11;

            z1 = (tmp12 + tmp13) * 0.707106781f;
            d2 = tmp13 + z1;
            d6 = tmp13 - z1;

            tmp10 = tmp4 + tmp5;
            tmp11 = tmp5 + tmp6;
            tmp12 = tmp6 + tmp7;

            z5 = (tmp10 - tmp12) * 0.382683433f;
            z2 = tmp10 * 0.541196100f + z5;
            z4 = tmp12 * 1.306562965f + z5;
            z3 = tmp11 * 0.707106781f;

            z11 = tmp7 + z3;
            z13 = tmp7 - z3;

            *d5p = z13 + z2;
            *d3p = z13 - z2;
            *d1p = z11 + z4;
            *d7p = z11 - z4;

            *d0p = d0;  *d2p = d2;  *d4p = d4;  *d6p = d6;
        }

        static int JpegProcessDu(Base::BitBuf& bitBuf, float* CDU, int stride, const float* fdtbl, int DC, const uint16_t HTDC[256][2], const uint16_t HTAC[256][2])
        {
            int offs, i, j, n, diff, end0pos, x, y;
            for (offs = 0; offs < 8; ++offs) 
                JpegDct(&CDU[offs], &CDU[offs + stride], &CDU[offs + stride * 2], &CDU[offs + stride * 3], &CDU[offs + stride * 4],
                    &CDU[offs + stride * 5], &CDU[offs + stride * 6], &CDU[offs + stride * 7]);
            for (offs = 0, n = stride * 8; offs < n; offs += stride)
                JpegDct(&CDU[offs], &CDU[offs + 1], &CDU[offs + 2], &CDU[offs + 3], &CDU[offs + 4], &CDU[offs + 5], &CDU[offs + 6], &CDU[offs + 7]);
            int DU[64];
            for (y = 0, j = 0; y < 8; ++y) 
            {
                for (x = 0; x < 8; ++x, ++j) 
                {
                    i = y * stride + x;
                    float v = CDU[i] * fdtbl[j];
                    DU[JpegZigZagD[j]] = Round(v);
                }
            }
            diff = DU[0] - DC;
            if (diff == 0) 
                bitBuf.Push(HTDC[0]);
            else 
            {
                uint16_t bits[2];
                JpegCalcBits(diff, bits);
                bitBuf.Push(HTDC[bits[1]]);
                bitBuf.Push(bits);
            }
            end0pos = 63;
            for (; (end0pos > 0) && (DU[end0pos] == 0); --end0pos);
            if (end0pos == 0) 
            {
                bitBuf.Push(HTAC[0x00]);
                return DU[0];
            }
            for (i = 1; i <= end0pos; ++i)
            {
                int startpos = i;
                int nrzeroes;
                uint16_t bits[2];
                for (; DU[i] == 0 && i <= end0pos; ++i);
                nrzeroes = i - startpos;
                if (nrzeroes >= 16) 
                {
                    int lng = nrzeroes >> 4;
                    int nrmarker;
                    for (nrmarker = 1; nrmarker <= lng; ++nrmarker)
                        bitBuf.Push(HTAC[0xF0]);
                    nrzeroes &= 15;
                }
                JpegCalcBits(DU[i], bits);
                bitBuf.Push(HTAC[(nrzeroes << 4) + bits[1]]);
                bitBuf.Push(bits);
            }
            if (end0pos != 63) 
                bitBuf.Push(HTAC[0x00]);
            return DU[0];
        }

        void JpegWriteBlockSubs(OutputMemoryStream & stream, int width, int height, const uint8_t * red,
            const uint8_t* green, const uint8_t* blue, int stride, const float * fY, const float* fUv, int dc[3])
        {
            int & DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            float Y[256], U[256], V[256];
            float subU[64], subV[64];
            bool gray = red == green && red == blue;
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 16)
            {
                for (int x = 0; x < width; x += 16)
                {
                    if (gray)
                        Base::GrayToY(red + x, stride, height - y, width - x, Y, 16);
                    else
                        Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 16);
                    DCY = JpegProcessDu(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        for (int yy = 0, pos = 0; yy < 8; ++yy)
                        {
                            for (int xx = 0; xx < 8; ++xx, ++pos)
                            {
                                int j = yy * 32 + xx * 2;
                                subU[pos] = (U[j + 0] + U[j + 1] + U[j + 16] + U[j + 17]) * 0.25f;
                                subV[pos] = (V[j + 0] + V[j + 1] + V[j + 16] + V[j + 17]) * 0.25f;
                            }
                        }
                        DCU = JpegProcessDu(bitBuf, subU, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, subV, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        Base::WriteBits(stream, bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
            }
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        void JpegWriteBlockFull(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            float Y[64], U[64], V[64];
            bool gray = red == green && red == blue;
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 8)
            {
                for (int x = 0; x < width; x += 8)
                {
                    if (gray)
                        Base::GrayToY(red + x, stride, height - y, width - x, Y, 8);
                    else
                        Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 8);
                    DCY = JpegProcessDu(bitBuf, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        Base::WriteBits(stream, bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
            }
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        void JpegWriteBlockNv12(OutputMemoryStream& stream, int width, int height, const uint8_t* ySrc, int yStride,
            const uint8_t* uvSrc, int uvStride, const float* fY, const float* fUv, int dc[3])
        {
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            float Y[256], U[64], V[64];
            bool gray = (uvSrc == NULL);
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 16)
            {
                for (int x = 0; x < width; x += 16)
                {
                    Base::GrayToY(ySrc + x, yStride, height - y, width - x, Y, 16);
                    DCY = JpegProcessDu(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        Nv12ToUv(uvSrc + x, uvStride, UvSize(height - y), UvSize(width - x), U, V);
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        Base::WriteBits(stream, bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
            }
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        void JpegWriteBlockYuv420p(OutputMemoryStream& stream, int width, int height, const uint8_t* ySrc, int yStride,
            const uint8_t* uSrc, int uStride, const uint8_t* vSrc, int vStride, const float* fY, const float* fUv, int dc[3])
        {
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            float Y[256], U[64], V[64];
            bool gray = (uSrc == NULL || vSrc == NULL);
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 16)
            {
                for (int x = 0; x < width; x += 16)
                {
                    Base::GrayToY(ySrc + x, yStride, height - y, width - x, Y, 16);
                    DCY = JpegProcessDu(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if(gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        Base::GrayToY(uSrc + UvSize(x), uStride, UvSize(height - y), UvSize(width - x), U, 8);
                        Base::GrayToY(vSrc + UvSize(x), vStride, UvSize(height - y), UvSize(width - x), V, 8);
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        Base::WriteBits(stream, bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
            }
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        //---------------------------------------------------------------------

        ImageJpegSaver::ImageJpegSaver(const ImageSaverParam& param)
            : ImageSaver(param)
            , _deintBgra(NULL)
            , _deintBgr(NULL)
            , _writeBlock(NULL)
            , _writeNv12Block(NULL)
            , _writeYuv420pBlock(NULL)
        {
        }

        void ImageJpegSaver::Init()
        {
            InitParams(false);
            if (_param.yuvType == SimdYuvUnknown)
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24:
                case SimdPixelFormatRgb24:
                    _deintBgr = Base::DeinterleaveBgr;
                    break;
                case SimdPixelFormatBgra32:
                case SimdPixelFormatRgba32:
                    _deintBgra = Base::DeinterleaveBgra;
                    break;
                default:
                    break;
                }
                _writeBlock = _subSample ? JpegWriteBlockSubs : JpegWriteBlockFull;
            }
            else
            {
                _writeNv12Block = JpegWriteBlockNv12;
                _writeYuv420pBlock = JpegWriteBlockYuv420p;
            }
        }

        void ImageJpegSaver::InitParams(bool trans)
        {
            static const int YQT[] = { 16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 
                16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 
                35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99 };
            static const int UVQT[] = { 17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 
                26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 
                99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99 };
            static const float AASF[] = { 1.0f * 2.828427125f, 1.387039845f * 2.828427125f, 
                1.306562965f * 2.828427125f, 1.175875602f * 2.828427125f, 1.0f * 2.828427125f, 
                0.785694958f * 2.828427125f, 0.541196100f * 2.828427125f, 0.275899379f * 2.828427125f };
            _quality = _param.quality;
            _quality = _quality ? _quality : 90;
            _subSample = (_quality <= 90 || _param.yuvType != SimdYuvUnknown) ? 1 : 0;
            _quality = _quality < 1 ? 1 : _quality > 100 ? 100 : _quality;
            _quality = _quality < 50 ? 5000 / _quality : 200 - _quality * 2;
            for (size_t i = 0; i < 64; ++i)
            {
                int uvti, yti = (YQT[i] * _quality + 50) / 100;
                _uY[Base::JpegZigZagD[i]] = uint8_t(yti < 1 ? 1 : yti > 255 ? 255 : yti);
                uvti = (UVQT[i] * _quality + 50) / 100;
                _uUv[Base::JpegZigZagD[i]] = uint8_t(uvti < 1 ? 1 : uvti > 255 ? 255 : uvti);
            }
            const uint8_t *ZigZag = trans ? Base::JpegZigZagT : Base::JpegZigZagD;
            for (size_t y = 0, i = 0; y < 8; ++y)
            {
                for (size_t x = 0; x < 8; ++x, ++i)
                {
                    _fY[i] = 1.0f / (_uY[ZigZag[i]] * AASF[y] * AASF[x]);
                    _fUv[i] = 1.0f / (_uUv[ZigZag[i]] * AASF[y] * AASF[x]);
                }
            }
            _block = _subSample ? 16 : 8;
            _width = (int)AlignHi(_param.width, _block);
            if (_param.format != SimdPixelFormatGray8 && _param.yuvType == SimdYuvUnknown)
                _buffer.Resize(_width * _block * 3);
        }

        void ImageJpegSaver::WriteHeader()
        {
            static const uint8_t DC_LUM_COD[] = { 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
            static const uint8_t DC_LUM_VAL[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            static const uint8_t AC_LUM_COD[] = { 0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
            static const uint8_t AC_LUM_VAL[] = {
               0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 
               0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 
               0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 
               0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 
               0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 
               0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 
               0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa
            };
            static const uint8_t DC_CHR_COD[] = { 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
            static const uint8_t DC_CHR_VAL[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            static const uint8_t AC_CHR_COD[] = { 0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
            static const uint8_t AC_CHR_VAL[] = {
               0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 
               0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 
               0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 
               0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 
               0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 
               0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 
               0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa
            };
            static const uint8_t head0[] = { 0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10, 'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0xFF, 0xDB, 0, 0x84, 0 };
            static const uint8_t head2[] = { 0xFF, 0xDA, 0, 0xC, 3, 1, 0, 2, 0x11, 3, 0x11, 0, 0x3F, 0 };
            const uint8_t head1[] = { 0xFF, 0xC0, 0, 0x11, 8,  uint8_t(_param.height >> 8),  uint8_t(_param.height),  uint8_t(_param.width >> 8),  
                uint8_t(_param.width), 3, 1, uint8_t(_subSample ? 0x22 : 0x11), 0, 2, 0x11, 1, 3, 0x11, 1, 0xFF, 0xC4, 0x01, 0xA2, 0 };
            _stream.Write(head0, sizeof(head0));
            _stream.Write(_uY, 64);
            _stream.Write8u(1);
            _stream.Write(_uUv, 64);
            _stream.Write(head1, sizeof(head1));
            _stream.Write(DC_LUM_COD + 1, sizeof(DC_LUM_COD) - 1);
            _stream.Write(DC_LUM_VAL, sizeof(DC_LUM_VAL));
            _stream.Write8u(0x10); // HTYACinfo
            _stream.Write(AC_LUM_COD + 1, sizeof(AC_LUM_COD) - 1);
            _stream.Write(AC_LUM_VAL, sizeof(AC_LUM_VAL));
            _stream.Write8u(1); // HTUDCinfo
            _stream.Write(DC_CHR_COD + 1, sizeof(DC_CHR_COD) - 1);
            _stream.Write(DC_CHR_VAL, sizeof(DC_CHR_VAL));
            _stream.Write8u(0x11); // HTUACinfo
            _stream.Write(AC_CHR_COD + 1, sizeof(AC_CHR_COD) - 1);
            _stream.Write(AC_CHR_VAL, sizeof(AC_CHR_VAL));
            _stream.Write(head2, sizeof(head2));
        }

        bool ImageJpegSaver::ToStream(const uint8_t* src, size_t stride)
        {
            Init();
            WriteHeader();
            uint8_t* r = _buffer.data, * g = r + _width * _block,* b = g + _width * _block;
            int dc[3] = { 0, 0, 0 };
            for (int row = 0; row < (int)_param.height; row += _block)
            {
                int block = Simd::Min(row + _block, (int)_param.height) - row;
                switch (_param.format)
                {
                case SimdPixelFormatBgr24:
                    _deintBgr(src, stride, _param.width, block, b, _width, g, _width, r, _width);
                    break;
                case SimdPixelFormatBgra32:
                    _deintBgra(src, stride, _param.width, block, b, _width, g, _width, r, _width, NULL, 0);
                    break;
                case SimdPixelFormatRgb24:
                    _deintBgr(src, stride, _param.width, block, r, _width, g, _width, b, _width);
                    break;
                case SimdPixelFormatRgba32:
                    _deintBgra(src, stride, _param.width, block, r, _width, g, _width, b, _width, NULL, 0);
                    break;
                default: 
                    break;
                }
                if(_param.format == SimdPixelFormatGray8)
                    _writeBlock(_stream, (int)_param.width, block, src, src, src, (int)stride, _fY, _fUv, dc);
                else
                    _writeBlock(_stream, (int)_param.width, block, r, g, b, _width, _fY, _fUv, dc);
                src += block * stride;
            }
            static const uint16_t FILL_BITS[] = { 0x7F, 7 };
            Base::WriteBits(_stream, FILL_BITS);
            _stream.Write8u(0xFF);
            _stream.Write8u(0xD9);
            return true;
        }

        bool ImageJpegSaver::ToStream(const uint8_t* y, size_t yStride, const uint8_t* uv, size_t uvStride)
        {
            Init();
            WriteHeader();
            int dc[3] = { 0, 0, 0 };
            for (int row = 0; row < (int)_param.height; row += _block)
            {
                int block = Simd::Min(row + _block, (int)_param.height) - row;
                _writeNv12Block(_stream, (int)_param.width, block, y, (int)yStride, uv, (int)uvStride, _fY, _fUv, dc);
                y += block * yStride;
                uv += (block / 2) * uvStride;
            }
            static const uint16_t FILL_BITS[] = { 0x7F, 7 };
            Base::WriteBits(_stream, FILL_BITS);
            _stream.Write8u(0xFF);
            _stream.Write8u(0xD9);
            return true;
        }

        bool ImageJpegSaver::ToStream(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride)
        {
            Init();
            WriteHeader();
            int dc[3] = { 0, 0, 0 };
            for (int row = 0; row < (int)_param.height; row += _block)
            {
                int block = Simd::Min(row + _block, (int)_param.height) - row;
                _writeYuv420pBlock(_stream, (int)_param.width, block, y, (int)yStride, u, (int)uStride, v, (int)vStride, _fY, _fUv, dc);
                y += block * yStride;
                u += (block / 2) * uStride;
                v += (block / 2) * vStride;
            }
            static const uint16_t FILL_BITS[] = { 0x7F, 7 };
            Base::WriteBits(_stream, FILL_BITS);
            _stream.Write8u(0xFF);
            _stream.Write8u(0xD9);
            return true;
        }

        //-----------------------------------------------------------------------------------------

        uint8_t* Nv12SaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* uv, size_t uvStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size)
        {
            ImageSaverParam param(width, height, quality, yuvType);
            if (param.Validate())
            {
                Holder<ImageJpegSaver> saver(new ImageJpegSaver(param));
                if (saver)
                {
                    if (saver->ToStream(y, yStride, uv, uvStride))
                        return saver->Release(size);
                }
            }
            return NULL;
        }

        uint8_t* Yuv420pSaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size)
        {
            ImageSaverParam param(width, height, quality, yuvType);
            if (param.Validate())
            {
                Holder<ImageJpegSaver> saver(new ImageJpegSaver(param));
                if (saver)
                {
                    if (saver->ToStream(y, yStride, u, uStride, v, vStride))
                        return saver->Release(size);
                }
            }
            return NULL;
        }
    }
}
