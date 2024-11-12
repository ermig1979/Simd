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
#ifndef __SimdImageLoadJpeg_h__
#define __SimdImageLoadJpeg_h__

#include "Simd/SimdImageLoad.h"

namespace Simd
{
    namespace Base
    {
        const int JpegFastBits = 9;
        const int JpegMaxDimensions = 1 << 24;

        const int JpegMarkerNone = 0xFF;
        const int JpegMarkerSoi = 0xD8;
        const int JpegMarkerEoi = 0xD9;
        const int JpegMarkerSos = 0xDA;
        const int JpegMarkerDnl = 0xDC;

        extern const uint8_t JpegDeZigZag[80];

        //-------------------------------------------------------------------------------------------------

        struct JpegHuffman
        {
            uint8_t  fast[1 << JpegFastBits];
            uint16_t code[256];
            uint8_t  values[256];
            uint8_t  size[257];
            unsigned int maxcode[18];
            int delta[17];
            int16_t fast_ac[1 << JpegFastBits];

            int Build(const int* count);
            void BuildFastAc();
        };

        //-------------------------------------------------------------------------------------------------

        struct JpegImgComp
        {
            int id;
            int h, v;
            int tq;
            int hd, ha;
            int dc_pred;
            int x, y, w2, h2;
            Array8u bufD, bufL;
            uint8_t* data;
            Array16i bufC;
            short* coeff;
            int coeffW, coeffH;
        };

        //-------------------------------------------------------------------------------------------------

        typedef void (*IdctBlockPtr)(const int16_t * src, uint8_t* dst, int stride);
        typedef uint8_t* (*ResampleRowPtr)(uint8_t* out, const uint8_t* in0, const uint8_t* in1, int w, int hs);
        typedef void (*YuvToRgbRowPtr)(uint8_t* out, const uint8_t* y, const uint8_t* pcb, const uint8_t* pcr, int count, int step);
        typedef void (*YuvToBgrPtr)(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType);
        typedef void (*YuvToBgraPtr)(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, uint8_t alpha, SimdYuvType yuvType);
        typedef void (*AnyToAnyPtr)(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride);

        //-------------------------------------------------------------------------------------------------

        struct JpegContext
        {
            InputMemoryStream* stream;
            uint32_t img_x, img_y;
            int img_n, img_out_n;
            JpegHuffman huff_dc[4];
            JpegHuffman huff_ac[4];
            uint16_t dequant[4][64];

            int img_h_max, img_v_max;
            int img_mcu_x, img_mcu_y;
            int img_mcu_w, img_mcu_h;

            JpegImgComp img_comp[4];

            uint32_t  code_buffer;
            int code_bits;
            unsigned char marker;
            int nomore;

            int progressive;
            int spec_start;
            int spec_end;
            int succ_high;
            int succ_low;
            int eob_run;
            int jfif;
            int app14_color_transform;
            int rgb;

            int scan_n, order[4];
            int restart_interval, todo;

            Array8u out;

            IdctBlockPtr idctBlock;
            ResampleRowPtr resampleRowHv2;
            YuvToRgbRowPtr yuvToRgbRow;

            YuvToBgrPtr yuv444pToBgr, yuv420pToBgr;
            YuvToBgraPtr yuv444pToBgra, yuv420pToBgra;
            AnyToAnyPtr rgbaToAny;

            JpegContext(InputMemoryStream* s);
            void Reset();

            SIMD_INLINE bool NeedRestart() const
            {
                return marker >= 0xd0 && marker <= 0xd7;
            }
        };

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int JpegIdctConst(float value)
        {
            return int(value * 4096.0f + 0.5f);
        }

        const int JpegIdctK00 = JpegIdctConst(0.5411961f);
        const int JpegIdctK01 = JpegIdctConst(-1.847759065f);
        const int JpegIdctK02 = JpegIdctConst(0.765366865f);
        const int JpegIdctK03 = JpegIdctConst(1.175875602f);
        const int JpegIdctK04 = JpegIdctConst(0.298631336f);
        const int JpegIdctK05 = JpegIdctConst(2.053119869f);
        const int JpegIdctK06 = JpegIdctConst(3.072711026f);
        const int JpegIdctK07 = JpegIdctConst(1.501321110f);
        const int JpegIdctK08 = JpegIdctConst(-0.899976223f);
        const int JpegIdctK09 = JpegIdctConst(-2.562915447f);
        const int JpegIdctK10 = JpegIdctConst(-1.961570560f);
        const int JpegIdctK11 = JpegIdctConst(-0.390180644f);

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE int JpegLoadError(const char* text, const char* type)
        {
            std::cout << "JPEG load error: " << text << ", " << type << "!" << std::endl;
            return 0;
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif
}

#endif
