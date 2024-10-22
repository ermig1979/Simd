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

        typedef uint8_t* (*ResampleRowPtr)(uint8_t* out, uint8_t* in0, uint8_t* in1, int w, int hs);
        typedef void (*YuvToBgrPtr)(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType);
        typedef void (*YuvToBgraPtr)(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, uint8_t alpha, SimdYuvType yuvType);
        typedef void (*AnyToAnyPtr)(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride);

        //-------------------------------------------------------------------------------------------------

        struct JpegContext
        {
            JpegContext(InputMemoryStream* s)
                : stream(s)
                , img_n(0)
            {
            }

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

            void (*idct_block_kernel)(uint8_t* out, int out_stride, short data[64]);
            void (*YCbCr_to_RGB_kernel)(uint8_t* out, const uint8_t* y, const uint8_t* pcb, const uint8_t* pcr, int count, int step);
            uint8_t* (*resample_row_hv_2_kernel)(uint8_t* out, uint8_t* in_near, uint8_t* in_far, int w, int hs);

            YuvToBgrPtr yuv444pToBgr;
            YuvToBgraPtr yuv444pToBgra;
            AnyToAnyPtr anyToAny;
        };

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
