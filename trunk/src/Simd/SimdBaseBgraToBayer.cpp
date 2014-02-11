/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        void BgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat)
        {
            assert((width%2 == 0) && (height%2 == 0));
            assert(bayerFormat >= SimdPixelFormatBayerGrbg && bayerFormat <= SimdPixelFormatBayerBggr);

            size_t offsets[2][2];
            switch(bayerFormat)
            {
            case SimdPixelFormatBayerGrbg:
                offsets[0][0] = 1;
                offsets[0][1] = 6;
                offsets[1][0] = 0;
                offsets[1][1] = 5;
                break;
            case SimdPixelFormatBayerGbrg:
                offsets[0][0] = 1;
                offsets[0][1] = 4;
                offsets[1][0] = 2;
                offsets[1][1] = 5;
                break;
            case SimdPixelFormatBayerRggb:
                offsets[0][0] = 2;
                offsets[0][1] = 5;
                offsets[1][0] = 1;
                offsets[1][1] = 4;
                break;
            case SimdPixelFormatBayerBggr:
                offsets[0][0] = 0;
                offsets[0][1] = 5;
                offsets[1][0] = 1;
                offsets[1][1] = 6;
                break;
            }

            for(size_t row = 0; row < height; row += 2)
            {
                for(size_t col = 0, offset = 0; col < width; col += 2, offset += 8)
                {
                    bayer[col + 0] = bgra[offset + offsets[0][0]];
                    bayer[col + 1] = bgra[offset + offsets[0][1]];
                }
                bgra += bgraStride;
                bayer += bayerStride;

                for(size_t col = 0, offset = 0; col < width; col += 2, offset += 8)
                {
                    bayer[col + 0] = bgra[offset + offsets[1][0]];
                    bayer[col + 1] = bgra[offset + offsets[1][1]];
                }
                bgra += bgraStride;
                bayer += bayerStride;
            }        
        }
    }
}