/*
* Simd Library Tests.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Test/TestUtils.h"

namespace Test
{
    void Copy(const View *pSrc, View *pDst)
    {
        assert(pSrc && pDst && pSrc->data && pDst->data);
        assert(pSrc->format == pDst->format && pSrc->width == pDst->width && pSrc->height == pDst->height);

        size_t size = pSrc->width*View::SizeOf(pSrc->format);
        unsigned char *src = pSrc->data;
        unsigned char *dst = pDst->data;
        for(size_t row = 0; row < pSrc->height; ++row)
        {
            memcpy(dst, src, size);
            src += pSrc->stride;
            dst += pDst->stride;
        }
    }

    void ResizeBilinear(const View *pSrc, View *pDst)
    {
        assert(pSrc && pDst && pSrc->data && pDst->data);
        assert(pSrc->format == pDst->format && 
            (pSrc->format == View::Gray8 || pSrc->format == View::Bgr24 || pSrc->format == View::Bgra32));
        assert(pSrc->width && pDst->width && pSrc->height && pDst->height);

        if(pSrc->width == pDst->width && pSrc->height == pDst->height)
        {
            Copy(pSrc, pDst);
        }
        else
        {
            Simd::ResizeBilinear(
				pSrc->data, pSrc->width, pSrc->height, pSrc->stride, 
				pDst->data, pDst->width, pDst->height, pDst->stride, 
                View::SizeOf(pSrc->format));
        }
    }

    void FillRandom(View & view)
    {
        assert(view.data);

        size_t width = view.width*View::SizeOf(view.format);
        for(size_t row = 0; row < view.height; ++row)
        {
            ptrdiff_t offset = row*view.stride;
            for(size_t col = 0; col < width; ++col, ++offset)
            {
                view.data[offset] = Random(256);
            }
        }
    }

    bool Compare(const View & a, const View & b, int differenceMax, bool printError, int errorCountMax, int valueCycle)
    {
        assert(a.data && b.data && a.height == b.height && a.width == b.width && a.format == b.format);
        assert(a.format == View::Gray8 || a.format == View::Bgr24 || a.format == View::Bgra32);

        int errorCount = 0;
        size_t colors = Simd::View::SizeOf(a.format);
        size_t width = colors*a.width;
        for(size_t row = 0; row < a.height; ++row)
        {
            uchar* pA = a.data + row*a.stride;
            uchar* pB = b.data + row*b.stride;
            for(size_t offset = 0; offset < width; ++offset)
            {
                if(pA[offset] != pB[offset])
                {
                    if(differenceMax > 0)
                    {
                        int difference = Simd::Base::Max(pA[offset], pB[offset]) - Simd::Base::Min(pA[offset], pB[offset]);
                        if(valueCycle > 0)
                            difference = Simd::Base::Min(difference, valueCycle - difference);
                        if(difference <= differenceMax)
                            continue;
                    }
                    errorCount++;
                    if(printError)
                    {
                        size_t col = offset/colors;
                        std::cout << "Error at [" << col << "," << row << "] : (" << (int)pA[col*colors];
                        for(size_t color = 1; color < colors; ++color)
                            std::cout << "," << (int)pA[col*colors + color]; 
                        std::cout << ") != (" << (int)pB[col*colors];
                        for(size_t color = 1; color < colors; ++color)
                            std::cout << "," << (int)pB[col*colors + color]; 
                        std::cout << ")." << std::endl;
                    }
                    if(errorCount > errorCountMax)
                    {
                        if(printError)
                            std::cout << "Stop comparison." << std::endl;
                        return false;
                    }
                }
            }
        }
        return errorCount == 0;
    }
}