/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#ifndef __SimdFont_h__
#define __SimdFont_h__

#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        struct FontImpl;

        struct Font : public Deletable
        {
            Font();

            virtual ~Font();

            SimdBool Resize(size_t height);

            size_t Height() const;

            void Measure(const char* text, size_t *width, size_t *height) const;

            void Draw(uint8_t* canvas, size_t stride, size_t width, size_t height, size_t channels, const char* text, size_t left, size_t top, const uint8_t * color) const;

        private:
            FontImpl* _impl;
        };
    }
}
#endif
