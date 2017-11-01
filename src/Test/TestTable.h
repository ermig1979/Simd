/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __TestTable_h__
#define __TestTable_h__

#include "Test/TestConfig.h"

namespace Test
{
    class Table
    {
    public:
        enum Format
        {
            Text,
            Html,
        };

        enum Aligment
        {
            None,
            Left,
            Center,
            Rigth,
        };

        struct Property
        {
            Aligment aligment;
            bool bold, separator, zero;
            int precision, foreground, background;
            Property() : aligment(None), bold(false), separator(false), zero(false), precision(3), foreground(0x000000), background(0xffffff) {}
        };

        Table(size_t width, size_t height);

        void SetColProperty(size_t col, const Property & property);
        void SetRowProperty(size_t row, const Property & property);

        void SetCell(size_t col, size_t row, const String & value);
        void SetCell(size_t col, size_t row, const double & value);

        String Generate(Format format);

    private:
        Size _size;
        typedef std::vector<Property> Properties;
        Properties _cols, _rows;
        Ints _widths;
        Strings _cells;
        int _indent;
        std::stringstream _stream;

        void GenerateText();

        void GenerateHtml();

    };
}

#endif//__TestTable_h__