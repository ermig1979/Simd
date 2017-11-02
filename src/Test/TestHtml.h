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
#ifndef __TestHtml_h__
#define __TestHtml_h__

#include "Test/TestConfig.h"

namespace Test
{
    struct Html
    {
        struct Attribute
        {
            String name, value;
            Attribute(const String & n = String(), const String & v = String());
        };
        typedef std::vector<Attribute> Attributes;

        static Attributes Attr();
        static Attributes Attr(
            const String & name0, const String & value0);
        static Attributes Attr(
            const String & name0, const String & value0,
            const String & name1, const String & value1);
        static Attributes Attr(
            const String & name0, const String & value0,
            const String & name1, const String & value1,
            const String & name2, const String & value2);

        Html(std::ostream & stream, size_t indent = 0);

        void WriteIndent();
        void WriteAtribute(const Attribute & attribute);
        void WriteBegin(const String & name, const Attributes & attributes, bool indent, bool line);
        void WriteEnd(const String & name, bool indent, bool line);
        void WriteValue(const String & name, const Attributes & attributes, const String & value, bool line);
        void WriteText(const String & text, bool indent, bool line);

        size_t Indent() const;

    private:
        std::ostream & _stream;
        size_t _indent;
        bool _line;
    };
}

#endif//__TestHtml_h__