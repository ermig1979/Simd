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
#include "Test/TestTable.h"
#include "Test/TestUtils.h"

namespace Test
{
    Table::Table(size_t width, size_t height)
        : _size(width, height)
    {
        _cells.resize(width*height);
        _cols.resize(width);
        _rows.resize(height);
        _widths.resize(width, 0);
    }

	Test::Size Table::Size() const
	{
		return _size;
	}

    void Table::SetColProperty(size_t col, const Property & property)
    {
        _cols[col] = property;
    }

    void Table::SetRowProperty(size_t row, const Property & property)
    {
        _rows[row] = property;
    }

    void Table::SetCell(size_t col, size_t row, const String & value)
    {
        _cells[row*_size.x + col] = value;
    }

    void Table::SetCell(size_t col, size_t row, const double & value)
    {
        std::stringstream ss;
        ss << std::setprecision(_cols[col].precision) << std::fixed << value;
        _cells[row*_size.x + col] = (value != 0 || _cols[col].zero) ? ss.str() : "";
    }

    String Table::Generate(Format format)
    {
        _stream.clear();
        switch (format)
        {
        case Text: 
            GenerateText(); 
            break;
        case Html: 
            GenerateHtml(); 
            break;
        default:
            break;
        }
        return _stream.str();
    }

    void Table::GenerateText()
    {
    }

    void Table::GenerateHtml()
    {
        _indent = 0;
    }
}