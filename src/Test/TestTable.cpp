/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Test/TestHtml.h"
#include "Test/TestUtils.h"
#include "Test/TestString.h"

namespace Test
{
    Table::Table(size_t width, size_t height)
        : _size(width, height)
    {
        _cells.resize(width*height);
        _headers.resize(width);
        _rows.resize(height);
    }

	Test::Size Table::Size() const
	{
		return _size;
	}

    void Table::SetHeader(size_t col, const String & name, bool separator, Alignment alignment)
    {
        _headers[col] = Header(name, separator, alignment);
    }

    void Table::SetRowProp(size_t row, bool separator, bool bold)
    {
        _rows[row] = RowProp(separator, bold);
    }

    void Table::SetCell(size_t col, size_t row, const String & value)
    {
        _cells[row*_size.x + col] = value;
        _headers[col].width = std::max(_headers[col].width, value.size());
    }

    String Table::GenerateText(size_t indent_)
    {
        std::stringstream header, separator, table, indent;
        for (size_t i = 0; i < indent_; ++i)
            indent << " ";
        header << "| ";
        for (ptrdiff_t col = 0; col < _size.x; ++col)
        {
            header << ExpandText(_headers[col].name, _headers[col]) << " ";
            if(_headers[col].separator)
                header << "|" << (col < _size.x - 1 ? " " : "");
        }
        for (size_t i = 0; i < header.str().size(); ++i)
            separator << "-";
        table << indent.str() << separator.str() << std::endl;
        table << indent.str() << header.str() << std::endl;
        table << indent.str() << separator.str() << std::endl;
        for (ptrdiff_t row = 0; row < _size.y; ++row)
        {
            table << indent.str() << "| ";
            for (ptrdiff_t col = 0; col < _size.x; ++col)
            {
                table << ExpandText(_cells[row*_size.x + col], _headers[col]) << " ";
                if (_headers[col].separator)
                    table << "|" << (col < _size.x - 1 ? " " : "");
            }
            table << std::endl;
            if(_rows[row].separator)
                table << indent.str() << separator.str() << std::endl;
        }
        table << indent.str() << separator.str() << std::endl;
        return table.str();
    }

    String Table::ExpandText(const String & value, const Header & header)
    {
        if (header.alignment == Left)
            return ExpandToRight(value, header.width);
        if (header.alignment == Right)
            return ExpandToLeft(value, header.width);
        return String();
    }

    String Table::GenerateHtml(size_t indent)
    {
        std::stringstream stream;
        Html html(stream, indent);

        html.WriteBegin("style", Html::Attr("type", "text/css"), true, true);
        html.WriteText("th.th0 { border-left: 0px solid #000000; border-top: 0px solid #000000; border-right: 0px solid #000000; border-bottom: 1px solid #000000;}", true, true);
        html.WriteText("th.th1 { border-left: 0px solid #000000; border-top: 0px solid #000000; border-right: 1px solid #000000; border-bottom: 1px solid #000000;}", true, true);
        html.WriteText("td.td0 { border-left: 0px solid #000000; border-top: 0px solid #000000; border-right: 0px solid #000000; border-bottom: 0px solid #000000;}", true, true);
        html.WriteText("td.td1 { border-left: 0px solid #000000; border-top: 0px solid #000000; border-right: 1px solid #000000; border-bottom: 0px solid #000000;}", true, true);
        html.WriteEnd("style", true, true);

        Html::Attributes attributes;
        attributes.push_back(Html::Attribute("align", "center"));
        attributes.push_back(Html::Attribute("cellpadding", "2"));
        attributes.push_back(Html::Attribute("cellspacing", "0"));
        attributes.push_back(Html::Attribute("border", "1"));
        attributes.push_back(Html::Attribute("cellpadding", "2"));
        attributes.push_back(Html::Attribute("width", "100%"));
        attributes.push_back(Html::Attribute("style", "border-collapse:collapse"));
        html.WriteBegin("table", attributes, true, true);

        html.WriteBegin("tr", Html::Attr("style", "background-color:#e0e0e0; font-weight:bold;"), true, false);
        for (ptrdiff_t col = 0; col < _size.x; ++col)
            html.WriteValue("th", Html::Attr("class", String("th") + ToString(_headers[col].separator)),_headers[col].name, false);
        html.WriteEnd("tr", true, true);

        for (ptrdiff_t row = 0; row < _size.y; ++row)
        {
            std::stringstream style;
            if (_rows[row].bold)
                style << "font-weight: bold; background-color:#f0f0f0";
            html.WriteBegin("tr", Html::Attr("align", "center", "style", style.str()), true, false);
            for (ptrdiff_t col = 0; col < _size.x; ++col)
                html.WriteValue("td", Html::Attr("class", String("td") + ToString(_headers[col].separator)), _cells[row*_size.x + col], false);
            html.WriteEnd("tr", true, true);
        }

        html.WriteEnd("table", true, true);

        return stream.str();
    }
}