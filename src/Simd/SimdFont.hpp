/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __SimdFont_hpp__
#define __SimdFont_hpp__

#include "Simd/SimdLib.hpp"

#include <vector>
#include <string>

namespace Simd
{
    class Font
    {
    public:
        typedef std::string String;
        typedef Simd::Point<ptrdiff_t> Point;
        typedef Simd::View<Simd::Allocator> View;

        Font(size_t height = 16);

        bool SetHeight(size_t height);

        size_t GetHeight() const;

        template <class Color> bool DrawText(View & canvas, const String & text, const Point & position, const Color & color) const;

    private:
        bool Load(const uint8_t * data, size_t size);
        bool LoadDefault();

        struct Symbol
        {
            char value;
            View image;
        };
        typedef std::vector<Symbol> Symbols;

        Symbols _originalSymbols, _currentSymbols;
        Point _originalSize, _currentSize;
    };

    Font::Font(size_t height)
    {
        LoadDefault();
        SetHeight(height);
    }

    bool Font::SetHeight(size_t height)
    {
        if (height < 4 || height > _originalSize.y)
            return false;

        _currentSize.y = height;
        _currentSize.x = height*_originalSize.x/_originalSize.y;

        _currentSymbols.resize(_originalSymbols.size());
        for (size_t i = 0; i < _originalSymbols.size(); ++i)
        {
            _currentSymbols[i].value = _originalSymbols[i].value;
            _currentSymbols[i].image.Recreate(_currentSize, View::Gray8);
            Simd::ResizeBilinear(_originalSymbols[i].image, _currentSymbols[i].image);
        }

        return true;
    }

    inline size_t Font::GetHeight() const
    {
        return _currentSize.y;
    }

    template <class Color> inline bool Font::DrawText(View & canvas, const String & text, const Point & position, const Color & color) const
    {
        return true;
    }

    inline bool Font::Load(const uint8_t * data, size_t size)
    {
        size_t symbolMin, symbolMax;


        return true;
    }

    inline bool Font::LoadDefault()
    {
        static const uint8_t data[] = { 
            32, 127,
        };

        return Load(data, sizeof(data));
    }
}

#endif//__SimdFont_hpp__
