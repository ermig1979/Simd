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
#ifndef __TestConsole_h__
#define __TestConsole_h__

#include "Test/TestConfig.h"

namespace Test
{
    namespace Console
    {
        enum Format
        {
            FormatDefault = 0,
            FormatBold = 1,
            FormatDim = 2,
            FormatItalics = 3,
            FormatUnderlined = 4,
            FormatBlink = 5,
            FormatReverse = 7,
            FormatHidden = 8,
        };

        enum Foreground
        {
            ForegroundDefault = 39,
            ForegroundBlack = 30,
            ForegroundRed = 31,
            ForegroundGreen = 32,
            ForegroundYellow = 33,
            ForegroundBlue = 34,
            ForegroundMagenta = 35,
            ForegroundCyan = 36,
            ForegroundLightGray = 37,
            ForegroundDarkGray = 90,
            ForegroundLightRed = 91,
            ForegroundLightGreen = 92,
            ForegroundLightYellow = 93,
            ForegroundLightBlue = 94,
            ForegroundLightMagenta = 95,
            ForegroundLightCyan = 96,
            ForegroundWhite = 97,
        };

        enum Background
        {
            BackgroundDefault = 49,
            BackgroundBlack = 40,
            BackgroundRed = 41,
            BackgroundGreen = 42,
            BackgroundYellow = 43,
            BackgroundBlue = 44,
            BackgroundMegenta = 45,
            BackgroundCyan = 46,
            BackgroundLightGray = 47,
            BackgroundDarkGray = 100,
            BackgroundLightRed = 101,
            BackgroundLightGreen = 102,
            BackgroundLightYellow = 103,
            BackgroundLightBlue = 104,
            BackgroundLightMagenta = 105,
            BackgroundLightCyan = 106,
            BackgroundWhite = 107,
        };

        enum Reset
        {
            ResetAll = 0,
            ResetBold = 21,
            ResetDim = 22,
            ResetUnderlined = 24,
            ResetBlink = 25,
            ResetReverse = 27,
            ResetHidden = 28,
        };

        String Stylized(const String & text, Format format = FormatDefault, Foreground foreground = ForegroundDefault, 
            Background background = BackgroundDefault, Reset reset = ResetAll)
        {
#ifdef __linux__
            std::stringstream ss;
            ss << "\033[" << (int)format;
            ss << ";" << (int)foreground;
            ss << ";" << (int)background << "m";
            ss << text;
            ss << "\033[" << (int)reset << "m";
            return ss.str();
#else
            return text;
#endif
        }
    }
}

#endif// __TestConsole_h__
