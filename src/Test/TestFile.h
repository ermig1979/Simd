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
#ifndef __TestFile_h__
#define __TestFile_h__

#include "Test/TestConfig.h"
#include "Test/TestLog.h"

namespace Test
{
    SIMD_INLINE String ExtensionByPath(const String& path)
    {
        size_t pos = path.find_last_of(".");
        if (pos == std::string::npos)
            return String();
        else
            return path.substr(pos + 1);
    }

    SIMD_INLINE String FolderSeparator()
    {
#ifdef WIN32
        return String("\\");
#elif defined(__unix__) || defined(__APPLE__)
        return String("/");
#else
        TEST_LOG_SS(Error, "FolderSeparator: Is not implemented yet!");
        return String("");
#endif
    }

    SIMD_INLINE String MakePath(const String& a, const String& b)
    {
        if (a.empty())
            return b;
        String s = FolderSeparator();
        return a + (a[a.size() - 1] == s[0] ? "" : s) + b;
    }

    bool DirectoryExists(const String & path);

    String DirectoryByPath(const String & path);

    bool CreatePath(const String & path);

    bool CreatePathIfNotExist(const String & path, bool file);
}

#endif//__TestFile_h__
