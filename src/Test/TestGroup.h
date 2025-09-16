/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __TestGroup_h__
#define __TestGroup_h__

#include "Test/TestOptions.h"

namespace Test
{
    typedef bool(*AutoTestPtr)(const Options& options);
    typedef bool(*SpecialTestPtr)(const Options& options);

    struct Group
    {
        String name;
        AutoTestPtr autoTest;
        SpecialTestPtr specialTest;
        double start, finish;
        Group(const String& n, const AutoTestPtr& a, const SpecialTestPtr& s)
            : name(n)
            , autoTest(a)
            , specialTest(s)
            , start(0.0)
            , finish(0.0)
        {
        }

        double Time() const
        {
            return finish - start;
        }
    };
    typedef std::vector<Group> Groups;

    //-------------------------------------------------------------------------------------------------

    inline bool Required(const Options& options, const Group& group)
    {
        if (options.mode == Options::Auto && group.autoTest == NULL)
            return false;
        if (options.mode == Options::Special && group.specialTest == NULL)
            return false;
        bool required = options.include.empty();
        for (size_t i = 0; i < options.include.size() && !required; ++i)
            if (group.name.find(options.include[i]) != std::string::npos)
                required = true;
        for (size_t i = 0; i < options.exclude.size() && required; ++i)
            if (group.name.find(options.exclude[i]) != std::string::npos)
                required = false;
        return required;
    }
}

#endif
