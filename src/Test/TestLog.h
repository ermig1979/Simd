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
#ifndef __TestLog_h__
#define __TestLog_h__

#include "Test/TestConfig.h"

namespace Test
{
    class Log
    {
    public:
        enum Level
        {
            Error = 0,
            Info,
        };

        Log();
        ~Log();

        void SetLogFile(String name);
        void SetLevel(Level level);
        void SetEnableThreadId(bool enable);
        void SetEnablePrefix(bool enable);

        void Write(Level level, const String & message);

        static Log s_log;
    private:
        std::mutex _mutex;
        std::ofstream _file;
        Level _level;
        bool _enableThreadId;
        bool _enablePrefix;

        typedef std::map<std::thread::id, String> Messages;
        Messages _lastSkippedMessages, _threadNames;
    };
}

#define TEST_LOG(level, message) \
    Test::Log::s_log.Write(Test::Log::level, message);

#define TEST_LOG_SS(level, message) \
    { \
        std::stringstream ___ss; \
        ___ss << message; \
        Test::Log::s_log.Write(Test::Log::level, ___ss.str()); \
    }

#endif// __TestLog_h__
