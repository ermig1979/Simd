/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Test/TestLog.h"

namespace Test
{
    Log Log::s_log;

    Log::Log()
        : _level(Log::Info)
        , _enablePrefix(true)
        , _enableThreadId(true)
    {

    }

    Log::~Log()
    {

    }

    void Log::SetLogFile(std::string name)
    {
        _file.open(name);
    }

    void Log::SetLevel(Level level)
    {
        _level = level;
    }

    void Log::SetEnableThreadId(bool enable)
    {
        _enableThreadId = enable;
   }

    void Log::SetEnablePrefix(bool enable)
    {
        _enablePrefix = enable;
    }

    void Log::Write(Level level, const std::string & message)
    {
        std::stringstream ss;
        if (_enableThreadId)
            ss << "[" << std::this_thread::get_id() << "] ";
        if (_enablePrefix)
        {
            switch (level)
            {
            case Log::Error : ss << "ERROR: "; break;
            case Log::Info: ss << "INFO: "; break;
            default:
                assert(0);
            }        
        }
        ss << message << std::endl;
        if(level > _level)
        {
            _lastSkippedMessages[std::this_thread::get_id()] = ss.str();
        }
        else
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if(_level == Error)
            {
                const std::string last = _lastSkippedMessages[std::this_thread::get_id()];
                if(last.size())
                {
                    std::cout << last;
                    if(_file.is_open())
                        _file << last;
                }
            }
            std::cout << ss.str();
            if(_file.is_open())
                _file << ss.str();
        }
    }
}