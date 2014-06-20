/*
* Simd Library Tests.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#ifndef __TestData_h__
#define __TestData_h__

#include "Test/TestConfig.h"

namespace Test
{
    class Data
    {
        std::string _path;

        std::string Path(const std::string & name) const;

        bool CreatePath(const std::string & path) const;

    public:
        Data(const std::string & name);

        bool Save(const View & image, const std::string & name) const;
        bool Load(View & image, const std::string & name) const;

        bool Save(const uint64_t & value, const std::string & name) const;
        bool Load(uint64_t & value, const std::string & name) const;

        inline bool Load(uint32_t & value, const std::string & name) const
        {
            uint64_t tmp;
            bool result = Load(tmp, name);
            value = (uint32_t)tmp;
            return result;
        }

        bool Save(const uint32_t * data, size_t size, const std::string & name) const;
        bool Load(uint32_t * data, size_t size, const std::string & name) const;

        inline bool Save(const Sums & sums, const std::string & name) const
        {
            return Save(sums.data(), sums.size(), name);
        }
        bool Load(Sums & sums, const std::string & name) const
        {
            return Load(sums.data(), sums.size(), name);
        }

        inline bool Save(const Histogram & histogram, const std::string & name) const
        {
            return Save(histogram, Simd::HISTOGRAM_SIZE, name);
        }
        bool Load(Histogram & histogram, const std::string & name) const
        {
            return Load(histogram, Simd::HISTOGRAM_SIZE, name);
        }

        static std::string Description(SimdCompareType type);
        static std::string Description(SimdOperationBinary8uType type);
        static std::string Description(View::Format format);
    };
}

#define TEST_SAVE(value) \
    if(!data.Save(value, #value)) return false;

#define TEST_LOAD(value) \
    if(!data.Load(value, #value)) return false;

#endif//__TestData_h__