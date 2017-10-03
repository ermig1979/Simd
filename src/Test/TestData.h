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
#ifndef __TestData_h__
#define __TestData_h__

#include "Test/TestConfig.h"

namespace Test
{
    class Data
    {
        String _path;

        String Path(const String & name) const;

        bool CreatePath(const String & path) const;

        template <class T> bool SaveArray(const T * data, size_t size, const String & name) const;
        template <class T> bool LoadArray(T * data, size_t size, const String & name) const;

    public:
        Data(const String & name);

        bool Save(const View & image, const String & name) const;
        bool Load(View & image, const String & name) const;

        bool Save(const uint64_t & value, const String & name) const;
        bool Load(uint64_t & value, const String & name) const;

        bool Save(const int64_t & value, const String & name) const;
        bool Load(int64_t & value, const String & name) const;

        bool Save(const uint32_t & value, const String & name) const;
        bool Load(uint32_t & value, const String & name) const;

        bool Save(const uint8_t & value, const String & name) const;
        bool Load(uint8_t & value, const String & name) const;

        bool Save(const double & value, const String & name) const;
        bool Load(double & value, const String & name) const;

        bool Save(const float & value, const String & name) const;
        bool Load(float & value, const String & name) const;

        bool Save(const Sums & sums, const String & name) const;
        bool Load(Sums & sums, const String & name) const;

        bool Save(const Histogram & histogram, const String & name) const;
        bool Load(Histogram & histogram, const String & name) const;

        bool Save(const Sums64 & sums, const String & name) const;
        bool Load(Sums64 & sums, const String & name) const;

        bool Save(const Rect & rect, const String & name) const;
        bool Load(Rect & rect, const String & name) const;

        bool Save(const std::vector<uint8_t> & data, const String & name) const;
        bool Load(std::vector<uint8_t> & data, const String & name) const;

        bool Save(const Buffer32f & buffer, const String & name) const;
        bool Load(Buffer32f & buffer, const String & name) const;

        static String Description(SimdCompareType type);
        static String Description(SimdOperationBinary8uType type);
        static String Description(SimdOperationBinary16iType type);
        static String Description(View::Format format);
    };
}

#define TEST_SAVE(value) \
    if(!data.Save(value, #value)) return false;

#define TEST_LOAD(value) \
    if(!data.Load(value, #value)) return false;

#endif//__TestData_h__
