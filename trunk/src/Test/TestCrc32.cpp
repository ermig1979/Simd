/*
* Simd Library Tests.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Test/TestConfig.h"
#include "Test/TestView.h"
#include "Test/TestUtils.h"
#include "Test/TestPerformance.h"

namespace Test
{
	namespace
	{
		struct Func
		{
			typedef unsigned __int32 (*FunkPtr)(const void *src, size_t size);

			FunkPtr func;
			std::string description;

			Func(const FunkPtr & f, const std::string & d) : func(f), description(d) {}

			unsigned __int32 Call(const void *src, size_t size) const
			{
				TEST_PERFORMANCE_TEST(description);
				return func(src, size);
			}
		};	
	}

#define FUNC(func) Func(func, #func)

    void SetRandom(unsigned char * data, size_t size) 
    {
        for(size_t i = 0; i < size; ++i)
            data[i] = Random(256);
    }

    bool Crc32Test(const std::vector<unsigned char> & data, const Func & f1, const Func & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " for size = " << data.size() << "." << std::endl;

        unsigned int crc1;
		TEST_EXECUTE_AT_LEAST_MIN_TIME(crc1 = f1.Call(&data[0], data.size()));

		unsigned int crc2;
		TEST_EXECUTE_AT_LEAST_MIN_TIME(crc2 = f2.Call(&data[0], data.size()));

        if(crc1 != crc2)
        {
            result = false;
            std::cout << "Crc32Test is failed (" << crc1 << " != " << crc2 <<")!" << std::endl;
        }
        return result;
    }

    bool Crc32Test()
    {
        bool result = true;

		std::vector<unsigned char> data(W*H - 1);
        SetRandom(&data[0], data.size());

        result = result && Crc32Test(data, FUNC(Simd::Base::Crc32), FUNC(Simd::Sse42::Crc32));

        return result;
    }
}