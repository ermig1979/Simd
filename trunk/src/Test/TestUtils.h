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
#ifndef __TestUtils_h__
#define __TestUtils_h__

#include "Test/TestConfig.h"

namespace Test
{
	inline int Random(int range)
	{
		return (::rand()*range)/RAND_MAX;
	}

    void FillRandom(View & view);

	void FillRandomMask(View & view, uchar index);
    
    bool Compare(const View & a, const View & b, 
		int differenceMax = 0, bool printError = false, int errorCountMax = 0, int valueCycle = 0, 
		const std::string & description = "");

	bool Compare(const Histogram a, const Histogram b, 
		int differenceMax = 0, bool printError = false, int errorCountMax = 0);

	std::string ColorDescription(View::Format format);
}

#endif//__TestUtils_h__