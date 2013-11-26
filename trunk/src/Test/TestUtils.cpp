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
#include "Test/TestUtils.h"

namespace Test
{
    void FillRandom(View & view)
    {
        assert(view.data);

        size_t width = view.width*View::PixelSize(view.format);
        for(size_t row = 0; row < view.height; ++row)
        {
            ptrdiff_t offset = row*view.stride;
            for(size_t col = 0; col < width; ++col, ++offset)
            {
                view.data[offset] = Random(256);
            }
        }
    }

	void FillRandomMask(View & view, uint8_t index)
	{
		assert(view.data);

		size_t width = view.width*View::PixelSize(view.format);
		for(size_t row = 0; row < view.height; ++row)
		{
			ptrdiff_t offset = row*view.stride;
			for(size_t col = 0; col < width; ++col, ++offset)
			{
				view.data[offset] = Random(2) ? index : 0;
			}
		}
	}

    bool Compare(const View & a, const View & b, int differenceMax, bool printError, int errorCountMax, int valueCycle, 
		const std::string & description)
    {
        assert(a.data && b.data && a.height == b.height && a.width == b.width && a.format == b.format);
        assert(a.format == View::Gray8 || a.format == View::Uv16 || a.format == View::Bgr24 || a.format == View::Bgra32);

        int errorCount = 0;
        size_t colors = Simd::View::PixelSize(a.format);
        size_t width = colors*a.width;
        for(size_t row = 0; row < a.height; ++row)
        {
            uint8_t* pA = a.data + row*a.stride;
            uint8_t* pB = b.data + row*b.stride;
            for(size_t offset = 0; offset < width; ++offset)
            {
                if(pA[offset] != pB[offset])
                {
                    if(differenceMax > 0)
                    {
                        int difference = Simd::Base::Max(pA[offset], pB[offset]) - Simd::Base::Min(pA[offset], pB[offset]);
                        if(valueCycle > 0)
                            difference = Simd::Base::Min(difference, valueCycle - difference);
                        if(difference <= differenceMax)
                            continue;
                    }
                    errorCount++;
                    if(printError)
                    {
						if(errorCount == 1 && description.length() > 0)
						{
							std::cout << "Fail comparison: " << description << std::endl;
						}
                        size_t col = offset/colors;
                        std::cout << "Error at [" << col << "," << row << "] : (" << (int)pA[col*colors];
                        for(size_t color = 1; color < colors; ++color)
                            std::cout << "," << (int)pA[col*colors + color]; 
                        std::cout << ") != (" << (int)pB[col*colors];
                        for(size_t color = 1; color < colors; ++color)
                            std::cout << "," << (int)pB[col*colors + color]; 
                        std::cout << ")." << std::endl;
                    }
                    if(errorCount >= errorCountMax)
                    {
                        if(printError)
                            std::cout << "Stop comparison." << std::endl;
                        return false;
                    }
                }
            }
        }
        return errorCount == 0;
    }

    template <class T> bool Compare(const T * a, const T * b, size_t size, int differenceMax, bool printError, int errorCountMax)
    {
        int errorCount = 0;
        for(size_t i = 0; i < size; ++i)
        {
            if(a[i] != b[i])
            {
                if(differenceMax > 0)
                {
                    int difference = Simd::Base::Max(a[i], b[i]) - Simd::Base::Min(a[i], b[i]);
                    if(difference <= differenceMax)
                        continue;
                }
                errorCount++;
                if(printError)
                {
                    std::cout << "Error at [" << i << "] : " << a[i] << " != " << b[i] << "." << std::endl;
                }
                if(errorCount > errorCountMax)
                {
                    if(printError)
                        std::cout << "Stop comparison." << std::endl;
                    return false;
                }
            }
        }
        return errorCount == 0;
    }

	bool Compare(const Histogram a, const Histogram b, int differenceMax, bool printError, int errorCountMax)
	{
        return Compare(a, b, Simd::HISTOGRAM_SIZE, differenceMax, printError, errorCountMax);
	}

    bool Compare(const Sums & a, const Sums b, int differenceMax, bool printError, int errorCountMax)
    {
        assert(a.size() == b.size());
        return Compare(a.data(), b.data(), a.size(), differenceMax, printError, errorCountMax);
    }
    
	std::string ColorDescription(View::Format format)
	{
		std::stringstream ss;
		ss << "<" << View::PixelSize(format) << ">";
		return ss.str();
	}

    std::string CompareTypeDescription(SimdCompareType type)
    {
        switch(type)
        {
        case SimdCompareEqual:
            return "(==)";
        case SimdCompareNotEqual:
            return "(!=)";
        case SimdCompareGreater:
            return "(> )";
        case SimdCompareGreaterOrEqual:
            return "(>=)";
        case SimdCompareLesser:
            return "(< )";
        case SimdCompareLesserOrEqual:
            return "(<=)";
        }
        assert(0);
        return "(Unknown)";
    }

    std::string ExpandToLeft(std::string value, size_t count)
    {
        assert(count <= value.size());
        std::stringstream ss;
        for(size_t i = value.size(); i < count; i++)
            ss << " ";
        ss << value;
        return ss.str();
    }

    std::string ExpandToRight(std::string value, size_t count)
    {
        assert(count <= value.size());
        std::stringstream ss;
        ss << value;
        for(size_t i = value.size(); i < count; i++)
            ss << " ";
        return ss.str();
    }

    std::string ToString(double value, size_t iCount, size_t fCount)
    {
        assert(iCount > 0);
        std::stringstream ss;
        ss << std::setprecision(fCount) << std::fixed << value;
        return ExpandToLeft(ss.str(), iCount + fCount + (fCount > 0 ? 1 : 0));
    }
}