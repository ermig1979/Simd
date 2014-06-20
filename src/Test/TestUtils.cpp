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

    void FillRhombMask(View & mask, const Rect & rect, uint8_t index)
    {
        assert(mask.format == View::Gray8 && Rect(mask.Size()).Contains(rect));

        Simd::Fill(mask, 0);

        Point c = rect.Center();
        for(ptrdiff_t row = rect.top; row < rect.bottom; ++row)
        {
            ptrdiff_t indent = std::abs(row - c.y)*rect.Width()/rect.Height();
            ptrdiff_t left = rect.left + indent;
            ptrdiff_t right = rect.right - indent;
            ptrdiff_t offset = row*mask.stride + left;
            for(ptrdiff_t col = left; col < right; ++col, ++offset)
                mask.data[offset] = Random(2) ? index : 0;
        }
    }

    template <class Channel> bool Compare(const View & a, const View & b, int differenceMax, bool printError, int errorCountMax, int valueCycle, 
        const std::string & description)
    {
        int errorCount = 0;
        size_t channelCount = a.ChannelCount();
        size_t width = channelCount*a.width;
        for(size_t row = 0; row < a.height; ++row)
        {
            const Channel * pA = (const Channel*)(a.data + row*a.stride);
            const Channel * pB = (const Channel*)(b.data + row*b.stride);
            for(size_t offset = 0; offset < width; ++offset)
            {
                if(pA[offset] != pB[offset])
                {
                    if(differenceMax > 0)
                    {
                        Channel difference = Simd::Max(pA[offset], pB[offset]) - Simd::Min(pA[offset], pB[offset]);
                        if(valueCycle > 0)
                            difference = Simd::Min<Channel>(difference, valueCycle - difference);
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
                        size_t col = offset/channelCount;
                        std::cout << "Error at [" << col << "," << row << "] : (" << (int)pA[col*channelCount];
                        for(size_t channel = 1; channel < channelCount; ++channel)
                            std::cout << "," << (int)pA[col*channelCount + channel]; 
                        std::cout << ") != (" << (int)pB[col*channelCount];
                        for(size_t channel = 1; channel < channelCount; ++channel)
                            std::cout << "," << (int)pB[col*channelCount + channel]; 
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

    bool Compare(const View & a, const View & b, int differenceMax, bool printError, int errorCountMax, int valueCycle, 
		const std::string & description)
    {
        assert(Simd::Compatible(a, b) && a.format != View::Float && a.format != View::Double);

        switch(a.ChannelSize())
        {
        case 1:
            return Compare<uint8_t>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
        case 2:
            return Compare<int16_t>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
        case 4:
            return Compare<int32_t>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
        case 8:
            return Compare<int64_t>(a, b, differenceMax, printError, errorCountMax, valueCycle, description);
        default:
            assert(0);
        }

        return false;
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

    bool Compare(const Sums & a, const Sums & b, int differenceMax, bool printError, int errorCountMax)
    {
        assert(a.size() == b.size());
        return Compare(a.data(), b.data(), a.size(), differenceMax, printError, errorCountMax);
    }

    bool Compare(const Rect & a, const Rect & b, bool printError)
    {
        bool result(a == b);
        if(!result && printError)
        {
            std::cout << "Rectangles is not equal: (" << a.left << ", " << a.top << ", " << a.right  << ", " << a.bottom << ") != (" 
                << b.left << ", " << b.top << ", " << b.right  << ", " << b.bottom << ") !" << std::endl;
        }
        return result;
    }
    
	std::string ColorDescription(View::Format format)
	{
		std::stringstream ss;
		ss << "<" << View::PixelSize(format) << ">";
		return ss.str();
	}

    std::string FormatDescription(View::Format format)
    {
        switch(format)
        {
        case View::None:      return "None";
        case View::Gray8:     return "8-bit Gray";
        case View::Uv16:      return "16-bit UV";
        case View::Bgr24:     return "24-bit BGR";
        case View::Bgra32:    return "32-bit BGRA";
        case View::Int16:     return "16-bit int";
        case View::Int32:     return "32-bit int";
        case View::Int64:     return "64-bit int";
        case View::Float:     return "32-bit float";
        case View::Double:    return "64-bit float";
        case View::BayerGrbg: return "Bayer GRBG";
        case View::BayerGbrg: return "Bayer GBRG";
        case View::BayerRggb: return "Bayer RGGB";
        case View::BayerBggr: return "Bayer BGGR";
        default: assert(0); return "";
        }
    }

    std::string ScaleDescription(const Point & scale)
    {
        std::stringstream ss;
        ss << "[" << scale.x << "x" << scale.y << "]";
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

    std::string ExpandToLeft(const std::string & value, size_t count)
    {
        assert(count >= value.size());
        std::stringstream ss;
        for(size_t i = value.size(); i < count; i++)
            ss << " ";
        ss << value;
        return ss.str();
    }

    std::string ExpandToRight(const std::string & value, size_t count)
    {
        assert(count >= value.size());
        std::stringstream ss;
        ss << value;
        for(size_t i = value.size(); i < count; i++)
            ss << " ";
        return ss.str();
    }

    std::string ToString(double value, size_t iCount, size_t fCount)
    {
        assert(iCount > 0);
        if(value > 0)
        {
            std::stringstream ss;
            ss << std::setprecision(fCount) << std::fixed << value;
            return ExpandToLeft(ss.str(), iCount + fCount + (fCount > 0 ? 1 : 0));
        }
        else
        {
            return ExpandToLeft("", iCount + fCount + 1);
        }
    }
}