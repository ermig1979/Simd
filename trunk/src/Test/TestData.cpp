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
#include "Test/TestData.h"

#ifdef _MSC_VER
#include <filesystem>
#else

#endif

namespace Test
{
    std::string Data::Path(const std::string & name) const
    {
        return _path + "/" + name + ".txt";
    }

    bool Data::CreatePath(const std::string & path) const
    {
#ifdef _MSC_VER
        if(!std::tr2::sys::exists(std::tr2::sys::path(path)))
        {
            if(!std::tr2::sys::create_directories(std::tr2::sys::path(path)))
            {
                std::cout << "Can't create path '" << path << "'!" << std::endl;
                return false;
            }
        }
        return true;
#else
        system((std::string("mkdir -p ") + path).c_str());
        return true;
#endif
    }

    template <class T> bool Data::SaveArray(const T * data, size_t size, const std::string & name) const
    {
        if(!CreatePath(_path))
            return false;

        std::string path = Path(name);
        std::ofstream ofs(path);
        if(ofs.bad())
        {
            std::cout << "Can't create file '" << path << "'!" << std::endl; 
            return false;
        }

        try
        {
            ofs << size << std::endl;
            if(sizeof(T) < sizeof(int))
            {
                for(size_t i = 0; i < size; ++i)
                    ofs << (int)data[i] << " ";
            }
            else
            {
                for(size_t i = 0; i < size; ++i)
                    ofs << data[i] << " ";
            }
            ofs << std::endl;
        }
        catch (std::exception e)
        {
            std::cout << "Can't save array to file '" << path << "'!" << std::endl; 
            std::cout << "There is an exception: " << e.what() << std::endl; 
            ofs.close();
            return false;
        }

        ofs.close();

        return true;
    }

    template <class T> bool Data::LoadArray(T * data, size_t size, const std::string & name) const
    {
        std::string path = Path(name);
        std::ifstream ifs(path);
        if(ifs.bad())
        {
            std::cout << "Can't open file '" << path << "'!" << std::endl; 
            return false;
        }

        try
        {
            uint64_t value;
            ifs >> value;
            if(value != (uint64_t)size)
                throw std::runtime_error("Invalid sums size!");

            if(sizeof(T) < sizeof(int))
            {
                int tmp;
                for(size_t i = 0; i < size; ++i)
                {
                    ifs >> tmp;
                    data[i] = (T)tmp;
                }
            }
            else
            {
                for(size_t i = 0; i < size; ++i)
                    ifs >> data[i];
            }
        }
        catch (std::exception e)
        {
            std::cout << "Can't load array from file '" << path << "'!" << std::endl; 
            std::cout << "There is an exception: " << e.what() << std::endl; 
            ifs.close();
            return false;
        }

        ifs.close();

        return true;
    }

    Data::Data(const std::string & test)
    {
        std::stringstream path;
        path << "../../test/";
        for(size_t i = 0; i < test.size(); ++i)
        {
            if(test[i] == '<')
                path << '_';
            else if(test[i] == '>')
                path << "";
            else
                path << test[i];
        }
        _path = path.str();
    }

    bool Data::Save(const View & image, const std::string & name) const
    {
        if(!CreatePath(_path))
            return false;

        std::string path = Path(name);
        std::ofstream ofs(path);
        if(ofs.bad())
        {
            std::cout << "Can't create file '" << path << "'!" << std::endl; 
            return false;
        }

        try
        {
            size_t channelCount = image.ChannelCount();
            size_t channelSize = image.ChannelSize();
            size_t pixelSize = image.PixelSize();

            ofs << (int)image.format << " " << image.width << " " << image.height << std::endl;
            if(image.format != View::Float && image.format != View::Double)
            {
                ofs << std::hex;
                for(size_t row = 0; row < image.height; ++row)
                {
                    for(size_t col = 0; col < image.width; ++col)
                    {
                        for(size_t channel = 0; channel < channelCount; ++channel)
                        {
                            const uint8_t * data = image.data + row*image.stride + col*pixelSize + channel*channelSize;
                            for(size_t i = 0; i < channelSize; ++i)
                            {
#ifdef SIMD_BIG_ENDIAN
                                ofs << (int)(data[i] >> 4);
                                ofs << (int)(data[i] & 0xF);
#else
                                ofs << (int)(data[channelSize - i - 1] >> 4);
                                ofs << (int)(data[channelSize - i - 1] & 0xF);
#endif                    
                            }
                            ofs << " ";
                        }
                        ofs << " ";
                    }
                    ofs << std::endl;
                } 
            }
            else
            {
                for(size_t row = 0; row < image.height; ++row)
                {
                    for(size_t col = 0; col < image.width; ++col)
                    {
                        if(image.format == View::Float)
                            ofs << image.At<float>(col, row);
                        else
                            ofs << image.At<double>(col, row);
                        ofs << " ";
                    }
                    ofs << std::endl;
                } 
            }
        }
        catch (std::exception e)
        {
            std::cout << "Can't save image to file '" << path << "'!" << std::endl; 
            std::cout << "There is an exception: " << e.what() << std::endl; 
            ofs.close();
            return false;
        }

        ofs.close();

        return true;
    }

    bool Data::Load(View & image, const std::string & name) const
    {
        std::string path = Path(name);
        std::ifstream ifs(path);
        if(ifs.bad())
        {
            std::cout << "Can't open file '" << path << "'!" << std::endl; 
            return false;
        }

        try
        {
            size_t channelCount = image.ChannelCount();
            size_t channelSize = image.ChannelSize();
            size_t pixelSize = image.PixelSize();

            uint64_t value;
            ifs >> value;
            if(value != (uint64_t)image.format)
                throw std::runtime_error("Invalid image format!");
            ifs >> value;
            if(value != (uint64_t)image.width)
                throw std::runtime_error("Invalid image width!");
            ifs >> value;
            if(value != (uint64_t)image.height)
                throw std::runtime_error("Invalid image height!");

            if(image.format != View::Float && image.format != View::Double)
            {
                ifs >> std::hex;
                for(size_t row = 0; row < image.height; ++row)
                {
                    for(size_t col = 0; col < image.width; ++col)
                    {
                        for(size_t channel = 0; channel < channelCount; ++channel)
                        {
                            uint8_t * data = image.data + row*image.stride + col*pixelSize + channel*channelSize;
                            ifs >> value;
                            for(size_t i = 0; i < channelSize; ++i)
                            {
#ifdef SIMD_BIG_ENDIAN
                                data[i] = (value >> 8*(channelSize - i - 1))&0xFF;
#else
                                data[i] = (value >> 8*i)&0xFF;
#endif                    
                            }
                        }
                    }
                }
            }
            else
            {
                for(size_t row = 0; row < image.height; ++row)
                {
                    for(size_t col = 0; col < image.width; ++col)
                    {
                        if(image.format == View::Float)
                            ifs >> image.At<float>(col, row);
                        else
                            ifs >> image.At<double>(col, row);
                    }
                }
            }
        }
        catch (std::exception e)
        {
            std::cout << "Can't load image from file '" << path << "'!" << std::endl; 
            std::cout << "There is an exception: " << e.what() << std::endl; 
            ifs.close();
            return false;
        }

        ifs.close();

        return true;
    }

    bool Data::Save(const uint64_t & value, const std::string & name) const
    {
        return SaveArray(&value, 1, name);
    }

    bool Data::Load(uint64_t & value, const std::string & name) const
    {
        return LoadArray(&value, 1, name);
    }

    bool Data::Save(const int64_t & value, const std::string & name) const
    {
        return SaveArray(&value, 1, name);
    }

    bool Data::Load(int64_t & value, const std::string & name) const
    {
        uint64_t _value;
        bool result = LoadArray(&_value, 1, name);
        value = (int64_t)_value;
        return result;
    }

    bool Data::Save(const uint32_t & value, const std::string & name) const
    {
        return SaveArray(&value, 1, name);
    }

    bool Data::Load(uint32_t & value, const std::string & name) const
    {
        return LoadArray(&value, 1, name);
    }

    bool Data::Save(const uint8_t & value, const std::string & name) const
    {
        uint32_t _value = value;
        return SaveArray(&_value, 1, name);
    }

    bool Data::Load(uint8_t & value, const std::string & name) const
    {
        uint32_t _value;
        bool result = LoadArray(&_value, 1, name);
        value = (uint8_t)_value;
        return result;
    }

    bool Data::Save(const double & value, const std::string & name) const
    {
        return SaveArray(&value, 1, name);
    }

    bool Data::Load(double & value, const std::string & name) const
    {
        return LoadArray(&value, 1, name);
    }

    bool Data::Save(const float & value, const std::string & name) const
    {
        return SaveArray(&value, 1, name);
    }

    bool Data::Load(float & value, const std::string & name) const
    {
        return LoadArray(&value, 1, name);
    }

    bool Data::Save(const Sums & sums, const std::string & name) const
    {
        return SaveArray(sums.data(), sums.size(), name);
    }

    bool Data::Load(Sums & sums, const std::string & name) const
    {
        return LoadArray(sums.data(), sums.size(), name);
    }

    bool Data::Save(const Histogram & histogram, const std::string & name) const
    {
        return SaveArray(histogram, Simd::HISTOGRAM_SIZE, name);
    }

    bool Data::Load(Histogram & histogram, const std::string & name) const
    {
        return LoadArray(histogram, Simd::HISTOGRAM_SIZE, name);
    }

    bool Data::Save(const Sums64 & sums, const std::string & name) const
    {
        return SaveArray(sums.data(), sums.size(), name);
    }

    bool Data::Load(Sums64 & sums, const std::string & name) const
    {
        return LoadArray(sums.data(), sums.size(), name);
    }

    bool Data::Save(const Rect & rect, const std::string & name) const
    {
        return SaveArray((const ptrdiff_t *)&rect, 4, name);
    }

    bool Data::Load(Rect & rect, const std::string & name) const
    {
        return LoadArray((ptrdiff_t *)&rect, 4, name);
    }

    bool Data::Save(const std::vector<uint8_t> & data, const std::string & name) const
    {
        return SaveArray(data.data(), data.size(), name);
    }

    bool Data::Load(std::vector<uint8_t> & data, const std::string & name) const
    {
        return LoadArray(data.data(), data.size(), name);
    }

    bool Data::Save(const Buffer32f & buffer, const std::string & name) const
    {
        return SaveArray(buffer.data(), buffer.size(), name);
    }

    bool Data::Load(Buffer32f & buffer, const std::string & name) const
    {
        return LoadArray(buffer.data(), buffer.size(), name);
    }

    std::string Data::Description(SimdCompareType type)
    {
        switch(type)
        {
        case SimdCompareEqual:
            return "_Equal";
        case SimdCompareNotEqual:
            return "_NotEqual";
        case SimdCompareGreater:
            return "_Greater";
        case SimdCompareGreaterOrEqual:
            return "_GreaterOrEqual";
        case SimdCompareLesser:
            return "_Lesser";
        case SimdCompareLesserOrEqual:
            return "_LesserOrEqual";
        }
        assert(0);
        return "_Unknown";
    }

    std::string Data::Description(SimdOperationBinary8uType type)
    {
        switch(type)
        {
        case SimdOperationBinary8uAverage:
            return "_Average";
        case SimdOperationBinary8uAnd:
            return "_And";
        case SimdOperationBinary8uMaximum:
            return "_Maximum";
        case SimdOperationBinary8uSaturatedSubtraction:
            return "_SaturatedSubtraction";
        case SimdOperationBinary8uSaturatedAddition:
            return "_SaturatedAddition";
        }
        assert(0);
        return "_Unknown";
    }

    std::string Data::Description(SimdOperationBinary16iType type)
    {
        switch(type)
        {
        case SimdOperationBinary16iAddition:
            return "_Addition";
        }
        assert(0);
        return "_Unknown";
    }

    std::string Data::Description(View::Format format)
    {
        switch(format)
        {
        case View::None:      
            return "_None";
        case View::Gray8:     
            return "_Gray8";
        case View::Uv16:      
            return "_Uv16";
        case View::Bgr24:     
            return "_Bgr24";
        case View::Bgra32:    
            return "_Bgra32";
        case View::Int16:     
            return "_Int16";
        case View::Int32:     
            return "_Int32";
        case View::Int64:     
            return "_Int64";
        case View::Float:     
            return "_Float";
        case View::Double:    
            return "_Double";
        case View::BayerGrbg: 
            return "_BayerGrgb";
        case View::BayerGbrg: 
            return "_BayerGbgr";
        case View::BayerRggb: 
            return "_BayerRggb";
        case View::BayerBggr: 
            return "_BayerBggr";
        }
        assert(0); 
        return "_Unknown";
    }
}