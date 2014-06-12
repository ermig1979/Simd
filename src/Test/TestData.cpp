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
        assert(image.format != View::Float && image.format != View::Double);

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
        assert(image.format != View::Float && image.format != View::Double);

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
                            data[i] = ((uint8_t*)&value)[7 - i];
#else
                            data[i] = ((uint8_t*)&value)[i];
#endif                    
                        }
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
}