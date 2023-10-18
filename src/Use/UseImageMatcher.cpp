/*
*  The use examples of Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#if (defined(__cplusplus) && (__cplusplus >= 201703L)) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201703L))
#ifdef SIMD_USE_INSIDE
#define main UseImageMatcher
#endif

#include <iostream>
#include <string>
#include <filesystem>

#include "Simd/SimdImageMatcher.hpp"

int main(int argc, char * argv[])
{
    typedef Simd::ImageMatcher<size_t, Simd::Allocator> ImageMatcher;
    typedef ImageMatcher::View Image;
    typedef std::vector<std::string> Strings;
    namespace fs = std::filesystem;

    if (argc < 2)
    {
        std::cout << "You have to set input directory with images!" << std::endl;
        return 1;
    }
    std::string path = argv[1];
    fs::file_status status = fs::status(path);
    if (!fs::is_directory(status))
    {
        std::cout << "Input '" << path << "' directory is not exist!" << std::endl;
        return 1;
    }
    double threshold = 0.05;
    if (argc > 2)
        threshold = std::strtod(argv[2], NULL);

    std::cout << "Start search images in '" << path << "' :" <<std::endl;
    Strings src;
    for (fs::recursive_directory_iterator it(path); it != fs::recursive_directory_iterator(); ++it)
    {
        if (it->is_regular_file() && it->path().extension() == ".jpg")
            src.push_back(it->path().string());
    }
    std::cout << src.size() << " images were found." << std::endl;

    ImageMatcher matcher;  
    matcher.Init(threshold, ImageMatcher::Hash32x32, src.size());
    for (size_t i = 0; i < src.size(); ++i)
    {
        Image image;
        if (!image.Load(src[i]))
        {
            std::cout << "Can't load image " << src[i] << " !" << std::endl;
            continue;
        }
        ImageMatcher::HashPtr hash = matcher.Create(image, i);
        ImageMatcher::Results results;
        std::cout << std::setprecision(2) << std::fixed;
        if (matcher.Find(hash, results))
        {
            for (size_t r = 0; r < results.size(); ++r)
                std::cout << "Duplicates: " << src[i] << " and " << src[results[r].hash->tag] << " , msd = " << results[r].difference*100 << "%." << std::endl;
        }
        matcher.Add(hash);
    }
    return 0;
}
#endif

