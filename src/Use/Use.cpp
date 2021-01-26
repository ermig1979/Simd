/*
* The use examples of Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
*
* In order to be enable of OpenCV examples for Visual Studio 2015/2019 you have to rename file 'Ocv.prop.default' into 'Ocv.prop' and set there correct paths to OpenCV.
*/
#include <iostream>
#include <string>
#include <vector>

struct Example
{
    typedef int(*Function)(int argc, char * argv[]);

    std::string name;
    Function func;

    Example(const std::string & n, const Function & f)
        : name(n)
        , func(f)
    {
    }
};
typedef std::vector<Example> Examples;
Examples g_examples;

#ifdef SIMD_OPENCV_ENABLE
#define USE_ADD_OPENCV_EXAMPLE(name) \
    int Use##name(int argc, char * argv[]); \
    bool Use##name##Add(){ g_examples.push_back(Example(#name, Use##name)); return true; } \
    bool Use##name##Added = Use##name##Add();
#else
#define USE_ADD_OPENCV_EXAMPLE(name)
#endif

USE_ADD_OPENCV_EXAMPLE(FaceDetection);
USE_ADD_OPENCV_EXAMPLE(MotionDetector);

void Print()
{
    std::cout << "The list of existed examples: " << std::endl;
    for (size_t i = 0; i < g_examples.size(); ++i)
        std::cout << g_examples[i].name << std::endl;
    std::cout << std::endl;
}

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        std::cout << "You have to set example name! For example: " << std::endl << std::endl;
        std::cout << " Use.exe " << g_examples[0].name << std::endl << std::endl;
        Print();
        return 1;
    }

    std::string name = argv[1];
    argv[1] = argv[0];

    for (size_t i = 0; i < g_examples.size(); ++i)
    {
        if (g_examples[i].name == name)
        {
            std::cout << "Start '" << name << "' example: " << std::endl;
            return g_examples[i].func(argc - 1, argv + 1);
        }
    }

    std::cout << "You use unknown example name '" << name << "'!" << std::endl;
    Print();
    return 1;
}
