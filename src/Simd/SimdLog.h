/*
* Simd Library (http://simd.sourceforge.net).
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
#ifndef __SimdLog_h__
#define __SimdLog_h__

#include "Simd/SimdConst.h"

#ifdef SIMD_LOG_ENABLE
#include <iostream>

namespace Simd
{
#ifdef SIMD_VMX_ENABLE
    namespace Vmx
    {
        template<class T> SIMD_INLINE void Log(const T * data, size_t size, const std::string & name)
        {
            std::cout << name << " = { ";
            for(int i = 0; i < size; i++)
            {
                std::cout << int(data[i]) << " ";
            }
            std::cout << "} " << std::endl;    
        }

        SIMD_INLINE void Log(const v128_u8 & value, const std::string & name)
        {
            std::cout << name << " = { ";
            for(int i = 0; i < 16; i++)
            {
                int element = vec_extract(value, i);
                std::cout << element << " ";
            }
            std::cout << "} " << std::endl;    
        }

        SIMD_INLINE void Log(const v128_u16 & value, const std::string & name)
        {
            std::cout << name << " = { ";
            for(int i = 0; i < 8; i++)
            {
                int element = vec_extract(value, i);
                std::cout << element << " ";
            }
            std::cout << "} " << std::endl;    
        }

        SIMD_INLINE void Log(const v128_s16 & value, const std::string & name)
        {
            std::cout << name << " = { ";
            for(int i = 0; i < 8; i++)
            {
                int element = vec_extract(value, i);
                std::cout << element << " ";
            }
            std::cout << "} " << std::endl;    
        }

        SIMD_INLINE void Log(const v128_u32 & value, const std::string & name)
        {
            std::cout << name << " = { ";
            for(int i = 0; i < 4; i++)
            {
                int element = vec_extract(value, i);
                std::cout << element << " ";
            }
            std::cout << "} " << std::endl;    
        }
    }
#endif//SIMD_VMX_ENABLE
}

#define SIMD_LOG(value) Log(value, #value)

#else//SIMD_LOG_ENABLE

#define SIMD_LOG(value)

#endif//SIMD_LOG_ENABLE 

#endif//__SimdLog_h__
