import os
from conans import ConanFile, CMake, MSBuild, tools


class SimdConan(ConanFile):
    name = "Simd"
    license = "MIT"
    author = "Ermig1979"
    url = "https://github.com/ermig1979/Simd"
    description = "The Simd Library is a free open source image processing and machine learning library, " \
                  "designed for C and C++ programmers"
    topics = ("performance", "optimization", "simd")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False}
    exports_sources = "Simd/*",

    def source(self):
        if not os.path.exists("Simd") or (os.path.isdir("Simd") and len(os.listdir("Simd")) == 0):
            self.output.info(f"Source not found. Downloading from {self.url}")
            git = tools.Git("Simd")
            git.clone(self.url)
            git.checkout("v" + self.version)

    def _configure_cmake(self):
        cmake = CMake(self)
        is_x86 = self.settings.arch == "x86" or self.settings.arch == "x86_64"
        cmake.definitions["SIMD_AVX512"] = is_x86
        cmake.definitions["SIMD_AVX512VNNI"] = is_x86
        cmake.definitions["SIMD_SHARED"] = self.options.shared
        cmake.definitions["SIMD_TEST"] = False
        return cmake

    def _win_static_option(self):
        prj_file = "Simd/prj/vs2019/Simd.vcxproj"
        with open(prj_file, 'r') as file:
            data = file.read()
            data = data.replace("<ConfigurationType>DynamicLibrary</ConfigurationType>",
                                "<ConfigurationType>StaticLibrary</ConfigurationType>")
        with open(prj_file, 'w') as file:
            file.write(data)

        config_file = "Simd/src/Simd/SimdConfig.h"
        with open(config_file, 'r') as file:
            data = file.read()
            data = data.replace("//#define SIMD_STATIC",
                                "#define SIMD_STATIC")
        with open(config_file, 'w') as file:
            file.write(data)

    def build(self):
        if self.settings.os == "Windows":
            if not self.options.shared:
                self._win_static_option()
            msbuild = MSBuild(self)
            msbuild.build("Simd/prj/vs2019/Simd.sln", upgrade_project=False)
        else:
            cmake = self._configure_cmake()
            cmake.configure(source_folder="Simd/prj/cmake")
            cmake.build()

    def _lib_pattern(self):
        if self.options.shared:
            return "*.dll" if self.settings.os == "Windows" else "*.so"
        else:
            return "*.lib" if self.settings.os == "Windows" else "*.a"

    def package(self):
        self.copy(self._lib_pattern(), dst="lib", keep_path=False)
        self.copy("*.h*", dst="include/Simd/", src=f"{self.source_folder}/Simd/src/Simd")

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
