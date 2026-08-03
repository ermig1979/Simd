import os

from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.build import can_run


class SimdTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires(self.tested_reference_str)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if can_run(self):
            self.run(os.path.join(self.cpp.build.bindir, "test_simd"), env="conanrun")
            simd = self.dependencies["simd"]
            # get_safe returns a Conan option object; compare its string form explicitly
            # (a non-empty string like "False" is truthy, so `if option:` is unreliable).
            if str(simd.options.get_safe("python")) == "True":
                wrapper_dir = simd.cpp_info.libdirs[0]
                script = os.path.join(self.source_folder, "test_python.py")
                self.run('python3 "%s" "%s"' % (script, wrapper_dir), env="conanrun")
