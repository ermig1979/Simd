import os

from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, load, collect_libs, replace_in_file
from conan.tools.scm import Git


class SimdConan(ConanFile):
    name = "simd"
    license = "MIT"
    author = "Ermig1979"
    url = "https://github.com/ermig1979/Simd"
    description = (
        "The Simd Library is a free open source image processing and machine learning library, "
        "designed for C and C++ programmers"
    )
    topics = ("performance", "optimization", "simd", "image-processing")
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "avx512": [True, False],
        "avx512vnni": [True, False],
        "amxbf16": [True, False],
        "synet": [True, False],
        "hide_internal": [True, False],
        "runtime": [True, False],
        "perf": [True, False],
        "python": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "avx512": True,
        "avx512vnni": True,
        "amxbf16": False,
        "synet": True,
        "hide_internal": False,
        "runtime": True,
        "perf": False,
        "python": False,
    }

    def _is_x86(self):
        return str(self.settings.arch) in ("x86", "x86_64")

    def _repo_root(self):
        return os.path.normpath(os.path.join(self.recipe_folder, "..", ".."))

    def set_version(self):
        # In local flow, read from repo; in cache, read from exported file
        for base in [self._repo_root(), self.recipe_folder]:
            version_file = os.path.join(base, "prj", "txt", "UserVersion.txt")
            if os.path.exists(version_file):
                self.version = load(self, version_file).strip()
                return

    def export(self):
        copy(self, "prj/txt/UserVersion.txt", src=self._repo_root(), dst=self.export_folder)

    def export_sources(self):
        root = self._repo_root()
        copy(self, "**", src=os.path.join(root, "src"), dst=os.path.join(self.export_sources_folder, "src"))
        copy(self, "**", src=os.path.join(root, "prj", "cmake"), dst=os.path.join(self.export_sources_folder, "prj", "cmake"))
        copy(self, "**", src=os.path.join(root, "prj", "sh"), dst=os.path.join(self.export_sources_folder, "prj", "sh"))
        copy(self, "**", src=os.path.join(root, "prj", "cmd"), dst=os.path.join(self.export_sources_folder, "prj", "cmd"))
        copy(self, "**", src=os.path.join(root, "prj", "txt"), dst=os.path.join(self.export_sources_folder, "prj", "txt"))
        copy(self, "**", src=os.path.join(root, "prj", "vs2022"), dst=os.path.join(self.export_sources_folder, "prj", "vs2022"))

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
        if not self._is_x86():
            del self.options.avx512
            del self.options.avx512vnni
            del self.options.amxbf16

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self, src_folder=".")

    def source(self):
        if not os.path.exists(os.path.join(self.source_folder, "src", "Simd")):
            self.output.info(f"Source not found. Downloading from {self.url}")
            git = Git(self)
            git.clone(self.url, target=".")
            git.checkout(f"v{self.version}")

    def generate(self):
        if self.settings.os == "Windows":
            from conan.tools.microsoft import MSBuildToolchain
            tc = MSBuildToolchain(self)
            tc.generate()
        else:
            tc = CMakeToolchain(self)
            tc.variables["SIMD_SHARED"] = bool(self.options.shared)
            tc.variables["SIMD_TEST"] = False
            tc.variables["SIMD_OPENCV"] = False
            tc.variables["SIMD_INSTALL"] = False
            tc.variables["SIMD_UNINSTALL"] = False
            tc.variables["SIMD_PYTHON"] = bool(self.options.python)
            tc.variables["SIMD_GET_VERSION"] = True
            tc.variables["SIMD_SYNET"] = bool(self.options.synet)
            tc.variables["SIMD_HIDE"] = bool(self.options.hide_internal)
            tc.variables["SIMD_RUNTIME"] = bool(self.options.runtime)
            tc.variables["SIMD_PERF"] = bool(self.options.perf)
            if self._is_x86():
                tc.variables["SIMD_AVX512"] = bool(self.options.avx512)
                tc.variables["SIMD_AVX512VNNI"] = bool(self.options.avx512vnni)
                tc.variables["SIMD_AMXBF16"] = bool(self.options.amxbf16)
            tc.generate()

    def _win_static_option(self):
        prj_file = os.path.join(self.source_folder, "prj", "vs2022", "Simd.vcxproj")
        replace_in_file(
            self, prj_file,
            "<ConfigurationType>DynamicLibrary</ConfigurationType>",
            "<ConfigurationType>StaticLibrary</ConfigurationType>",
        )
        config_file = os.path.join(self.source_folder, "src", "Simd", "SimdConfig.h")
        replace_in_file(
            self, config_file,
            "//#define SIMD_STATIC",
            "#define SIMD_STATIC",
        )

    def build(self):
        if self.settings.os == "Windows":
            from conan.tools.microsoft import MSBuild
            if not self.options.shared:
                self._win_static_option()
            msbuild = MSBuild(self)
            msbuild.build("prj/vs2022/Simd.sln")
        else:
            cmake = CMake(self)
            cmake.configure(build_script_folder="prj/cmake")
            cmake.build()

    def package(self):
        copy(
            self, "*.h",
            src=os.path.join(self.source_folder, "src", "Simd"),
            dst=os.path.join(self.package_folder, "include", "Simd"),
        )
        copy(
            self, "*.hpp",
            src=os.path.join(self.source_folder, "src", "Simd"),
            dst=os.path.join(self.package_folder, "include", "Simd"),
        )
        # Linux libraries
        copy(
            self, "*.a",
            src=self.build_folder,
            dst=os.path.join(self.package_folder, "lib"),
            keep_path=False,
        )
        copy(
            self, "*.so*",
            src=self.build_folder,
            dst=os.path.join(self.package_folder, "lib"),
            keep_path=False,
        )
        # Windows libraries
        copy(
            self, "*.lib",
            src=self.build_folder,
            dst=os.path.join(self.package_folder, "lib"),
            keep_path=False,
        )
        copy(
            self, "*.dll",
            src=self.build_folder,
            dst=os.path.join(self.package_folder, "bin"),
            keep_path=False,
        )

    def package_info(self):
        self.cpp_info.libs = collect_libs(self)
        if self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.system_libs.extend(["pthread", "dl"])
