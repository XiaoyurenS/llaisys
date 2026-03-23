add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

option("cpu-openmp")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to enable OpenMP for CPU operators")
option_end()

option("cpu-openblas")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to enable OpenBLAS-backed CPU linear")
option_end()

option("openblas-incdir")
    set_default("")
    set_showmenu(true)
    set_description("OpenBLAS include directory")
option_end()

option("openblas-linkdir")
    set_default("")
    set_showmenu(true)
    set_description("OpenBLAS library directory")
option_end()

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    if has_config("cpu-openmp") and not is_plat("windows") then
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
        add_defines("LLAISYS_USE_OPENMP")
    end
    if has_config("cpu-openblas") then
        add_defines("LLAISYS_USE_OPENBLAS")
        local incdir = get_config("openblas-incdir")
        if incdir ~= nil and incdir ~= "" then
            add_includedirs(incdir)
        end
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    if has_config("cpu-openmp") and not is_plat("windows") then
        add_ldflags("-fopenmp")
    end
    if has_config("cpu-openblas") then
        local linkdir = get_config("openblas-linkdir")
        if linkdir ~= nil and linkdir ~= "" then
            add_linkdirs(linkdir)
        end
        if is_plat("linux") then
            add_ldflags("-Wl,--no-as-needed")
        end
    end
    add_files("src/llaisys/*.cc")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
        if is_plat("macosx") then
            os.cp("lib/*.dylib", "python/llaisys/libllaisys/")
        end
    end)

    if has_config("cpu-openblas") then
        add_syslinks("openblas")
    end
target_end()