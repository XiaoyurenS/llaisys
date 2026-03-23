target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("cuda")
    add_syslinks("cudart")
    -- 强制 CUDA 代码链接动态 cudart，避免静态 cudart 与运行环境中的 driver/runtime 组合不一致。
    add_cuflags("-cudart=shared")
    add_culdflags("-cudart=shared")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_rules("cuda")
    add_syslinks("cudart")
    -- 与 device target 保持一致，统一使用动态 cudart。
    add_cuflags("-cudart=shared")
    add_culdflags("-cudart=shared")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
