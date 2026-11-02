# CANNTEST

昇腾 **CANN 9.0.0** 自定义算子工程（`AddCustom`），目标硬件 **Ascend 910B3**。

在 CANNLab 云容器（A2 规格）里生成、编译、打包全流程实测通过。

| 项 | 值 |
|---|---|
| 算子 | `AddCustom`（x + y·alpha → z，ND，fp16/fp32） |
| 生成工具 | `msopgen`（CANN 9.0.0 自带） |
| 工程形态 | op_host + op_kernel + framework(tf_plugin) |
| 编译产物 | `build_out/custom_opp_ubuntu_aarch64.run`（约 356 KB） |
| 编译耗时 | **17 秒**（16 vCPU 容器，冷启动） |
| 运行环境 | Ubuntu 20.04 aarch64 · CANN 9.0.0 · 910B3 · HBM 64 GB |

---

## 目录结构

```
CANNTEST/
├── CMakeLists.txt              # 顶层构建入口
├── CMakePresets.json           # 预设构建配置
├── build.sh                    # 一键编译 + CPack 打包
├── op_host/
│   ├── add_custom.cpp          # 算子原型注册 + tiling（Shape/数据类型推导）
│   └── CMakeLists.txt
├── op_kernel/
│   ├── add_custom.cpp          # Ascend C kernel 实现
│   ├── add_custom_tiling.h     # tiling 结构体
│   └── CMakeLists.txt
└── framework/
    └── tf_plugin/
        └── tensorflow_add_custom_plugin.cc
```

---

## 快速开始

```bash
# 1) 环境（容器内必须先做，否则找不到 libhccl.so / 编译器）
source /home/developer/Ascend/ascend-toolkit/set_env.sh

# 2) 编译 + 打包
cd CANNTEST
bash build.sh

# 3) 产物
ls -lh build_out/*.run
# build_out/custom_opp_ubuntu_aarch64.run
```

解包内容（编译日志实测）：

```
packages/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/add_custom/*.o   # 2 个 kernel bin
packages/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so
packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so
packages/vendors/customize/version.info
```

> `build_out/` 与 `*.run` 不入库（见 `.gitignore`），换机器重新 `bash build.sh` 即可。

---

## 改算子 / 重新生成工程

改算子签名（新增输入、改 dtype、加 attr）建议用 `msopgen` 重新生成，再迁移业务代码：

```bash
umask 077                                   # 关键：msopgen 拒绝 group/other 可写的路径
mkdir -p ~/tmpgen && chmod 700 ~/tmpgen && cd ~/tmpgen
cat > add_custom.json <<'JSON'
[
  {
    "op": "AddCustom",
    "language": "cpp",
    "input_desc": [
      {"name": "x", "param_type": "required", "format": ["ND"], "type": ["float16", "float"]},
      {"name": "y", "param_type": "required", "format": ["ND"], "type": ["float16", "float"]}
    ],
    "output_desc": [
      {"name": "z", "param_type": "required", "format": ["ND"], "type": ["float16", "float"]}
    ],
    "attr_desc": [{"name": "alpha", "type": "float", "param_type": "optional", "default_value": 1.0}]
  }
]
JSON
chmod 600 add_custom.json
msopgen gen -i add_custom.json -f tf -c ai_core-Ascend910B3 -lan cpp -out ./AddCustom
```

`-f tf` 生成 tensorflow 插件；纯 torch / aclnn 场景可换 `-f pytorch` 或去掉 framework 层只留 op_host/op_kernel。

---

## 踩坑记录

| # | 现象 | 原因 | 解法 |
|---|---|---|---|
| 1 | `msopgen` 报 `The path xxx.json should not be written by user group or others` | 同组/其他用户可写 → 安全校验 | `umask 077` + `chmod 600` 文件、`chmod 700` 目录 |
| 2 | `build.sh` 报找不到编译器 / `libhccl.so` | 未 source 工具链 | `source .../ascend-toolkit/set_env.sh` |
| 3 | 打包阶段才失败 | 磁盘/permission | 确认 `build_out/` 可写、磁盘 ≥ 1 GB |

---

## 许可

工程骨架由 `msopgen` 生成，含 Huawei CANN Open Software License Agreement Version 2.0 头注释，遵循原许可。
