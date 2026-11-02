# CANNTEST

昇腾 **CANN 9.0.0** 自定义算子工程（`AddCustom`：`z = x + y`），目标硬件 **Ascend 910B3**。

工程骨架由 `msopgen` 生成，kernel 与 tiling 为本仓库实现，**已在真机 910B3 上跑通数值验证**。

| 项 | 值 |
|---|---|
| 算子 | `AddCustom`（x + y → z，ND，fp16 / fp32） |
| 生成工具 | `msopgen`（CANN 9.0.0 自带，MindStudio 26.0.0） |
| 工程形态 | op_host（原型/tiling）+ op_kernel（Ascend C）+ framework(tf_plugin) + **aclnn 接口（自动生成）** |
| 编译产物 | `build_out/custom_opp_ubuntu_aarch64.run`（约 358 KB） |
| 编译耗时 | **17 秒**（16 vCPU 容器，冷启动） |
| 运行环境 | Ubuntu 20.04 aarch64 · CANN 9.0.0 · 910B3 · HBM 64 GB |
| 真机验证 | ✅ **6/6 shape 全 PASS**（含 100 万元素、非对齐尾长 1000003） |

---

## 目录结构

```
CANNTEST/
├── CMakeLists.txt / CMakePresets.json / build.sh   # 构建入口（msopgen 生成）
├── LICENSE                                          # 木兰宽松许可证 v2
├── op_host/
│   └── add_custom.cpp        # 算子原型 + InferShape/InferDataType + TilingFunc（本仓库实现）
├── op_kernel/
│   ├── add_custom.cpp        # Ascend C kernel：双缓冲 + 多核切分 + DataCopyPad 尾块
│   └── add_custom_tiling.h   # tiling 结构：totalLength / tileLength / tileNum
├── framework/tf_plugin/      # TensorFlow 插件（msopgen 生成，未启用）
├── tests/
│   ├── test_add_custom.cpp   # 真机数值验证（aclnn + ascendcl）
│   └── run_test.sh           # 一键编译并运行
└── .github/workflows/static-check.yml   # 静态检查（GitHub runner 无 NPU）
```

---

## 快速开始（CANNLab 容器内）

```bash
# 1) 环境
source /home/developer/Ascend/ascend-toolkit/set_env.sh

# 2) 编译 + 打包（约 17 秒）
cd CANNTEST && bash build.sh
ls -lh build_out/custom_opp_ubuntu_aarch64.run

# 3) 安装算子包（装到用户目录，避免动只读的 toolkit opp）
./build_out/custom_opp_ubuntu_aarch64.run --quiet --install-path=$HOME/Ascend/custom_opp

# 4) 真机验证
bash tests/run_test.sh
```

输出：

```
AddCustom on Ascend 910B3, float32
  n=6        PASS (mismatch=0)
  n=8        PASS (mismatch=0)
  n=1000     PASS (mismatch=0)
  n=1024     PASS (mismatch=0)
  n=1048576  PASS (mismatch=0)
  n=1000003  PASS (mismatch=0)
SUMMARY: ALL PASS (0/6 cases failed)
```

### 四个 shape 档位各自验证了什么

| shape | 验证点 |
|---|---|
| 6 / 8 | 小于一个 tile，走单核 + 尾块路径；6 元素即 24 B，**非 32B 对齐** |
| 1000 / 1024 | 单 tile 边界（1000 不是 tileLength 的整数倍） |
| 1048576 | 8 核 × 多 tile 全流程，1M 元素 |
| 1000003 | 非对齐总长 + 多核，检验最后一块的 `DataCopyPad` 与每核长度向上对齐逻辑 |

---

## kernel 实现要点

```cpp
// 每核长度按 32B（8 个 fp32）向上对齐，保证 GM 起始地址对齐
uint32_t per = (total + blockDim - 1) / blockDim;
per = (per + 7u) / 8u * 8u;
uint32_t offset = idx * per;
len_ = (offset >= total) ? 0u : ((total - offset) < per ? (total - offset) : per);
```

1. **多核切分**：`GetBlockNum()/GetBlockIdx()` 均分，块长向上取 32B 整数倍。
2. **双缓冲**：`TQue<QuePosition::VECIN, 2>` / `VECOUT`，搬运与计算流水。
3. **尾块非对齐**：GM↔UB 用 `DataCopyPad`（`DataCopyExtParams`）而不是 `DataCopy`，否则 24 B 这类长度会丢数或越界。
4. **小 shape**：tiling 里 `total < TILE_LENGTH` 时把 `blockDim` 设为 1，避免 8 核各分 0 个元素（这是骨架默认 `SetBlockDim(8)` 最容易踩的坑）。

---

## 改算子 / 重新生成工程

改签名（新增输入、改 dtype、加 attr）建议用 `msopgen` 重新生成再迁移业务代码：

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
    ]
  }
]
JSON
chmod 600 add_custom.json
msopgen gen -i add_custom.json -f tf -c ai_core-Ascend910B3 -lan cpp -out ./AddCustom
```

> 重新生成会覆盖 `op_host/` 与 `op_kernel/`，先 `git stash` 或复制业务代码。

---

## 踩坑记录

| # | 现象 | 原因 | 解法 |
|---|---|---|---|
| 1 | `msopgen` 报 `The path xxx.json should not be written by user group or others` | 同组/其他用户可写 → 安全校验 | `umask 077` + `chmod 600` 文件、`chmod 700` 目录 |
| 2 | `build.sh` 找不到编译器 / `libhccl.so` | 未 source 工具链 | `source .../ascend-toolkit/set_env.sh` |
| 3 | 真机执行结果全 0 | `msopgen` 生成的 kernel 是 `// TODO: user kernel impl` **空壳** | 必须自己实现 kernel |
| 4 | 小 shape 无输出 | 骨架默认 `SetBlockDim(8)`，6 个元素时每核分到 0 个 | tiling 里按 shape 调整 blockDim |
| 5 | 非 32B 对齐长度丢尾部数据 | `DataCopy` 对非对齐长度不可靠 | 用 `DataCopyPad` + `DataCopyExtParams` |
| 6 | 装算子包到 toolkit 的 `opp/` 失败 | `opp/` 只读（`dr-xr-x---`） | `--install-path=$HOME/Ascend/custom_opp`，运行时设 `ASCEND_CUSTOM_OPP_PATH` |

---

## CI

`.github/workflows/static-check.yml` 在 GitHub 托管 runner 上跑**静态检查**（结构完整性、`bash -n`、JSON 校验、tiling 契约一致、kernel 非空壳、产物未入库）。

真机编译与执行必须进 CANNLab 容器：`bash build.sh && bash tests/run_test.sh`。

---

## 许可

- 仓库内**手写实现与测试**（`op_kernel/add_custom.cpp`、`op_host/add_custom.cpp`、`tests/`）采用 **木兰宽松许可证 v2（Mulan PSL v2）**，全文见 [LICENSE](LICENSE)。
- 由 `msopgen` 生成的骨架文件（`build.sh`、`CMakeLists.txt` 等）保留华为原始版权与许可头（CANN Open Software License Agreement Version 2.0 / Mulan PSL v2），遵循原许可。
