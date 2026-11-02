# CANNTEST

昇腾 **CANN 9.0.0** 自定义算子工程（`AddCustom`：`z = x + y`），目标硬件 **Ascend 910B3**。

工程骨架由 `msopgen` 生成，kernel 与 tiling 为本仓库实现，**已在真机 910B3 上跑通数值验证并做过 msprof 性能调优**。

| 项 | 值 |
|---|---|
| 算子 | `AddCustom`（x + y → z，ND，**fp16 / fp32**） |
| 生成工具 | `msopgen`（CANN 9.0.0 自带，MindStudio 26.0.0） |
| 工程形态 | op_host（原型/tiling）+ op_kernel（Ascend C）+ framework(tf_plugin) + **aclnn 接口（自动生成）** |
| 编译产物 | `build_out/custom_opp_ubuntu_aarch64.run`（约 358 KB） |
| 编译耗时 | **17 秒**（16 vCPU 容器，冷启动） |
| 运行环境 | Ubuntu 20.04 aarch64 · CANN 9.0.0 · 910B3 · HBM 64 GB |
| 真机验证 | ✅ **12/12 全 PASS**（fp32 + fp16 × 6 档 shape，含 100 万元素、非对齐尾长 1000003） |
| 性能（1M 元素） | fp32 **18.8 µs**（638 GB/s）· fp16 **10.9 µs**（569 GB/s），8 核 |

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
│   ├── test_add_custom.cpp   # 真机数值验证（aclnn + ascendcl，fp32/fp16 × 6 档 shape）
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
AddCustom on Ascend 910B3
  fp32 n=6        PASS (mismatch=0)
  fp32 n=8        PASS (mismatch=0)
  fp32 n=1000     PASS (mismatch=0)
  fp32 n=1024     PASS (mismatch=0)
  fp32 n=1048576  PASS (mismatch=0)
  fp32 n=1000003  PASS (mismatch=0)
  fp16 n=6        PASS (mismatch=0)
  fp16 n=8        PASS (mismatch=0)
  fp16 n=1000     PASS (mismatch=0)
  fp16 n=1024     PASS (mismatch=0)
  fp16 n=1048576  PASS (mismatch=0)
  fp16 n=1000003  PASS (mismatch=0)
SUMMARY: ALL PASS (0/12 cases failed)
```

### 六档 shape 各自验证了什么

| shape | 验证点 |
|---|---|
| 6 / 8 | 小于一个 tile，走单核 + 尾块路径；6 元素 \(4×6=24 B\)（fp32）**非 32B 对齐** |
| 1000 / 1024 | 单 tile 边界（1000 不是 tileLength 的整数倍） |
| 1048576 | 8 核 × 多 tile 全流程，1M 元素 |
| 1000003 | 非对齐总长 + 多核，检验最后一块的 `DataCopyPad` 与每核长度向上对齐逻辑 |

fp16 用例：Host 侧用位运算做 fp32→fp16 / fp16→fp32 转换（该 g++ 9.4 不支持 `_Float16`），容差 0.25（fp16 在 64 附近 ULP=0.0625）。

---

## 性能与调优（msprof）

用 `msprof op` 单算子采集（`--kernel-name=AddCustom --aic-metrics=PipeUtilization`），1M 元素：

| `TILE_BYTES` | fp32 1M | fp16 1M | 说明 |
|---|---|---|---|
| 4096 | 31.28 µs | 16.80 µs | 等于最初「1024 元素」配置（fp32） |
| 8192 | 19.42 µs | 10.86 µs | |
| **16384** | **18.42 µs** | **11.06 µs** | **最终采用**（UB 占用 96 KB，安全） |
| 32768 | 19.36 µs | 10.54 µs | 收益已饱和，UB 占用 192 KB 有风险 |
| 65536 | 运行失败 | 运行失败 | 超出单核 UB 容量 |

**优化前后**：

| dtype | 优化前（1024 元素） | 优化后（16 KB） | 提升 |
|---|---|---|---|
| fp32 1M | 31.44 µs | **18.80 µs** | **1.67×** |
| fp16 1M | 31.00 µs | **10.88 µs** | **2.85×** |

关键结论：

1. **tiling 必须按字节而不是按元素**：原配置固定 1024 元素，fp16 每趟只搬 2 KB，等于把带宽砍半 —— 改成 `TILE_BYTES / sizeof(dtype)` 后 fp16 直接快 2.85 倍。
2. **单算子添加类瓶颈在 MTE2**：`aiv_mte2_ratio ≈ 0.93`（搬运占核时间 93%），`aiv_vec_ratio ≈ 0.16`（向量计算只占 16%）→ 纯访存瓶颈，双缓冲已经打满，换算法无收益。
3. **UB 是硬上限**：`TILE_BYTES=65536` 时三路缓冲 ×2 双缓冲 = 384 KB > 单核 UB（192 KB），算子直接跑挂。

有效带宽：fp32 12 MB / 18.8 µs ≈ **638 GB/s**；fp16 6 MB / 10.9 µs ≈ **569 GB/s**（8 核合计，每核 MTE2 约 58 GB/s）。

---

## kernel 实现要点

```cpp
// 每核长度按 32B 向上对齐（元素数随 dtype 变化：fp32=8、fp16=16）
constexpr uint32_t ALIGN_ELEMS = 32u / sizeof(DTYPE_X);
uint32_t per = (total + blockNum - 1) / blockNum;
per = (per + ALIGN_ELEMS - 1u) / ALIGN_ELEMS * ALIGN_ELEMS;
```

```cpp
// Host tiling：按字节定 tile，按 dtype 换算元素数
constexpr uint32_t TILE_BYTES = 16384;
uint32_t dsize = (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16) ? 2u : 4u;
uint32_t tileLength = TILE_BYTES / dsize;
```

1. **多核切分**：`GetBlockNum()/GetBlockIdx()` 均分，块长向上取 32B 整数倍。
2. **双缓冲**：`TQue<QuePosition::VECIN, 2>` / `VECOUT`，搬运与计算流水。
3. **尾块非对齐**：GM↔UB 用 `DataCopyPad`（`DataCopyExtParams`）而不是 `DataCopy`，否则 24 B 这类长度会丢数或越界。
4. **小 shape**：tiling 里 `total < tileLength` 时把 `blockDim` 设为 1，避免 8 核各分 0 个元素（这是骨架默认 `SetBlockDim(8)` 最容易踩的坑）。

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
| 7 | fp16 比 fp32 还慢一倍 | tiling 按元素数固定，fp16 每趟只搬一半字节 | tile 按字节定，再除以 `sizeof(dtype)` |
| 8 | `msprof op` 报 `xxx is writable by any other users or group users` | 同类安全校验 | `chmod 700` 工作目录 |
| 9 | `msprof op --kernel-name=add_custom` 采不到数据（`No profiling data dumped`） | 实际 op name 是 `AddCustom_<hash>_0`，大小写敏感 | 用 `--kernel-name=AddCustom`（子串匹配） |
| 10 | `TILE_BYTES=65536` 算子跑挂 | 三路缓冲 × 双缓冲 = 384 KB > UB 192 KB | 控制在 16 KB（96 KB 占用） |

---

## CI

`.github/workflows/static-check.yml` 在 GitHub 托管 runner 上跑**静态检查**（结构完整性、`bash -n`、JSON 校验、tiling 契约一致、kernel 非空壳、产物未入库）。

真机编译与执行必须进 CANNLab 容器：`bash build.sh && bash tests/run_test.sh`。

---

## 许可

- 仓库内**手写实现与测试**（`op_kernel/add_custom.cpp`、`op_host/add_custom.cpp`、`tests/`）采用 **木兰宽松许可证 v2（Mulan PSL v2）**，全文见 [LICENSE](LICENSE)。
- 由 `msopgen` 生成的骨架文件（`build.sh`、`CMakeLists.txt` 等）保留华为原始版权与许可头（CANN Open Software License Agreement Version 2.0 / Mulan PSL v2），遵循原许可。
