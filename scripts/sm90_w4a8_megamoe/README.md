# SM90 packed-W4A8 MegaMoE

本地分支：`v4.1/h20-dev-pd`。集成 DeepGEMM [PR #53](https://github.com/sgl-project/DeepGEMM/pull/53)
的 Hopper W4A8 融合专家路径，固定源码版本见 [source.lock.json](source.lock.json)。
PR 尚未合入；普通 `sgl-deep-gemm==0.2.2` wheel 不能替代此 PR 的构建。

## 接入与数据布局

- 加载 DSV4.1 的 `Fp8Config(is_fp4_experts=True)` 或原生 `Mxfp4Config`
  权重时，在 SM90 + `megamoe` 路径使用专用 transform。
- 专家权重保持 E2M1 packed FP4。checkpoint 的 gate/up 分半布局转成
  gran-8 行交错布局；逐行 K32 scale 转成 K-major packed UE8M0 int32，
  四个指数字节合成一个 int32。gate/up 的 scale 同步重排。
- 参数直接指向重排后的张量，释放旧布局，不保留整份 FP8 专家副本。
  加载过程中仍存在当前层的重排临时张量，不承诺零额外峰值内存。
- 输入 BF16 先由 `mega_moe_pre_dispatch_sm90` 做 FP8 g128 量化和路由元数据写入，
  然后调用 `fp8_fp4_mega_moe(recipe=(1,1,32))`。融合核完成跨 rank dispatch、
  两次 GEMM、SwiGLU 和 combine，第二次 GEMM 的激活为 FP8 g64；输出 BF16。
  shared expert 继续走已有的独立路径。
- 使用 PR 自带的 FP4 解码、流水和小 batch swapAB 调度，不在 SGLang 中复制 CUDA 实现。

## CP、BCG 与其他路径

- 沿用现有 CP/DP 本地行路由：每个 rank 传自己的 MoE 行；CP 非连续视图在
  native API 边界转成连续输入。跨 rank dispatch/combine 由 MegaMoE 完成。
- 空 rank 仍参加两次 native 调用，避免其他 rank 等待；输出返回零行。
- 图捕获沿用现有 symmetric buffer 缓存，输入的物理 padding 行完整保留，
  无效 top-k id 保持 `-1`。没有新增 `.item()`、CPU 同步或逐 forward 内存注册。
- 保留现有 BCG 分段、decoder SWA bounded replay、auto128 indexer 及
  target/draft 各自的 MoE 后端选择；本次不修改 attention/indexer 的 metadata。
- 专家重排后不能回退到普通 grouped/indexed GEMM。容量不足时显式报错，
  包括图捕获 bucket 超出容量的情况。
- 此适配支持 bias-free SwiGLU，不支持 W4A4 激活、SiTU/OAI 特殊激活参数、
  router weight 乘在输入上的变体。H/I 必须为 128 的倍数，I 不超过 4096。
  DSV4.1-Flash 的 H=5120、I=2304 满足这些形状条件。

## 构建依赖（Linux CUDA 机器）

当前应使用相邻的 `../DeepGEMM` 仓库：`0.2.2-sm90-megamoe` 分支，
commit `3d47168f6015c7d454c4ec0f4e41db3c388dca2c`，已推送 AntCode。
其中 `100525e` 为内核移植，`3d47168` 将固定版本的 CUTLASS/DeepJIT 完整源码
纳入普通 Git 文件。工作机器无需访问 GitHub 或执行 submodule 初始化即可获得源码：

```bash
git clone --branch 0.2.2-sm90-megamoe https://code.alipay.com/csy512356/DeepGEMM.git ../DeepGEMM
```

该移植以 `v0.2.2`
（`ec226bcf`）为 base，保留其原生 SM90 block32 GEMM、SM100 TRT-LLM
权重布局 API，并接入 PR #53。冲突分析和静态验证记录在该仓库的
`SM90_W4A8_PORT.md`，版本映射记录在本目录 `source.lock.json` 的 `local_port`。

`3rdparty/DeepGEMM` 仅保留原始 PR 的独立源码 checkout，作为本地移植参考，
不是 SGLang 的 gitlink/submodule，不参与当前构建。原始 PR 和依赖版本均记录在
`source.lock.json`，当前应构建的是其 `local_port` 指向的 AntCode 版本。

以下构建/安装步骤本次未执行。应在目标 CUDA Python 环境中运行，先检查 PyTorch、
CUDA toolkit、C++20 编译器和 upstream 构建依赖；具体依赖以此固定版本的
`sgl_deep_gemm/pyproject.toml` 和 `build_sgl_deep_gemm.sh` 为准。

```bash
cd ../DeepGEMM
bash build_sgl_deep_gemm.sh
python -m pip install --no-deps --force-reinstall dist/sgl_deep_gemm-*.whl
```

必须使用 `build_sgl_deep_gemm.sh` 的 TVM-FFI distribution，不能用根目录
`pip install .` 构建另一套 pybind API。upstream 脚本会重建其自身 `build/` 目录。
安装后检查：

```bash
python -c 'import deep_gemm as d; print(d.__file__); assert all(hasattr(d, n) for n in ("fp8_fp4_mega_moe", "mega_moe_pre_dispatch_sm90", "transform_weights_for_mega_moe_sm90_fp4"))'
```

## 服务配置

在原有 CP/BCG/Engram 服务命令中选择：

```text
--moe-a2a-backend megamoe
--moe-runner-backend auto
```

W4A8 不需要 W4A4 开关。`SGLANG_DSV4_FP4_DEQUANT` 必须保持未设置或 `0`，
显式设成 `1` 会沿用原有 FP4→FP8 展开路径。DeepGEMM 的
`DG_USE_FP4_ACTS`、`DG_USE_FP8_COMBINE` 应保持未设置或 `0`。

`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` 必须覆盖每个 rank 的
最大物理 MoE 行数，包括 prefill chunk、prefill BCG bucket、decode/verify
图 bucket × 每请求投机 token 数。CP prefill 使用分片后的行数；DP decode
使用各 rank 本地 bucket。增大容量也会增大 symmetric buffer，不保证任何配置都能放入显存。
保留 `--enable-prefill-cp`、原有 CP decode mode、BCG 和
`--enable-decoder-swa-bounded-replay` 等其他启动参数。

## 验证

CPU 接口回归（需要完整 SGLang CPU 测试依赖）：

```bash
PYTHONPATH=python python -m unittest discover -s test/registered/unit/layers/moe -p test_mega_moe_deepgemm_api.py
```

SM90 分布式最小数值/图回放检查（需要至少两卡，建议四卡）：

```bash
PYTHONPATH=python torchrun --standalone --nproc-per-node=4 test/manual/kernels/test_sm90_w4a8_megamoe.py
```

该 fixture 检查真实 FP4 权重重排、非连续 CP 输入、空 rank、带 `-1` padding
的图捕获，以及改变输入和路由后的重复 replay；相对 FP32 解包权重参考设置
固定 5% relative-L2 上限，用于检查两次 FP8 激活量化后的误差，不代表模型精度结论。

本机为 macOS，缺少 CUDA/PyTorch，当前仅能进行静态验证。
尚需目标 H20 上的 native 构建、上述 smoke、完整模型 CP/BCG/bounded replay
正确性与端到端性能验证；不能将代码接入等同于这些测试已通过。
