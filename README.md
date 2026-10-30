# imgsort —— 用 Rust 写的批量图片识别分类工具

给一个目录、给一串类别名，`imgsort` 会把目录里的每张图用 **CLIP ViT-B/32** 跑一遍零样本分类，按内容分文件夹放好，并输出 CSV 报告（文件名、类别、置信度、Top3）。

整套东西是**纯 Rust**：推理用 [tract](https://github.com/sonos/tract) 读 ONNX，不需要 PyTorch、不需要 Python、不需要 GPU，也不需要 onnxruntime —— 最后这点对龙芯 loongarch64 尤其关键（没有官方预编译包）。本文所有命令和性能数据都在龙芯 3B6000M + AOSC OS 上实测过。

- 类别名随便写：`-c '猫,狗,发票,截图'` 甚至 `-c '猫的表情包,带二维码的截图'`
- 支持中文类别名（内置 80+ 条中文 → 英文 prompt 映射）
- 多核并行（rayon），8 核约 200 张/分钟
- 坏图不会中断任务，会被标成 `_失败` 写进报告
- 落盘方式可选：复制 / 移动 / 软链接 / 只出报告不动文件

---

## 目录结构

```
imgsort/
├── Cargo.toml
├── Cargo.lock
├── LICENSE                  # MIT
├── README.md                # 就是这篇
├── scripts/
│   └── download-models.sh   # 拉模型（600MB，不进 git）
├── examples/
│   └── inspect.rs           # 打印模型输入输出，换模型时用来核对
└── src/
    ├── main.rs              # CLI、并行调度、自动 batch、缓存、落盘、CSV 报告
    ├── clip.rs              # ONNX 加载 + 图像/文本编码 + CLIP 预处理 + batch 推理
    └── taxonomy.rs          # 中文类别 -> CLIP 英文 prompt 映射表
```

`models/` 目录不进版本库，用脚本拉。

---

## 一、从零跑起来（教程）

### 0. 前提

- Rust 工具链 1.99+（`rustc -vV` 看看）
- 能访问某个 HuggingFace 镜像；本机 huggingface.co 不通，走 `hf-mirror.com`
- 磁盘：模型 580MB + 编译产物

crates.io 直连被墙的话配个镜像（`~/.cargo/config.toml`）：

```toml
[source.crates-io]
replace-with = "tuna"

[source.tuna]
registry = "sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/"
```

### 1. 拿代码

```bash
git clone https://git.loongfans.cn/tsuibin/imgsort.git
cd imgsort
```

### 2. 下载模型

模型是 Xenova 导出的 CLIP ViT-B/32 ONNX：**视觉塔 + 文本塔 + 分词器**，共约 580MB。

```bash
bash scripts/download-models.sh
# 等价于：
# base=https://hf-mirror.com/Xenova/clip-vit-base-patch32/resolve/main
# curl -sL -o models/vision_model.onnx $base/onnx/vision_model.onnx
# curl -sL -o models/text_model.onnx   $base/onnx/text_model.onnx
# curl -sL -o models/tokenizer.json    $base/tokenizer.json
```

### 3. 编译

```bash
cargo build --release      # 产物：target/release/imgsort
```

首次要把 tract 全家桶编一遍，慢一点正常。

### 4. 第一次分类

先别动你的原图：

```bash
./target/release/imgsort \
    -i ~/图片 \
    -o ~/图片_分类 \
    -c '人物,动物,食物,风景,建筑,车辆,植物,文档,截图' \
    --action none
```

`-c` 是类别列表，`--action none` 表示只出报告不落盘。看一眼 `<out>/report.csv` 确认效果：

```csv
文件,类别,置信度,Top3,源文件
cat.png,动物,0.7885,动物:0.788 | 风景:0.201 | 人物:0.011,/tmp/imgsort-test/cat.png
pic33.jpg,风景,0.9780,风景:0.978 | 人物:0.013 | 动物:0.009,/tmp/imgsort-test/pic33.jpg
broken.jpg,_失败,0.0000,解码图片失败: /tmp/x/broken.jpg,/tmp/x/broken.jpg
```

### 5. 正式落盘

```bash
# 复制（推荐，原图不动）
./target/release/imgsort -i ~/图片 -o ~/图片_分类 -c '人物,动物,风景' --action copy

# 软链接：几乎不占空间，原图可以原位不动
./target/release/imgsort -i ~/图片 -o ~/图片_分类 -c '人物,动物,风景' --action symlink

# 移动：会改动原图位置，跑之前务必先用 --action none 看报告
./target/release/imgsort -i ~/图片 -o ~/图片_分类 -c '人物,动物,风景' --action move
```

输出结构：

```
图片_分类/
├── 动物/cat.png            # （或指向原图的软链）
├── 人物/pic11.jpg
├── 风景/pic33.jpg
└── report.csv
```

### 6. 常用姿势

```bash
# 没把握的丢进“其他”技能 contingency：最高置信度低于 0.6 就归到“其他”
./target/release/imgsort -i ~/图片 -c '猫,狗,风景' --min-score 0.6

# 直接写英文类别也行（没命中映射表就原样用）
./target/release/imgsort -i ~/图片 -c 'a cat, a dog, a screenshot of code, an invoice'

# 只看顶层目录、指定线程数、指定 batch
./target/release/imgsort -i ~/图片 -c '人物,风景' --no-recursive --jobs 4 --batch 1
```

---

## 二、它是怎么工作的

```
图片目录 ──> 遍历 ──> rayon 线程池（按 batch 分块）
                        │
                        ├─ image 解码 -> 短边缩放到 224 -> 中心裁剪 224x224
                        ├─ /255 -> (x-mean)/std -> NCHW f32
                        └─ tract 跑 vision_model.onnx -> 512 维向量 -> L2 归一化
                                                          │
类别名 ──> 中文映射 ----> "a photo of xxx" ---> tokenizer -> text_model.onnx -> 512 维
                                                          │
                                        cosine * 100 -> softmax -> 取最高 -> 落盘 + CSV
```

三个要点：

1. **零样本分类**：CLIP 把图片和文本映射到同一个向量空间，所以类别不需要训练，写句话就行 —— 这是它能用中文主题名当类别的原因。
2. **中文类别必须翻译成英文再喂**. CLIP 的文本塔是英文语料训练的，中文 token 基本没意义。实测把「动物」直接当 prompt 时，一张猫图会被判成「风景」；映射成 `a photo of an animal` 后才正常。
3. **prompt 模板要统一**：不带模板（裸 `an animal`）效果也差一截，`taxonomy.rs` 里统一套 `a photo of ...`。

---

## 三、参数

| 参数 | 说明 |
| --- | --- |
| `-i/--input` | 输入目录（默认递归子目录，`--no-recursive` 只看顶层） |
| `-o/--out` | 输出目录，默认 `<input>/_classified` |
| `-c/--categories` | 类别列表，逗号分隔 |
| `--model-dir` | 模型目录，默认 `./models` |
| `--action` | `copy` / `move` / `symlink` / `none`（只出报告） |
| `--min-score` | 最高概率低于该值归入「其他」，0 表示不启用 |
| `--report` | CSV 报告路径，默认 `<out>/report.csv` |
| `--jobs` | 并发线程数，默认等于 CPU 核数 |
| `--batch` | 一次前向同时推理几张图；**不写就自动选**（见下） |
| `--force-calibrate` | 忽略 batch 校准缓存，重新跑一遍探针 |

支持的图片格式：jpg/jpeg/png/webp/bmp/gif/tiff/avif/jfif。

---

## 四、性能与 batch 自动选择

龙芯 3B6000M，8 核，AOSC：

- 模型加载（含 tract 图优化）约 3 秒
- 单张 CPU 耗时约 1.1 秒，8 线程并行约 **3~4 张/秒**（≈200 张/分钟）

### 为什么默认是 batch=1

直觉上“把图堆一起批量推理更快”，但在这台机器上是反的：

| 场景 | batch=1 | batch=4 | batch=8 | batch=16 | batch=32 |
| --- | --- | --- | --- | --- | --- |
| 240 张 / 8 线程 | **41.5s** | 71.9s | 122.6s | 95.1s | 99.3s |
| 40 张 / 单线程 | 46.4s | 44.9s | 45.9s | 53.1s | — |

tract 在 loongarch64 上跑的是通用 matmul 内核，batch 一大中间激活就把 cache 撑爆、多线程还互相抢内存带宽，比乘积效应还狠。所以别手动调大 `--batch`。

### 自动选择（默认行为）

不给 `--batch` 时程序自己挑：

1. **先按图片数筛候选**：候选 `1/2/4/8`，要求 `ceil(图片数/batch) >= 线程数`（并行度不能被 batch 吃掉）、末尾补零浪费 ≤ 20%，图片数不够多就不考虑 8。
2. **再实跑一小段探针**：拿 `线程数×2` 张假图，候选值逐个建图 + 走一遍和正式流程**一样的 rayon 分块路径**，取「每张耗时」最小的。必须走多线程路径 —— 串行测是测不出争用的。测得票面接近时（差 3% 以内）维持小 batch（省内存）。
3. 图片数 < 256 张跳过校准，直接 batch=1。
4. **结果有缓存**：`$XDG_CACHE_HOME/imgsort/batch-calibration.json`（默认 `~/.cache/imgsort/`），key 是 `架构 + 线程数 + 探针数 + 候选集 + 视觉模型 size/mtime`，还记着每个候选的实测每张耗时。换线程数、换模型会自动失效；`--force-calibrate` 强制重测；写不进去就静默跳过。

实际输出：

```
$ ./target/release/imgsort -i ~/Pictures -c '动物,风景,人物' --action none
自动选 batch：16 张探针图，候选 [1, 2, 4]
  batch=1     166 ms/张
  batch=2     228 ms/张
  batch=4     348 ms/张
加载 CLIP 模型 (3 类别, batch=1, jobs=8)…

$ ./target/release/imgsort -i ~/Pictures -c '动物,风景,人物' --action none   # 第二次
batch 校准命中缓存 -> batch=1（batch=1 180ms/张, batch=2 248ms/张, batch=4 380ms/张）
```

320 张图实测：首次含校准 84s，之后命中缓存 69s，省下约 15 秒的重复建图 + 探针。

---

## 五、加自己的类别

`src/taxonomy.rs` 顶部有一张表，左边中文、右边 CLIP 英文 prompt：

```rust
const MAP: &[(&str, &str)] = &[
    ("猫", "a cat"),
    ("发票", "an invoice or receipt"),
    // 加你自己的：左边是你在命令行里写的名字，右边是喂给 CLIP 的英文
];
```

没命中的类别会原样拼上 `a photo of` 前缀，所以直接写英文也生效。加完重新 `cargo build --release`。

---

## 六、排错

**Q：中文类别分不准 / 全归到一类**
A：大概率是类别名没在映射表里，被原样喂给了英文文本塔。加映射到 `taxonomy.rs`，或者直接用英文类别。

**Q：某张图排 GPU 爆了 / 报没找到图**
A：报告里 `_失败` 行会写原因（解码失败之类）。

**Q：报 `Impossible to unify TDim with I64` 之类**
A：见下面的“LoongArch 踩坑”，先确认 tract 版本 ≥ 0.23。

**Q：换了自己的 ONNX 模型要改什么**
A：先 `cargo run --release --example inspect` 看一眼输入输出：

```
== models/vision_model.onnx
  IN  pixel_values batch_size,num_channels,height,width,F32
  OUT /visual_projection/MatMul ..,?
```

然后按实际名字/shape 改 `clip.rs`。

---

## 七、在 loongarch64 上重写的几个坑（血泪版）

1. **没有 onnxruntime**。PyPI 的 `onnxruntime` 轮子在这台机器上 import 就报缺 `GLIBC_2.28` 符号版本；也没有 loongarch64 预编译包。`tract` 是纯 Rust，是唯一省事的路子。
2. **tract 必须 ≥ 0.23**。0.21 处理 CLIP 文本塔时类型推导失败：`Impossible to unify TDim with I64`，而且是在 `into_optimized()` 阶段静默 `exit(1)` —— 记得把 stderr 留好（`cmd 2>&1 1>/dev/null`）才能看到 anyhow 报的错。
3. **0.23 的 API 变了**：`to_array_view` → `to_plain_array_view`；`TypedRunnableModel` 不再带泛型参数，`into_runnable()` 返回 `Arc<SimplePlan>`，`run()` 要在 `Arc` 上调。
4. **必须显式 `set_input_fact`**，否则卷积的通道数推不出来（`Impossible to unify Sym(num_channels) with Val(3)`）。
5. **batch 维不能用符号**。这个 ONNX 里 attention 的 `Reshape` 在 batch 未知时推不动（`Reshape volume mismatch`），所以 batch 维在图优化阶段就得定死，末尾不足的补零凑齐。这也是 batch 自动选择的由来。
6. **Xenova 这个导出只有 `input_ids` 一个输入**，没有 `attention_mask`；池化位置由 `argmax(input_ids)` 决定，所以末尾补 0 即可，别再去喂第二个输入。
7. **huggingface.co 不通**，模型走 `hf-mirror.com`；crates.io 直连 403，走 tuna 镜像。

---

## 八、已知限制

- 准确率取决于 CLIP ViT-B/32：大类区分很稳（动物/食物/风景/截图），细粒度不行（狗的品种、车型）。要更准就换 ViT-L/14 或中文微调模型，自己导出 ONNX 再改 `clip.rs`。
- `--action move` 会移动原文件，跑之前先 `--action none` 看报告。
- tract 算子覆盖不如 onnxruntime，换模型前先跑 `examples/inspect.rs` 核对。
- 「其他」桶不参与 softmax 竞争，只在置信度低于 `--min-score` 时兜底，所以它不会偷偷抢走其他类别的概率。

## License

MIT，见 [LICENSE](LICENSE)。模型本身的许可请遵循上游 Xenova/clip-vit-base-patch32。
