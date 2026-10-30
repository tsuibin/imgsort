mod clip;
mod taxonomy;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use walkdir::WalkDir;

use clip::Clip;

const EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff", "tif", "avif", "jfif"];

#[derive(Parser, Debug)]
#[command(
    name = "imgsort",
    about = "批量识别图片内容并按内容分类（CLIP ViT-B/32 + tract/ONNX，纯 Rust，无需 onnxruntime）"
)]
struct Args {
    /// 输入目录
    #[arg(short, long)]
    input: PathBuf,

    /// 输出目录，默认 <input>/_classified
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// 类别列表，逗号分隔；支持中文（内置中文->英文 prompt 映射）
    #[arg(
        short,
        long,
        value_delimiter = ',',
        default_value = "人物,动物,食物,风景,建筑,车辆,植物,文档,截图,其他"
    )]
    categories: Vec<String>,

    /// 模型目录（含 vision_model.onnx / text_model.onnx / tokenizer.json）
    #[arg(long, default_value = "./models")]
    model_dir: PathBuf,

    /// 并发线程数，默认等于 CPU 核数
    #[arg(long)]
    jobs: Option<usize>,

    /// 每次前向同时推理多少张图；不给就让程序按图片数 + 线程数自己选（会跑一小段校准）
    #[arg(long)]
    batch: Option<usize>,

    /// 忽略 batch 校准缓存，重新跑一遍探针
    #[arg(long)]
    force_calibrate: bool,

    /// 分类后的落盘方式
    #[arg(long, default_value = "copy")]
    action: Action,

    /// 最高分低于该概率时归入"其他"（0 表示不启用）
    #[arg(long, default_value_t = 0.0)]
    min_score: f32,

    /// CSV 报告路径，默认 <out>/report.csv
    #[arg(long)]
    report: Option<PathBuf>,

    /// 只看顶层目录，不递归
    #[arg(long)]
    no_recursive: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum Action {
    /// 复制
    Copy,
    /// 移动
    Move,
    /// 软链接
    Symlink,
    /// 只出报告，不动文件
    None,
}

struct Hit {
    path: PathBuf,
    category: String,
    score: f32,
    top3: String,
}

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn collect_images(root: &Path, recursive: bool) -> Vec<PathBuf> {
    let walker = WalkDir::new(root).follow_links(true);
    let walker = if recursive { walker } else { walker.max_depth(1) };
    walker
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && is_image(e.path()))
        .map(|e| e.path().to_path_buf())
        .collect()
}

fn unique_dest(dir: &Path, name: &str) -> PathBuf {
    let candidate = dir.join(name);
    if !candidate.exists() {
        return candidate;
    }
    let stem = Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    let ext = Path::new(name)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    for i in 1..10000 {
        let n = if ext.is_empty() {
            format!("{stem}-{i}")
        } else {
            format!("{stem}-{i}.{ext}")
        };
        let p = dir.join(n);
        if !p.exists() {
            return p;
        }
    }
    dir.join(format!("{stem}-dup-{:?}", std::time::SystemTime::now()))
}

/// 图像嵌入 -> 余弦相似度 -> logits(100x) -> softmax -> 取最高类别
fn score_hit(
    path: &Path,
    img_vec: &[f32],
    text_embeds: &[Vec<f32>],
    categories: &[String],
    min_score: f32,
) -> Hit {
    let sims: Vec<f32> = text_embeds
        .iter()
        .map(|t| img_vec.iter().zip(t.iter()).map(|(a, b)| a * b).sum::<f32>() * 100.0)
        .collect();

    // 兜底桶不参与竞争，只在阈值不达标时兜底，避免它抢走概率
    let other_idx = categories.iter().position(|c| c == "其他");
    let in_run: Vec<usize> = (0..categories.len())
        .filter(|i| Some(*i) != other_idx)
        .collect();

    let max_s = in_run.iter().map(|i| sims[*i]).fold(f32::MIN, f32::max);
    let exps: Vec<f32> = in_run.iter().map(|i| (sims[*i] - max_s).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut ranked: Vec<(usize, f32)> = in_run
        .iter()
        .zip(exps.iter())
        .map(|(i, e)| (*i, e / sum))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let (best_idx, best) = ranked[0];
    let low = min_score > 0.0 && best < min_score;
    let category = if low && other_idx.is_some() {
        categories[other_idx.unwrap()].clone()
    } else {
        categories[best_idx].clone()
    };
    let top3 = ranked
        .iter()
        .take(3)
        .map(|(i, p)| format!("{}:{:.3}", categories[*i], p))
        .collect::<Vec<_>>()
        .join(" | ");

    Hit {
        path: path.to_path_buf(),
        category,
        score: best,
        top3,
    }
}

/// 少于这个张数就懒得校准了：直接 batch=1（本机实测最稳）
const CALIB_MIN_IMAGES: usize = 256;

/// 按图片数和线程数筛出候选 batch：
/// - 每个线程至少要分到一个任务，否则并行度被 batch 吃掉
/// - 末尾补零的浪费不超过 20%
fn candidate_batches(n: usize, jobs: usize) -> Vec<usize> {
    let max_b = if n >= jobs * 64 { 8 } else { 4 };
    let mut cands = Vec::new();
    for b in [1usize, 2, 4, 8] {
        if b > max_b || b > n {
            break;
        }
        let chunks = n.div_ceil(b);
        if chunks < jobs {
            break;
        }
        if (chunks * b - n) * 5 > n {
            continue;
        }
        cands.push(b);
    }
    if cands.is_empty() {
        cands.push(1);
    }
    cands
}

/// 自校准：用假的零图跑一小段探针，走的是和正式流程一样的 rayon 分块路径，
/// 所以能把"多线程抢内存带宽"这个因素也算进去（串行测是测不出来的）。
fn auto_batch(model_dir: &Path, n: usize, jobs: usize, force: bool) -> Result<usize> {
    if n < CALIB_MIN_IMAGES {
        return Ok(1);
    }
    let cands = candidate_batches(n, jobs);
    if cands.len() == 1 {
        return Ok(cands[0]);
    }

    let probe = (jobs * 2).max(4);
    let (model_len, model_mtime) = model_fingerprint(model_dir);
    let wanted = BatchCalibCache {
        version: CALIB_CACHE_VERSION.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        jobs,
        probe,
        candidates: cands.clone(),
        model_len,
        model_mtime,
        batch: 0,
        per_image_ms: Vec::new(),
    };

    // 命中缓存就直接复用，省掉给每个候选重建一遍图的时间
    if !force {
        let cached = cache_file()
            .and_then(|p| fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<BatchCalibCache>(&s).ok())
            .filter(|hit| hit.key_matches(&wanted));
        if let Some(hit) = cached {
            let detail = hit
                .per_image_ms
                .iter()
                .map(|(b, ms)| format!("batch={b} {ms:.0}ms/张"))
                .collect::<Vec<_>>()
                .join(", ");
            println!("batch 校准命中缓存 -> batch={}（{}）", hit.batch, detail);
            return Ok(hit.batch);
        }
    }

    let planes: Vec<Vec<f32>> = vec![vec![0f32; 3 * clip::PLANE]; probe];
    println!("自动选 batch：{} 张探针图，候选 {:?}", probe, cands);

    let mut best = cands[0];
    let mut best_per = f32::MAX;
    let mut per_image_ms = Vec::new();
    for b in cands {
        let vision = clip::VisionEncoder::load(model_dir, b)?;
        let refs: Vec<&[f32]> = planes.iter().map(|v| v.as_slice()).collect();
        let t0 = std::time::Instant::now();
        let rows = refs
            .par_chunks(b)
            .map(|c| vision.encode_planes(c))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .count();
        let per = t0.elapsed().as_secs_f32() / rows.max(1) as f32;
        let ms = per * 1000.0;
        println!("  batch={b:<2} {:>6.0} ms/张", ms);
        per_image_ms.push((b, ms));
        // 要有 3% 以上的提升才换，否则维持小 batch（省内存，探针噪声不大值得冒险）
        if per < best_per * 0.97 {
            best = b;
            best_per = per;
        }
    }

    // 缓存写失败无所谓（只读家目录等），下次重新校准就是了
    if let Some(path) = cache_file() {
        if let Some(dir) = path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let entry = BatchCalibCache {
            batch: best,
            per_image_ms,
            ..wanted
        };
        if let Ok(s) = serde_json::to_string_pretty(&entry) {
            let _ = fs::write(&path, s);
        }
    }
    Ok(best)
}


/// 探针结果缓存：同一台机器 + 同一个模型 + 同样的线程数只校准一次
const CALIB_CACHE_VERSION: &str = "1";

#[derive(serde::Serialize, serde::Deserialize)]
struct BatchCalibCache {
    version: String,
    arch: String,
    jobs: usize,
    probe: usize,
    candidates: Vec<usize>,
    model_len: u64,
    model_mtime: u64,
    batch: usize,
    per_image_ms: Vec<(usize, f32)>,
}

impl BatchCalibCache {
    fn key_matches(&self, o: &Self) -> bool {
        self.version == o.version
            && self.arch == o.arch
            && self.jobs == o.jobs
            && self.probe == o.probe
            && self.candidates == o.candidates
            && self.model_len == o.model_len
            && self.model_mtime == o.model_mtime
    }
}

fn cache_file() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("imgsort").join("batch-calibration.json"))
}

/// 用模型文件的大小 + mtime 当指纹，模型换了就自动失效
fn model_fingerprint(model_dir: &Path) -> (u64, u64) {
    match fs::metadata(model_dir.join("vision_model.onnx"))
        .and_then(|m| Ok((m.len(), m)))
    {
        Ok((len, m)) => {
            let secs = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            (len, secs)
        }
        Err(_) => (0, 0),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = &args.input;
    if !input.is_dir() {
        anyhow::bail!("输入目录不存在: {}", input.display());
    }
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| input.join("_classified"));

    let mut categories: Vec<String> = args
        .categories
        .iter()
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .collect();
    if categories.is_empty() {
        anyhow::bail!("类别列表为空");
    }
    // 启用阈值时保证有个兜底桶
    if args.min_score > 0.0 && !categories.iter().any(|c| c == "其他") {
        categories.push("其他".to_string());
    }

    if let Some(j) = args.jobs {
        rayon::ThreadPoolBuilder::new().num_threads(j).build_global()?;
    }
    let jobs = args.jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });

    // 先扫图，才知道图片总数 -> 自动选 batch 要用
    let files = collect_images(input, !args.no_recursive)
        .into_iter()
        .filter(|p| !p.starts_with(&out))
        .collect::<Vec<_>>();
    if files.is_empty() {
        println!("没有找到图片文件：{}", input.display());
        return Ok(());
    }

    let batch = match args.batch {
        Some(b) => b.max(1),
        None => auto_batch(&args.model_dir, files.len(), jobs, args.force_calibrate)?,
    };
    println!(
        "加载 CLIP 模型 ({} 类别, batch={}, jobs={})…",
        categories.len(),
        batch,
        jobs
    );
    let clip = Clip::load(&args.model_dir, batch)?;

    let mut text_embeds: Vec<Vec<f32>> = Vec::with_capacity(categories.len());
    for c in &categories {
        let prompt = taxonomy::to_prompt(c);
        println!("  {c:<8} -> {prompt}");
        text_embeds.push(clip.encode_text(&prompt)?);
    }

    println!("待处理 {} 张图片，输出目录 {}", files.len(), out.display());

    let pb = ProgressBar::new(files.len() as u64).with_style(
        ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} {msg}")?
            .progress_chars("=>-"),
    );

    let hits: Vec<Hit> = files
        .par_chunks(batch)
        .map(|chunk| -> Result<Vec<Hit>> {
            // 解码 + 预处理先做掉，坏图单独记一笔，不挡整个 batch
            let mut prepared: Vec<(usize, Vec<f32>)> = Vec::with_capacity(chunk.len());
            let mut failed: Vec<Hit> = Vec::new();
            for (i, path) in chunk.iter().enumerate() {
                match clip::preprocess_vec(path) {
                    Ok(v) => prepared.push((i, v)),
                    Err(e) => failed.push(Hit {
                        path: path.clone(),
                        category: "_失败".to_string(),
                        score: 0.0,
                        top3: format!("{e}"),
                    }),
                }
            }

            // 一个 chunk 拼成 [N,3,224,224] 一起过图像编码器
            let planes: Vec<&[f32]> = prepared.iter().map(|(_, v)| v.as_slice()).collect();
            let embs = clip.encode_planes(&planes)?;

            let mut rows = Vec::with_capacity(chunk.len());
            for (row, (idx, _)) in prepared.iter().enumerate() {
                let path = &chunk[*idx];
                rows.push(score_hit(
                    path,
                    &embs[row],
                    &text_embeds,
                    &categories,
                    args.min_score,
                ));
                pb.inc(1);
                pb.set_message(
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?")
                        .chars()
                        .take(28)
                        .collect::<String>(),
                );
            }
            rows.extend(failed);
            Ok(rows)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    pb.finish_and_clear();

    // 落盘
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    if args.action != Action::None {
        fs::create_dir_all(&out)?;
    }
    for h in &hits {
        *counts.entry(h.category.clone()).or_insert(0) += 1;
        if args.action == Action::None || h.category == "_失败" {
            continue;
        }
        let dir = out.join(&h.category);
        fs::create_dir_all(&dir)?;
        let name = h.path.file_name().and_then(|n| n.to_str()).unwrap_or("image");
        let dest = unique_dest(&dir, name);
        match args.action {
            Action::Copy => {
                fs::copy(&h.path, &dest).with_context(|| format!("复制失败: {}", h.path.display()))?;
            }
            Action::Move => {
                if fs::rename(&h.path, &dest).is_err() {
                    fs::copy(&h.path, &dest)?;
                    fs::remove_file(&h.path)?;
                }
            }
            Action::Symlink => {
                let _ = fs::remove_file(&dest);
                std::os::unix::fs::symlink(&h.path, &dest)?;
            }
            Action::None => {}
        }
    }

    // 报告
    let report_path = args.report.clone().unwrap_or_else(|| out.join("report.csv"));
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut wtr = csv::Writer::from_path(&report_path)?;
    wtr.write_record(["文件", "类别", "置信度", "Top3", "源文件"])?;
    for h in &hits {
        wtr.write_record([
            h.path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string(),
            h.category.clone(),
            format!("{:.4}", h.score),
            h.top3.clone(),
            h.path.to_string_lossy().to_string(),
        ])?;
    }
    wtr.flush()?;

    println!("\n分类统计：");
    for (c, n) in &counts {
        println!("  {c:<10} {n} 张");
    }
    println!("\n报告：{}", report_path.display());
    if args.action != Action::None {
        println!("图片目录：{}", out.display());
    }
    Ok(())
}
