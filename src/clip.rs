use anyhow::{Context, Result};
use image::{imageops::FilterType, ImageReader, RgbImage};
use std::path::Path;
use tokenizers::Tokenizer;
use tract_onnx::prelude::*;

pub const IMAGE_SIZE: u32 = 224;
pub const CONTEXT_LENGTH: usize = 77;
pub const PLANE: usize = (IMAGE_SIZE * IMAGE_SIZE) as usize;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

type Runnable = std::sync::Arc<TypedRunnableModel>;

/// 图像编码器。batch 维在图优化阶段就得定死：这个 ONNX 的 attention Reshape
/// 推不动符号维，只能给一个具体值，末尾不足 batch 的图补零凑齐。
pub struct VisionEncoder {
    model: Runnable,
    batch: usize,
}

impl VisionEncoder {
    pub fn load(model_dir: &Path, batch: usize) -> Result<Self> {
        let batch = batch.max(1);
        // 必须显式声明输入 shape，否则卷积层通道数推不出来
        let mut vision = tract_onnx::onnx()
            .model_for_path(model_dir.join("vision_model.onnx"))
            .with_context(|| format!("加载模型失败: {}", model_dir.display()))?;
        vision.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec!(batch, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize),
            ),
        )?;
        let model = vision
            .into_optimized()
            .context("图像编码器 优化失败")?
            .into_runnable()
            .context("图像编码器 编译失败")?;
        Ok(Self { model, batch })
    }

    /// 一次前向跑多张图。planes 每个元素是一张图预处理后的 Vec<f32>（通道优先，长度 3*224*224）。
    /// 内部按 batch 切块，返回顺序与入参一致，不足 batch 的补零。
    pub fn encode_planes(&self, planes: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if planes.is_empty() {
            return Ok(Vec::new());
        }
        let mut out_all = Vec::with_capacity(planes.len());
        for chunk in planes.chunks(self.batch) {
            let mut data = vec![0f32; self.batch * 3 * PLANE];
            for (i, p) in chunk.iter().enumerate() {
                if p.len() != 3 * PLANE {
                    anyhow::bail!("图像预处理结果长度不对: {} != {}", p.len(), 3 * PLANE);
                }
                data[i * 3 * PLANE..(i + 1) * 3 * PLANE].copy_from_slice(p);
            }
            let input: Tensor = tract_ndarray::Array4::from_shape_vec(
                (self.batch, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize),
                data,
            )?
            .into();
            let out = self.model.run(tvec![input.into()])?;
            let emb = out[0].to_plain_array_view::<f32>()?;
            let flat = emb.as_slice().context("图像嵌入不是连续内存")?;
            let dim = flat.len() / self.batch;
            out_all.extend(
                flat.chunks(dim)
                    .take(chunk.len())
                    .map(|c| normalize(c.to_vec())),
            );
        }
        Ok(out_all)
    }
}

pub struct Clip {
    vision: VisionEncoder,
    text: Runnable,
    tokenizer: Tokenizer,
}

impl Clip {
    pub fn load(model_dir: &Path, batch: usize) -> Result<Self> {
        let vision = VisionEncoder::load(model_dir, batch)?;

        // 这个导出只有 input_ids 一个输入（没有 attention_mask），
        // 池化位置由 argmax(input_ids) 决定，所以末尾补 0 即可
        let mut text = tract_onnx::onnx()
            .model_for_path(model_dir.join("text_model.onnx"))
            .with_context(|| format!("加载模型失败: {}", model_dir.display()))?;
        text.set_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), tvec!(1, CONTEXT_LENGTH)))?;
        let text = text
            .into_optimized()
            .context("文本编码器 优化失败")?
            .into_runnable()
            .context("文本编码器 编译失败")?;

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("加载 tokenizer.json 失败: {e}"))?;
        Ok(Self { vision, text, tokenizer })
    }

    /// 文本编码器：返回 L2 归一化后的向量
    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("分词失败: {e}"))?;

        let mut ids = vec![0i64; CONTEXT_LENGTH];
        for (i, id) in encoding.get_ids().iter().take(CONTEXT_LENGTH).enumerate() {
            ids[i] = *id as i64;
        }

        let ids_t: Tensor = tract_ndarray::Array2::from_shape_vec((1, CONTEXT_LENGTH), ids)?.into();
        let out = self.text.run(tvec![ids_t.into()])?;
        let emb = out[0].to_plain_array_view::<f32>()?;
        Ok(normalize(emb.iter().copied().collect()))
    }

    pub fn encode_planes(&self, planes: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        self.vision.encode_planes(planes)
    }
}

pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

/// CLIP 预处理：短边缩放 224 -> 中心裁剪 224x224 -> /255 -> (x-mean)/std -> 通道优先
pub fn preprocess_vec(path: &Path) -> Result<Vec<f32>> {
    let img = ImageReader::open(path)
        .with_context(|| format!("打开图片失败: {}", path.display()))?
        .decode()
        .with_context(|| format!("解码图片失败: {}", path.display()))?
        .into_rgb8();

    let (w, h) = img.dimensions();
    let (rw, rh) = if w < h {
        (IMAGE_SIZE, ((IMAGE_SIZE as f32 * h as f32 / w as f32).round() as u32).max(IMAGE_SIZE))
    } else {
        (((IMAGE_SIZE as f32 * w as f32 / h as f32).round() as u32).max(IMAGE_SIZE), IMAGE_SIZE)
    };
    let resized = image::imageops::resize(&img, rw, rh, FilterType::CatmullRom);
    let x0 = (rw - IMAGE_SIZE) / 2;
    let y0 = (rh - IMAGE_SIZE) / 2;
    let crop: RgbImage = image::imageops::crop_imm(&resized, x0, y0, IMAGE_SIZE, IMAGE_SIZE).to_image();

    let mut data = vec![0f32; 3 * PLANE];
    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let px = crop.get_pixel(x, y);
            let idx = (y * IMAGE_SIZE + x) as usize;
            for c in 0..3usize {
                data[c * PLANE + idx] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    Ok(data)
}
