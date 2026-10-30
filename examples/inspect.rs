use anyhow::Result;
use tract_onnx::prelude::*;

fn main() -> Result<()> {
    eprintln!("step: parse vision");
    let mut v = tract_onnx::onnx().model_for_path("models/vision_model.onnx")?;
    v.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1usize, 3, 224, 224)))?;
    eprintln!("step: optimize vision");
    let v = v.into_optimized()?.into_runnable()?;
    eprintln!("step: run vision (zeros)");
    let zeros = tract_ndarray::Array4::<f32>::zeros((1, 3, 224, 224));
    let out = v.run(tvec![Tensor::from(zeros).into()])?;
    let arr = out[0].to_plain_array_view::<f32>()?;
    eprintln!("vision out len={} first={:?}", arr.len(), &arr.iter().take(4).collect::<Vec<_>>());

    eprintln!("step: parse text");
    let mut t = tract_onnx::onnx().model_for_path("models/text_model.onnx")?;
    t.set_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), tvec!(1usize, 77)))?;
    eprintln!("step: optimize text");
    let t = t.into_optimized()?.into_runnable()?;
    let mut ids = vec![0i64; 77];
    ids[0] = 49406; ids[1] = 320; ids[2] = 49407;
    let ids_t = tract_ndarray::Array2::from_shape_vec((1, 77), ids)?;
    eprintln!("step: run text");
    let out = t.run(tvec![Tensor::from(ids_t).into()])?;
    let arr = out[0].to_plain_array_view::<f32>()?;
    let norm: f32 = arr.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("text out len={} norm={}", arr.len(), norm);
    Ok(())
}
