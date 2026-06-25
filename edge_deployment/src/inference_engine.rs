//! ONNX 推理引擎模块
//! 
//! 【功能】
//! - 加载 ONNX 模型
//! - 执行推理
//! - 返回控制建议

use anyhow::Result;
use ndarray::Array;

pub struct InferenceEngine {
    session: ort::Session,
    input_name: String,
    output_name: String,
}

impl InferenceEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        log::info!("[INFERENCE] 加载 ONNX 模型: {}", model_path);
        
        let session = ort::Session::from_file(model_path)?;
        
        let input_name = session.inputs()[0].name.clone();
        let output_name = session.outputs()[0].name.clone();
        
        Ok(Self {
            session,
            input_name,
            output_name,
        })
    }
    
    pub fn infer(&self, input_data: &[f32]) -> Result<Vec<f32>> {
        let input_shape = vec![1, 1440, 13];
        
        let input_array = Array::from_shape_vec(
            ndarray::Dim(input_shape.clone()),
            input_data.to_vec()
        )?;
        
        let outputs = self.session.run(
            ort::SessionInputs::from(vec![
                ort::Input::from_array(self.input_name.clone(), input_array.into_dyn())
            ])
        )?;
        
        let output = &outputs[0];
        let output_slice = output.as_slice::<f32>()?;
        
        Ok(output_slice.to_vec())
    }
    
    pub fn infer_batch(&self, batch_data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        batch_data.iter()
            .map(|data| self.infer(data))
            .collect()
    }
}
