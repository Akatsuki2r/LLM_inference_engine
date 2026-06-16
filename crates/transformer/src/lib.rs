use aether_tensor::{TensorView, TensorMut, TensorType};
use aether_kernels::{NaiveEngine, KernelError};
use aether_arena::{UnifiedArena, MemoryCategory};

/// Embedding layer: looks up token IDs in a weight matrix.
pub struct Embedding {
    pub weight: Vec<f32>,
    pub vocab_size: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let weight = vec![0.0f32; vocab_size * embedding_dim];
        Self { weight, vocab_size, embedding_dim }
    }

    pub fn forward(&self, token_ids: &[usize], output: &mut TensorMut) -> Result<(), KernelError> {
        let seq_len = token_ids.len();
        let output_data = unsafe {
            std::slice::from_raw_parts_mut(output.data_mut().as_mut_ptr() as *mut f32, output.data().len() / 4)
        };
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            let start = token_id * self.embedding_dim;
            let end = start + self.embedding_dim;
            let embedding = &self.weight[start..end];
            
            let out_start = i * self.embedding_dim;
            output_data[out_start..out_start + self.embedding_dim].copy_from_slice(embedding);
        }
        Ok(())
    }
}

/// Root Mean Square Layer Normalization (RMSNorm).
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
    pub hidden_size: usize,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        let weight = vec![1.0f32; hidden_size];
        Self { weight, eps, hidden_size }
    }

    pub fn forward(&self, input: &TensorView, output: &mut TensorMut) -> Result<(), KernelError> {
        let shape = input.shape();
        let total_tokens = shape[..shape.len()-1].iter().product();
        let hidden_size = self.hidden_size;

        for i in 0..total_tokens {
            let mut sum_sq = 0.0f32;
            for j in 0..hidden_size {
                let val = input.get_2d(i, j)?;
                sum_sq += val * val;
            }
            let rms = (sum_sq / hidden_size as f32 + self.eps).sqrt();
            for j in 0..hidden_size {
                let val = input.get_2d(i, j)?;
                output.set_2d(i, j, (val / rms) * self.weight[j])?;
            }
        }
        Ok(())
    }
}

/// Linear (Dense) layer.
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = vec![0.0f32; out_features * in_features];
        Self { weight, bias: None, in_features, out_features }
    }

    pub fn forward(&self, input: &TensorView, output: &mut TensorMut) -> Result<(), KernelError> {
        let weight_view = TensorView::from_raw_parts(
            unsafe { std::slice::from_raw_parts(self.weight.as_ptr() as *const u8, self.weight.len() * 4) },
            &[self.out_features, self.in_features],
            TensorType::F32
        )?;
        
        let weight_t = weight_view.transpose()?;
        NaiveEngine::gemm(input, &weight_t, output)?;
        
        if let Some(ref bias) = self.bias {
            let shape = output.shape();
            let total_tokens = shape[..shape.len()-1].iter().product();
            for i in 0..total_tokens {
                for j in 0..self.out_features {
                    let val = output.get_2d(i, j)?;
                    output.set_2d(i, j, val + bias[j])?;
                }
            }
        }
        Ok(())
    }
}

/// A single Transformer block (layer).
pub struct TransformerBlock {
    pub attention_norm: RmsNorm,
    pub attention_q_weight: Linear,
    pub attention_k_weight: Linear,
    pub attention_v_weight: Linear,
    pub attention_o_weight: Linear,
    pub ffn_norm: RmsNorm,
    pub ffn_gate_weight: Linear,
    pub ffn_up_weight: Linear,
    pub ffn_down_weight: Linear,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
}

impl TransformerBlock {
    pub fn new(hidden_size: usize, intermediate_size: usize, num_attention_heads: usize) -> Self {
        Self {
            attention_norm: RmsNorm::new(hidden_size, 1e-5),
            attention_q_weight: Linear::new(hidden_size, hidden_size),
            attention_k_weight: Linear::new(hidden_size, hidden_size),
            attention_v_weight: Linear::new(hidden_size, hidden_size),
            attention_o_weight: Linear::new(hidden_size, hidden_size),
            ffn_norm: RmsNorm::new(hidden_size, 1e-5),
            ffn_gate_weight: Linear::new(hidden_size, intermediate_size),
            ffn_up_weight: Linear::new(hidden_size, intermediate_size),
            ffn_down_weight: Linear::new(intermediate_size, hidden_size),
            num_attention_heads,
            intermediate_size,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &TensorView,
        output: &mut TensorMut,
        arena: &mut UnifiedArena,
    ) -> Result<(), KernelError> {
        let shape = hidden_states.shape();
        let total_tokens = shape[..shape.len()-1].iter().product();
        let hidden_size = self.attention_norm.hidden_size;

        // Allocate all needed temporary buffers at once to avoid multiple mutable borrows
        let buf_size = total_tokens * hidden_size * 4;
        let inter_buf_size = total_tokens * self.intermediate_size * 4;
        
        let attn_input_ptr = arena.alloc(buf_size, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;
        let q_ptr = arena.alloc(buf_size, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;
        let k_ptr = arena.alloc(buf_size, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;
        let v_ptr = arena.alloc(buf_size, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;
        let attn_output_ptr = arena.alloc(buf_size, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;
        let ffn_input_ptr = arena.alloc(buf_size, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;
        let gate_ptr = arena.alloc(inter_buf_size, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;
        let up_ptr = arena.alloc(inter_buf_size, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;
        let ff_out_ptr = arena.alloc(buf_size, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;

        unsafe {
            let mut attn_input_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(attn_input_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            self.attention_norm.forward(hidden_states, &mut attn_input_view)?;

            let mut q_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(q_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            let mut k_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(k_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            let mut v_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(v_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;

            let attn_input_readonly = TensorView::from_raw_parts(std::slice::from_raw_parts(attn_input_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            self.attention_q_weight.forward(&attn_input_readonly, &mut q_view)?;
            self.attention_k_weight.forward(&attn_input_readonly, &mut k_view)?;
            self.attention_v_weight.forward(&attn_input_readonly, &mut v_view)?;

            let head_dim = hidden_size / self.num_attention_heads;
            let scale = (head_dim as f32).sqrt();
            let mut attn_output_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(attn_output_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;

            for i in 0..total_tokens {
                let scores_ptr = arena.alloc(total_tokens * 4, MemoryCategory::Activations).map_err(|_| KernelError::UnsupportedHardware)?;
                let scores = std::slice::from_raw_parts_mut(scores_ptr as *mut f32, total_tokens);
                
                for j in 0..total_tokens {
                    let mut dot = 0.0f32;
                    for h in 0..self.num_attention_heads {
                        let h_off = h * head_dim;
                        for d in 0..head_dim {
                            dot += q_view.get_2d(i, h_off + d)? * k_view.get_2d(j, h_off + d)?;
                        }
                    }
                    scores[j] = dot / scale;
                }
                
                let mut max_s = scores[0];
                for j in 1..total_tokens { if scores[j] > max_s { max_s = scores[j]; } }
                let mut sum_e = 0.0f32;
                for j in 0..total_tokens {
                    scores[j] = (scores[j] - max_s).exp();
                    sum_e += scores[j];
                }
                for j in 0..total_tokens { scores[j] /= sum_e; }

                for h in 0..self.num_attention_heads {
                    let h_off = h * head_dim;
                    for d in 0..head_dim {
                        let mut sum_v = 0.0f32;
                        for j in 0..total_tokens {
                            sum_v += scores[j] * v_view.get_2d(j, h_off + d)?;
                        }
                        attn_output_view.set_2d(i, h_off + d, sum_v)?;
                    }
                }
            }

            let mut residual_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(output.data_mut().as_mut_ptr(), buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            let attn_output_readonly = TensorView::from_raw_parts(std::slice::from_raw_parts(attn_output_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            self.attention_o_weight.forward(&attn_output_readonly, &mut residual_view)?;

            for i in 0..total_tokens {
                for j in 0..hidden_size {
                    let x_val = hidden_states.get_2d(i, j)?;
                    let res_val = residual_view.get_2d(i, j)?;
                    residual_view.set_2d(i, j, x_val + res_val)?;
                }
            }

            let mut ffn_input_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(ffn_input_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            let residual_readonly = TensorView::from_raw_parts(std::slice::from_raw_parts(residual_view.data().as_ptr(), buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            self.ffn_norm.forward(&residual_readonly, &mut ffn_input_view)?;

            let mut gate_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(gate_ptr, inter_buf_size), &[total_tokens, self.intermediate_size], TensorType::F32)?;
            let mut up_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(up_ptr, inter_buf_size), &[total_tokens, self.intermediate_size], TensorType::F32)?;

            let ffn_input_readonly = TensorView::from_raw_parts(std::slice::from_raw_parts(ffn_input_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            self.ffn_gate_weight.forward(&ffn_input_readonly, &mut gate_view)?;
            self.ffn_up_weight.forward(&ffn_input_readonly, &mut up_view)?;

            for i in 0..total_tokens {
                for j in 0..self.intermediate_size {
                    let g = gate_view.get_2d(i, j)?;
                    let u = up_view.get_2d(i, j)?;
                    let silu = g * (1.0 / (1.0 + (-g).exp()));
                    gate_view.set_2d(i, j, silu * u)?;
                }
            }

            let mut ff_out_view = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(ff_out_ptr, buf_size), &[total_tokens, hidden_size], TensorType::F32)?;
            let gate_readonly = TensorView::from_raw_parts(std::slice::from_raw_parts(gate_ptr, inter_buf_size), &[total_tokens, self.intermediate_size], TensorType::F32)?;
            self.ffn_down_weight.forward(&gate_readonly, &mut ff_out_view)?;

            for i in 0..total_tokens {
                for j in 0..hidden_size {
                    let current_val = residual_view.get_2d(i, j)?;
                    let ff_val = ff_out_view.get_2d(i, j)?;
                    residual_view.set_2d(i, j, current_val + ff_val)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_arena::{UnifiedArena, MemoryCategory};
    use aether_tensor::{TensorView, TensorMut, TensorType};

    #[test]
    fn test_embedding_forward() {
        let vocab_size = 10;
        let embedding_dim = 4;
        let embedding = Embedding::new(vocab_size, embedding_dim);
        let token_ids = [0, 2, 5, 9];
        let mut arena = UnifiedArena::new(1024).unwrap();
        let output_buf = arena.alloc_slice(4 * embedding_dim * 4, MemoryCategory::Scratch).unwrap();
        let mut output = TensorMut::from_raw_parts(output_buf, &[4, embedding_dim], TensorType::F32).unwrap();
        embedding.forward(&token_ids, &mut output).unwrap();
        for i in 0..4 {
            for j in 0..embedding_dim {
                assert_eq!(embedding.weight[token_ids[i] * embedding_dim + j], output.get_2d(i, j).unwrap());
            }
        }
    }

    #[test]
    fn test_rmsnorm_forward() {
        let hidden_size = 4;
        let norm = RmsNorm::new(hidden_size, 1e-5);
        let mut arena = UnifiedArena::new(2048).unwrap();
        
        let in_ptr = arena.alloc(2 * hidden_size * 4, MemoryCategory::Scratch).unwrap();
        let out_ptr = arena.alloc(2 * hidden_size * 4, MemoryCategory::Scratch).unwrap();
        
        unsafe {
            let mut input = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(in_ptr, 2 * hidden_size * 4), &[2, hidden_size], TensorType::F32).unwrap();
            for i in 0..2 {
                for j in 0..hidden_size {
                    *input.get_mut(&[i, j]).unwrap() = (i * hidden_size + j) as f32 + 1.0;
                }
            }
            
            let mut output = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(out_ptr, 2 * hidden_size * 4), &[2, hidden_size], TensorType::F32).unwrap();
            
            let input_view = TensorView::from_raw_parts(std::slice::from_raw_parts(in_ptr, 2 * hidden_size * 4), &[2, hidden_size], TensorType::F32).unwrap();
            norm.forward(&input_view, &mut output).unwrap();
            assert!(!output.get_2d(0, 0).unwrap().is_nan());
        }
    }

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(2, 3);
        let mut arena = UnifiedArena::new(2048).unwrap();
        
        let in_ptr = arena.alloc(2 * 2 * 4, MemoryCategory::Scratch).unwrap();
        let out_ptr = arena.alloc(2 * 3 * 4, MemoryCategory::Scratch).unwrap();

        unsafe {
            let mut input = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(in_ptr, 2 * 2 * 4), &[2, 2], TensorType::F32).unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    *input.get_mut(&[i, j]).unwrap() = (i * 2 + j) as f32 + 1.0;
                }
            }

            let mut output = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(out_ptr, 2 * 3 * 4), &[2, 3], TensorType::F32).unwrap();

            let input_view = TensorView::from_raw_parts(std::slice::from_raw_parts(in_ptr, 2 * 2 * 4), &[2, 2], TensorType::F32).unwrap();
            linear.forward(&input_view, &mut output).unwrap();
            assert!(!output.get_2d(0, 0).unwrap().is_nan());
        }
    }

    #[test]
    fn test_transformer_block_forward() {
        let block = TransformerBlock::new(4, 8, 2);
        let mut arena = UnifiedArena::new(128 * 1024).unwrap();
        
        let in_ptr = arena.alloc(1 * 2 * 4 * 4, MemoryCategory::Scratch).unwrap();
        let out_ptr = arena.alloc(1 * 2 * 4 * 4, MemoryCategory::Scratch).unwrap();

        unsafe {
            let mut input = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(in_ptr, 1 * 2 * 4 * 4), &[2, 4], TensorType::F32).unwrap();
            for i in 0..2 {
                for j in 0..4 {
                    *input.get_mut(&[i, j]).unwrap() = ((i * 4 + j) as f32 + 1.0) / 10.0;
                }
            }

            let mut output = TensorMut::from_raw_parts(std::slice::from_raw_parts_mut(out_ptr, 1 * 2 * 4 * 4), &[2, 4], TensorType::F32).unwrap();

            let input_view = TensorView::from_raw_parts(std::slice::from_raw_parts(in_ptr, 1 * 2 * 4 * 4), &[2, 4], TensorType::F32).unwrap();
            block.forward(&input_view, &mut output, &mut arena).unwrap();
            assert!(!output.get_2d(0, 0).unwrap().is_nan());
        }
    }
}
