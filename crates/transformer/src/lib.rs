use aether_tensor::{TensorView, TensorMut};
use aether_kernels::{NaiveEngine, TiledEngine, quant::Q4Engine, KernelError};
use aether_arena::{UnifiedArena, MemoryCategory};
use std::marker::PhantomData;

/// Embedding layer: looks up embeddings for token IDs.
pub struct Embedding {
    pub weight: Vec<f32>, // shape: [vocab_size, embedding_dim]
    pub vocab_size: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Initialize with random values for now
        let mut weight = vec![0.0f32; vocab_size * embedding_dim];
        // In a real model, we would load from GGUF or initialize properly
        for i in 0..weight.len() {
            weight[i] = (i as f32) * 0.01;
        }
        Self {
            weight,
            vocab_size,
            embedding_dim,
        }
    }

    /// Look up embeddings for token IDs.
    /// # Arguments
    /// * `token_ids` - slice of token IDs (length: seq_len)
    /// * `output` - output tensor of shape [seq_len, embedding_dim] (to be filled)
    pub fn forward(&self, token_ids: &[u32], output: &mut TensorMut) -> Result<(), KernelError> {
        let seq_len = token_ids.len();
        if output.shape()[0] != seq_len || output.shape()[1] != self.embedding_dim {
            return Err(KernelError::DimensionMismatch);
        }

        for i in 0..seq_len {
            let token_id = token_ids[i] as usize;
            if token_id >= self.vocab_size {
                return Err(KernelError::DimensionMismatch);
            }
            // Copy the embedding vector for this token
            for j in 0..self.embedding_dim {
                let src_idx = token_id * self.embedding_dim + j;
                let dst_idx = i * self.embedding_dim + j;
                output.data_mut()[dst_idx] = self.weight[src_idx];
            }
        }
        Ok(())
    }
}

/// Root Mean Square Layer Normalization.
pub struct RmsNorm {
    pub weight: Vec<f32>, // shape: [hidden_size]
    pub hidden_size: usize,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        // Initialize weight to ones
        let weight = vec![1.0f32; hidden_size];
        Self {
            weight,
            hidden_size,
            eps,
        }
    }

    /// Apply RMSNorm to the input tensor.
    /// # Arguments
    /// * `input` - input tensor of shape [*, hidden_size]
    /// * `output` - output tensor of same shape as input (to be filled)
    pub fn forward(&self, input: &TensorView, output: &mut TensorMut) -> Result<(), KernelError> {
        // Flatten all but the last dimension
        let mut flat_size = 1;
        for dim in 0..input.shape().len() - 1 {
            flat_size *= input.shape()[dim];
        }
        let hidden_size = self.hidden_size;
        if hidden_size != input.shape()[input.shape().len() - 1] {
            return Err(KernelError::DimensionMismatch);
        }
        if output.shape() != input.shape() {
            return Err(KernelError::DimensionMismatch);
        }

        // Get the input data as f32 slice
        let input_data = input.data();
        let output_data = output.data_mut();

        for i in 0..flat_size {
            // Compute the RMS of the slice [i * hidden_size, (i+1) * hidden_size)
            let mut sum_squares = 0.0f32;
            for j in 0..hidden_size {
                let val = input_data[i * hidden_size + j];
                sum_squares += val * val;
            }
            let rms = (sum_squares / hidden_size as f32 + self.eps).sqrt();

            // Normalize and scale
            for j in 0..hidden_size {
                let val = input_data[i * hidden_size + j];
                let normalized = val / rms * self.weight[j];
                output_data[i * hidden_size + j] = normalized;
            }
        }
        Ok(())
    }
}

/// Linear layer (matrix multiplication: y = x * W^T + b, but we assume no bias for now).
pub struct Linear {
    pub weight: Vec<f32>, // shape: [out_features, in_features]
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Initialize with random values (small)
        let mut weight = vec![0.0f32; out_features * in_features];
        for i in 0..weight.len() {
            weight[i] = (i as f32) * 0.01;
        }
        Self {
            weight,
            in_features,
            out_features,
        }
    }

    /// Forward pass: y = x * W^T
    /// # Arguments
    /// * `x` - input tensor of shape [*, in_features]
    /// * `y` - output tensor of shape [*, out_features] (to be filled)
    pub fn forward(
        &self,
        x: &TensorView,
        y: &mut TensorMut,
    ) -> Result<(), KernelError> {
        // Flatten all but the last dimension of x.
        let mut batch_size = 1;
        for dim in 0..x.shape().len() - 1 {
            batch_size *= x.shape()[dim];
        }
        let in_features = self.in_features;
        let out_features = self.out_features;

        if in_features != x.shape()[x.shape().len() - 1] {
            return Err(KernelError::DimensionMismatch);
        }
        if y.shape()[0] != batch_size || y.shape()[1] != out_features {
            return Err(KernelError::DimensionMismatch);
        }

        // Create a tensor view for the weight matrix [out_features, in_features]
        let weight_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.weight.as_ptr() as *const u8,
                self.weight.len() * std::mem::size_of::<f32>(),
            )
        };
        let weight_view = TensorView::from_raw_parts(weight_bytes, vec![self.out_features, self.in_features])?;

        // Get the input and output data as f32 slices
        let input_data = x.data();
        let output_data = y.data_mut();

        // Compute y = x * W^T
        for i in 0..batch_size {
            for j in 0..out_features {
                let mut sum = 0.0f32;
                for k in 0..in_features {
                    let x_val = input_data[i * in_features + k];
                    let w_val = weight_view.get_2d(j, k)?; // weight[j, k]
                    sum += x_val * w_val;
                }
                output_data[i * out_features + j] = sum;
            }
        }
        Ok(())
    }
}

/// A single transformer block (pre-norm architecture).
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
}

impl TransformerBlock {
    pub fn new(hidden_size: usize, intermediate_size: usize, num_attention_heads: usize) -> Self {
        // For simplicity, we set intermediate_size = 4 * hidden_size (common in LLMs)
        let intermediate_size = intermediate_size.max(4 * hidden_size);

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
        }
    }

    /// Forward pass of the transformer block.
    /// # Arguments
    /// * `hidden_states` - input tensor of shape [batch_size, seq_len, hidden_size]
    /// * `output` - output tensor of same shape as input (to be filled)
    pub fn forward(
        &self,
        hidden_states: &TensorView,
        output: &mut TensorMut,
    ) -> Result<(), KernelError> {
        // We assume hidden_states is 3D: [batch_size, seq_len, hidden_size]
        // For simplicity, we will treat the first two dimensions as batch.
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];
        let hidden_size = self.attention_norm.hidden_size;

        if hidden_states.shape().len() != 3 {
            return Err(KernelError::DimensionMismatch);
        }
        if hidden_states.shape()[2] != hidden_size {
            return Err(KernelError::DimensionMismatch);
        }
        if output.shape() != hidden_states.shape() {
            return Err(KernelError::DimensionMismatch);
        }

        // Flatten the batch and seq_len dimensions for the norms and linear layers.
        let total_tokens = batch_size * seq_len;

        // Get the data as f32 slices
        let input_data = hidden_states.data();
        let output_data = output.data_mut();

        // 1. Self-attention pre-norm
        // We'll use the same output buffer for intermediate results, but we need to be careful.
        // We'll create a tensor view for the input and output as 2D: [total_tokens, hidden_size]
        let mut attn_input_data = vec![0.0f32; total_tokens * hidden_size]; // temporary buffer for attn input
        let mut attn_output_data = vec![0.0f32; total_tokens * hidden_size]; // temporary buffer for attn output

        // Normalize the input and store in attn_input_data
        {
            // Create a tensor view for the input as 2D
            let input_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    input_data.as_ptr() as *const u8,
                    input_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let input_view = TensorView::from_raw_parts(input_bytes, vec![total_tokens, hidden_size])?;
            let attn_input_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    attn_input_data.as_mut_ptr() as *mut u8,
                    attn_input_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let mut attn_input_view = TensorMut::from_raw_parts(attn_input_bytes, vec![total_tokens, hidden_size])?;
            self.attention_norm.forward(&input_view, &mut attn_input_view)?;
        }

        // 2. Self-attention
        let hidden_size = self.attention_norm.hidden_size;
        let head_dim = hidden_size / self.num_attention_heads;
        let mut q_data = vec![0.0f32; total_tokens * hidden_size];
        let mut k_data = vec![0.0f32; total_tokens * hidden_size];
        let mut v_data = vec![0.0f32; total_tokens * hidden_size];

        // Project normalized input to Q, K, V
        {
            let q_weight_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.attention_q_weight.weight.as_ptr() as *const u8,
                    self.attention_q_weight.weight.len() * std::mem::size_of::<f32>(),
                )
            };
            let q_weight_view = TensorView::from_raw_parts(q_weight_bytes, vec![self.attention_q_weight.out_features, self.attention_q_weight.in_features])?;
            let q_data_byte_len = q_data.len() * std::mem::size_of::<f32>();
            let mut q_out_view = TensorMut::from_raw_parts(
                unsafe { std::slice::from_raw_parts_mut(q_data.as_mut_ptr() as *mut u8, q_data_byte_len) },
                vec![total_tokens, self.attention_q_weight.out_features],
            )?;
            let attn_input_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    attn_input_data.as_ptr() as *const u8,
                    attn_input_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let attn_input_view = TensorView::from_raw_parts(attn_input_bytes, vec![total_tokens, hidden_size])?;
            self.attention_q_weight.forward(&attn_input_view, &mut q_out_view)?;
        }
        {
            let k_weight_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.attention_k_weight.weight.as_ptr() as *const u8,
                    self.attention_k_weight.weight.len() * std::mem::size_of::<f32>(),
                )
            };
            let k_weight_view = TensorView::from_raw_parts(k_weight_bytes, vec![self.attention_k_weight.out_features, self.attention_k_weight.in_features])?;
            let k_data_byte_len = k_data.len() * std::mem::size_of::<f32>();
            let mut k_out_view = TensorMut::from_raw_parts(
                unsafe { std::slice::from_raw_parts_mut(k_data.as_mut_ptr() as *mut u8, k_data_byte_len) },
                vec![total_tokens, self.attention_k_weight.out_features],
            )?;
            let attn_input_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    attn_input_data.as_ptr() as *const u8,
                    attn_input_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let attn_input_view = TensorView::from_raw_parts(attn_input_bytes, vec![total_tokens, hidden_size])?;
            self.attention_k_weight.forward(&attn_input_view, &mut k_out_view)?;
        }
        {
            let v_weight_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    self.attention_v_weight.weight.as_ptr() as *const u8,
                    self.attention_v_weight.weight.len() * std::mem::size_of::<f32>(),
                )
            };
            let v_weight_view = TensorView::from_raw_parts(v_weight_bytes, vec![self.attention_v_weight.out_features, self.attention_v_weight.in_features])?;
            let v_data_byte_len = v_data.len() * std::mem::size_of::<f32>();
            let mut v_out_view = TensorMut::from_raw_parts(
                unsafe { std::slice::from_raw_parts_mut(v_data.as_mut_ptr() as *mut u8, v_data_byte_len) },
                vec![total_tokens, self.attention_v_weight.out_features],
            )?;
            let attn_input_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    attn_input_data.as_ptr() as *const u8,
                    attn_input_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let attn_input_view = TensorView::from_raw_parts(attn_input_bytes, vec![total_tokens, hidden_size])?;
            self.attention_v_weight.forward(&attn_input_view, &mut v_out_view)?;
        }

        // Compute attention: softmax(QK^T / sqrt(d)) * V
        let scale = (head_dim as f32).sqrt();
        for i in 0..total_tokens {
            // Compute scores for token i against all tokens j
            let mut scores = vec![0.0f32; total_tokens];
            for j in 0..total_tokens {
                let mut dot = 0.0f32;
                for h in 0..self.num_attention_heads {
                    let h_start = h * head_dim;
                    // Q[i, h]
                    let q_offset = i * hidden_size + h_start;
                    // K[j, h]
                    let k_offset = j * hidden_size + h_start;
                    for d in 0..head_dim {
                        dot += q_data[q_offset + d] * k_data[k_offset + d];
                    }
                }
                scores[j] = dot / scale;
            }
            // Softmax
            let mut max_score = scores[0];
            for j in 1..total_tokens {
                if scores[j] > max_score {
                    max_score = scores[j];
                }
            }
            let mut exp_sum = 0.0f32;
            for j in 0..total_tokens {
                exp_sum += (scores[j] - max_score).exp();
            }
            let mut weights = vec![0.0f32; total_tokens];
            for j in 0..total_tokens {
                weights[j] = ((scores[j] - max_score).exp()) / exp_sum;
            }
            // Weighted sum of V
            for h in 0..self.num_attention_heads {
                let h_start = h * head_dim;
                let out_offset = i * hidden_size + h_start;
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..total_tokens {
                        let v_offset = j * hidden_size + h_start + d;
                        sum += weights[j] * v_data[v_offset];
                    }
                    attn_output_data[out_offset + d] = sum;
                }
            }
        }

        // 3. Skip connection: x = x + attn_output
        for i in 0..total_tokens {
            for j in 0..hidden_size {
                let x_val = input_data[i * hidden_size + j];
                let attn_val = attn_output_data[i * hidden_size + j];
                output_data[i * hidden_size + j] = x_val + attn_val;
            }
        }

        // 4. Feed-forward pre-norm
        // Normalize the current output (which is x + attn_output) and store in ffn_input_data
        let mut ffn_input_data = vec![0.0f32; total_tokens * hidden_size];
        {
            // Create a tensor view for the current output as 2D
            let output_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    output_data.as_ptr() as *const u8,
                    output_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let output_view = TensorView::from_raw_parts(output_bytes, vec![total_tokens, hidden_size])?;
            let ffn_input_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    ffn_input_data.as_mut_ptr() as *mut u8,
                    ffn_input_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let mut ffn_input_view = TensorMut::from_raw_parts(ffn_input_bytes, vec![total_tokens, hidden_size])?;
            self.ffn_norm.forward(&output_view, &mut ffn_input_view)?;
        }

        // 5. Feed-forward network: SwiGLU
        //   gate = ffn_gate_weight * ffn_input   [total_tokens, intermediate_size]
        //   up = ffn_up_weight * ffn_input       [total_tokens, intermediate_size]
        //   gate_act = silu(gate)
        //   ff = (gate_act * up) * ffn_down_weight   [total_tokens, hidden_size]

        // Allocate temporary buffers
        let mut gate_out_data = vec![0.0f32; total_tokens * self.ffn_gate_weight.out_features];
        let mut up_out_data = vec![0.0f32; total_tokens * self.ffn_up_weight.out_features];
        let mut gate_act_out_data = vec![0.0f32; total_tokens * self.ffn_gate_weight.out_features];
        let mut ff_out_data = vec![0.0f32; total_tokens * hidden_size];

        // Gate projection
        {
            let ffn_input_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    ffn_input_data.as_ptr() as *const u8,
                    ffn_input_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let ffn_input_view = TensorView::from_raw_parts(ffn_input_bytes, vec![total_tokens, self.ffn_gate_weight.in_features])?;
            let gate_out_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    gate_out_data.as_mut_ptr() as *mut u8,
                    gate_out_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let mut gate_out_view = TensorMut::from_raw_parts(gate_out_bytes, vec![total_tokens, self.ffn_gate_weight.out_features])?;
            self.ffn_gate_weight.forward(&ffn_input_view, &mut gate_out_view)?;
        }

        // Up projection
        {
            let ffn_input_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    ffn_input_data.as_ptr() as *const u8,
                    ffn_input_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let ffn_input_view = TensorView::from_raw_parts(ffn_input_bytes, vec![total_tokens, self.ffn_up_weight.in_features])?;
            let up_out_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    up_out_data.as_mut_ptr() as *mut u8,
                    up_out_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let mut up_out_view = TensorMut::from_raw_parts(up_out_bytes, vec![total_tokens, self.ffn_up_weight.out_features])?;
            self.ffn_up_weight.forward(&ffn_input_view, &mut up_out_view)?;
        }

        // SiLU activation: x * sigmoid(x)
        for i in 0..gate_out_data.len() {
            let x = gate_out_data[i];
            let silu = x * (1.0 / (1.0 + (-x).exp()));
            gate_act_out_data[i] = silu;
        }

        // Element-wise multiplication: gate_act * up
        let mut gate_up_out_data = vec![0.0f32; total_tokens * self.ffn_gate_weight.out_features];
        for i in 0..gate_act_out_data.len() {
            let gate_val = gate_act_out_data[i];
            let up_val = up_out_data[i];
            gate_up_out_data[i] = gate_val * up_val;
        }

        // Down projection
        {
            let gate_up_out_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    gate_up_out_data.as_ptr() as *const u8,
                    gate_up_out_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let gate_up_out_view = TensorView::from_raw_parts(gate_up_out_bytes, vec![total_tokens, self.ffn_down_weight.in_features])?;
            let ff_out_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(
                    ff_out_data.as_mut_ptr() as *mut u8,
                    ff_out_data.len() * std::mem::size_of::<f32>(),
                )
            };
            let mut ff_out_view = TensorMut::from_raw_parts(ff_out_bytes, vec![total_tokens, self.ffn_down_weight.out_features])?;
            self.ffn_down_weight.forward(&gate_up_out_view, &mut ff_out_view)?;
        }

        // 6. Skip connection: x = x + ff_out
        for i in 0..total_tokens {
            for j in 0..hidden_size {
                let x_val = output_data[i * hidden_size + j];
                let ff_val = ff_out_data[i * hidden_size + j];
                output_data[i * hidden_size + j] = x_val + ff_val;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_arena::{UnifiedArena, MemoryCategory};
    use aether_tensor::{TensorView, TensorMut};

    #[test]
    fn test_embedding_forward() {
        let vocab_size = 10;
        let embedding_dim = 4;
        let embedding = Embedding::new(vocab_size, embedding_dim);

        let token_ids = [0, 2, 5, 9];
        let seq_len = token_ids.len();

        // Allocate memory for output
        let mut arena = UnifiedArena::new(1024).unwrap();
        let output_buf = arena.alloc_slice(seq_len * embedding_dim * 4, MemoryCategory::Scratch).unwrap();
        let mut output = TensorMut::from_raw_parts(output_buf, vec![seq_len, embedding_dim]).unwrap();

        embedding.forward(&token_ids, &mut output).unwrap();

        // Check that the output matches the embedding weights
        for i in 0..seq_len {
            let token_id = token_ids[i] as usize;
            for j in 0..embedding_dim {
                let expected = embedding.weight[token_id * embedding_dim + j];
                let actual = output.get_2d(i, j).unwrap();
                assert_eq!(expected, actual, "mismatch at token {}, dim {}", i, j);
            }
        }
    }

    #[test]
    fn test_rmsnorm_forward() {
        let hidden_size = 4;
        let eps = 1e-5;
        let norm = RmsNorm::new(hidden_size, eps);

        // Input: [2, 4] (batch_size=2, hidden_size=4)
        let mut input_arena = UnifiedArena::new(1024).unwrap();
        let input_buf = input_arena.alloc_slice(2 * hidden_size * 4, MemoryCategory::Scratch).unwrap();
        let mut input = TensorMut::from_raw_parts(input_buf, vec![2, hidden_size]).unwrap();
        let mut output_arena = UnifiedArena::new(1024).unwrap();
        let output_buf = output_arena.alloc_slice(2 * hidden_size * 4, MemoryCategory::Scratch).unwrap();
        let mut output = TensorMut::from_raw_parts(output_buf, vec![2, hidden_size]).unwrap();

        // Set input values
        let input_vals = [
            1.0f32, 2.0f32, 3.0f32, 4.0f32,
            2.0f32, 4.0f32, 6.0f32, 8.0f32,
        ];
        for i in 0..input.data().len() {
            input.data_mut()[i] = input_vals[i];
        }

        // Create a TensorView from the input TensorMut
        let input_view = {
            let input_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    input.data().as_ptr() as *const u8,
                    input.data().len() * std::mem::size_of::<f32>(),
                )
            };
            TensorView::from_raw_parts(input_bytes, input.shape().to_vec()).unwrap()
        };

        norm.forward(&input_view, &mut output).unwrap();

        // Compute expected output for each row
        let mut expected = vec![0.0f32; 2 * hidden_size];
        for row in 0..2 {
            let row_start = row * hidden_size;
            let row_end = row_start + hidden_size;
            let row_vals = &input_vals[row_start..row_end];
            // Compute sum of squares
            let mut sum_squares = 0.0f32;
            for &val in row_vals {
                sum_squares += val * val;
            }
            let mean_squares = sum_squares / hidden_size as f32;
            let rms = (mean_squares + eps).sqrt();
            for j in 0..hidden_size {
                let expected_val = row_vals[j] / rms * norm.weight[j]; // weight is 1.0
                expected[row_start + j] = expected_val;
            }
        }

        for j in 0..hidden_size {
            let actual = output.get_2d(0, j).unwrap();
            assert!((actual - expected[j]).abs() < 1e-5, "row0 col{} mismatch: {} vs {}", j, actual, expected[j]);
            let actual = output.get_2d(1, j).unwrap();
            assert!((actual - expected[hidden_size + j]).abs() < 1e-5, "row1 col{} mismatch: {} vs {}", j, actual, expected[hidden_size + j]);
        }
    }
    }

    #[test]
    fn test_linear_forward() {
        let in_features = 2;
        let out_features = 3;
        let linear = Linear::new(in_features, out_features);

        // Input: [2, 2] (batch_size=2, in_features=2)
        let mut input_arena = UnifiedArena::new(1024).unwrap();
        let input_buf = input_arena.alloc_slice(2 * in_features * 4, MemoryCategory::Scratch).unwrap();
        let mut input = TensorMut::from_raw_parts(input_buf, vec![2, in_features]).unwrap();
        let mut output_arena = UnifiedArena::new(1024).unwrap();
        let output_buf = output_arena.alloc_slice(2 * out_features * 4, MemoryCategory::Scratch).unwrap();
        let mut output = TensorMut::from_raw_parts(output_buf, vec![2, out_features]).unwrap();

        // Set input values
        let input_vals = [
            1.0f32, 2.0f32,
            3.0f32, 4.0f32,
        ];
        for i in 0..input.data().len() {
            input.data_mut()[i] = input_vals[i];
        }

        // Create a TensorView from the input TensorMut
        let input_view = {
            let input_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    input.data().as_ptr() as *const u8,
                    input.data().len() * std::mem::size_of::<f32>(),
                )
            };
            TensorView::from_raw_parts(input_bytes, input.shape().to_vec()).unwrap()
        };

        linear.forward(&input_view, &mut output).unwrap();

        // Compute expected output for first row:
        //   x = [1,2]
        //   W = [
        //       [w00, w01],   // out_feature 0
        //       [w10, w11],   // out_feature 1
        //       [w20, w21],   // out_feature 2
        //   ]
        //   y0 = 1*w00 + 2*w01
        //   y1 = 1*w10 + 2*w11
        //   y2 = 1*w20 + 2*w21
        // We initialized weight with (i as f32)*0.01, so:
        //   w00 = 0*0.01 = 0.0, w01 = 1*0.01 = 0.01 -> y0 = 1*0.0 + 2*0.01 = 0.02
        //   w10 = 2*0.01 = 0.02, w11 = 3*0.01 = 0.03 -> y1 = 1*0.02 + 2*0.03 = 0.02+0.06=0.08
        //   w20 = 4*0.01 = 0.04, w21 = 5*0.01 = 0.05 -> y2 = 1*0.04 + 2*0.05 = 0.04+0.10=0.14
        let expected_row0 = [0.02f32, 0.08f32, 0.14f32];
        // Second row: x = [3,4]
        //   y0 = 3*0.0 + 4*0.01 = 0.04
        //   y1 = 3*0.02 + 4*0.03 = 0.06+0.12=0.18
        //   y2 = 3*0.04 + 4*0.05 = 0.12+0.20=0.32
        let expected_row1 = [0.04f32, 0.18f32, 0.32f32];

        for j in 0..out_features {
            let actual = output.get_2d(0, j).unwrap();
            assert!((actual - expected_row0[j]).abs() < 1e-5, "row0 col{} mismatch: {} vs {}", j, actual, expected_row0[j]);
            let actual = output.get_2d(1, j).unwrap();
            assert!((actual - expected_row1[j]).abs() < 1e-5, "row1 col{} mismatch: {} vs {}", j, actual, expected_row1[j]);
        }
    }

    #[test]
    fn test_transformer_block_forward() {
        let hidden_size = 4;
        let intermediate_size = 8; // 2 * hidden_size for test
        let block = TransformerBlock::new(hidden_size, intermediate_size, 4);

        // Input: [1, 2, 4] (batch_size=1, seq_len=2, hidden_size=4)
        let mut input_arena = UnifiedArena::new(4096).unwrap();
        let input_buf = input_arena.alloc_slice(1 * 2 * hidden_size * 4, MemoryCategory::Scratch).unwrap();
        let mut input = TensorMut::from_raw_parts(input_buf, vec![1, 2, hidden_size]).unwrap();
        let mut output_arena = UnifiedArena::new(4096).unwrap();
        let output_buf = output_arena.alloc_slice(1 * 2 * hidden_size * 4, MemoryCategory::Scratch).unwrap();
        let mut output = TensorMut::from_raw_parts(output_buf, vec![1, 2, hidden_size]).unwrap();

        // Set input values to something simple
        let input_vals = [
            1.0f32, 0.0f32, 0.0f32, 0.0f32,   // first token
            0.0f32, 1.0f32, 0.0f32, 0.0f32,   // second token
        ];
        for i in 0..input.data().len() {
            input.data_mut()[i] = input_vals[i];
        }

        // Create a TensorView from the input TensorMut
        let input_view = {
            let input_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    input.data().as_ptr() as *const u8,
                    input.data().len() * std::mem::size_of::<f32>(),
                )
            };
            TensorView::from_raw_parts(input_bytes, input.shape().to_vec()).unwrap()
        };

        block.forward(&input_view, &mut output).unwrap();

        // Since we used identity for attention and the feed-forward is initialized with small weights,
        // we expect the output to be close to the input (but not exactly due to the feed-forward).
        // We'll just check that the output is not NaN and has reasonable values.
        for i in 0..output.data().len() {
            let val = output.data()[i];
            assert!(!val.is_nan(), "output is NaN at {}", i);
            assert!(!val.is_infinite(), "output is infinite at {}", i);
        }
    }