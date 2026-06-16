use aether_transformer::{Embedding, Linear, TransformerBlock};
use aether_tensor::{TensorView, TensorMut, TensorType};
use aether_arena::{UnifiedArena, MemoryCategory};

/// Transformer language model consisting of an embedding layer, a stack of transformer blocks, and a language modeling head.
pub struct TransformerModel {
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub head: Linear,
}

impl TransformerModel {
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Self {
        let embedding = Embedding::new(vocab_size, hidden_size);
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new(hidden_size, intermediate_size, num_heads));
        }
        let head = Linear::new(hidden_size, vocab_size);

        Self {
            embedding,
            layers,
            head,
        }
    }

    /// Run inference for the given prompt tokens and return the next token ID.
    pub fn generate_next_token(
        &self,
        prompt: &[usize],
        arena: &mut UnifiedArena,
    ) -> Result<usize, String> {
        let seq_len = prompt.len();
        let hidden_size = self.embedding.embedding_dim;
        let vocab_size = self.head.out_features;

        // Use raw pointers from arena.alloc to avoid multiple mutable borrow errors
        let embedding_size = hidden_size * seq_len * 4;
        let embedding_ptr = arena.alloc(embedding_size, MemoryCategory::Activations).map_err(|e| e.to_string())?;
        
        let mut hidden_states_buf = unsafe { std::slice::from_raw_parts_mut(embedding_ptr, embedding_size) };
        let mut hidden_states = TensorMut::from_raw_parts(hidden_states_buf, &[seq_len, hidden_size], TensorType::F32).map_err(|e| e.to_string())?;
        self.embedding.forward(prompt, &mut hidden_states).map_err(|e| format!("{:?}", e))?;

        // 2. Transformer layers
        for block in &self.layers {
            let next_buf_ptr = arena.alloc(embedding_size, MemoryCategory::Activations).map_err(|e| e.to_string())?;
            let mut next_hidden_states = unsafe {
                let next_buf = std::slice::from_raw_parts_mut(next_buf_ptr, embedding_size);
                TensorMut::from_raw_parts(next_buf, &[seq_len, hidden_size], TensorType::F32).map_err(|e| e.to_string())?
            };
            
            let hidden_states_view = unsafe {
                let bytes = std::slice::from_raw_parts(hidden_states.data().as_ptr(), hidden_states.data().len());
                TensorView::from_raw_parts(bytes, &[seq_len, hidden_size], TensorType::F32).map_err(|e| e.to_string())?
            };
            block.forward(&hidden_states_view, &mut next_hidden_states, arena).map_err(|e| format!("{:?}", e))?;
            hidden_states = next_hidden_states;
        }

        // 3. Final head
        let logits_size = vocab_size * 4;
        let logits_ptr = arena.alloc(logits_size, MemoryCategory::Activations).map_err(|e| e.to_string())?;
        let mut logits = unsafe {
            let logits_buf = std::slice::from_raw_parts_mut(logits_ptr, logits_size);
            TensorMut::from_raw_parts(logits_buf, &[1, vocab_size], TensorType::F32).map_err(|e| e.to_string())?
        };
        
        let final_hidden_states_view = unsafe {
            let last_token_offset = (seq_len - 1) * hidden_size * 4;
            let bytes = std::slice::from_raw_parts(hidden_states.data().as_ptr().add(last_token_offset), hidden_size * 4);
            TensorView::from_raw_parts(bytes, &[1, hidden_size], TensorType::F32).map_err(|e| e.to_string())?
        };
        self.head.forward(&final_hidden_states_view, &mut logits).map_err(|e| format!("{:?}", e))?;

        // 4. Argmax
        let logits_data = unsafe { std::slice::from_raw_parts(logits.data().as_ptr() as *const f32, vocab_size) };
        let mut max_val = logits_data[0];
        let mut next_token = 0;
        for (i, &val) in logits_data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                next_token = i;
            }
        }

        Ok(next_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_arena::UnifiedArena;

    #[test]
    fn test_generate_next_token() {
        let vocab_size = 100;
        let hidden_size = 16;
        let intermediate_size = 32;
        let num_layers = 2;
        let num_heads = 4;

        let model = TransformerModel::new(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
        );

        let prompt = vec![0, 1, 2, 3];
        let mut arena = UnifiedArena::new(256 * 1024).unwrap(); // 256KB for temp buffers
        let token = model.generate_next_token(&prompt, &mut arena).unwrap();
        assert!(token < vocab_size);
    }
}
