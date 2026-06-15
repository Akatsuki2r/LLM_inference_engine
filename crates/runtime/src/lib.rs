use aether_transformer::{Embedding, Linear, TransformerBlock};
use aether_tensor::{TensorView, TensorMut};
use aether_arena::{UnifiedArena, MemoryCategory};

/// Transformer language model consisting of an embedding layer, a stack of transformer blocks,
/// and a linear head for logits.
pub struct TransformerModel {
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub head: Linear,
}

impl TransformerModel {
    /// Create a new TransformerModel with random weights.
    /// # Arguments
    /// * `vocab_size` - size of the vocabulary
    /// * `hidden_size` - dimensionality of the model
    /// * `num_layers` - number of transformer blocks
    /// * `num_attention_heads` - number of attention heads per block
    /// * `intermediate_size` - intermediate size in the feed-forward network
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
    ) -> Self {
        let embedding = Embedding::new(vocab_size, hidden_size);
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(
                hidden_size,
                intermediate_size,
                num_attention_heads,
            ));
        }
        let head = Linear::new(hidden_size, vocab_size);
        Self {
            embedding,
            blocks,
            head,
        }
    }

    /// Forward pass of the model.
    /// # Arguments
    /// * `token_ids` - input token IDs of shape [batch_size, seq_len]
    /// * `logits` - output logits of shape [batch_size, seq_len, vocab_size] (to be filled)
    /// * `arena` - arena for allocating temporary buffers
    pub fn forward(
        &self,
        token_ids: &[usize],
        logits: &mut TensorMut,
        arena: &mut UnifiedArena,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batch_size = 1; // We assume batch size 1 for simplicity
        let seq_len = token_ids.len();
        let hidden_size = self.embedding.embedding_dim;

        if logits.shape() != &[batch_size, seq_len, self.head.out_features] {
            return Err(format!(
                "Expected logits shape [{}, {}, {}], got {:?}",
                batch_size,
                seq_len,
                self.head.out_features,
                logits.shape()
            ).into());
        }

        // 1. Embedding lookup
        let mut embedding_output = arena.alloc_slice(seq_len * hidden_size * 4, MemoryCategory::Scratch)?;
        let mut embedding_output_view = TensorMut::from_raw_parts(embedding_output, vec![seq_len, hidden_size])?;
        self.embedding.forward(token_ids, &mut embedding_output_view)?;

        // 2. Pass through transformer blocks
        let mut hidden_states = embedding_output_view;
        for block in &self.blocks {
            let mut block_output = arena.alloc_slice(seq_len * hidden_size * 4, MemoryCategory::Scratch)?;
            let mut block_output_view = TensorMut::from_raw_parts(block_output, vec![seq_len, hidden_size])?;
            block.forward(&hidden_states, &mut block_output_view)?;
            hidden_states = block_output_view;
        }

        // 3. Apply head to get logits
        let mut head_output = arena.alloc_slice(seq_len * self.head.out_features * 4, MemoryCategory::Scratch)?;
        let mut head_output_view = TensorMut::from_raw_parts(head_output, vec![seq_len, self.head.out_features])?;
        self.head.forward(&hidden_states, &mut head_output_view)?;

        // Copy the result to the output logits tensor
        let logits_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                logits.data_mut().as_ptr() as *mut u8,
                logits.data_mut().len() * std::mem::size_of::<f32>(),
            )
        };
        let mut logits_view = TensorMut::from_raw_parts(logits_bytes, vec![seq_len, self.head.out_features])?;
        let head_output_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                head_output_view.data().as_ptr() as *const u8,
                head_output_view.data().len() * std::mem::size_of::<f32>(),
            )
        };
        let head_output_view = TensorView::from_raw_parts(head_output_bytes, vec![seq_len, self.head.out_features])?;
        logits_view.copy_from_slice(&head_output_view)?;

        Ok(())
    }

    /// Generate the next token given a prompt (greedy sampling).
    /// # Arguments
    /// * `prompt` - input token IDs (slice)
    /// * `arena` - arena for allocating temporary buffers
    /// * Returns the predicted next token ID.
    pub fn generate_next_token(
        &self,
        prompt: &[usize],
        arena: &mut UnifiedArena,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        // We'll allocate a tensor for the logits of shape [1, prompt.len(), vocab_size]
        let mut logits_data = arena.alloc_slice((1 * prompt.len() * self.head.out_features) * 4, MemoryCategory::Scratch)?;
        let mut logits = TensorMut::from_raw_parts(logits_data, vec![1, prompt.len(), self.head.out_features])?;
        self.forward(prompt, &mut logits, arena)?;

        // Get the logits for the last token
        let last_token_logits = {
            let start = (prompt.len() - 1) * self.head.out_features;
            &logits.data()[start..start + self.head.out_features]
        };

        // Find the argmax
        let mut max_idx = 0;
        let mut max_val = last_token_logits[0];
        for i in 1..last_token_logits.len() {
            if last_token_logits[i] > max_val {
                max_val = last_token_logits[i];
                max_idx = i;
            }
        }
        Ok(max_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_arena::{UnifiedArena, MemoryCategory};

    #[test]
    fn test_model_forward() {
        let vocab_size = 10;
        let hidden_size = 4;
        let num_layers = 2;
        let num_attention_heads = 2;
        let intermediate_size = 8;
        let model = TransformerModel::new(vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size);

        let prompt = vec![0, 1, 2, 3];
        let mut arena = UnifiedArena::new(1024).unwrap();
        let mut logits_data = arena.alloc_slice((1 * prompt.len() * vocab_size) * 4, MemoryCategory::Scratch).unwrap();
        let mut logits = TensorMut::from_raw_parts(logits_data, vec![1, prompt.len(), vocab_size]).unwrap();

        let result = model.forward(&prompt, &mut logits, &mut arena);
        assert!(result.is_ok());

        // Check that logits are not NaN and not infinite
        for &val in logits.data() {
            assert!(!val.is_nan(), "logits is NaN");
            assert!(!val.is_infinite(), "logits is infinite");
        }
    }

    #[test]
    fn test_generate_next_token() {
        let vocab_size = 10;
        let hidden_size = 4;
        let num_layers = 2;
        let num_attention_heads = 2;
        let intermediate_size = 8;
        let model = TransformerModel::new(vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size);

        let prompt = vec![0, 1, 2, 3];
        let mut arena = UnifiedArena::new(1024).unwrap();
        let token = model.generate_next_token(&prompt, &mut arena).unwrap();
        assert!(token < vocab_size);
    }
}
