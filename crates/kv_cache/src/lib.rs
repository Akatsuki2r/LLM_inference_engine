use aether_tensor::{TensorView, TensorMut};

/// Cache for a single transformer layer's key and value states.
pub struct LayerCache {
    /// Keys tensor of shape [max_seq_len, num_heads * head_dim] or [num_heads, max_seq_len, head_dim]?
    /// We'll store as a flat vector and keep track of the current length.
    pub keys: Vec<f32>,
    /// Values tensor of same shape as keys.
    pub values: Vec<f32>,
    /// Current sequence length stored (number of tokens so far).
    pub len: usize,
    /// Maximum sequence length the cache can hold.
    pub capacity: usize,
    /// Dimension of the key/value per head (head_dim).
    pub head_dim: usize,
    /// Number of heads.
    pub num_heads: usize,
}

impl LayerCache {
    pub fn new(capacity: usize, num_heads: usize, head_dim: usize) -> Self {
        let len = num_heads * capacity * head_dim;
        Self {
            keys: vec![0.0f32; len],
            values: vec![0.0f32; len],
            len: 0,
            capacity,
            head_dim,
            num_heads,
        }
    }

    /// Update the cache with new keys and values for the current token.
    /// `keys` and `values` should have shape [num_heads, head_dim] (i.e., length num_heads * head_dim).
    pub fn update(&mut self, keys: &[f32], values: &[f32]) -> Result<(), &'static str> {
        if self.len >= self.capacity {
            return Err("KV cache capacity exceeded");
        }
        if keys.len() != self.num_heads * self.head_dim {
            return Err("Keys length mismatch");
        }
        if values.len() != self.num_heads * self.head_dim {
            return Err("Values length mismatch");
        }
        let offset = self.num_heads * self.head_dim * self.len;
        self.keys[offset..offset + keys.len()].copy_from_slice(keys);
        self.values[offset..offset + values.len()].copy_from_slice(values);
        self.len += 1;
        Ok(())
    }

    /// Get the cached keys and values up to the current length as tensors.
    /// Returns (keys, values) each as a slice of length [num_heads * self.len * head_dim].
    pub fn get(&self) -> (&[f32], &[f32]) {
        let len = self.num_heads * self.head_dim * self.len;
        (&self.keys[..len], &self.values[..len])
    }
}

/// Cache for all transformer layers.
pub struct KVCache {
    pub layers: Vec<LayerCache>,
}

impl KVCache {
    pub fn new(num_layers: usize, capacity: usize, num_heads: usize, head_dim: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerCache::new(capacity, num_heads, head_dim))
            .collect();
        Self { layers }
    }

    /// Update the cache for a specific layer.
    pub fn update_layer(&mut self, layer_idx: usize, keys: &[f32], values: &[f32]) -> Result<(), &'static str> {
        self.layers[layer_idx].update(keys, values)
    }

    /// Get the cache for a specific layer.
    pub fn get_layer(&self, layer_idx: usize) -> (&[f32], &[f32]) {
        self.layers[layer_idx].get()
    }
}
