/// Cache for a single transformer layer's key and value states.
pub struct LayerCache<'a> {
    /// Keys tensor of shape [num_heads, max_seq_len, head_dim]
    keys: &'a mut [f32],
    /// Values tensor of shape [num_heads, max_seq_len, head_dim]
    values: &'a mut [f32],
    /// Current sequence length in the cache
    cur_len: usize,
    /// Maximum sequence length capacity
    max_len: usize,
    /// Hidden dimension per head
    head_dim: usize,
    /// Number of heads
    num_heads: usize,
}

impl<'a> LayerCache<'a> {
    pub fn new(
        keys: &'a mut [f32],
        values: &'a mut [f32],
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            keys,
            values,
            cur_len: 0,
            max_len: max_seq_len,
            head_dim,
            num_heads,
        }
    }

    pub fn update(&mut self, keys: &[f32], values: &[f32]) -> Result<(), &'static str> {
        let seq_len = keys.len() / (self.num_heads * self.head_dim);
        if self.cur_len + seq_len > self.max_len {
            return Err("KV cache overflow");
        }

        // Store keys and values at the current sequence offset
        let offset = self.cur_len * self.num_heads * self.head_dim;
        self.keys[offset..offset + keys.len()].copy_from_slice(keys);
        self.values[offset..offset + values.len()].copy_from_slice(values);

        self.cur_len += seq_len;
        Ok(())
    }

    pub fn get(&self) -> (&[f32], &[f32]) {
        (self.keys, self.values)
    }

    pub fn cur_len(&self) -> usize {
        self.cur_len
    }
}

/// KV Cache for the entire model
pub struct KVCache<'a> {
    layers: Vec<LayerCache<'a>>,
}

impl<'a> KVCache<'a> {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        arena_data: &'a mut [f32],
    ) -> Self {
        let layer_size = num_heads * max_seq_len * head_dim;
        let total_needed = num_layers * layer_size * 2;
        assert!(arena_data.len() >= total_needed, "Insufficient arena memory for KVCache");

        let mut layers = Vec::with_capacity(num_layers);
        let mut offset = 0;

        // Split the provided arena data into segments for each layer
        let (mut all_keys, mut all_values) = arena_data.split_at_mut(num_layers * layer_size);

        for _ in 0..num_layers {
            let (keys, next_keys) = all_keys.split_at_mut(layer_size);
            let (values, next_values) = all_values.split_at_mut(layer_size);
            
            layers.push(LayerCache::new(keys, values, num_heads, max_seq_len, head_dim));
            
            all_keys = next_keys;
            all_values = next_values;
        }

        Self { layers }
    }

    pub fn update_layer(&mut self, layer_idx: usize, keys: &[f32], values: &[f32]) -> Result<(), &'static str> {
        self.layers[layer_idx].update(keys, values)
    }

    pub fn get_layer(&self, layer_idx: usize) -> (&[f32], &[f32]) {
        self.layers[layer_idx].get()
    }
}
