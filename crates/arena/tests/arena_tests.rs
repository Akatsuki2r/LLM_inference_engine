use aether_arena::{UnifiedArena, MemoryCategory};

#[test]
fn test_arena_alignment() {
    let mut arena = UnifiedArena::new(1024).unwrap();

    let p1 = arena.alloc(1, MemoryCategory::Weights).unwrap();
    let p2 = arena.alloc(1, MemoryCategory::Activations).unwrap();
    let p3 = arena.alloc(1, MemoryCategory::KvCache).unwrap();

    assert_eq!(p1 as usize % 64, 0);
    assert_eq!(p2 as usize % 64, 0);
    assert_eq!(p3 as usize % 64, 0);
}

#[test]
fn test_arena_oom() {
    let mut arena = UnifiedArena::new(64).unwrap();
    let _ = arena.alloc(1, MemoryCategory::Weights);
    let result = arena.alloc(128, MemoryCategory::Weights);
    assert!(result.is_err());
}
