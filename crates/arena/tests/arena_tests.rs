use aether_arena::{UnifiedArena, AllocationCategory};

#[test]
fn test_arena_alignment() {
    let mut arena = UnifiedArena::new(1024).unwrap();

    let p1 = arena.alloc::<u8>(1, AllocationCategory::Weights).unwrap();
    let p2 = arena.alloc::<f32>(1, AllocationCategory::Activations).unwrap();
    let p3 = arena.alloc::<u64>(1, AllocationCategory::KvCache).unwrap();

    assert_eq!(p1 as usize % 64, 0);
    assert_eq!(p2 as usize % 64, 0);
    assert_eq!(p3 as usize % 64, 0);
}

#[test]
fn test_arena_oom() {
    let mut arena = UnifiedArena::new(64).unwrap();
    let _ = arena.alloc::<u8>(1, AllocationCategory::Weights);
    let result = arena.alloc::<u8>(128, AllocationCategory::Weights);
    assert!(result.is_err());
}
