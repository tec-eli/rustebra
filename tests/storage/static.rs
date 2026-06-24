use rustebra::storage::{StaticStorage, Storage};

#[test]
fn static_storage_construction_and_access() {
    let storage = StaticStorage::new([1, 2, 3]);

    assert_eq!(storage.len(), 3);
    assert!(!storage.is_empty());
    assert_eq!(storage.get(1), Some(&2));
    assert_eq!(storage.get(3), None);
}
