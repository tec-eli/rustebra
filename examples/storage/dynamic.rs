use rustebra::storage::{DynamicStorage, Storage};

pub(crate) fn run() {
    println!("\n== DynamicStorage ==");
    let storage = DynamicStorage::new(vec![1, 2, 3]);

    println!("len = {}", storage.len());
    println!("is_empty = {}", storage.is_empty());
    println!("get(1) = {:?}", storage.get(1));
    println!("get(3) = {:?}", storage.get(3));
}
