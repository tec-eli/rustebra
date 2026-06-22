#[cfg(not(feature = "alloc"))]
use core::alloc::{GlobalAlloc, Layout};

#[cfg(not(feature = "alloc"))]
struct PanicAllocator;

#[cfg(not(feature = "alloc"))]
unsafe impl GlobalAlloc for PanicAllocator {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        panic!("heap allocation attempted in a no_std build");
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        panic!("heap deallocation attempted in a no_std build");
    }
}

#[cfg(not(feature = "alloc"))]
#[global_allocator]
static ALLOCATOR: PanicAllocator = PanicAllocator;

#[cfg(not(feature = "alloc"))]
fn main() {
    use rustebra::matrix::StaticMatrix;
    use rustebra::vector::StaticVector;

    let a = StaticVector::new([1.0, 2.0, 3.0]);
    let b = StaticVector::new([4.0, 5.0, 6.0]);

    assert_eq!(a.add(&b), StaticVector::new([5.0, 7.0, 9.0]));
    assert_eq!(a.dot(&b), 32.0);
    assert_eq!(StaticVector::new([3.0, 4.0]).norm(), 5.0);

    let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let n = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);

    assert_eq!(m.add(&n), StaticMatrix::new([[6.0, 8.0], [10.0, 12.0]]));
}

#[cfg(feature = "alloc")]
fn main() {}
