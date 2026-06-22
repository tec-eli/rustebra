#[cfg(not(feature = "alloc"))]
use core::alloc::{GlobalAlloc, Layout};

#[cfg(not(feature = "alloc"))]
struct PanicAllocator;

#[cfg(not(feature = "alloc"))]
unsafe impl GlobalAlloc for PanicAllocator {
    // `panic!` would need to allocate here too (to box its payload for unwinding), which
    // would recurse straight back into this same function and abort with a confusing
    // panic-during-panic instead of reporting the real problem; `eprintln!` plus
    // `process::exit` reports a genuine accidental allocation plainly instead.
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        eprintln!("heap allocation attempted in a no_std build");
        std::process::exit(1);
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        eprintln!("heap deallocation attempted in a no_std build");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "alloc"))]
#[global_allocator]
static ALLOCATOR: PanicAllocator = PanicAllocator;

#[cfg(not(feature = "alloc"))]
fn main() {
    use rustebra::matrix::StaticMatrix;
    use rustebra::vector::StaticVector;

    // `eprintln!` + `process::exit` instead of `assert_eq!`/`panic!`, for the same reason
    // `PanicAllocator` avoids `panic!` above: unwinding would need to allocate to box the
    // panic payload, masking a genuine mismatch behind a confusing panic-during-panic abort.

    let a = StaticVector::new([1.0, 2.0, 3.0]);
    let b = StaticVector::new([4.0, 5.0, 6.0]);

    if a.add(&b) != StaticVector::new([5.0, 7.0, 9.0]) {
        eprintln!("StaticVector::add produced an unexpected result");
        std::process::exit(1);
    }
    if a.dot(&b) != 32.0 {
        eprintln!("StaticVector::dot produced an unexpected result");
        std::process::exit(1);
    }
    if StaticVector::new([3.0, 4.0]).norm() != 5.0 {
        eprintln!("StaticVector::norm produced an unexpected result");
        std::process::exit(1);
    }

    let m = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let n = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);

    if m.add(&n) != StaticMatrix::new([[6.0, 8.0], [10.0, 12.0]]) {
        eprintln!("StaticMatrix::add produced an unexpected result");
        std::process::exit(1);
    }
}

#[cfg(feature = "alloc")]
fn main() {}
