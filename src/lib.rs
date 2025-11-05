#![doc = include_str!("../README.md")]

/// Hazard-pointer family used by [`AtomicList`](crate::list::AtomicList).
#[non_exhaustive]
pub struct AtomicListFamily;

pub mod list;
pub mod list_iter;
pub mod pointer_guard;

mod node;
mod node_ptr;
mod tests;

pub use list::AtomicList;
pub use list_iter::AtomicListIter;
pub use pointer_guard::PointerGuard;
