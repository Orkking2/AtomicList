#![doc = include_str!("../README.md")]

pub type OptNode<T> = Option<sync::Node<T>>;

pub mod atm_p;
pub mod cursor;
pub mod list;
pub mod list_iter;
pub mod sync;

mod tests;

pub use list::AtomicList;
pub use list_iter::AtomicListIter;
