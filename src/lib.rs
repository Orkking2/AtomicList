#![doc = include_str!("../README.md")]

pub type OptNode<T> = Option<sync::Node<T>>;

pub mod atm_p;
pub mod cursor;
pub mod sync;

mod tests;

