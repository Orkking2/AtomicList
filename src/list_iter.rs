//! Iterator utilities for [`AtomicList`](crate::list::AtomicList).

use crate::{list::AtomicList, node::AtomicListNode, pointer_guard::PointerGuard};

/// Iterator that keeps a hazard pointer to the current node in an [`AtomicList`].
pub struct AtomicListIter<'a, T: Sync + Send> {
    list: &'a AtomicList<T>,
    current: Option<PointerGuard<'a, AtomicListNode<T>>>,
}

impl<'a, T: Sync + Send> AtomicListIter<'a, T> {
    /// Acquire a fresh guard to the current root.
    pub fn root(&self) -> Option<PointerGuard<'a, AtomicListNode<T>>> {
        self.list.root()
    }
}

impl<'a, T: Sync + Send> From<&'a AtomicList<T>> for AtomicListIter<'a, T> {
    /// Begin iterating from the current root of `list`.
    fn from(list: &'a AtomicList<T>) -> Self {
        Self {
            list,
            current: list.root(),
        }
    }
}

impl<'a, T: Sync + Send> Iterator for AtomicListIter<'a, T> {
    type Item = PointerGuard<'a, AtomicListNode<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.current = self
            .list
            .next_if(self.current.take()?)
            .or_else(|| self.root());

        self.root()
    }
}
