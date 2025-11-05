use crate::{list::AtomicList, node::AtomicListNode, pointer_guard::PointerGuard};

pub struct AtomicListIter<'a, T: Sync + Send> {
    list: &'a AtomicList<T>,
    current: Option<PointerGuard<'a, AtomicListNode<T>>>,
}

impl<'a, T: Sync + Send> AtomicListIter<'a, T> {
    pub fn root(&self) -> Option<PointerGuard<'a, AtomicListNode<T>>> {
        self.list.root()
    }
}

impl<'a, T: Sync + Send> From<&'a AtomicList<T>> for AtomicListIter<'a, T> {
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
