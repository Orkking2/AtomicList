use std::{
    ops::Deref,
    ptr::{self, NonNull},
};

use haphazard::{AtomicPtr, Domain};

use crate::{AtomicListFamily, node_ptr::NodePtr, pointer_guard::PointerGuard};

pub struct AtomicListNode<T> {
    pub(crate) next: AtomicPtr<Self, AtomicListFamily, NodePtr<AtomicListNode<T>>>,
    inner: T,
}

impl<T> Deref for AtomicListNode<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: Sync + Send> AtomicListNode<T> {
    pub fn new_self_linked(inner: T) -> NodePtr<Self> {
        let node = NodePtr::new(Self {
            next: unsafe { AtomicPtr::new(ptr::null_mut()) },
            inner,
        });

        node.next.store(node);

        node
    }

    pub fn store_next(&self, next: NonNull<Self>) {
        unsafe { self.next.store_ptr(next.as_ptr()) }
    }

    pub fn next<'a>(
        &self,
        domain: &'a Domain<AtomicListFamily>,
    ) -> Option<PointerGuard<'a, AtomicListNode<T>>> {
        PointerGuard::new(&self.next, domain)
    }

    pub fn next_expected<'a>(
        &self,
        domain: &'a Domain<AtomicListFamily>,
    ) -> PointerGuard<'a, AtomicListNode<T>> {
        self.next(domain)
            .expect("root != None => node.next != None for all nodes")
    }
}
