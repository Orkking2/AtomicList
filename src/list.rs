use std::ptr;

use haphazard::{AtomicPtr, Domain, raw::Pointer};

use crate::{
    AtomicListFamily, list_iter::AtomicListIter, node::AtomicListNode, node_ptr::NodePtr,
    pointer_guard::PointerGuard,
};

/// Minimal lock-free singly linked list that allows multiple producers to
/// append nodes while agreeing on the insertion point via a predicate.
pub struct AtomicList<T: Sync + Send> {
    domain: Domain<AtomicListFamily>,
    root: AtomicPtr<AtomicListNode<T>, AtomicListFamily, NodePtr<AtomicListNode<T>>>,
}

impl<T: Sync + Send> AtomicList<T> {
    pub fn new() -> Self {
        Self {
            domain: Domain::new(&AtomicListFamily),
            root: unsafe { AtomicPtr::new(ptr::null_mut()) },
        }
    }

    pub fn domain(&self) -> &Domain<AtomicListFamily> {
        &self.domain
    }

    /// Insert a value immediately before the first node whose reference causes the
    /// predicate to return `true`. When the list is empty the element is always
    /// inserted. The call loops until it can install the node.
    ///
    /// Intended for pushing blocks where predicate is `Block::is_full`.
    /// Specifically, it looks for the first block which has not been fully conusmed.
    pub fn push_before<F: Fn(&T) -> bool>(&self, elem: T, predicate: F) {
        let new_node = AtomicListNode::new_self_linked(elem);

        'outer: loop {
            match self.root() {
                None => match self.root.compare_exchange(ptr::null_mut(), new_node) {
                    Ok(_) => break 'outer,
                    Err(_) => continue 'outer,
                },
                Some(root) => {
                    let root_ptr = root.ptr();
                    let mut current = root;

                    loop {
                        let next = current.next_expected(&self.domain);

                        if predicate(&*next) {
                            new_node.store_next(next.ptr());

                            match current.next.compare_exchange(next.ptr().as_ptr(), new_node) {
                                Ok(_) => {
                                    if current == root_ptr {
                                        let _ =
                                            self.root.compare_exchange(root_ptr.as_ptr(), new_node);
                                    }

                                    break 'outer;
                                }
                                // Err(x) here returns new_node, not what was actually in next
                                Err(_) => {
                                    current = current.next_expected(&self.domain);

                                    if current == root_ptr {
                                        continue 'outer;
                                    }

                                    continue;
                                }
                            }
                        }

                        current = next;

                        if current == root_ptr {
                            continue 'outer;
                        }
                    }
                }
            }
        }
    }

    /// Mark the first node for which predicate is `true` for reclamation.
    /// Returns true if we successfully marked a node for retirement, false otherwise.
    pub fn pop_when<F: Fn(&T) -> bool>(&self, predicate: F) -> bool {
        if let Some(root) = self.root() {
            let root_ptr = root.ptr();
            let mut current = root;

            loop {
                let next = current.next_expected(&self.domain);

                if predicate(&next) {
                    let new_next = unsafe {
                        NodePtr::from_raw(next.next_expected(&self.domain).ptr().as_ptr())
                    };

                    match current.next.compare_exchange(next.ptr().as_ptr(), new_next) {
                        Ok(replaced) => {
                            let _ = self.root.compare_exchange(current.ptr().as_ptr(), new_next);

                            // Safety: current.next and root cannot be loaded to produce `next`, as both are CAS'd.
                            unsafe {
                                replaced
                                    .expect("every `next` is guaranteed to be NonNull")
                                    .retire_in(&self.domain);
                            }

                            break true;
                        }
                        Err(_) => {
                            current = current.next_expected(&self.domain);
                        }
                    }
                }

                if current == root_ptr {
                    break false;
                }
            }
        } else {
            false
        }
    }

    /// Returns the current root pointer for use with `next_if`.
    /// Returns null if the list is empty.
    pub fn root<'a>(&'a self) -> Option<PointerGuard<'a, AtomicListNode<T>>> {
        PointerGuard::new(&self.root, &self.domain)
    }

    pub fn next<'a>(&'a self) -> Option<PointerGuard<'a, AtomicListNode<T>>> {
        self.root().map(|guard| guard.next(&self.domain)).flatten()
    }

    /// Attempts to push root forward one step. Returns `next` if we are able to set it,
    /// since this can avoid a loading of root.
    pub fn next_if<'a, 'b>(
        &'a self,
        expected_current: PointerGuard<'b, AtomicListNode<T>>,
    ) -> Option<PointerGuard<'a, AtomicListNode<T>>> {
        expected_current
            .next(&self.domain)
            .map(|next| {
                match self
                    .root
                    .compare_exchange(expected_current.ptr().as_ptr(), unsafe {
                        NodePtr::from_raw(next.ptr().as_ptr())
                    }) {
                    Ok(_) => Some(next),
                    Err(_) => None,
                }
            })
            .flatten()
    }

    pub fn iter<'a>(&'a self) -> AtomicListIter<'a, T> {
        AtomicListIter::from(self)
    }
}

impl<T: Sync + Send> Default for AtomicList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send> Drop for AtomicList<T> {
    fn drop(&mut self) {
        if let Some(root) = self.root() {
            let root_ptr = root.ptr();
            let mut current = root;

            loop {
                let next = current.next_expected(&self.domain);

                drop(unsafe { Box::from_raw(current.ptr().as_ptr()) });

                current = next;
                if current == root_ptr {
                    break;
                }
            }
        }
    }
}
