//! Core lock-free list implementation built on hazard pointers.

use crate::{
    OptNode,
    atm_p::CASErr,
    cursor::Cursor,
    sync::{Node, RawExt},
};
use std::sync::atomic::Ordering;

/// Lock-free circular list.
#[derive(Clone)]
pub struct AtomicList<T: Sync + Send> {
    root: OptNode<T>,
}

impl<T: Sync + Send> AtomicList<T> {
    /// Create an empty list with its own hazard-pointer domain.
    pub fn new() -> Self {
        Self { root: None }
    }

    /// Insert `elem` immediately before the first node for which `predicate` returns `true`.
    ///
    /// When the list is empty the new node becomes the root.
    ///
    /// If we scan the whole list and predicate returns false for each element in the list, returns Err(elem).
    pub fn push_before<F: Fn(&T) -> bool>(&mut self, elem: T, predicate: F) -> Result<(), T> {
        let new_node = Node::new(elem);

        if let Some(ref root) = self.root {
            let mut current = root.clone();

            loop {
                let next = current.load_next_strong();

                if predicate(&next) {
                    drop(new_node.next().strong.swap(next.clone(), Ordering::Release));

                    match current.next().strong.compare_exchange(
                        &current,
                        new_node.clone(),
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break Ok(()),
                        Err(CASErr { new, .. }) => drop(new),
                    }
                }

                current = next;

                if Node::ptr_eq(root, &current) {
                    break Err(Node::try_unwrap(new_node)
                        .map_err(|_| "unwrapping Node<T> failure")
                        .expect("new_node has not been given to any AtomicNode's"));
                }
            }
        } else {
            self.root = Some(new_node);

            Ok(())
        }
    }

    /// Returns the Node which contains the element and acts as guard for it.
    /// Node is a thin wrapper that can be undone with `Node::into_inner`.
    /// Node::into_inner will return None if there are other Nodes alive out there
    /// that are pointing to the same data.
    ///
    /// Otherwise Node<T> still deref's to &T, so it'll work for most things anyway.
    /// Dropping this without unwrapping it is well defined, appropriate behaviour (do not fear it).
    pub fn pop_when<F: Fn(&T) -> bool>(&self, predicate: F) -> OptNode<T> {
        if let Some(ref root) = self.root {
            let mut current = root.clone();

            loop {
                let next = current.load_next_strong();

                if predicate(&next) {
                    let new_next = next.load_next_strong();

                    match current.next().strong.compare_exchange(
                        &next,
                        new_next,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    ) {
                        Ok(old) => break Some(old),
                        Err(CASErr {
                            // This is the actual next of current as of this moment.
                            actual: next,
                            ..
                        }) => {
                            current = next;
                        }
                    }
                }

                if Node::ptr_eq(root, &current) {
                    break None;
                }
            }
        } else {
            None
        }
    }

    pub fn cursor(&self) -> Option<Cursor<T, Node<T>>> {
        self.root.clone().map(|node| Cursor::new(node))
    }
}

impl<T: Sync + Send> Default for AtomicList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sync + Send> Drop for AtomicList<T> {
    fn drop(&mut self) {
        if let Some(mut current) = self.root.take() {
            while let Some(next) = current.next().strong.as_atomic_p().take(Ordering::Relaxed) {
                current = next;
            }
        }
    }
}
