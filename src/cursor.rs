//! Shared atomic cursors for coordinated traversal.

use crate::{
    atm_p::{CASErr, NonNullAtomicNode, NonNullAtomicP},
    sync::Node,
};
use std::{
    fmt::Debug,
    ops::Deref,
    sync::{Arc, atomic::Ordering},
};

/// Shared iteration cursor backed by a [`Node<T>`]. This is the ergonomic entry
/// point when you need to walk or coordinate around
/// [`Node<T>`] values. For other pointer types, see the generic [`Cursor`].
///
/// Cloning a `Cursor` keeps both clones synchronized on the same atomically
/// stored pointer, so advancing in one thread updates the view in every other
/// holder. This allows callers to build independent cursors that can be moved
/// in separate tasks while still coordinating which node each cursor refers to.
///
/// ```
/// use atomic_list::{cursor::Cursor, sync::Node};
///
/// let head = Node::new("root");
/// head.push_before("tail", |_| true).unwrap();
///
/// let mut a = Cursor::new(head.clone());
/// let mut b = a.clone();
///
/// // Either cursor advancing updates the shared position for all holders.
/// assert_eq!(a.next().as_deref(), Some(&"tail"));
/// assert_eq!(b.next().as_deref(), Some(&"tail"));
/// ```
pub struct Cursor<T> {
    atm_ptr: Arc<NonNullAtomicNode<T>>,
    current: Node<T>,
}

impl<T> Debug for Cursor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cursor")
            .field("atm_ptr", &self.atm_ptr)
            .field("current", &self.current)
            .finish()
    }
}

impl<T> Cursor<T> {
    /// Start a shared cursor at `node`.
    pub fn new(node: Node<T>) -> Self {
        Self {
            current: node.clone(),
            atm_ptr: Arc::new(NonNullAtomicP::new(node)),
        }
    }

    /// Update current to what is most currently stored in the atomic pointer.
    ///
    /// This lets a cursor that has been idle catch up to a sibling that already
    /// advanced the shared position.
    pub fn reload(&mut self) -> &Self {
        self.current = self.atm_ptr.load(Ordering::Relaxed);
        self
    }

    /// Inspect the next reachable node without advancing the shared cursor.
    ///
    /// This resolves through weak breadcrumbs just like [`Iterator::next`],
    /// but leaves the shared atomic pointer untouched.
    pub fn peek(&self) -> Option<Node<T>> {
        Node::resolve_next(&self.current)
    }

    /// Get access to the underlying node the cursor currently references.
    pub fn get_current(this: &Self) -> &Node<T> {
        &this.current
    }

    /// Try to reclaim the backing pointer when this is the last cursor.
    pub fn into_current(this: Self) -> Option<Node<T>> {
        Arc::into_inner(this.atm_ptr).map(NonNullAtomicP::into_p)
    }

    /// Attempt to unwrap the cursor into its backing pointer.
    ///
    /// Returns `Ok(P)` when this was the only live cursor; otherwise yields
    /// back the cursor for continued shared use.
    ///
    /// If you are not interested in the potential Err(Self), use [`into_p`](Self::into_p)
    /// instead, as it optimizes for the guaranteed drop of self.
    pub fn try_unwrap(this: Self) -> Result<Node<T>, Self> {
        let Self { atm_ptr, current } = this;

        Arc::try_unwrap(atm_ptr)
            .map(NonNullAtomicP::into_p)
            .map_err(|atm_ptr| Self { atm_ptr, current })
    }

    pub fn increment(this: &mut Self) -> bool {
        loop {
            // Acquire to observe any node another thread published.
            let loaded = this.atm_ptr.load(Ordering::Acquire);

            // Nobody else has advanced the shared pointer yet.
            if Node::ptr_eq(&this.current, &loaded) {
                // If we are unable to resolve a next, incrementing does not make sense.
                if let Some(next) = Node::resolve_next(&this.current) {
                    // Attempt to publish the successor. On success we hand that
                    // node out; on failure we yield the node installed by another
                    // thread to keep all holders in sync.
                    match this.atm_ptr.compare_exchange(
                        &this.current,
                        next.clone(),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            this.current = next;
                            break true;
                        }
                        Err(CASErr { actual, .. }) => {
                            this.current = actual;
                            break true;
                        }
                    }
                } else {
                    break false;
                };
            } else {
                // Another thread moved the cursor; follow along.
                this.current = loaded;
                break true;
            }
        }
    }
}

impl<T> Deref for Cursor<T> {
    type Target = Node<T>;

    fn deref(&self) -> &Self::Target {
        &self.current
    }
}

impl<T> Clone for Cursor<T> {
    fn clone(&self) -> Self {
        Self {
            atm_ptr: Arc::clone(&self.atm_ptr),
            current: self.current.clone(),
        }
    }
}

impl<T> AsRef<T> for Cursor<T> {
    fn as_ref(&self) -> &T {
        self.current.as_ref()
    }
}

// impl<T> Iterator for Cursor<T> {
//     type Item = Node<T>;

//     /// Yield the next node while following any position published by sibling
//     /// cursors. If another holder advances first, this cursor observes that
//     /// move and aligns to the shared node before continuing traversal.
//     fn next(&mut self) -> Option<Self::Item> {
//         if Self::increment(self) {
//             Some(self.current.clone())
//         } else {
//             None
//         }
//     }
// }
