use crate::{
    atm_p::{CASErr, NonNullAtomicP},
    sync::{Node, RawExt},
};
use std::sync::{Arc, atomic::Ordering};

/// Alias for the common node-backed cursor.
pub type Cursor<T> = CursorP<T, Node<T>>;

/// Shared iteration cursor backed by an atomic pointer.
///
/// Cloning a `CursorP` keeps both clones synchronized on the same atomically
/// stored pointer, so advancing in one thread updates the view in every other
/// holder. This allows callers to build independent “head” and “tail” cursors
/// that can be moved in separate tasks while still coordinating which node
/// each cursor refers to.
pub struct CursorP<T, P = Node<T>>
where
    P: RawExt<T> + Clone,
{
    atm_ptr: Arc<NonNullAtomicP<T, P>>,
    current: P,
}

impl<T, P> CursorP<T, P>
where
    P: RawExt<T> + Clone,
{
    pub fn new(node: P) -> Self {
        Self {
            current: node.clone(),
            atm_ptr: Arc::new(NonNullAtomicP::new(node)),
        }
    }

    /// Try to reclaim the backing pointer when this is the last cursor.
    pub fn into_p(this: Self) -> Option<P> {
        Arc::into_inner(this.atm_ptr).map(|p| p.into_p())
    }

    pub fn try_unwrap(this: Self) -> Result<P, Self> {
        let Self { atm_ptr, current } = this;

        Arc::try_unwrap(atm_ptr)
            .map(|p| p.into_p())
            .map_err(|atm_ptr| Self { atm_ptr, current })
    }
}

impl<T, P> Clone for CursorP<T, P>
where
    P: RawExt<T> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            atm_ptr: Arc::clone(&self.atm_ptr),
            current: self.current.clone(),
        }
    }
}

impl<T> Iterator for CursorP<T, Node<T>> {
    type Item = Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Acquire to observe any node another thread published.
            let loaded = self.atm_ptr.load(Ordering::Acquire);

            // Fast path: nobody else has advanced the shared pointer yet.
            if Node::ptr_eq(&self.current, &loaded) {
                let next = self.current.find_next_strong()?;

                // Attempt to publish the successor. On success we hand that
                // node out; on failure we yield the node installed by another
                // thread to keep all holders in sync.
                match self.atm_ptr.compare_exchange(
                    &self.current,
                    next.clone(),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        self.current = next.clone();
                        return Some(next);
                    }
                    Err(CASErr { actual, .. }) => {
                        self.current = actual.clone();
                        return Some(actual);
                    }
                }
            } else {
                // Another thread moved the cursor; follow along.
                self.current = loaded.clone();
                return Some(loaded);
            }
        }
    }
}
