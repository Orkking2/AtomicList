use crate::{
    atm_p::{CASErr, NonNullAtomicNode, NonNullAtomicWeakNode},
    cursor::Cursor,
};
use std::{
    borrow::Borrow,
    fmt,
    hash::{Hash, Hasher},
    hint,
    mem::{ManuallyDrop, offset_of},
    ops::Deref,
    ptr::{self, NonNull},
    sync::{
        Arc, Weak,
        atomic::{self, AtomicUsize, Ordering},
    },
};

const MAX_REFCOUNT: usize = (isize::MAX) as usize;
const INTERNAL_OVERFLOW_ERROR: &str = "Node counter overflow";

// #[cfg(not(sanitize = "thread"))]
macro_rules! acquire {
    ($x:expr) => {
        atomic::fence(Ordering::Acquire)
    };
}

// ThreadSanitizer does not support memory fences. To avoid false positive
// reports in Arc / Weak implementation use atomic loads for synchronization
// instead.
// #[cfg(sanitize = "thread")]
// macro_rules! acquire {
//     ($x:expr) => {
//         $x.load(Ordering::Acquire)
//     };
// }

/// Pointer-like interface expected by [`AtomicP`](crate::atm_p::AtomicP).
///
/// The implementor must be able to roundtrip through a `*const T` without
/// running constructors or destructors. All atomic operations in `AtomicP`
/// convert `P` to a raw pointer on store and reconstruct `P` from that pointer
/// on load/compare-exchange, trusting the type’s internal control structure
/// (e.g., ref counts in `Arc`/`Weak` or `Node`/`WeakNode`) to manage lifecycle.
pub trait RawExt<T>
where
    Self: Sized,
{
    /// Return the underlying pointer for identity comparisons.
    ///
    /// Implementations should mirror [`Arc::as_ptr`](std::sync::Arc::as_ptr) or
    /// [`Node::as_ptr`](crate::sync::Node::as_ptr): no ref counts are touched
    /// and the pointer stays valid as long as `self` is.
    fn as_ptr(this: &Self) -> *const T;

    /// Pointer equality using [`as_ptr`](RawExt::as_ptr) by default.
    fn ptr_eq(lhs: &Self, rhs: &Self) -> bool {
        ptr::addr_eq(Self::as_ptr(lhs), Self::as_ptr(rhs))
    }

    /// Consume `self` and yield the raw pointer without running a destructor.
    ///
    /// This must behave like [`Arc::into_raw`](std::sync::Arc::into_raw) or
    /// [`Node::into_raw`](crate::sync::Node::into_raw): the allocation and
    /// control block remain owned by the caller, with ref counts unchanged.
    fn into_raw(this: Self) -> *const T;

    /// Recreate `Self` from a pointer produced by [`into_raw`](RawExt::into_raw).
    ///
    /// Implementations must treat the pointer as owned and restore whatever
    /// control structure is needed (e.g., ref counts for
    /// [`Arc`](std::sync::Arc) / [`Weak`](std::sync::Weak) or
    /// [`Node`](crate::sync::Node) / [`WeakNode`](crate::sync::WeakNode)).
    unsafe fn from_raw(ptr: *const T) -> Self;
}

impl<T> RawExt<T> for Arc<T> {
    /// Forwards [`Arc::as_ptr`]
    fn as_ptr(this: &Self) -> *const T {
        Self::as_ptr(this)
    }

    /// Forwards [`Arc::ptr_eq`]
    fn ptr_eq(lhs: &Self, rhs: &Self) -> bool {
        Self::ptr_eq(lhs, rhs)
    }

    /// Forwards [`Arc::into_raw`]
    fn into_raw(this: Self) -> *const T {
        Self::into_raw(this)
    }

    /// Forwards [`Arc::from_raw`]
    unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self::from_raw(ptr) }
    }
}

impl<T> RawExt<T> for Weak<T> {
    /// Forwards [`Weak::as_ptr`]
    fn as_ptr(this: &Self) -> *const T {
        Self::as_ptr(this)
    }

    /// Forwards [`Weak::ptr_eq`]
    fn ptr_eq(lhs: &Self, rhs: &Self) -> bool {
        Self::ptr_eq(lhs, rhs)
    }

    /// Forwards [`Weak::into_raw`]
    fn into_raw(this: Self) -> *const T {
        Self::into_raw(this)
    }

    /// Forwards [`Weak::from_raw`]
    unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self::from_raw(ptr) }
    }
}

impl<T> RawExt<T> for Node<T> {
    /// Forwards [`Node::as_ptr`]
    fn as_ptr(this: &Self) -> *const T {
        Self::as_ptr(this)
    }

    /// Forwards [`Node::into_raw`]
    fn into_raw(this: Self) -> *const T {
        Self::into_raw(this)
    }

    /// Forwards [`Node::from_raw`]
    unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self::from_raw(ptr) }
    }
}

impl<T> RawExt<T> for WeakNode<T> {
    /// Forwards [`WeakNode::as_ptr`]
    fn as_ptr(this: &Self) -> *const T {
        Self::as_ptr(this)
    }

    /// Forwards [`WeakNode::into_raw`]
    fn into_raw(this: Self) -> *const T {
        Self::into_raw(this)
    }

    /// Forwards [`WeakNode::from_raw`]
    unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self::from_raw(ptr) }
    }
}

/// Borrowed access to a node's successor slots.
pub struct Next<'a, Strong, Weak> {
    pub strong: &'a Strong,
    pub weak: &'a Weak,
}

#[repr(C)]
/// Stored metadata and payload for a `Node<T>`.
struct NodeInner<T> {
    strong: AtomicUsize,
    weak: AtomicUsize,

    next: NonNullAtomicNode<T>,
    weak_next: NonNullAtomicWeakNode<T>,

    data: T,
}

#[repr(transparent)]
/// Intrusive, reference-counted list node.
///
/// A `Node<T>` is always part of a circular ring. Holding a strong reference
/// gives you both ownership of the payload and of a *strong* `next` pointer that
/// advances the ring. Newly constructed nodes automatically point `next` back to
/// themselves, so the ring invariants stay valid even before the node is spliced
/// into a list. When the node eventually links to some other node, the strong
/// reference count of that successor implicitly accounts for the `next` edge.
///
/// Because every strong pointer implies a live strong `next`, self-referential
/// nodes behave correctly: dropping or unwrapping the node treats the dependent
/// `next` reference as if it were the caller’s, ensuring that no extra strong
/// count remains.
///
/// A `Node` can either be a part of a ring or not. If it is a part of a ring,
/// its next and weak_next will point to the same `next`.
pub struct Node<T> {
    ptr: NonNull<NodeInner<T>>,
}

/// Local iterator that walks a ring once, starting from a given node.
///
/// Each call to [`Iterator::next`] yields the current node and advances to its
/// successor via [`Node::resolve_next`]. Iteration stops after the iterator
/// would wrap back to the starting node or if the traversal falls off the ring
/// (e.g., the nodes were detached).
pub struct UniqueNodeIter<T> {
    start: Option<Node<T>>,
    next: Option<Node<T>>,
}

/// A local iterator that walks the ring that a node is a component of.
///
/// If and when this iterator reaches the last node and returns None for
/// the first time, it will drop the node it is holding onto, so it does
/// not need to be dropped itself for list deallocation, it just needs
/// to run out.
pub struct NodeIter<T> {
    current: Option<Node<T>>,
}

impl<T> Deref for Node<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner().data
    }
}

impl<T> Node<T> {
    /// Allocate a node that starts as a singleton ring pointing to itself.
    ///
    /// ```
    /// use atomic_list::sync::Node;
    ///
    /// let node = Node::new("root");
    /// assert_eq!(*node, "root");
    /// // The initial successor is the node itself.
    /// assert!(Node::ptr_eq(&node, &Node::load_next_strong(&node)));
    /// ```
    pub fn new(data: T) -> Self {
        let x = Box::new(NodeInner {
            strong: AtomicUsize::new(1),
            weak: AtomicUsize::new(1),
            // Safety: See below.
            next: unsafe { NonNullAtomicNode::null() },
            weak_next: unsafe { NonNullAtomicWeakNode::null() },
            data,
        });

        let this = Self {
            ptr: Box::leak(x).into(),
        };

        let Next { strong, weak } = Self::next(&this);

        // Here we guarantee the safety of NonNullAtomicP::null
        strong.store(this.clone(), Ordering::Relaxed);
        weak.store(Self::downgrade(&this), Ordering::Relaxed);

        this
    }

    /// Firstly we remove ourself from any list of which we are a part.
    ///
    /// Then we consider that the indication that a node is a singleton in a
    /// list vs a free node is whether or not its weak_next refers to itself
    /// or to a different allocation. If it points to self, then we are a valid
    /// list, but if it points to someone else, then they are the list and we
    /// are a popped (free) node.
    ///
    /// Ownership mechanics are arbitrary tbh.
    pub fn into_new_ring(this: Self) -> Self {
        drop(Self::remove(&this));

        Self::next(&this)
            .weak
            .store(Self::downgrade(&this), Ordering::Release);
        this
    }

    /// As mentioned in [`into_new_ring`](Self::into_new_ring), the indication
    /// of whether we are a free node or part of a list is whether out weak_next
    /// points to ourselves or to another node, given of course that our strong
    /// already points to self.
    pub fn is_list(this: &Self) -> bool {
        Self::ptr_eq(&Self::next(this).strong.load(Ordering::Acquire), this)
            && Self::weak_ptr_eq(this, &Self::next(this).weak.load(Ordering::Acquire))
    }

    /// Insert `elem` before the first successor that satisfies `predicate`.
    ///
    /// Traverses the circular ring starting at `self`, splices a new node in
    /// front of the first match, and returns `Err(elem)` if no match is found
    /// after a full lap so the caller can recover ownership.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    ///
    /// let root = Node::new("root");
    ///
    /// // Insert a worker right before the element matching "root".
    /// root.push_before("worker-1", |cur| *cur == "root")
    ///     .expect("insert before root");
    ///
    /// // The ring now goes: root -> worker-1 -> root.
    /// let worker = Node::load_next_strong(&root);
    /// assert_eq!(*worker, "worker-1");
    /// assert!(Node::ptr_eq(&Node::load_next_strong(&worker), &root));
    /// ```
    ///
    /// Treats self as a root into the atomic list of which this node is a node
    pub fn push_before<F: Fn(&T) -> bool>(&self, elem: T, predicate: F) -> Result<(), T> {
        let new_node = Node::new(elem);

        // Walk the ring looking for the first node whose value satisfies the predicate,
        // splicing the new node in immediately before that successor.
        let mut current = self.clone();

        loop {
            let next = Self::load_next_strong(&current);

            if predicate(&next) {
                // Pre-link the new node so that, if the CAS succeeds, the ring is valid in
                // a single atomic write to `current.next`.
                let Next { strong, weak: _ } = Self::next(&new_node);

                strong.store(next.clone(), Ordering::Release);

                // We don't actually need to greedily assign a weak to this node.
                // This is because popping a node (which is what modifies its strong
                // edge) already updates that nodes weak to ensure that it can
                // recover the live ring, even after detachment.
                //
                // The thought process goes: as long as we have a strong edge our weak
                // edge is completely irrelevant. So modifications must only happen to
                // our weak edge when (a) we are creating a new list, see `into_new_ring`
                // or (b) we are modifying our strong edge, see `pop_when`.
                // weak.store(next.downgrade(), Ordering::Release);

                // Attempt to install the new node as `current.next`. If another thread
                // changed the pointer in the meantime, restart with the observed successor
                // so we keep following the live ring.
                match Self::next(&current).strong.compare_exchange(
                    &next,
                    new_node.clone(),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        break Ok(());
                    }
                    Err(CASErr { new, .. }) => drop(new),
                }
            }

            current = next;

            // We have circled back to the start without finding a match; return the
            // element to the caller rather than leaking the allocation.
            if Node::ptr_eq(self, &current) {
                break Err(Node::try_unwrap(new_node)
                    .map_err(|_| "unwrapping Node<T> failure")
                    .expect("new_node has not been given to any AtomicNode's"));
            }
        }
    }

    /// Remove and return the first successor whose value satisfies `predicate`.
    ///
    /// Starts at `self.next` and only checks `self` after completing a full lap
    /// of the ring. If `self` is likely to match, check it manually before
    /// calling `pop_when` to avoid an extra `clone` of `self`; the clone is
    /// cheap, but skipping it is a small optimization.
    ///
    /// On removal, the returned node's **strong** `next` is reset to refer to
    /// itself, so it no longer keeps the list alive. Its **weak** `next`
    /// continues to point into the list so `resolve_next` can still locate
    /// the current successor if the node is reinserted later.
    ///
    /// Returns `None` if a full lap completes without finding a match.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    ///
    /// let root = Node::new("root");
    /// root.push_before("temp", |cur| *cur == "root").unwrap();
    ///
    /// // Remove the node holding "temp".
    /// let removed = Node::pop_when(&root, |cur| *cur == "temp")
    ///     .expect("found matching node");
    /// assert_eq!(*removed, "temp");
    ///
    /// // The removed node now owns only a self-looping strong edge.
    /// assert!(Node::ptr_eq(&Node::load_next_strong(&removed), &removed));
    ///
    /// // The ring closes back into a singleton, and weak navigation from the
    /// // removed node still finds the live list.
    /// assert!(Node::ptr_eq(&Node::load_next_strong(&root), &root));
    /// assert!(Node::ptr_eq(&Node::resolve_next(&removed).unwrap(), &root));
    /// ```
    pub fn pop_when<F: Fn(&T) -> bool>(this: &Self, predicate: F) -> Option<Self> {
        // Track the node whose successor we are inspecting; removing happens by
        // relinking that successor around the matching node.
        let mut current = this.clone();

        loop {
            let next = Self::load_next_strong(&current);

            if predicate(&next) {
                // Capture `next`'s successor up front so a successful CAS can bypass
                // `next` in one step.
                let new_next = Self::load_next_strong(&next);

                // Try to swing `current.next` from `next` to `new_next`. A failure means
                // the list changed underneath us, so follow the observed successor and
                // continue searching.
                match Self::next(&current).strong.compare_exchange(
                    &next,
                    new_next.clone(),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(old) => {
                        // Detach the removed node's strong edge so it no longer owns the
                        // list, but keep a weak edge so it can still find the live ring.
                        let Next { strong, weak } = Self::next(&old);
                        weak.store(Self::downgrade(&new_next), Ordering::Release);
                        strong.store(old.clone(), Ordering::Release);

                        break Some(old);
                    }
                    Err(CASErr {
                        // This is the actual next of current as of this moment.
                        actual: next,
                        new: _,
                    }) => {
                        current = next;
                    }
                }
            }

            // Completed a full traversal without a match.
            if Node::ptr_eq(this, &current) {
                break None;
            }
        }
    }

    /// Remove this node from the ring without cloning it first.
    ///
    /// Finds a predecessor, swings its `next` pointer around `self`, and returns
    /// a node that remains in the list (the successor) if one exists. If the
    /// ring only contains `self`, the function leaves the self-loop intact and
    /// returns `None`.
    ///
    /// After a successful removal:
    /// - `self.next().strong` is reset to a self-loop so this node no longer
    ///   holds an owning reference to the ring.
    /// - `self.next().weak` still points into the ring so `resolve_next` can
    ///   recover a reference to the original ring.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let root = Node::new("root");
    /// root.push_before("worker", |cur| *cur == "root").unwrap();
    ///
    /// // Remove the root in-place without cloning it.
    /// let survivor = Node::remove(&root).expect("another node remains");
    /// assert_eq!(*survivor, "worker");
    ///
    /// // `root` now has a self-looping strong edge but can still find the live ring.
    /// assert!(Node::ptr_eq(&Node::load_next_strong(&root), &root));
    /// assert!(Node::ptr_eq(&Node::resolve_next(&root).unwrap(), &survivor));
    /// ```
    pub fn remove(this: &Self) -> Option<Self> {
        // Fast path: singleton ring. Keep links self-referential.
        let successor = Self::load_next_strong(this);
        if Node::ptr_eq(this, &successor) {
            Self::next(this)
                .weak
                .store(Self::downgrade(this), Ordering::Release);
            Self::next(this).strong.store(successor, Ordering::Release);
            return None;
        }

        // Search for a predecessor whose next points to `self`.
        let mut pred = successor.clone();
        loop {
            let next = Self::load_next_strong(&pred);

            if Node::ptr_eq(&next, this) {
                let new_next = successor.clone();

                match Self::next(&pred).strong.compare_exchange(
                    &next,
                    new_next.clone(),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // Detach self's ownership of the ring but keep a weak breadcrumb.
                        let Next { strong, weak } = Self::next(this);
                        weak.store(Self::downgrade(&new_next), Ordering::Release);
                        strong.store(this.clone(), Ordering::Release);

                        return Some(new_next);
                    }
                    Err(CASErr { actual, .. }) => {
                        pred = actual;
                        continue;
                    }
                }
            }

            pred = next;

            // If we looped the ring without finding a stable predecessor, abort.
            if Node::ptr_eq(&pred, this) {
                return None;
            }
        }
    }

    /// Obtain a raw pointer to the stored payload.
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr: *mut NodeInner<T> = NonNull::as_ptr(this.ptr);

        unsafe { &raw mut (*ptr).data }
    }

    /// Check if two nodes point to the same allocation.
    /// Uses the default [`RawExt::ptr_eq`].
    pub fn ptr_eq(lhs: &Self, rhs: &Self) -> bool {
        RawExt::ptr_eq(lhs, rhs)
    }

    /// Try to reclaim the payload, treating the self-loop as the only other strong ref.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let node = Node::new(1);
    /// assert_eq!(Node::into_inner(node), Some(1));
    ///
    /// let a = Node::new("shared");
    /// let b = a.clone();
    /// assert_eq!(Node::into_inner(a), None);
    /// drop(b);
    /// ```
    pub fn into_inner(this: Self) -> Option<T> {
        let this = ManuallyDrop::new(this);

        match this.inner().strong.fetch_sub(1, Ordering::Release) {
            1 => {
                acquire!(this.inner().strong);

                let elem = unsafe { ptr::read(&mut (*this.ptr.as_ptr()).data) };

                drop(WeakNode { ptr: this.ptr });

                Some(elem)
            }
            2 => {
                let next = Self::next(&this).strong.load_noclone(Ordering::Acquire);

                if next
                    .inner()
                    .strong
                    .compare_exchange(1, 0, Ordering::Acquire, Ordering::Relaxed)
                    .is_err()
                {
                    return None;
                }

                let is_self = Self::ptr_eq(&next, &this);

                next.inner().strong.store(1, Ordering::Release);

                if !is_self {
                    return None;
                }

                return Self::into_inner(unsafe {
                    Self::next(&this).strong.take(Ordering::Relaxed)
                });
            }
            _ => None,
        }
    }

    /// Equivalent of [`Arc::try_unwrap`], but aware of the implicit self edge.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let node = Node::new(5);
    /// assert_eq!(Node::try_unwrap(node), Ok(5));
    ///
    /// let shared = Node::new(7);
    /// let alias = shared.clone();
    /// assert!(Node::try_unwrap(shared.clone()).is_err());
    /// drop(alias);
    /// assert_eq!(Node::try_unwrap(shared), Ok(7));
    /// ```
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        match this
            .inner()
            .strong
            .compare_exchange(1, 0, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => {
                acquire!(this.inner().strong);

                let this = ManuallyDrop::new(this);
                let elem: T = unsafe { ptr::read(&this.ptr.as_ref().data) };

                // Make a weak pointer to clean up the implicit strong-weak reference
                drop(WeakNode { ptr: this.ptr });

                Ok(elem)
            }
            Err(actual) => {
                if actual != 2 {
                    return Err(this);
                }

                let next = Self::next(&this).strong.load_noclone(Ordering::Acquire);

                if !Self::ptr_eq(&next, &this) {
                    return Err(this);
                }

                let this = ManuallyDrop::new(this);

                this.inner().strong.fetch_sub(1, Ordering::Release);

                let next_self = unsafe { Self::next(&this).strong.take(Ordering::Relaxed) };

                match Self::try_unwrap(next_self) {
                    Ok(value) => Ok(value),
                    Err(_) => unreachable!("self-referential Node::try_unwrap failed"),
                }
            }
        }
    }

    #[must_use = "losing the pointer will leak memory"]
    /// Leak-safe escape hatch that turns a [`Node<T>`] into a raw pointer.
    pub fn into_raw(this: Self) -> *const T {
        let this = ManuallyDrop::new(this);
        Self::as_ptr(&*this)
    }

    /// Recreate a [`Node<T>`] from the pointer returned by [`into_raw`](Self::into_raw).
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self {
            ptr: unsafe {
                NonNull::new_unchecked(
                    ptr.byte_sub(offset_of!(NodeInner<T>, data)) as *mut NodeInner<T>
                )
            },
        }
    }

    /// Expose the underlying metadata block.
    fn inner(&self) -> &NodeInner<T> {
        unsafe { self.ptr.as_ref() }
    }

    /// Load the current strong count with the requested ordering.
    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong.load(Ordering::Relaxed)
    }

    /// Load the current weak count with the requested ordering.
    pub fn weak_count(this: &Self) -> usize {
        this.inner().weak.load(Ordering::Relaxed)
    }

    /// Borrow the strong/weak successor slots.
    pub fn next(this: &Self) -> Next<'_, NonNullAtomicNode<T>, NonNullAtomicWeakNode<T>> {
        let inner = this.inner();

        Next {
            strong: &inner.next,
            weak: &inner.weak_next,
        }
    }

    /// Create a `WeakNode` that observes the same allocation.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let node = Node::new(10);
    /// let weak = Node::downgrade(&node);
    /// assert!(weak.upgrade().is_some());
    /// drop(node);
    /// assert!(weak.upgrade().is_none());
    /// ```
    pub fn downgrade(this: &Self) -> WeakNode<T> {
        let inner = this.inner();
        let mut cur = inner.weak.load(Ordering::Relaxed);

        loop {
            if cur == usize::MAX {
                hint::spin_loop();
                cur = inner.weak.load(Ordering::Relaxed);
                continue;
            }

            assert!(cur <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);

            match inner.weak.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return WeakNode { ptr: this.ptr };
                }
                Err(old) => cur = old,
            }
        }
    }

    /// Convenience for acquiring the current strong successor.
    ///
    /// This lets callers walk the ring without manually touching the atomic
    /// wrappers. If the node currently has a self-loop (for example, after it
    /// was removed by `pop_when`), this returns `self`.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let root = Node::new(1);
    /// root.push_before(2, |cur| *cur == 1).unwrap();
    /// assert_eq!(*Node::load_next_strong(&root), 2);
    /// ```
    pub fn load_next_strong(this: &Self) -> Self {
        Self::next(this).strong.load(Ordering::Acquire)
    }

    /// Load the strong successor, returning `None` when the edge is a self-loop.
    ///
    /// Handy for traversal code that wants to stop when it finds the end of a
    /// temporarily detached node.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let root = Node::new("root");
    ///
    /// // A singleton ring has no unique successor.
    /// assert!(Node::load_next_strong_unique(&root).is_none());
    ///
    /// root.push_before("next", |cur| *cur == "root").unwrap();
    /// assert_eq!(
    ///     Node::load_next_strong_unique(&root)
    ///         .unwrap()
    ///         .to_string(),
    ///     "next"
    /// );
    /// ```
    pub fn load_next_strong_unique(this: &Self) -> Option<Self> {
        let next = Self::load_next_strong(this);

        if Self::ptr_eq(this, &next) {
            None
        } else {
            Some(next)
        }
    }

    /// Load the weak successor of this node.
    pub fn load_next_weak(this: &Self) -> WeakNode<T> {
        Self::next(this).weak.load(Ordering::Acquire)
    }

    /// Load the weak successor, returning `None` when the edge is a self-loop.
    pub fn load_next_weak_unique(this: &Self) -> Option<WeakNode<T>> {
        let next = Self::load_next_weak(this);

        if Self::weak_ptr_eq(this, &next) {
            None
        } else {
            Some(next)
        }
    }

    /// Find the next reachable strong node, following weak links as needed.
    ///
    /// If the strong edge is a self-loop (e.g., a node popped out of the list),
    /// this follows the weak edge until it can upgrade to a live strong node or
    /// returns `None` if the search falls off the ring entirely.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let root = Node::new("root");
    /// root.push_before("temp", |cur| *cur == "root").unwrap();
    ///
    /// let removed = Node::pop_when(&root, |cur| *cur == "temp").unwrap();
    /// assert!(Node::ptr_eq(&Node::load_next_strong(&removed), &removed));
    /// // Weak edge still points into the ring, so we can walk back to root.
    /// assert!(Node::ptr_eq(&Node::resolve_next(&removed).unwrap(), &root));
    /// ```
    pub fn resolve_next(this: &Self) -> Option<Self> {
        if let Some(strong_next) = Self::load_next_strong_unique(this) {
            Some(strong_next)
        } else {
            let weak_next = Self::load_next_weak(this);

            if let Some(strong_next) = weak_next.upgrade() {
                Some(strong_next)
            } else {
                weak_next.find_next_strong()
            }
        }
    }

    /// Create a local iterator that walks this ring once starting at `self`.
    ///
    /// The iterator yields `self` first, then advances successors using
    /// [`resolve_next`](Self::resolve_next). Iteration ends after a
    /// full lap or if traversal cannot find another strong node (for example,
    /// when the node has been detached).
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let root = Node::new("root");
    /// root.push_before("worker-1", |cur| *cur == "root").unwrap();
    /// root.push_before("worker-2", |_| true).unwrap();
    ///
    /// let seen: Vec<_> = root.unique_iter().map(|n| n.to_string()).collect();
    /// assert_eq!(seen.len(), 3);
    /// assert!(seen.contains(&"root".to_string()));
    /// ```
    pub fn unique_iter(&self) -> UniqueNodeIter<T> {
        UniqueNodeIter {
            start: Some(self.clone()),
            next: Some(self.clone()),
        }
    }

    /// Build a shared `Cursor` rooted at this node.
    ///
    /// Clones the node and seeds a new [`cursor::Cursor`] so callers can
    /// coordinate traversal across threads.
    ///
    /// ```
    /// use atomic_list::{cursor::Cursor, sync::Node};
    ///
    /// let head = Node::new(0);
    /// let mut cursor: Cursor<_> = head.cursor();
    /// assert!(cursor.next().is_none());
    /// head.push_before(1, |_| true).unwrap();
    /// assert_eq!(*cursor.next().unwrap(), 1);
    /// ```
    pub fn cursor(&self) -> Cursor<T> {
        Cursor::new(self.clone())
    }

    /// Compare `self` and a `WeakNode` by address of the underlying payload.
    ///
    /// Useful when you need to check identity across strong/weak references
    /// without upgrading the weak pointer.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let node = Node::new(10);
    /// let weak = Node::downgrade(&node);
    /// assert!(Node::weak_ptr_eq(&node, &weak));
    /// ```
    pub fn weak_ptr_eq(lhs: &Self, rhs: &WeakNode<T>) -> bool {
        ptr::addr_eq(Node::as_ptr(lhs), WeakNode::as_ptr(rhs))
    }
}

impl<T: PartialEq> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        (**self).eq(&**other)
    }

    fn ne(&self, other: &Self) -> bool {
        (**self).ne(&**other)
    }
}
impl<T: Eq> Eq for Node<T> {}

impl<T: PartialOrd> PartialOrd for Node<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }

    fn lt(&self, other: &Self) -> bool {
        (**self).lt(&**other)
    }

    fn le(&self, other: &Self) -> bool {
        (**self).le(&**other)
    }

    fn gt(&self, other: &Self) -> bool {
        (**self).gt(&**other)
    }

    fn ge(&self, other: &Self) -> bool {
        (**self).ge(&**other)
    }
}

impl<T: Ord> Ord for Node<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: Hash> Hash for Node<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: fmt::Display> fmt::Display for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: fmt::Debug> fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T> fmt::Pointer for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ptr = Self::as_ptr(self);
        fmt::Pointer::fmt(&ptr, f)
    }
}

impl<T: Default> Default for Node<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<T> for Node<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> AsRef<T> for Node<T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T> Borrow<T> for Node<T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T> Node<T> {
    /// Insert all items from `iter` before the first successor that satisfies `predicate`.
    ///
    /// Elements are inserted in iteration order; each successful insertion
    /// starts its search from the previously inserted node to avoid reversing
    /// the sequence when `predicate` always matches.
    ///
    /// Returns the items that could not be inserted (their predicates never
    /// matched during traversal).
    pub fn extend_using<I, F>(&mut self, iter: I, predicate: F) -> Vec<T>
    where
        I: IntoIterator<Item = T>,
        F: Fn(&T) -> bool,
    {
        let mut tail = self.clone();
        let mut failed = Vec::new();

        for value in iter {
            match tail.push_before(value, &predicate) {
                Ok(()) => tail = Node::load_next_strong(&tail),
                Err(v) => failed.push(v),
            }
        }

        failed
    }

    /// Insert items paired with their own predicate, returning any that failed to insert.
    pub fn extend_with_predicates<I, F>(&mut self, iter: I) -> Vec<(T, F)>
    where
        I: IntoIterator<Item = (T, F)>,
        F: Fn(&T) -> bool,
    {
        let mut tail = self.clone();
        let mut failed = Vec::new();

        for (value, predicate) in iter {
            match tail.push_before(value, &predicate) {
                Ok(()) => tail = Node::load_next_strong(&tail),
                Err(v) => failed.push((v, predicate)),
            }
        }

        failed
    }
}

impl<T> Extend<T> for Node<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let _ = self.extend_using(iter, |_| true);
    }
}

unsafe impl<T: Send + Sync> Send for Node<T> {}
unsafe impl<T: Send + Sync> Sync for Node<T> {}

impl<T> Clone for Node<T> {
    fn clone(&self) -> Self {
        let old_size = self.inner().strong.fetch_add(1, Ordering::Relaxed);

        assert!(old_size <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);

        Self { ptr: self.ptr }
    }
}

impl<T> Iterator for NodeIter<T> {
    type Item = Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.take()?;

        self.current = Node::resolve_next(&current);

        self.current.clone()
    }
}

impl<T> Iterator for UniqueNodeIter<T> {
    type Item = Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.start.as_ref()?;
        let current = self.next.take()?;

        let successor = Node::resolve_next(&current);

        self.next = successor.and_then(|next| {
            if Node::ptr_eq(&next, &start) {
                None
            } else {
                Some(next)
            }
        });

        if self.next.is_none() {
            drop(self.start.take());
        }

        Some(current)
    }
}

impl<T> Drop for Node<T> {
    #[inline(never)]
    fn drop(&mut self) {
        match self.inner().strong.fetch_sub(1, Ordering::Release) {
            1 => {
                acquire!(self.inner().strong);

                // Implicit weak created on construction.
                drop(WeakNode { ptr: self.ptr });

                unsafe { ptr::drop_in_place(&raw mut self.ptr.as_mut().next) };
                unsafe { ptr::drop_in_place(&raw mut self.ptr.as_mut().data) };
            }
            // There is 1 other reference to self out there.
            // We need to make sure that reference is not ourselves.
            2 => {
                let next = Self::next(self).strong.load_noclone(Ordering::Acquire);

                // If we are at any point able to CAS 1 with 0 then we guarantee that we are the only strong reference.
                // No upgrades can happen with strong == 0, and no clones can happen without breaking &mut self.
                // If this fails, we are not the only holder of a strong reference and next cannot possibly be self anyway.
                if next
                    .inner()
                    .strong
                    .compare_exchange(1, 0, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    if Self::ptr_eq(&next, &self) {
                        // We need this counter to be 1 for drop
                        next.inner().strong.store(1, Ordering::Release);
                        // We have confirmed that it is our own strong next that is our only strong reference left.
                        // This will drop self where self.next.strong.load() == None
                        unsafe { drop(Self::next(self).strong.take(Ordering::Relaxed)) }
                    } else {
                        // We mistakenly consumed the last strong reference to next which is not self
                        next.inner().strong.store(1, Ordering::Release);
                    }
                }
            }
            _ => return,
        }
    }
}

#[repr(transparent)]
/// Non-owning view of a `Node<T>` sharing the weak count.
///
/// Much like `Arc::downgrade`, every `WeakNode<T>` owns a *weak* `next`
/// reference (`weak_next`). The pointer is initialized to refer back to the node
/// itself and later follows the real successor once the node is linked into a
/// ring. Dropping the last `WeakNode` tears down both the weak counter and the
/// stored `weak_next` edge.
pub struct WeakNode<T> {
    ptr: NonNull<NodeInner<T>>,
}

/// Borrowed view into `WeakNode` counters and links.
struct WeakInner<'a, T> {
    strong: &'a AtomicUsize,
    weak: &'a AtomicUsize,
    next: &'a NonNullAtomicWeakNode<T>,
}

impl<T> WeakNode<T> {
    pub fn as_ptr(&self) -> *const T {
        let ptr: *mut NodeInner<T> = NonNull::as_ptr(self.ptr);

        unsafe { &raw mut (*ptr).data }
    }

    #[must_use = "losing the pointer will leak memory"]
    pub fn into_raw(self) -> *const T {
        ManuallyDrop::new(self).as_ptr()
    }

    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self {
            ptr: unsafe {
                NonNull::new_unchecked(
                    ptr.byte_sub(offset_of!(NodeInner<T>, data)) as *mut NodeInner<T>
                )
            },
        }
    }

    pub fn upgrade(&self) -> Option<Node<T>> {
        #[inline]
        fn checked_increment(n: usize) -> Option<usize> {
            // Any write of 0 we can observe leaves the field in permanently zero state.
            if n == 0 {
                return None;
            }

            assert!(n <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);
            Some(n + 1)
        }

        // We use a CAS loop to increment the strong count instead of a
        // fetch_add as this function should never take the reference count
        // from zero to one.
        //
        // Relaxed is fine for the failure case because we don't have any expectations about the new state.
        // Acquire is necessary for the success case to synchronise with `Arc::new_cyclic`, when the inner
        // value can be initialized after `Weak` references have already been created. In that case, we
        // expect to observe the fully initialized value.
        if self
            .inner()
            .strong
            .fetch_update(Ordering::Acquire, Ordering::Relaxed, checked_increment)
            .is_ok()
        {
            Some(Node { ptr: self.ptr })
        } else {
            None
        }
    }

    pub fn next(&self) -> &NonNullAtomicWeakNode<T> {
        self.inner().next
    }

    pub fn load_next(&self) -> WeakNode<T> {
        self.next().load(Ordering::Acquire)
    }

    pub fn load_next_unique(&self) -> Option<WeakNode<T>> {
        let next = self.load_next();

        if Self::ptr_eq(self, &next) {
            None
        } else {
            Some(next)
        }
    }

    pub fn find_next_strong(&self) -> Option<Node<T>> {
        let mut next = self.load_next_unique()?;

        loop {
            if let Some(strong_next) = next.upgrade() {
                break Some(strong_next);
            } else {
                if let Some(weak_next) = next.load_next_unique() {
                    next = weak_next;
                    continue;
                } else {
                    break None;
                }
            }
        }
    }

    #[inline]
    fn inner(&self) -> WeakInner<'_, T> {
        let ptr = self.ptr.as_ptr();

        unsafe {
            WeakInner {
                strong: &(*ptr).strong,
                weak: &(*ptr).weak,
                next: &(*ptr).weak_next,
            }
        }
    }
}

impl<T> fmt::Debug for WeakNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(WeakNode)")
    }
}

impl<T> fmt::Pointer for WeakNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ptr = self.as_ptr();
        fmt::Pointer::fmt(&ptr, f)
    }
}

unsafe impl<T: Send + Sync> Send for WeakNode<T> {}
unsafe impl<T: Send + Sync> Sync for WeakNode<T> {}

impl<T> Clone for WeakNode<T> {
    fn clone(&self) -> Self {
        let old_size = self.inner().weak.fetch_add(1, Ordering::Relaxed);

        assert!(old_size <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);

        WeakNode { ptr: self.ptr }
    }
}

impl<T> Drop for WeakNode<T> {
    // See Node::drop
    fn drop(&mut self) {
        match self.inner().weak.fetch_sub(1, Ordering::Release) {
            1 => {
                acquire!(self.inner().weak);

                let NodeInner { next, data, .. } = unsafe { *Box::from_raw(self.ptr.as_ptr()) };

                // Already dropped by strong referrer.
                let _ = ManuallyDrop::new(next);
                let _ = ManuallyDrop::new(data);
            }
            2 => {
                let next = self.next().load_noclone(Ordering::Acquire);

                if next
                    .inner()
                    .weak
                    .compare_exchange(1, 0, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
                {
                    if Self::ptr_eq(&next, &self) {
                        next.inner().weak.store(1, Ordering::Release);
                        unsafe { drop(self.next().take(Ordering::Relaxed)) }
                    } else {
                        next.inner().weak.store(1, Ordering::Release);
                    }
                }
            }
            _ => return,
        }
    }

    // fn drop(&mut self) {
    //     let inner = self.inner();

    //     match inner.weak.fetch_sub(1, Ordering::Release) {
    //         1 => {
    //             acquire!(inner.weak);

    //             let NodeInner { next, data, .. } = unsafe { *Box::from_raw(self.inner.as_ptr()) };

    //             // Already dropped by strong referrer.
    //             let _ = ManuallyDrop::new(next);
    //             let _ = ManuallyDrop::new(data);
    //         }
    //         // See Node::drop
    //         2 => drop(self.next().compare_exchange(
    //             Some(&self),
    //             None,
    //             Ordering::Acquire,
    //             Ordering::Relaxed,
    //         )),
    //         _ => return,
    //     }
    // }
}
