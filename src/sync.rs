use crate::atm_p::{NonNullAtomicNode, NonNullAtomicWeakNode};
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

pub trait RawExt<T>
where
    Self: Sized,
{
    fn as_ptr(this: &Self) -> *const T;

    fn ptr_eq(lhs: &Self, rhs: &Self) -> bool {
        ptr::addr_eq(Self::as_ptr(lhs), Self::as_ptr(rhs))
    }

    fn into_raw(this: Self) -> *const T;
    unsafe fn from_raw(ptr: *const T) -> Self;
}

impl<T> RawExt<T> for Arc<T> {
    fn as_ptr(this: &Self) -> *const T {
        Self::as_ptr(this)
    }

    fn into_raw(this: Self) -> *const T {
        Self::into_raw(this)
    }

    unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self::from_raw(ptr) }
    }
}

impl<T> RawExt<T> for Weak<T> {
    fn as_ptr(this: &Self) -> *const T {
        Self::as_ptr(this)
    }

    fn into_raw(this: Self) -> *const T {
        Self::into_raw(this)
    }

    unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self::from_raw(ptr) }
    }
}

impl<T> RawExt<T> for Node<T> {
    fn as_ptr(this: &Self) -> *const T {
        Self::as_ptr(this)
    }

    fn into_raw(this: Self) -> *const T {
        Self::into_raw(this)
    }

    unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe { Self::from_raw(ptr) }
    }
}

impl<T> RawExt<T> for WeakNode<T> {
    fn as_ptr(this: &Self) -> *const T {
        Self::as_ptr(this)
    }

    fn into_raw(this: Self) -> *const T {
        Self::into_raw(this)
    }

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
pub struct Node<T> {
    ptr: NonNull<NodeInner<T>>,
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
    /// assert_eq!(**node, "root");
    /// // The initial successor is the node itself.
    /// assert!(Node::ptr_eq(&node, &node.load_next_strong()));
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

        let Next { strong, weak } = this.next();

        // Here we guarantee the safety of NonNullAtomicP::null
        strong.store(this.clone(), Ordering::Relaxed);
        weak.store(Self::downgrade(&this), Ordering::Relaxed);

        this
    }

    /// Obtain a raw pointer to the stored payload.
    pub fn as_ptr(this: &Self) -> *const T {
        let ptr: *mut NodeInner<T> = NonNull::as_ptr(this.ptr);

        unsafe { &raw mut (*ptr).data }
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
                let next = this.next().strong.load_noclone(Ordering::Acquire);

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

                return Self::into_inner(unsafe { this.next().strong.take(Ordering::Relaxed) });
            }
            _ => None,
        }
    }

    /// Equivalent of `Arc::try_unwrap`, but aware of the implicit self edge.
    ///
    /// ```
    /// # use atomic_list::sync::Node;
    /// let node = Node::new(5);
    /// assert_eq!(Node::try_unwrap(node), Ok(5));
    ///
    /// let shared = Node::new(7);
    /// let alias = shared.clone();
    /// assert!(Node::try_unwrap(shared).is_err());
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

                let next = this.next().strong.load_noclone(Ordering::Acquire);

                if !Self::ptr_eq(&next, &this) {
                    return Err(this);
                }

                let this = ManuallyDrop::new(this);

                this.inner().strong.fetch_sub(1, Ordering::Release);

                let next_self = unsafe { this.next().strong.take(Ordering::Relaxed) };

                match Self::try_unwrap(next_self) {
                    Ok(value) => Ok(value),
                    Err(_) => unreachable!("self-referential Node::try_unwrap failed"),
                }
            }
        }
    }

    #[must_use = "losing the pointer will leak memory"]
    /// Leak-safe escape hatch that turns a `Node<T>` into a raw pointer.
    pub fn into_raw(this: Self) -> *const T {
        let this = ManuallyDrop::new(this);
        Self::as_ptr(&*this)
    }

    /// Recreate a `Node<T>` from the pointer returned by `into_raw`.
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
    pub fn strong(&self, order: Ordering) -> usize {
        self.inner().strong.load(order)
    }

    /// Load the current weak count with the requested ordering.
    pub fn weak(&self, order: Ordering) -> usize {
        self.inner().weak.load(order)
    }

    /// Borrow the strong/weak successor slots.
    pub fn next(&self) -> Next<'_, NonNullAtomicNode<T>, NonNullAtomicWeakNode<T>> {
        let inner = self.inner();

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
        let mut cur = this.inner().weak.load(Ordering::Relaxed);

        loop {
            if cur == usize::MAX {
                hint::spin_loop();
                cur = this.inner().weak.load(Ordering::Relaxed);
                continue;
            }

            assert!(cur <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);

            match this.inner().weak.compare_exchange_weak(
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
    /// wrappers.
    pub fn load_next_strong(&self) -> Self {
        self.next().strong.load(Ordering::Acquire)
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

unsafe impl<T: Send + Sync> Send for Node<T> {}
unsafe impl<T: Send + Sync> Sync for Node<T> {}

impl<T> Clone for Node<T> {
    fn clone(&self) -> Self {
        let old_size = self.inner().strong.fetch_add(1, Ordering::Relaxed);

        assert!(old_size <= MAX_REFCOUNT, "{}", INTERNAL_OVERFLOW_ERROR);

        Self { ptr: self.ptr }
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
                let next = self.next().strong.load_noclone(Ordering::Acquire);

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
                        unsafe { drop(self.next().strong.take(Ordering::Relaxed)) }
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
