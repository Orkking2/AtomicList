use crate::sync::{Node, RawExt, WeakNode};
use std::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr,
    sync::{
        Arc, Weak,
        atomic::{AtomicPtr, Ordering},
    },
};

pub type AtomicArc<T> = AtomicP<T, Arc<T>>;
pub type AtomicWeak<T> = AtomicP<T, Weak<T>>;

pub type AtomicNode<T> = AtomicP<T, Node<T>>;
pub type AtomicWeakNode<T> = AtomicP<T, WeakNode<T>>;

pub type NonNullAtomicNode<T> = NonNullAtomicP<T, Node<T>>;
pub type NonNullAtomicWeakNode<T> = NonNullAtomicP<T, WeakNode<T>>;

/// Drop of NonNullAtomicP is exactly the drop of AtomicP, so even if it is in the null state
/// which is technically invalid (for the sake of the methods) it can still be dropped safely.
#[repr(transparent)]
#[derive(Clone)]
pub struct NonNullAtomicP<T, P: RawExt<T>> {
    inner: AtomicP<T, P>,
}

impl<T, P: Clone + RawExt<T>> TryFrom<AtomicP<T, P>> for NonNullAtomicP<T, P> {
    type Error = AtomicP<T, P>;

    fn try_from(value: AtomicP<T, P>) -> Result<Self, Self::Error> {
        if value.load(Ordering::Relaxed).is_some() {
            Ok(Self { inner: value })
        } else {
            Err(value)
        }
    }
}

impl<T, P: RawExt<T>> From<NonNullAtomicP<T, P>> for AtomicP<T, P> {
    fn from(value: NonNullAtomicP<T, P>) -> Self {
        value.inner
    }
}

impl<T, P: RawExt<T>> NonNullAtomicP<T, P> {
    pub fn new(p: P) -> Self {
        Self {
            inner: AtomicP::new(p),
        }
    }

    /// Safety:
    ///
    /// Caller must ensure a store is called with
    /// a valid P before any other methods are called (drop is always allowed).
    pub unsafe fn null() -> Self {
        Self {
            inner: AtomicP::null(),
        }
    }

    pub fn as_atomic_p(&self) -> &AtomicP<T, P> {
        &self.inner
    }

    /// Safety:
    ///
    /// Caller must ensure a valid store is called before
    /// loading `self` again (drop is always allowed).
    /// No method should be called on `Self` besides store.
    pub unsafe fn take(&self, order: Ordering) -> P {
        self.inner.take(order).unwrap()
    }

    pub fn store(&self, new: P, order: Ordering) {
        self.inner.store(new, order)
    }

    pub fn as_ptr(&self, order: Ordering) -> *mut T {
        self.inner.as_ptr(order)
    }

    pub fn load(&self, order: Ordering) -> P
    where
        P: Clone,
    {
        unsafe { self.inner.load_unchecked(order) }
    }

    pub fn load_noclone(&self, order: Ordering) -> ManuallyDrop<P> {
        unsafe { self.inner.load_manuallydrop_unchecked(order) }
    }

    pub fn swap(&self, new: P, order: Ordering) -> P {
        unsafe { self.inner.swap_unchecked(new, order) }
    }

    pub fn compare_exchange(
        &self,
        current: &P,
        new: P,
        success: Ordering,
        failure: Ordering,
    ) -> Result<P, CASErr<P>>
    where
        P: Clone,
    {
        unsafe {
            self.inner
                .compare_exchange_unchecked(current, new, success, failure)
        }
    }
}

pub struct AtomicP<T, P: RawExt<T>> {
    ptr: AtomicPtr<T>,
    _marker: PhantomData<P>,
}

impl<T, P: RawExt<T>> AtomicP<T, P> {
    pub fn new(p: P) -> Self {
        let raw = P::into_raw(p);

        Self {
            ptr: AtomicPtr::new(raw as *mut T),
            _marker: PhantomData,
        }
    }

    /// Create an empty cell (no value).
    pub fn null() -> Self {
        Self {
            ptr: AtomicPtr::new(ptr::null_mut()),
            _marker: PhantomData,
        }
    }

    /// Produces an extra P that doesn't get dropped.
    pub fn load(&self, order: Ordering) -> Option<P>
    where
        P: Clone,
    {
        let p = self.ptr.load(order);

        if p.is_null() {
            return None;
        }

        unsafe {
            let out = P::from_raw(p);
            let _extra = ManuallyDrop::new(out.clone());

            Some(out)
        }
    }

    /// Safety:
    ///
    /// Caller must guarantee that `self` contains a valid pointer.
    /// e.g. Has only stored `Some(e)`, never `None`.
    pub unsafe fn load_unchecked(&self, order: Ordering) -> P
    where
        P: Clone,
    {
        unsafe {
            let out = P::from_raw(self.ptr.load(order));
            let _extra = ManuallyDrop::new(out.clone());

            out
        }
    }

    pub unsafe fn load_manuallydrop_unchecked(&self, order: Ordering) -> ManuallyDrop<P> {
        unsafe { ManuallyDrop::new(P::from_raw(self.ptr.load(order))) }
    }

    /// Get the current raw pointer for use in CAS.
    ///
    /// This does *not* touch the refcount.
    pub fn as_ptr(&self, order: Ordering) -> *mut T {
        self.ptr.load(order)
    }

    /// Swap the stored `Arc<T>` with a new one, returning the old one.
    ///
    /// If `new` is `Some`, the cell takes ownership of its ref.
    /// If `new` is `None`, the cell becomes empty.
    pub fn swap(&self, new: Option<P>, order: Ordering) -> Option<P> {
        unsafe { ptr_to_opt_p(self.ptr.swap(opt_p_to_ptr(new), order)) }
    }

    /// Safety:
    ///
    /// Caller must guarantee that `self` contains a valid pointer.
    /// e.g. Has only stored `Some(e)`, never `None`.
    pub unsafe fn swap_unchecked(&self, new: P, order: Ordering) -> P {
        unsafe { ptr_to_p(self.ptr.swap(p_to_ptr(new), order)) }
    }

    pub fn store(&self, new: P, order: Ordering) {
        drop(self.swap(Some(new), order))
    }

    /// Take the value out of the cell (cell becomes empty).
    /// The returned Arc owns what used to be the cell's strong ref.
    pub fn take(&self, order: Ordering) -> Option<P> {
        self.swap(None, order)
    }

    /// Safety:
    ///
    /// Caller must guarantee that `self` contains a valid pointer.
    /// e.g. Has only stored `Some(e)`, never `None`.
    pub unsafe fn compare_exchange_unchecked(
        &self,
        current: &P,
        new: P,
        success: Ordering,
        failure: Ordering,
    ) -> Result<P, CASErr<P>>
    where
        P: Clone,
    {
        let new_raw = p_to_ptr(new);

        match self
            .ptr
            .compare_exchange(P::as_ptr(current) as *mut T, new_raw, success, failure)
        {
            Ok(old_raw) => Ok(unsafe { ptr_to_p(old_raw) }),
            Err(actual_raw) => {
                let actual = unsafe { P::from_raw(actual_raw) };
                let _extra = ManuallyDrop::new(actual.clone());

                Err(CASErr {
                    actual,
                    new: unsafe { ptr_to_p(new_raw) },
                })
            }
        }
    }

    /// On success: you get the old `Arc<T>` that was in the cell.
    ///
    /// On failure: you get back both your `new` with no overal ref count change,
    /// and what is essentially a `load` of the previous value, which should just be dropped if not needed.
    pub fn compare_exchange(
        &self,
        current: Option<&P>,
        new: Option<P>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<P>, CASErr<Option<P>>>
    where
        P: Clone,
    {
        let new_raw = opt_p_to_ptr(new);

        match self.ptr.compare_exchange(
            current
                .map(P::as_ptr)
                .map(|p| p as *mut T)
                .unwrap_or(ptr::null_mut()),
            new_raw,
            success,
            failure,
        ) {
            Ok(old_raw) => Ok(unsafe { ptr_to_opt_p(old_raw) }),
            Err(actual_raw) => {
                let actual = if actual_raw.is_null() {
                    None
                } else {
                    let out = unsafe { P::from_raw(actual_raw) };
                    let _extra = ManuallyDrop::new(out.clone());

                    Some(out)
                };

                Err(CASErr {
                    actual,
                    new: unsafe { ptr_to_opt_p(new_raw) },
                })
            }
        }
    }
}

impl<T, P: Clone + RawExt<T>> Clone for AtomicP<T, P> {
    fn clone(&self) -> Self {
        let out = Self::null();
        let _ = out.swap(self.load(Ordering::Relaxed), Ordering::Relaxed);
        out
    }
}

impl<T, P: RawExt<T>> Drop for AtomicP<T, P> {
    fn drop(&mut self) {
        // Safe because &mut self guarantees no concurrent access.
        let raw = *self.ptr.get_mut();

        if !raw.is_null() {
            unsafe {
                drop(P::from_raw(raw));
            }
        }
    }
}

pub struct CASErr<T> {
    pub actual: T,
    pub new: T,
}

fn p_to_ptr<T, P: RawExt<T>>(p: P) -> *mut T {
    P::into_raw(p) as *mut T
}

fn opt_p_to_ptr<T, P: RawExt<T>>(opt_p: Option<P>) -> *mut T {
    match opt_p {
        Some(p) => p_to_ptr(p),
        None => ptr::null_mut(),
    }
}

unsafe fn ptr_to_opt_p<T, P: RawExt<T>>(ptr: *mut T) -> Option<P> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { ptr_to_p(ptr) })
    }
}

unsafe fn ptr_to_p<T, P: RawExt<T>>(ptr: *mut T) -> P {
    unsafe { P::from_raw(ptr) }
}
