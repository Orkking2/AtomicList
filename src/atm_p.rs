//! Atomic storage for pointer-like smart pointers.
//!
//! `AtomicP` mirrors the ergonomics of `AtomicPtr`, but it works with any `P`
//! that implements [`RawExt`]. That trait is satisfied by types such as
//! [`Arc`](std::sync::Arc), [`Weak`](std::sync::Weak), and the custom
//! [`Node`](crate::sync::Node) / [`WeakNode`](crate::sync::WeakNode) pairs used
//! throughout this crate. The contract is that `P` can be losslessly converted
//! to and from a `*const T` without running constructors or destructors. This
//! lets `AtomicP` exchange ownership by only manipulating raw pointers.
//!
//! Every atomic operation converts `P` to a raw pointer on the way in and
//! reconstructs it from the raw pointer on the way out. No drops happen inside
//! the atomic operations themselves; ref counts are adjusted by `from_raw` /
//! `into_raw` on the edges. This mirrors how `AtomicPtr` behaves and keeps the
//! control structures of `P` (like the ref counts inside `Arc`/`Weak`) in charge
//! of lifecycle.
//!
//! Because the contract is only about raw-pointer roundtrips, users can define
//! additional pointer types with custom control structures (for example,
//! `Arc`-like wrappers around intrusive nodes) and still gain atomic sharing by
//! plugging them into `AtomicP`.

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

/// An atomic cell for [`Arc`](std::sync::Arc).
pub type AtomicArc<T> = AtomicP<T, Arc<T>>;
/// An atomic cell for [`Weak`](std::sync::Weak).
pub type AtomicWeak<T> = AtomicP<T, Weak<T>>;

/// An atomic cell for [`Node`](crate::sync::Node).
pub type AtomicNode<T> = AtomicP<T, Node<T>>;
/// An atomic cell for [`WeakNode`](crate::sync::WeakNode).
pub type AtomicWeakNode<T> = AtomicP<T, WeakNode<T>>;

pub type NonNullAtomicNode<T> = NonNullAtomicP<T, Node<T>>;
pub type NonNullAtomicWeakNode<T> = NonNullAtomicP<T, WeakNode<T>>;

/// An [`AtomicP`] that promises to avoid the null state in normal use.
///
/// This is a convenience wrapper for call sites that do not want to juggle
/// `Option<P>`. Construction still goes through [`AtomicP`] and is therefore
/// subject to the same raw-pointer contract; dropping behaves identically.
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
    /// See [`AtomicP::new`]
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

    /// Get a reference to the inner AtomicP. Generally
    /// this is not necessary, but is exposed for 
    /// convenience.
    pub fn as_atomic_p(&self) -> &AtomicP<T, P> {
        &self.inner
    }

    /// Safety:
    ///
    /// Caller must ensure a valid store is called before
    /// loading `self` again (drop is always allowed).
    /// No method should be called on `Self` besides store (or drop).
    pub unsafe fn take(&self, order: Ordering) -> P {
        unsafe { self.inner.take(order).unwrap_unchecked() }
    }

    /// See [`AtomicP::store`]
    pub fn store(&self, new: P, order: Ordering) {
        self.inner.store(new, order)
    }

    /// See [`AtomicP::as_ptr`]
    pub fn as_ptr(&self, order: Ordering) -> *mut T {
        self.inner.as_ptr(order)
    }

    /// See [`AtomicP::load_unchecked`]
    pub fn load(&self, order: Ordering) -> P
    where
        P: Clone,
    {
        unsafe { self.inner.load_unchecked(order) }
    }

    /// See [`AtomicP::load_manuallydrop_unchecked`]
    pub fn load_noclone(&self, order: Ordering) -> ManuallyDrop<P> {
        unsafe { self.inner.load_manuallydrop_unchecked(order) }
    }

    /// See [`AtomicP::swap`]
    pub fn swap(&self, new: P, order: Ordering) -> P {
        unsafe { self.inner.swap_unchecked(new, order) }
    }

    /// See [`AtomicP::compare_exchange_unchecked`]
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

    /// Convert `self` into the P that it holds. Safety
    /// ensured by ownership requirement.
    pub fn into_p(self) -> P {
        unsafe { ptr_to_p(self.inner.ptr.swap(ptr::null_mut(), Ordering::AcqRel)) }
    }
}

/// An atomic cell for pointer-like smart pointers.
///
/// `P` must implement [`RawExt`], which promises that it can roundtrip through
/// a `*const T` without constructing or destroying the underlying allocation.
/// `AtomicP` relies on that promise to perform atomic operations directly on
/// raw pointers, skipping destructor calls on stores and constructor work on
/// loads. Control structures embedded in `P` (such as ref counts inside
/// [`Arc`](std::sync::Arc) or the custom [`Node`](crate::sync::Node)) stay
/// responsible for lifecycle.
pub struct AtomicP<T, P: RawExt<T>> {
    ptr: AtomicPtr<T>,
    _marker: PhantomData<P>,
}

impl<T, P: RawExt<T>> AtomicP<T, P> {
    /// Create a new atomic cell containing `p`.
    ///
    /// No destructor is run when storing `p`; the value is turned into a raw
    /// pointer via [`RawExt::into_raw`] and written directly. Likewise,
    /// reconstructing a value on subsequent loads relies solely on
    /// [`RawExt::from_raw`], so any ref counting or control state embedded in
    /// `P` remains authoritative.
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

    /// Load the current value, cloning to preserve ref counts.
    ///
    /// This converts the raw pointer back into `P` and immediately clones it,
    /// so the returned value is an *additional* handle (e.g. one more `Arc`).
    /// The clone keeps ref counting balanced while the original raw pointer
    /// stays in the atomic cell.
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

    /// Load without cloning, returning a `ManuallyDrop` wrapper.
    ///
    /// This exposes the exact pointer stored in the cell without touching ref
    /// counts. The caller must ensure the pointer remains valid.
    pub unsafe fn load_manuallydrop_unchecked(&self, order: Ordering) -> ManuallyDrop<P> {
        unsafe { ManuallyDrop::new(P::from_raw(self.ptr.load(order))) }
    }

    /// Get the current raw pointer for use in CAS.
    ///
    /// This does *not* touch the refcount.
    pub fn as_ptr(&self, order: Ordering) -> *mut T {
        self.ptr.load(order)
    }

    /// Swap the stored pointer-like `P` with a new one, returning the old one.
    ///
    /// If `new` is `Some`, the cell takes ownership of its ref (by writing the
    /// raw pointer). If `new` is `None`, the cell becomes empty. No drop occurs
    /// inside the atomic operation; the returned value should be dropped by the
    /// caller if it is not needed.
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

    /// Store a new value, replacing whatever was inside.
    ///
    /// The incoming `P` is turned into a raw pointer; no destructor runs for
    /// the overwritten value until it is dropped by whoever receives it from
    /// [`swap`](Self::swap) or [`take`](Self::take).
    pub fn store(&self, new: P, order: Ordering) {
        drop(self.swap(Some(new), order))
    }

    /// Take the value out of the cell (cell becomes empty).
    /// The returned pointer-like `P` owns what used to be the cell's ref.
    ///
    /// As with other operations, the transfer happens via raw pointers—no
    /// destructors run inside the atomic primitive.
    pub fn take(&self, order: Ordering) -> Option<P> {
        unsafe { ptr_to_opt_p(self.ptr.swap(ptr::null_mut(), order)) }
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

    /// On success: you get the old pointer-like `P` that was in the cell.
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

    pub fn try_into_p(self) -> Option<P> {
        self.take(Ordering::Relaxed)
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
        drop(self.take(Ordering::Relaxed));
    }
}

/// Error returned by compare-and-swap operations.
///
/// The caller receives both what was actually found in the cell and the
/// `new` value they attempted to write (reconstructed via `from_raw`) so ref
/// counts remain balanced.
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
