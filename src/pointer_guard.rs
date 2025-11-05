//! Hazard-pointer backed guard for nodes stored in the list.

use crate::{AtomicListFamily, node_ptr::NodePtr};
use haphazard::{AtomicPtr, Domain, HazardPointer};
use std::{ops::Deref, ptr::NonNull};

/// Protects a node loaded from an [`AtomicPtr`] using a hazard pointer.
pub struct PointerGuard<'a, T> {
    _hp: HazardPointer<'a, AtomicListFamily>,
    ptr: NonNull<T>,
}

impl<'a, T: Sync + Send> PointerGuard<'a, T> {
    /// Create a guard for the pointer stored in `atm_ptr`.
    pub fn new(
        atm_ptr: &AtomicPtr<T, AtomicListFamily, NodePtr<T>>,
        domain: &'a Domain<AtomicListFamily>,
    ) -> Option<Self> {
        let mut hp = HazardPointer::new_in_domain(domain);

        let node = unsafe { NonNull::new((atm_ptr.load(&mut hp)?) as *const _ as *mut _) }?;

        Some(Self { _hp: hp, ptr: node })
    }

    /// Expose the protected raw pointer.
    pub fn ptr(&self) -> NonNull<T> {
        self.ptr.clone()
    }
}

impl<'a, T: 'a> AsRef<T> for PointerGuard<'a, T> {
    fn as_ref(&self) -> &'a T {
        // Safety: `hp` protects `ptr` for the entire lifetime of `'a`
        unsafe { self.ptr.as_ref() }
    }
}

impl<'a, T: 'a> Deref for PointerGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, T> PartialEq<NonNull<T>> for PointerGuard<'a, T> {
    fn eq(&self, other: &NonNull<T>) -> bool {
        self.ptr == *other
    }
}
