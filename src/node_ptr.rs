use std::{ops::Deref, ptr::NonNull};

use haphazard::raw::Pointer;

pub struct NodePtr<T> {
    inner: NonNull<T>,
}

impl<T> Clone for NodePtr<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Copy for NodePtr<T> {}

impl<T> NodePtr<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(value))) },
        }
    }
}

impl<T> Deref for NodePtr<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.inner.as_ptr() }
    }
}

unsafe impl<T> Pointer<T> for NodePtr<T> {
    fn into_raw(self) -> *mut T {
        self.inner.as_ptr()
    }

    unsafe fn from_raw(ptr: *mut T) -> Self {
        Self {
            inner: unsafe { NonNull::new_unchecked(ptr) },
        }
    }
}
