use crate::{atm_p::NonNullAtomicP, sync::RawExt};

pub struct Cursor<T, P: RawExt<T>> {
    atm_ptr: NonNullAtomicP<T, P>,
}

impl<T, P: RawExt<T>> Cursor<T, P> {
    pub fn new(p: P) -> Self {
        Self {
            atm_ptr: NonNullAtomicP::new(p),
        }
    }

    
}
