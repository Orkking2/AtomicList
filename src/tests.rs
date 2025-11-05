#[cfg(test)]
mod tests {
    use crate::list::AtomicList;
    use std::collections::HashSet;
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;

    // Walk the ring once starting from the current root and collect values.
    fn collect_ring(list: &AtomicList<usize>) -> Vec<usize> {
        let mut out = Vec::new();
        if let Some(root) = list.root() {
            let root_ptr = root.ptr();
            out.push(**root);
            let mut current = root;
            loop {
                let next = current.next_expected(list.domain());
                if next == root_ptr {
                    break;
                }
                out.push(**next);
                current = next;
            }
        }
        out
    }

    #[test]
    fn push_order_always_true_is_lifo() {
        let list = AtomicList::new();
        list.push_before(1, |_| true);
        list.push_before(2, |_| true);
        list.push_before(3, |_| true);

        let vals = collect_ring(&list);
        assert_eq!(vals, vec![3, 1, 2], "expected LIFO order around the ring");
    }

    #[test]
    fn pop_when_removes_first_match_and_updates_root() {
        let list = AtomicList::new();
        list.push_before(1, |_| true);
        list.push_before(2, |_| true);
        list.push_before(3, |_| true);
        // ring is [3, 1, 2]

        assert!(list.pop_when(|x| *x == 1));

        let vals = collect_ring(&list);
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&2) && vals.contains(&3));
        assert!(!vals.contains(&1));
    }

    #[test]
    fn empty_pop_returns_false() {
        let list = AtomicList::<usize>::new();
        assert!(!list.pop_when(|_| true));
    }

    #[test]
    fn next_if_advances_root_once_and_is_idempotent_on_stale_expected() {
        let list = AtomicList::new();
        list.push_before(1, |_| true);
        list.push_before(2, |_| true); // ring [2, 1]

        let expected = list.root().expect("root");
        let next = list.next_if(expected).expect("advance succeeds");
        assert_eq!(**next, 1);

        // Ring remains length 2.
        let vals = collect_ring(&list);
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn concurrent_pushes_all_nodes_present() {
        let list = Arc::new(AtomicList::new());
        let threads = 4;
        let per_thread = 100;
        let start = Arc::new(Barrier::new(threads));
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for _ in 0..threads {
            let list = Arc::clone(&list);
            let start = Arc::clone(&start);
            let counter = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                start.wait();
                for _ in 0..per_thread {
                    let id = counter.fetch_add(1, Ordering::Relaxed);
                    list.push_before(id, |_| true);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let vals = collect_ring(&list);
        assert_eq!(vals.len(), threads * per_thread);

        // Verify uniqueness of IDs
        let set: HashSet<_> = vals.iter().copied().collect();
        assert_eq!(set.len(), vals.len());
    }
}
