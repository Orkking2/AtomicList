#![cfg(test)]

use crate::{cursor::Cursor, sync::Node};
use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier, Mutex,
    },
    thread::{self, yield_now},
};

fn push_before_anywhere(root: &Node<usize>, mut value: usize) {
    for _ in 0..1_000 {
        match root.push_before(value, |_| true) {
            Ok(_) => return,
            Err(v) => {
                value = v;
                yield_now();
            }
        }
    }
    panic!("unable to insert {value} after retries");
}

#[test]
fn concurrent_push_before_keeps_all_nodes() {
    let root = Node::new(0usize);
    let threads = 8;
    let per_thread = 25;

    let next_id = Arc::new(AtomicUsize::new(1));
    let barrier = Arc::new(Barrier::new(threads));

    let mut handles = Vec::new();
    for _ in 0..threads {
        let root = root.clone();
        let barrier = Arc::clone(&barrier);
        let next_id = Arc::clone(&next_id);

        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..per_thread {
                let id = next_id.fetch_add(1, Ordering::Relaxed);
                push_before_anywhere(&root, id);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let mut seen = HashSet::new();
    for node in root.unique_iter_from() {
        seen.insert(*node);
    }

    assert_eq!(seen.len(), threads * per_thread + 1);
    for id in 0..=threads * per_thread {
        assert!(seen.contains(&id), "missing {id} after concurrent pushes");
    }
}

#[test]
fn push_and_pop_race_leaves_singleton_ring() {
    let root = Node::new(0usize);
    let initial = 4usize;
    for i in 1..=initial {
        push_before_anywhere(&root, i);
    }

    let push_total = 20usize;
    let next_id = Arc::new(AtomicUsize::new(initial + 1));
    let remaining = Arc::new(AtomicUsize::new(initial));
    let popped_values: Arc<Mutex<Vec<Node<usize>>>> = Arc::new(Mutex::new(Vec::new()));

    let pushers = 2;
    let poppers = 2;
    let barrier = Arc::new(Barrier::new(pushers + poppers));
    let mut handles = Vec::new();

    for _ in 0..pushers {
        let root = root.clone();
        let barrier = Arc::clone(&barrier);
        let next_id = Arc::clone(&next_id);
        let remaining = Arc::clone(&remaining);

        handles.push(thread::spawn(move || {
            barrier.wait();
            loop {
                let id = next_id.fetch_add(1, Ordering::Relaxed);
                if id > initial + push_total {
                    break;
                }
                push_before_anywhere(&root, id);
                remaining.fetch_add(1, Ordering::AcqRel);
            }
        }));
    }

    for _ in 0..poppers {
        let root = root.clone();
        let barrier = Arc::clone(&barrier);
        let remaining = Arc::clone(&remaining);
        let popped_values = Arc::clone(&popped_values);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let max_attempts = (initial + push_total) * 20;

            for _ in 0..max_attempts {
                if remaining.load(Ordering::Acquire) == 0 {
                    break;
                }

                if let Some(node) = root.pop_when(|cur| *cur != 0) {
                    popped_values.lock().unwrap().push(node);
                    remaining.fetch_sub(1, Ordering::AcqRel);
                } else {
                    // Once all pushes are published, switch to yielding until the main
                    // thread drains any stragglers.
                    yield_now();
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    while let Some(node) = root.pop_when(|cur| *cur != 0) {
        popped_values.lock().unwrap().push(node);
    }

    let popped = popped_values.lock().unwrap();
    let values: Vec<_> = popped.iter().map(|node| **node).collect();
    let unique: HashSet<_> = values.iter().copied().collect();
    assert_eq!(unique.len(), popped.len());
    assert!(popped.len() >= initial);
    assert!(popped.len() <= initial + push_total);
    assert!(values.iter().all(|v| *v != 0));

    assert!(Node::ptr_eq(&root.load_next_strong(), &root));
    assert!(root.resolve_next().is_none());
}

#[test]
fn independent_rings_handle_concurrent_work() {
    let ring_a = Node::new(0usize);
    let ring_b = Node::new(10_000usize);

    let initial = 8usize;
    for i in 1..=initial {
        push_before_anywhere(&ring_a, i);
        push_before_anywhere(&ring_b, 10_000 + i);
    }

    let extra = 16usize;
    let barrier = Arc::new(Barrier::new(4));
    let popped_a: Arc<Mutex<Vec<Node<usize>>>> = Arc::new(Mutex::new(Vec::new()));
    let popped_b: Arc<Mutex<Vec<Node<usize>>>> = Arc::new(Mutex::new(Vec::new()));
    let next_a = Arc::new(AtomicUsize::new(1_000));
    let next_b = Arc::new(AtomicUsize::new(20_000));

    let mut handles = Vec::new();

    {
        let ring_a = ring_a.clone();
        let barrier = Arc::clone(&barrier);
        let next_a = Arc::clone(&next_a);

        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..extra {
                let id = next_a.fetch_add(1, Ordering::Relaxed);
                push_before_anywhere(&ring_a, id);
            }
        }));
    }

    {
        let ring_b = ring_b.clone();
        let barrier = Arc::clone(&barrier);
        let next_b = Arc::clone(&next_b);

        handles.push(thread::spawn(move || {
            barrier.wait();
            for _ in 0..extra {
                let id = next_b.fetch_add(1, Ordering::Relaxed);
                push_before_anywhere(&ring_b, id);
            }
        }));
    }

    {
        let ring_a = ring_a.clone();
        let barrier = Arc::clone(&barrier);
        let popped_a = Arc::clone(&popped_a);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let mut popped = 0;
            let mut attempts = 0;
            while popped < initial + extra && attempts < (initial + extra) * 20 {
                if let Some(node) = ring_a.pop_when(|cur| *cur != 0) {
                    popped += 1;
                    popped_a.lock().unwrap().push(node);
                } else {
                    attempts += 1;
                    yield_now();
                }
            }
        }));
    }

    {
        let ring_b = ring_b.clone();
        let barrier = Arc::clone(&barrier);
        let popped_b = Arc::clone(&popped_b);

        handles.push(thread::spawn(move || {
            barrier.wait();
            let mut popped = 0;
            let mut attempts = 0;
            while popped < initial + extra && attempts < (initial + extra) * 20 {
                if let Some(node) = ring_b.pop_when(|cur| *cur != 10_000) {
                    popped += 1;
                    popped_b.lock().unwrap().push(node);
                } else {
                    attempts += 1;
                    yield_now();
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    while let Some(node) = ring_a.pop_when(|cur| *cur != 0) {
        popped_a.lock().unwrap().push(node);
    }
    while let Some(node) = ring_b.pop_when(|cur| *cur != 10_000) {
        popped_b.lock().unwrap().push(node);
    }

    let popped_a = popped_a.lock().unwrap();
    let popped_b = popped_b.lock().unwrap();

    assert!(popped_a.len() >= initial);
    assert!(popped_a.len() <= initial + extra);
    assert!(popped_b.len() >= initial);
    assert!(popped_b.len() <= initial + extra);

    let values_a: Vec<_> = popped_a.iter().map(|n| **n).collect();
    let values_b: Vec<_> = popped_b.iter().map(|n| **n).collect();

    assert!(values_a.iter().all(|v| *v < 10_000));
    assert!(values_b.iter().all(|v| *v > 10_000));

    assert!(Node::ptr_eq(&ring_a.load_next_strong(), &ring_a));
    assert!(Node::ptr_eq(&ring_b.load_next_strong(), &ring_b));
}

#[test]
fn ownership_helpers_only_succeed_when_unique() {
    let with_alias = Node::new(7usize);
    let alias = with_alias.clone();
    assert!(Node::into_inner(with_alias).is_none());
    assert_eq!(Node::into_inner(alias), Some(7));

    let unique = Node::new(11usize);
    assert_eq!(Node::try_unwrap(unique), Ok(11));

    let cursor = Cursor::new(Node::new(25usize));
    let cursor_clone = cursor.clone();
    assert!(Cursor::into_current(cursor).is_none());
    assert_eq!(*Cursor::into_current(cursor_clone).unwrap(), 25);
}
