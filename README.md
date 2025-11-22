# AtomicList

`AtomicList` is an experimental circular list built directly from intrusive,
reference-counted nodes. The crate ships nodes and cursors only—there is no
separate list container—so you keep a `Node<T>` (and optionally a shared
`cursor::Cursor`) as your handle into the ring. A `sync::Node<T>` acts like a
tiny `Arc<T>` that also stores **strong** and **weak** successors, letting CAS
loops splice nodes in and out of a ring without locks or external hazard-pointer
schemes.

## Current surface area
- `Node<T>` / `WeakNode<T>`: Arc/Weak-like handles with embedded successor edges.
- Lock-free splice helpers: `push_before`, `pop_when`, and `remove_self` rearrange
  a ring while maintaining the self-loop invariants on detached nodes.
- Successor introspection: `load_next_*` plus `find_next_strong` follow weak
  breadcrumbs after a node has been popped out.
- Shared traversal: `cursor::Cursor` is an atomic cursor that multiple holders
  advance together.
- Reusable atomics: `atm_p` exports generic atomic pointer wrappers used by the
  list but also usable for `Arc`/`Weak`.

## Working with nodes

```rust,no_run
use atomic_list::sync::Node;

let root = Node::new("root");
root.push_before("worker-1", |cur| *cur == "root").unwrap();
root.push_before("worker-2", |_| true).unwrap();

// Remove the first node whose payload ends with "1".
let removed = root.pop_when(|cur| cur.ends_with('1')).unwrap();
assert_eq!(*removed, "worker-1");

// The ring closes back over the gap.
assert_eq!(root.load_next_strong().to_string(), "worker-2");
assert!(Node::ptr_eq(
    &root.load_next_strong().load_next_strong(),
    &root
));

// Detached nodes keep a weak breadcrumb into the live ring.
assert_eq!(
    removed.find_next_strong().unwrap().to_string(),
    "root"
);
assert_eq!(Node::into_inner(removed), Some("worker-1"));
```

## Coordinated traversal

```rust
use atomic_list::{cursor::Cursor, sync::Node};

let head = Node::new(0);
for i in 1..=3 {
    head.push_before(i, |_| true).unwrap();
}

// Multiple holders share the same atomic cursor position.
let mut cursor_a = Cursor::new(head.clone());
let mut cursor_b = cursor_a.clone();

assert_eq!(*cursor_a.next().unwrap(), 3);
assert_eq!(*cursor_b.next().unwrap(), 3);

// Advancing from either handle moves everyone forward.
assert_eq!(*cursor_a.next().unwrap(), 2);
assert_eq!(*cursor_b.next().unwrap(), 2);
```

## Notes
- Pure atomics: no hazard pointers or epoch GC, just ref-counted nodes.
- There is no separate `AtomicList` struct; hang on to a `Node<T>` as your entry
  point into the ring.
- The API is experimental and may still shift as iteration ergonomics improve.

## License

This project is distributed under the terms of the MIT license. See `LICENSE` for details.
