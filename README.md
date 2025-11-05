# AtomicList

`AtomicList` is a small Rust crate that implements a lock-free, hazard-pointer protected,
circularly linked list. It is designed for workloads where many threads concurrently append
or retire nodes while keeping their local iterators in sync with one another.

The list is built on top of the [`haphazard`](https://docs.rs/haphazard/latest/haphazard/)
hazard-pointer library. Every access to a node is protected by a `PointerGuard`, allowing
producers and consumers to cooperate without resorting to global locks and while still
retiring elements safely.

## Key capabilities
- **Non-blocking inserts:** `push_before` atomically splices a node before the first element
  matching a predicate, allowing multiple producers to converge on a shared insertion point.
- **Predicate-driven removal:** `pop_when` detaches the first matching node and schedules it
  for reclamation inside the list’s hazard-pointer domain.
- **Hazard-pointer aware iteration:** `AtomicListIter` keeps a local view of the ring that
  advances together with the global root via `next_if`, so several iterators can walk the list
  without observing torn state.
- **Automatic memory management:** Nodes are retired once no iterator protects them, avoiding
  ABA issues while staying compatible with `Send + Sync` payloads.

## How it is structured

Internally the list is a ring of `AtomicListNode<T>` values. Each node owns its successor
pointer via `NodePtr`, a thin wrapper that implements the raw pointer contract required by
`haphazard`. A list owns its own `Domain<AtomicListFamily>`, so hazard pointers issued by the
list stay scoped to that domain and can retire nodes cooperatively.

The root pointer always participates in the ring. Inserting or removing elements is expressed
in terms of moving that root forward while performing compare-and-swap operations on the links:

- `push_before` walks the ring until the predicate is satisfied, relinks the predecessor to the
  new node, and, when necessary, advances the root to keep it near recent insertions.
- `pop_when` uses the same traversal pattern to unlink a node, advances the root past it, and
  retires the node within the hazard-pointer domain.

Iterators are lightweight: they snapshot a hazard-protected pointer to the current root, then
use `next_if` to stay synchronized with any concurrent root advancement. Because each yielded
item is a `PointerGuard`, consumers retain safe read access for as long as their guard lives.

## Example

```rust
use atomic_list::list::AtomicList;

// Create a list that will store work items shared between threads.
let list = AtomicList::new();

// Multiple producers can insert concurrently. The predicate chooses the insertion position.
list.push_before("item-0", |_| true);
list.push_before("item-1", |current| current.starts_with("item"));

// Consumers can retire nodes cooperatively.
let removed = list.pop_when(|current| current == &"item-0");
assert!(removed);

// Iterators hold hazard pointers so each step is synchronized with other walkers.
for guard in list.iter().take(4) {
    println!("saw {}", **guard);
}
```

## Testing

Integration tests in `src/tests.rs` stress concurrent producers, ensure predicate-based ordering,
and verify that removals keep the ring consistent. Run them with:

```shell
cargo test
```

## License

This project is distributed under the terms of the MIT license. See `LICENSE` for details.
