# AtomicList

`AtomicList` is an experimental circular linked list built from intrusive,
reference-counted nodes. Each element lives inside a `sync::Node<T>`, which embeds
its successor pointer and uses atomic reference counts that resemble a hand-rolled
`Arc`. Splicing and removal are expressed as compare-and-swap loops on those
next pointers, so the data structure can stay lock-free internally even though
the public API is still conservative about aliasing.

Unlike an earlier iteration of this crate, the current implementation does **not**
use hazard pointers or the `haphazard` crate. Inserts require `&mut self`, and the
ring never exposes iterators or automatic retirement hooks yet.

## What is implemented today
- **Circular storage.** The first inserted node becomes the root of the ring and every
  new node keeps a strong pointer to its successor via `NonNullAtomicNode<T>`.
- **Predicate-based insertion.** `AtomicList::push_before` allocates a new node and
  attempts to splice it before the first element whose value satisfies the predicate.
  If no element matches after a full traversal the function returns `Err(elem)` so
  the caller can recover ownership.
- **Predicate-based removal.** `AtomicList::pop_when` walks the ring, detaches the
  first matching node with a CAS, and returns it as `Option<Node<T>>`. Nodes act as
  guards – you can keep reading through `Deref` or reclaim the payload with
  `Node::into_inner`.
- **Manual traversal helpers (WIP).** `AtomicList::cursor` hands out a `cursor::Cursor`
  rooted at the current `Node<T>`. The cursor only stores the pointer today, so any
  higher-level iteration logic still has to be written manually on top of it.

Because `push_before` requires `&mut self`, the structure is not yet shareable
between threads even though the internals rely on atomics. `pop_when` can be invoked
through shared references, so it may be used by read-only consumers that observe
and remove nodes inserted elsewhere under exclusive access.

## Working with nodes

`sync::Node<T>` behaves a lot like an `Arc<T>` plus an embedded `next` and `next_weak` pointer:

- Cloning a `Node<T>` simply bumps the strong count.
- `Node::into_inner` returns `Some(T)` if the caller held the last strong reference,
  otherwise it yields `None`.
- Dropping a `Node<T>` automatically decrements the count and, once it reaches zero,
  tears down the stored successor pointer and payload.

`pop_when` hands ownership of a removed node back to the caller, so you can either
inspect it via `Deref<Target = T>` or reclaim the payload when you know no other
strong references exist.

## Example

```rust
use atomic_list::{list::AtomicList, sync::Node};

fn main() {
    let mut list = AtomicList::new();

    // First insertion initializes the ring with a single node.
    list.push_before("root", |_| true).expect("insert root");

    // Insert another value before the element that equals "root".
    list.push_before("worker-1", |cur| *cur == "root")
        .expect("insert worker");

    // Remove the first node equal to "root".
    if let Some(node) = list.pop_when(|cur| *cur == "root") {
        // Node<T> dereferences to &T for as long as the guard lives.
        assert_eq!(**node, "root");
        // Or reclaim ownership when we know we hold the last ref.
        assert_eq!(Node::into_inner(node), Some("root"));
    }
}
```

## Implementation notes

The list stores a single `Option<Node<T>>` called `root` (`src/list.rs`). Every node
starts as a singleton ring by pointing its `next` field back to itself. Insertion
allocates a fresh node, copies the successor pointer from the node it should precede,
and uses `compare_exchange` to swing the predecessor toward the new node. Removal
performs the inverse operation.

The custom atomic pointer types live in `src/atm_p.rs` and `src/sync.rs`. They wrap
raw pointers with manual reference counting tailored to this structure, which is why
the crate currently has no external dependencies.

## License

This project is distributed under the terms of the MIT license. See `LICENSE` for details.
