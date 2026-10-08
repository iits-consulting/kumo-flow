# How structural signatures work

`_sigs` in `backend/graph.py` gives every node a **structural signature**: a
SHA1 hex string that fingerprints everything the node's result can depend on —
its kind, its config, its `cache_extra()`, and (transitively) the same for
every node wired into it. That string is the key into the cross-run result
cache (`CACHE`): if a node in a later run hashes to the same sig, its cached
`{port: value}` outputs are reused without running it *or anything upstream of
it*. There is no run-to-run diffing — the sig is computed from the graph JSON
alone, so "did anything relevant change?" collapses into a dict lookup.

"Merkle" means: a node's hash includes the finished hashes of its inputs,
which include the hashes of *their* inputs, and so on — one digest transitively
pins the entire upstream subgraph, the same way a git commit hash pins the
whole tree behind it.

## The shape of the function

```python
def _sigs(by_id, incoming):
    sigs = {}          # memo: node id -> hex sig | None
    visiting = set()   # nodes currently mid-computation -> cycle detection

    def sig(nid):      # memoized recursive wrapper
        ...
    def compute(nid):  # the actual hash for one node
        ...

    for nid in by_id:  # driver: force a sig for every node
        sig(nid)
    return sigs
```

- `sig(nid)` returns the memoized value if present; otherwise it marks the node
  as `visiting`, calls `compute`, unmarks, stores. Re-entering a node that is
  still `visiting` means the recursion looped back on itself →
  `ValueError("graph has a cycle")`. This is why `run_graph` calls `_sigs`
  before evaluating anything: it doubles as cycle rejection for hand-POSTed
  graphs (the frontend's `createsCycle` guard never creates them).
- `compute(nid)` does the work in two halves: seed the hash with the node's own
  identity, then fold in the sigs of its inputs.

### Half 1 — the node's own identity

```python
inst = cls(**node.config)
cfg = json.dumps(inst.model_dump(), sort_keys=True, default=str)
h = hashlib.sha1(f"{node.kind}\0{cfg}\0{inst.cache_extra()}".encode())
```

Three deliberate choices:

- **The validated dump is hashed, not the raw config.** A config of `{}` and
  one with the default spelled out (`{"threshold": 0.5}`) dump identically, so
  the same effective config always gets the same sig. `sort_keys` makes key
  order irrelevant too. If validation fails, the raw config is hashed as a
  fallback — the sig only needs to be deterministic; `run()` raises the real
  error later.
- **`cache_extra()` mixes in state the config can't see** — e.g. `LoadVideo`
  returns `(path, mtime_ns, size)` per file, so editing a video on disk changes
  the sig and invalidates the cache.
- **The node id is *not* hashed.** Two copy-pasted identical branches hash
  identically and share one cache entry.

Unknown kinds and nodes with `cacheable = False` return `None` instead of a
hash: uncacheable.

### Half 2 — the Merkle fold

```python
for (n, port), srcs in incoming.items():
    if n != nid:
        continue
    for src_id, src_port in srcs:
        s = sig(src_id)                 # recurse upstream
        if s is None:
            return None                 # uncacheable upstream poisons this node
        h.update(f"\0{port}\0{src_port}\0{s}".encode())
return h.hexdigest()
```

For every edge feeding this node, recursively obtain the source node's
*finished* sig and fold `(input_port, source_port, source_sig)` into the hash.
Two properties fall out:

- **Poisoning.** Any `None` upstream makes this node `None` too — a result
  whose inputs can't be fingerprinted can't be cached, and neither can anything
  built on it.
- **Edge order matters.** Edges fold in the order they arrived in the request,
  which is the order `evaluate()` consumes them (this matters for multi-edge
  ports like two Text Prompts merging into one `prompts` input). Reordering
  the edges changes the sig — a cache *miss*, never a wrong *hit*. A miss
  costs a re-run; a wrong hit would be a wrong result.

## The traversal is a post-order DFS

There is no "hash everything, then merge" phase — the fold *is* the descent.
`compute(A)` opens A's hash object, and the first `sig(source)` call recurses
all the way to the graph's roots before returning; only then is the source's
digest folded into A's still-open hash. A node's digest is finalized only
*after* all its ancestors' digests are finished, because those digests are
ingredients of its own. That ordering constraint is exactly what makes it
Merkle-style.

### Worked example

Four nodes, SAM3 with two inputs (fake 4-char digests instead of 40-char SHA1s):

```
L: load_image {path:"cat.jpg"} ──image──▶ R: resize {size:512} ──image──▶ S: sam3 {}
                                          T: text_prompt {text:"cat"} ──prompts──▶ S
```

Say `by_id` iterates `[S, L, R, T]`, so the driver starts at the most
downstream node and forces the deepest recursion immediately:

```
sig(S)                                      visiting={S}
│ compute(S)
│   h_S = sha1("sam3\0{}\0")                         ← seed: S's own identity
│   edge (S,"image") ← (R,"image") — need R's sig first:
│   │
│   ├─ sig(R)                               visiting={S,R}
│   │  │ compute(R)
│   │  │   h_R = sha1("resize\0{\"size\": 512}\0")   ← seed R
│   │  │   edge (R,"image") ← (L,"image"):
│   │  │   │
│   │  │   ├─ sig(L)                        visiting={S,R,L}
│   │  │   │  │ compute(L)
│   │  │   │  │   h_L = sha1("load_image\0{\"path\": \"cat.jpg\"}\0")
│   │  │   │  │   no incoming edges → finalize: "aaaa"   ★ first done: a LEAF
│   │  │   │  └ sigs[L]="aaaa"              visiting={S,R}
│   │  │   │
│   │  │   h_R.update("\0image\0image\0aaaa")        ← fold L's digest into R
│   │  │   → finalize: "bbbb"               ★ R done only AFTER L
│   │  └ sigs[R]="bbbb"                     visiting={S}
│   │
│   h_S.update("\0image\0image\0bbbb")               ← fold R's digest into S
│   edge (S,"prompts") ← (T,"prompts"):
│   │
│   ├─ sig(T)                               visiting={S,T}
│   │  │ compute(T)
│   │  │   h_T = sha1("text_prompt\0{\"text\": \"cat\"}\0")
│   │  │   no incoming edges → finalize: "cccc"
│   │  └ sigs[T]="cccc"                     visiting={S}
│   │
│   h_S.update("\0prompts\0prompts\0cccc")           ← fold T's digest into S
│   → finalize: "dddd"                      ★ S done LAST
└ sigs[S]="dddd"                            visiting={}

driver continues:  sig(L), sig(R), sig(T) → memo hits, no recursion
```

Finalize order **L, R, T, S** — leaves first, downstream last. Each digest is
effectively:

```
aaaa = H(load_image, {path: cat.jpg})
bbbb = H(resize,     {size: 512},   image←image: aaaa)
cccc = H(text_prompt, {text: cat})
dddd = H(sam3,       {},            image←image: bbbb,  prompts←prompts: cccc)
```

### Change one leaf and watch it ripple

Edit the load node: `cat.jpg → dog.jpg`.

```
L's seed changes         → aaaa becomes eeee
R hashes "…\0eeee"       → bbbb becomes ffff     (R's own config untouched)
S hashes "…\0ffff\0cccc" → dddd becomes gggg
T unchanged              → still cccc
```

The change ripples to every descendant without anyone comparing configs — it's
carried purely by the digests. On the next run S misses the cache (correct:
different image), while any graph still using the same text prompt subtree
hits `cccc`. And the flip side: copy-paste L→R as new nodes with identical
configs and they hash to `aaaa`/`bbbb` again — the copy hits the cache the
original filled.

## Where the sigs are consumed

`evaluate()`'s `node_outputs` checks `_cache_get(sig)` **before recursing into
the node's inputs** — a hit therefore skips the entire upstream subgraph, not
just one node; the sig already vouches for all of it. On success the result is
written back with `_cache_put`. Failures and cancellations never reach the
write. There is a second consumer: `/run` registers the run's whole sig set in
`RUNS` (`run["sigs"]`), and eviction skips those entries until the run ends —
a live run's not-yet-touched cache hits can't be evicted out from under it.
See [graph-evaluation.md](graph-evaluation.md) for the evaluation side and the
cache's LRU/eviction story.

## Limits

- The sig can't see state outside the graph that no `cache_extra()` reports —
  a Custom Code node reading files or the network will serve stale results
  while its code and inputs are unchanged. `POST /cache/clear` is the escape
  hatch.
- Cycle detection only covers cacheable subgraphs: a cycle routed through an
  uncacheable node returns `None` early without visiting its inputs, and dies
  later as that target's RecursionError.
- The edge scan (`if n != nid: continue`) is O(edges) per node — a linear
  filter instead of a pre-grouped index, fine at UI graph sizes.
