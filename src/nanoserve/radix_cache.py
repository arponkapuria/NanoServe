import time

class RadixNode:
    """One block-aligned edge: exactly `block_size` tokens -> one physical block.
    Block-alignment removes the need for edge-splitting a true (token-granular)
radix tree needs — a block either matches an existing child whole, or it doesn't
    match at all, so there's never a partial-edge divergence to split mid-chunk."""

    __slots__ = ("tokens", "block_id", "children", "ref_count", "last_access")

    def __init__(self, tokens: tuple = (), block_id: int | None = None):
        self.tokens = tokens
        self.block_id = block_id
        self.children: dict[tuple, "RadixNode"] = {}
        self.ref_count = 0
        self.last_access = 0.0


class RadixCache:
    """Trie over fixed-size token chunks (block_size each), refcounted per node,
    shared across requests via NanoServeEngine.pool. Caches PROMPT tokens only —
    generated tokens are never inserted.

    Eviction is NOT implemented: nodes persist for the engine's process lifetime once
    inserted, so the block pool only ever shrinks. Deliberate scope cut at this
    project's scale (a handful of distinct prefixes in testing) — a real deployment
    would add LRU eviction over ref_count==0 leaves, keyed on `last_access`."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = RadixNode()

    def match(self, tokens: list[int]) -> tuple[list[int], list[RadixNode]]:
        """Walk complete block-chunks from the root. Returns (block_ids, nodes) for
        the longest matching prefix — nodes[i] is the node owning block_ids[i]."""

        node = self.root
        block_ids, nodes = [], []
        B = self.block_size

        for i in range(len(tokens) // B):
            chunk = tuple(tokens[i * B:(i + 1) * B])
            child = node.children.get(chunk)
            if child is None:
                break
            block_ids.append(child.block_id)
            nodes.append(child)
            node = child

        return block_ids, nodes

    def touch(self, nodes: list[RadixNode]) -> None:
        """Marks cached nodes as being used."""

        now = time.time()
        for n in nodes:
            n.ref_count += 1
            n.last_access = now

    def insert(self, tokens: list[int], parent: RadixNode, start_block: int, new_block_ids: list[int]) -> list[RadixNode]:
        """Insert freshly-computed complete blocks (new_block_ids, starting at block
        index start_block in `tokens`) below `parent`. Returns the nodes touched, in
        order, for the caller's ref-count bookkeeping."""

        node = parent
        inserted = []
        now = time.time()
        B = self.block_size

        for j, block_id in enumerate(new_block_ids):
            i = start_block + j
            chunk = tuple(tokens[i * B:(i+1) * B])
            child = node.children.get(chunk)
            if child is None:
                child = RadixNode(tokens=chunk, block_id=block_id)
                node.children[chunk] = child
            child.ref_count += 1
            child.last_access = now
            inserted.append(child)
            node = child

        return inserted

    def release(self, nodes: list[RadixNode]) -> None:
        """Releases when a request is finished with cached blocks. 
        Not deleted since the implementation intentionally has no eviction"""

        for n in nodes:
            n.ref_count = max(0, n.ref_count - 1)

    def stats(self) -> dict:
        """Reports how much is cached."""
        def walk(node):
            count = 1
            blocks = 1 if node.block_id is not None else 0
            for c in node.children.values():
                c_count, c_blocks = walk(c)
                count += c_count
                blocks += c_blocks
            return count, blocks

        node_count, block_count = walk(self.root)
        return {"node_count": node_count - 1, "cached_blocks": block_count}

