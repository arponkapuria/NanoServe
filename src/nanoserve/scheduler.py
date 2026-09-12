class Scheduler:
    """FCFS admission queue with fixed-depth backpressure and a fixed per-request
    wait timeout. Pure state-transition logic, no model calls and no real sleeping —
    it takes explicit `now` timestamps so admission/rejection decisions are testable
    independent of MPS step-time jitter, and separate from run_continuous_batch's
    execution loop (see progress.md log for why: keeps a reproducible decision layer
    separate from real wall-clock timing noise)."""

    def __init__(self, max_queue_depth: int, max_wait_s: float):
        self.max_queue_depth = max_queue_depth
        self.max_wait_s = max_wait_s
        self.queue: list[dict] = []  # FCFS: index 0 = next in line to be admitted
        self.rejections = {"queue_full": 0, "timeout": 0}

    def submit(self, request: dict, now: float) -> str:
        """Called once per request, the instant its arrival_delay elapses. This is
        where backpressure is checked. Returns 'queued' or 'rejected_queue_full'."""
        if len(self.queue) >= self.max_queue_depth:
            self.rejections["queue_full"] += 1
            return "rejected_queue_full"
        self.queue.append({"request": request, "arrival_time": now})
        return "queued"

    def expire(self, now: float) -> list[dict]:
        """Drop anything that's waited past max_wait_s. Call before pull() every
        scheduling pass. Returns the dropped entries."""
        keep, expired = [], []
        for entry in self.queue:
            if now - entry["arrival_time"] > self.max_wait_s:
                expired.append(entry)
                self.rejections["timeout"] += 1
            else:
                keep.append(entry)
        self.queue = keep
        return expired

    def pull(self, now: float, num_slots: int) -> list[dict]:
        """Admit up to num_slots requests, FCFS. Attaches queue_time_s to each."""
        if num_slots <= 0:
            return []
        admitted, self.queue = self.queue[:num_slots], self.queue[num_slots:]
        for entry in admitted:
            entry["queue_time_s"] = now - entry["arrival_time"]
        return admitted

    @property
    def depth(self) -> int:
        return len(self.queue)