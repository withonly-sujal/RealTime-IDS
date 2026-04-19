import asyncio
from threading import Lock


class EventBus:
    def __init__(self):
        self._subscribers = []
        self._lock = Lock()
        self._loop = None
        self._event_history = []
        self._latest_stats = None
        self.MAX_HISTORY = 100

    def set_loop(self, loop):
        self._loop = loop

    def subscribe(self):
        queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue):
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def emit(self, event: dict):
        with self._lock:
            # Maintain history for AI context
            if event.get("type") == "prediction":
                self._event_history.append(event)
                if len(self._event_history) > self.MAX_HISTORY:
                    self._event_history.pop(0)
            elif event.get("type") == "stats":
                self._latest_stats = event.get("data")

            for queue in self._subscribers:
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(queue.put_nowait, event)

    def get_history(self):
        """Get a copy of the prediction event history."""
        with self._lock:
            return list(self._event_history)

    def get_stats_summary(self):
        with self._lock:
            return dict(self._latest_stats) if self._latest_stats else {}
