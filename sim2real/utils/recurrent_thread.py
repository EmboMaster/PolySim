import threading
import time

class RecurrentThread(threading.Thread):
    def __init__(self, interval, task, *args, **kwargs):
        super().__init__()
        self.interval = interval
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.task(*self.args, **self.kwargs)
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()