from abc import ABC
from typing_extensions import Literal


class TrainerStateManagerMixin(ABC):
    """
    Handles the state (e.g., timeline, intervals) for the trainer.
    Uses book-keeping variables such as `current_step`, `current_epoch` and `current_task`.
    """
    current_step: int = 0
    current_epoch: int = 0
    current_task: int = 0
    
    def _tick_step(self):
        self.current_step += 1
    
    def _tick_epoch(self):
        self.current_epoch += 1
    
    def _tick_task(self):
        self.current_task += 1
    
    def tick(self, interval: Literal['step', 'epoch', 'task']):
        if interval == 'step':
            self._tick_step()
        elif interval == 'epoch':
            self._tick_epoch()
        elif interval == 'task':
            self._tick_task()
        else:
            raise ValueError("Supported intervals are 'step', 'epoch', and 'task")
