from abc import ABC
from cl_gym.utils.callbacks import ContinualCallback
from typing import List


class TrainerCallbackHookMixin(ABC):

    callbacks: List[ContinualCallback] = []

    def on_before_setup(self):
        for cb in self.callbacks:
            cb.on_before_setup(self)

    def on_after_setup(self):
        for cb in self.callbacks:
            cb.on_after_setup(self)

    def on_before_teardown(self):
        for cb in self.callbacks:
            cb.on_before_teardown(self)

    def on_after_teardown(self):
        for cb in self.callbacks:
            cb.on_after_teardown(self)

    def on_before_fit(self):
        for cb in self.callbacks:
            cb.on_before_fit(self)

    def on_after_fit(self):
        for cb in self.callbacks:
            cb.on_after_fit(self)

    def on_before_training_task(self):
        for cb in self.callbacks:
            cb.on_before_training_task(self)

    def on_after_training_task(self):
        for cb in self.callbacks:
            cb.on_after_training_task(self)

    def on_before_training_epoch(self):
        for cb in self.callbacks:
            cb.on_before_training_epoch(self)

    def on_after_training_epoch(self):
        for cb in self.callbacks:
            cb.on_after_training_epoch(self)

    def on_before_training_step(self):
        for cb in self.callbacks:
            cb.on_before_training_step(self)

    def on_after_training_step(self):
        for cb in self.callbacks:
            cb.on_after_training_step(self)
