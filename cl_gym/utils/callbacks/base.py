import os
from pathlib import Path
from typing import List, Optional


class ContinualCallback:
    """
    Base class for continual callback.
    Simply, a callback can listen to one or several events that the Trainer emits during the learning experience:
    
    .. image:: https://user-images.githubusercontent.com/8312051/122718892-4f888480-d222-11eb-87be-301e462de551.png
        :width: 400
    Please see the introduction video on what the Trainer's timeline looks like.
    
    """
    def __init__(self, name: str = '', save_dirs: Optional[List[str]] = None):
        self.name = name
        self.params = {}
        self.save_dirs = save_dirs
        self.save_paths = {}
    
    def _mkdir_save_path(self, directory_name: str):
        path = os.path.join(self.params['output_dir'], directory_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
    
    def connect(self, trainer):
        self.params = trainer.params
        self._prepare()
    
    def _prepare(self):
        if self.save_dirs:
            self._prepare_save_paths()
    
    def _prepare_save_paths(self):
        for directory in self.save_dirs:
            self.save_paths[directory] = self._mkdir_save_path(directory)
    
    def on_before_setup(self, trainer):
        pass

    def on_after_setup(self, trainer):
        pass

    def on_before_teardown(self, trainer):
        pass

    def on_after_teardown(self, trainer):
        pass

    def on_before_fit(self, trainer):
        pass
    
    def on_after_fit(self, trainer):
        pass
    
    def on_before_training_task(self, trainer):
        pass
    
    def on_after_training_task(self, trainer):
        pass
    
    def on_before_training_epoch(self, trainer):
        pass
    
    def on_after_training_epoch(self, trainer):
        pass
    
    def on_before_training_step(self, trainer):
        pass
    
    def on_after_training_step(self, trainer):
        pass
    
    def __repr__(self):
        return f"Callback[{self.name}]"

