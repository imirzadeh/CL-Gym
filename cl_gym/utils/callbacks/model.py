import os
import torch
from pathlib import Path
from cl_gym.utils.callbacks import ContinualCallback


class ModelCheckpoint(ContinualCallback):
    """
    Model Checkpoint callback.
    Saves the backbone (model) to disk at the end of every epoch or task (depending on the input settings).
    """
    def __init__(self, interval='task', name_prefix=None):
        """
        
        Args:
            interval: The intervals at which the checkpointing will be done. can be either 'task' or 'epoch'
            name_prefix: Optional prefix for model names.
        """
        if interval.lower() not in ['task', 'epoch']:
            raise ValueError("Checkpoint callback supports can only save after each 'task' or each 'epoch'")
        self.interval = interval
        self.name_prefix = name_prefix
        super(ModelCheckpoint, self).__init__()
    
    def __get_save_path(self, trainer):
        checkpoint_folder = os.path.join(trainer.params['output_dir'], 'checkpoints')
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        model_name = self.name_prefix if self.name_prefix else ''
        if trainer.current_step <= 1:
            model_name += 'init.pt'
        else:
            model_name += f"{self.interval}_"
            if self.interval == 'task':
                model_name += f"{trainer.current_task}.pt"
            else:
                model_name += f"{trainer.current_epoch}.pt"
        
        filepath = os.path.join(checkpoint_folder, model_name)
        return filepath
    
    def __save_model(self, trainer):
        # get the model (backbone)
        model = trainer.algorithm.backbone
        model.eval()
        file_path = self.__get_save_path(trainer)
        torch.save(model.to('cpu').state_dict(), file_path)
    
    def on_before_fit(self, trainer):
        # save init params
        self.__save_model(trainer)
    
    def on_after_training_task(self, trainer):
        if self.interval == 'task':
            self.__save_model(trainer)
    
    def on_after_training_epoch(self, trainer):
        if self.interval == 'epoch':
            self.__save_model(trainer)


class BackboneFreeze(ContinualCallback):
    def __init__(self):
        super(BackboneFreeze, self).__init__()
    
    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def on_after_training_task(self, trainer):
        if trainer.current_task >= 1:
            backbone = trainer.algorithm.backbone
            self._freeze_module(backbone.block_1)
            # self._freeze_module(backbone.block_2)
