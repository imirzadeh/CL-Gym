from cl_gym.utils.callbacks import ContinualCallback


class ExperimentManager(ContinualCallback):
    """
    Experiment manager callback: logs parameters before training starts, updates server when training ends.
    """
    def __init__(self):
        super(ExperimentManager, self).__init__('ExperimentManager')
    
    def on_before_fit(self, trainer):
        if trainer.logger:
            trainer.logger.log_parameters(trainer.params)
    
    def on_before_teardown(self, trainer):
        if trainer.logger:
            path = trainer.params['output_dir']
            trainer.logger.log_folder(folder_path=path)
            trainer.logger.terminate()
