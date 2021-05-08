from cl_gym.utils.callbacks import ContinualCallback


class ExperimentManager(ContinualCallback):
    def __init__(self):
        super(ExperimentManager, self).__init__()
    
    def on_before_fit(self, trainer):
        if trainer.logger:
            trainer.logger.log_parameters(trainer.params)
    
    def on_after_fit(self, trainer):
        if trainer.logger:
            path = trainer.params['output_dir']
            trainer.logger.log_folder(folder_path=path)
            trainer.logger.terminate()
