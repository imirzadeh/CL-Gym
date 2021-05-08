class ContinualCallback:
    def __init__(self, name=''):
        self.name = name
    
    def log_text(self, trainer, text):
        if trainer.logger:
            trainer.logger.log_text(text)
        print(text)
    
    def log_metric(self, trainer, metric_name, metric_value, metric_step=None):
        if trainer.logger:
            trainer.logger.log_metric(metric_name, metric_value, metric_step)
    
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

