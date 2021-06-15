import comet_ml
from comet_ml import Experiment
import os


class Logger:
    def __init__(self, api_key, project_name, trial_name):
        self.api_key = api_key
        self.trial_id = trial_name
        self.project_name = project_name
        self.experiment = self._build_experiment()
    
    def _build_experiment(self):
        raise NotImplementedError
    
    def log_parameters(self, params: dict):
        raise NotImplementedError
    
    def log_metric(self, key: str, value, step=None):
        raise NotImplementedError
    
    def log_figure(self, figure, name, step=None):
        raise NotImplementedError
    
    def log_text(self, text):
        raise NotImplementedError
    
    def log_folder(self, folder_path):
        raise NotImplementedError
    
    def terminate(self):
        raise NotImplementedError


class CometLogger(Logger):
    def __init__(self, api_key=None, project_name=None, trial_name=None, workspace=None):
        if not api_key:
            api_key = os.environ.get("LOGGER_API_KEY")
            if not api_key:
                raise ValueError("Either set `LOGGER_API_KEY` environment variable or pass the API key to the class")
        if not project_name or not workspace:
            raise ValueError("Project & workspace name for the comet experiment manager is required.")
        self.workspace = workspace
        super(CometLogger, self).__init__(api_key, project_name, trial_name)
    
    def _build_experiment(self):
        exp = Experiment(self.api_key, self.project_name, self.workspace,
                          auto_metric_logging=False, auto_param_logging=False,
                          log_graph=False, disabled=False)
        exp.disable_mp()
        return exp
    
    def log_parameters(self, params: dict):
        self.experiment.log_parameters(params)
    
    def log_metric(self, key: str, value, step=None):
        self.experiment.log_metric(key, value, step)
    
    def log_text(self, text):
        self.experiment.log_text(text)
    
    def log_figure(self, figure, name, step=None):
        self.experiment.log_figure(name, figure, step)
    
    def log_folder(self, folder_path):
        self.experiment.log_asset_folder(folder_path, log_file_name=False, recursive=True)
    
    def terminate(self):
        self.experiment.end()
