import clearml
import datetime
import utils


class MLLogger:

    def __init__(self, task_name: str) -> None:
        
        self.cmltask = clearml.Task.init(
            task_name=f"{task_name}_{utils.get_current_iso_datetime()}",
            project_name="hackathon")

        # get logger object for current task
        self.cmllogger: clearml.Logger = self.cmltask.get_logger()
        # Increase logger limit
        self.cmllogger.set_default_debug_sample_history(2000);


    def report_train_loss(self, current: float, mean: float, iteration: int) -> None:
        self.cmllogger.report_scalar(title="Train Loss", series="current", iteration=iteration, value=current)
        self.cmllogger.report_scalar(title="Train Loss", series="mean",    iteration=iteration, value=mean)
        # print(f"MSE: {mse:0.3f}")

    def report_val_metric(self, acc: float, iteration: int) -> None:
        self.cmllogger.report_scalar(title="Validation Metric", series="acc", iteration=iteration, value=acc)
    

    def save_weights(self) -> None:
        pass