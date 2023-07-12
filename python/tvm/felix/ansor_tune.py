import logging
from typing import List, Optional

from tvm import auto_scheduler as ansor
from tvm.auto_scheduler.task_scheduler import TaskScheduler, TaskSchedulerCallback

from . import utils

logger = logging.getLogger(__file__)
__all__ = ["ansor_tune_full", "make_ansor_tuner"]


def ansor_tune_full(
    tasks,
    cost_model_path: str,
    json_log: Optional[utils.PathLike],
    n_total_measure: int,
    pop_size: int = 2048,
    round_n_steps: int = 4,
    round_n_measure: int = 64,
):
    tuner = make_ansor_tuner(tasks, cost_model_path, json_log)
    tuning_options = ansor.TuningOptions(
        num_measure_trials=n_total_measure,
        num_measures_per_round=round_n_measure,
        measure_callbacks=[ansor.RecordToFile(tuner.load_log_file)],
    )
    tuner.tune(
        tuning_options,
        search_policy_params={
            "evolutionary_search_population": pop_size,
            "evolutionary_search_num_iters": round_n_steps,
        },
    )
    return tuner


def make_ansor_tuner(
    tasks: List[utils.AnsorTaskWeight],
    model_file: Optional[str] = None,
    config_file: Optional[utils.PathLike] = None,
):
    tasks_info = [(task, weight, *utils.parse_task(task)) for task, weight in tasks]
    tasks_info = sorted(tasks_info, key=lambda x: x[2][0])
    task_desc_strs = []
    for _, weight, task_desc, shapes in tasks_info:
        task_name = utils.short_print_names(task_desc)
        first_line = f"{task_name}: \t\tshape = {shapes}, weight = {weight}"
        task_desc_strs.append(first_line)
    logger.info("Listing all tasks: \n%s", "\n".join(task_desc_strs))
    config_file_ = None if config_file is None else str(config_file)
    tasks_, weights, _, _ = zip(*tasks_info)
    tuner = ansor.TaskScheduler(
        tasks_,
        weights,
        load_model_file=model_file,  # type: ignore
        load_log_file=config_file_,  # type: ignore
        callbacks=[PrintTableInfo()],
    )
    return tuner


class PrintTableInfo(TaskSchedulerCallback):
    """The callback that prints a table of current progress."""

    def _print_line(self, task_scheduler: TaskScheduler, i: int):
        id_str = str(i)
        name = utils.parse_task_name(task_scheduler.tasks[i].desc)
        if task_scheduler.best_costs[i] < 1e9:
            latency_str = f"{1e6 * task_scheduler.best_costs[i]:.1f}"
            flops = task_scheduler.tasks[i].compute_dag.flop_ct
            speed = flops / task_scheduler.best_costs[i] / 1e9
            speed_str = f"{speed:.2f}"
        else:
            latency_str = speed_str = "-"
        n_measures: int = task_scheduler.num_measures_per_round  # type: ignore
        trials_str = str(task_scheduler.task_cts[i] * n_measures)
        return (
            f"| {id_str:4s} | {name:25s} | {latency_str:12s} | {speed_str:14s} | {trials_str:6s} |"
        )

    def post_tune(self, task_scheduler: TaskScheduler, task_id: int):
        import time

        # content
        lines = [self._print_line(task_scheduler, i) for i in range(len(task_scheduler.tasks))]
        text = "\n".join(lines)
        log_table = f"""
-----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
-----------------------------------------------------------------------
|  ID  |         Task Name         | Latency (us) | Speed (GFLOPS) | Trials |
-----------------------------------------------------------------------
{text}
-----------------------------------------------------------------------
"""
        logger.info(log_table)
        # overall info
        if all(cost < 1e9 for cost in task_scheduler.best_costs):
            total_latency_str = "%.3f" % (task_scheduler.cur_score * 1e3)
        else:
            total_latency_str = "-"
        tic: float = task_scheduler.tic  # type: ignore
        logger.info(
            "Estimated total latency: %s ms \tTrials: %d \tUsed time: %.0f s \tNext ID: %d",
            total_latency_str,
            task_scheduler.ct,
            time.time() - tic,
            task_id,
        )
