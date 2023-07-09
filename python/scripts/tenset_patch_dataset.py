import json
import pickle as pkl
from pathlib import Path

from tqdm import tqdm
from tvm import auto_scheduler as ansor
from tvm.auto_scheduler import SearchTask

PREFIX = Path("lightning_logs/tenset")


def patched_setstate(self, state):
    from tvm.auto_scheduler import _ffi_api
    from tvm.auto_scheduler.workload_registry import (
        WORKLOAD_FUNC_REGISTRY,
        register_workload_tensors,
    )
    from tvm.target import Target

    # Register the workload if needed
    try:
        workload = json.loads(state["workload_key"])
    except Exception:  # pylint: disable=broad-except
        raise RuntimeError("Invalid workload key %s" % state["workload_key"])
    # workload[0] is either the compute function name or the ComputeDAG hash.
    # The compute functions are already registered when importing TVM, so here
    # we only register the ComputeDAG workloads. If the same workload has
    # already been registered, the later registration overrides the prvious one.
    if workload[0] not in WORKLOAD_FUNC_REGISTRY:
        register_workload_tensors(state["workload_key"], state["compute_dag"].tensors)
    state["target"], state["target_host"] = Target.check_and_update_host_consist(
        state["target"], state["target_host"]
    )
    self.__init_handle_by_constructor__(
        _ffi_api.SearchTask,  # type: ignore # pylint: disable=no-member
        state["compute_dag"],
        state["workload_key"],
        state["target"],
        state["target"].host,
        state["hardware_params"],
        state["layout_rewrite_option"],
        state["task_input_names"],
        "",
    )


SearchTask.__setstate__ = patched_setstate


def clean_name(x):
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace('"', "")
    x = x.replace("'", "")
    return x


def make_structured_key(old_wlk_str: str, tensors):
    old_wlk = json.loads(old_wlk_str)
    wlk_hash, wlk_shapes = old_wlk[0], old_wlk[1:]
    target_shapes = [[int(x) for x in t.shape] for t in tensors]
    assert [x for shape in target_shapes for x in shape] == wlk_shapes
    new_wlk = [wlk_hash, *target_shapes]
    return json.dumps(new_wlk)


def process_single_task_file(pkl_file):
    """Use this to process these single-task files in the dataset."""
    with open(pkl_file, "rb") as f:
        tasks, weights = pkl.load(f)
    for task in tqdm(tasks):
        # Create structured key from flattened key
        task.workload_key = make_structured_key(task.workload_key, task.compute_dag.tensors)
    task_infos = list(zip(tasks, weights))
    with open(pkl_file, "wb") as f:
        pkl.dump(task_infos, f)


def main():
    # process_single_task_file(
    #     "lightning_logs/tenset/network_info/((resnet_50,[(4,3,256,256)]),cuda).task.pkl"
    # )
    # process_single_task_file(
    #     "lightning_logs/tenset/network_info/((wide_resnet_50,[(4,3,256,256)]),cuda).task.pkl"
    # )

    all_tasks_pkl = PREFIX / "network_info/all_tasks.pkl"
    with open(all_tasks_pkl, "rb") as f:
        tasks = pkl.load(f)
    for task in tqdm(tasks):
        # Create structured key from flattened key
        old_wlk_str = task.workload_key
        new_wlk_str = task.workload_key = make_structured_key(old_wlk_str, task.compute_dag.tensors)
        # Register the tasks with the old key so the old configs can be loaded
        ansor.workload_registry.register_workload_tensors(old_wlk_str, task.compute_dag.tensors)
        # Load old config file, replace the workload keys, and save to new file
        old_task_key = clean_name((old_wlk_str, str(task.target.kind)))
        from_conf_file = PREFIX / f"to_measure_programs/{old_task_key}.json"
        new_task_key = clean_name((new_wlk_str, str(task.target.kind)))
        to_conf_file = PREFIX / f"to_measure_programs/{new_task_key}.json"
        tqdm.write(f"Moving {from_conf_file} to {to_conf_file}")
        with open(from_conf_file, "r") as fr, open(to_conf_file, "w") as fw:
            lines = fr.readlines()
            replaced = []
            for line in lines:
                entry = json.loads(line.strip())
                entry["i"][0][0] = task.workload_key
                replaced.append(json.dumps(entry) + "\n")
            fw.writelines(replaced)
        from_conf_file.unlink()
    # task.workload_key has been modified in-place so we need to save the tasks again
    tasks = [(t, 1) for t in tasks]
    with open(all_tasks_pkl, "wb") as f:
        pkl.dump(tasks, f)


if __name__ == "__main__":
    main()
