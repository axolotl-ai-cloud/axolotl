"""
axolotl CLI for running lm_eval tasks
"""

import subprocess  # nosec
from collections import defaultdict
from datetime import datetime
from typing import Optional

import click
import yaml

from axolotl.utils.dict import DictDefault


def build_lm_eval_command(
    tasks: list[str],
    bfloat16=True,
    flash_attention=False,
    output_dir="./",
    batch_size=8,
    wandb_project=None,
    wandb_entity=None,
    wandb_name=None,
    model=None,
    revision=None,
    apply_chat_template=None,
    fewshot_as_multiturn=None,
):
    tasks_by_num_fewshot: dict[str, list] = defaultdict(list)
    if isinstance(tasks, str):
        tasks = [tasks]
    for task in tasks:
        num_fewshot = "-1"
        task_parts = task.split(":")
        task_name = task_parts[0]
        if len(task_parts) == 2:
            task_name, num_fewshot = task_parts
        tasks_by_num_fewshot[str(num_fewshot)].append(task_name)

    for num_fewshot, tasks_list in tasks_by_num_fewshot.items():
        tasks_str = ",".join(tasks_list)
        num_fewshot_val = num_fewshot if num_fewshot != "-1" else None
        pretrained = "pretrained="
        pretrained += model if model else output_dir
        fa2 = ",attn_implementation=flash_attention_2" if flash_attention else ""
        dtype = ",dtype=bfloat16" if bfloat16 else ",dtype=float16"
        revision = f",revision={revision}" if revision else ""
        output_path = output_dir
        output_path += "" if output_dir.endswith("/") else "/"
        output_path += "lm_eval_results/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        lm_eval_args = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"{pretrained}{fa2}{dtype}{revision}",
            "--tasks",
            tasks_str,
            "--batch_size",
            str(batch_size),
            "--output_path",
            output_path,
        ]
        wandb_args = []
        if wandb_project:
            wandb_args.append(f"project={wandb_project}")
        if wandb_entity:
            wandb_args.append(f"entity={wandb_entity}")
        if wandb_name:
            wandb_args.append(f"name={wandb_name}")
        if wandb_args:
            lm_eval_args.append("--wandb_args")
            lm_eval_args.append(",".join(wandb_args))
        if apply_chat_template:
            lm_eval_args.append("--apply_chat_template")
        if num_fewshot_val:
            lm_eval_args.append("--num_fewshot")
            lm_eval_args.append(str(num_fewshot_val))
            if apply_chat_template and fewshot_as_multiturn:
                lm_eval_args.append("--fewshot_as_multiturn")

        yield lm_eval_args


@click.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option("--cloud", default=None, type=click.Path(exists=True, path_type=str))
def lm_eval(config: str, cloud: Optional[str] = None):
    """
    use lm eval to evaluate a trained language model
    """

    if cloud:
        from axolotl.cli.cloud import do_cli_lm_eval

        do_cli_lm_eval(cloud_config=cloud, config=config)
    else:
        with open(config, encoding="utf-8") as file:
            cfg: DictDefault = DictDefault(yaml.safe_load(file))

        # pylint: disable=duplicate-code
        for lm_eval_args in build_lm_eval_command(
            cfg.lm_eval_tasks,
            bfloat16=cfg.bfloat16 or cfg.bf16,
            flash_attention=cfg.flash_attention,
            output_dir=cfg.output_dir,
            batch_size=cfg.lm_eval_batch_size,
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            wandb_name=cfg.wandb_name,
            model=cfg.lm_eval_model or cfg.hub_model_id,
            revision=cfg.revision,
            apply_chat_template=cfg.apply_chat_template,
            fewshot_as_multiturn=cfg.fewshot_as_multiturn,
        ):
            subprocess.run(  # nosec
                lm_eval_args,
                check=True,
            )
