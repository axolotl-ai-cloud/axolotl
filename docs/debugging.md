# Debugging Axolotl

This document provides some tips and tricks for debugging Axolotl.  It also provides an example configuration for debugging with VSCode.  A good debugging setup is essential to understanding how Axolotl code works behind the scenes.

## Table of Contents

- [General Tips](#general-tips)
- [Debugging with VSCode](#debugging-with-vscode)
    - [Background](#background)
    - [Configuration](#configuration)
    - [Customizing your debugger](#customizing-your-debugger)
    - [Video Tutorial](#video-tutorial)

## General Tips

While debugging it's helpful to simplify your test scenario as much as possible.  Here are some tips for doing so:

> [!Important]
> All of these tips are incorporated into the [example configuration](#configuration) for debugging with VSCode below.

1. **Eliminate Concurrency**: Restrict the number of processes to 1 for both training and data preprocessing:
    - Set `CUDA_VISIBLE_DEVICES` to a single GPU, ex: `export CUDA_VISIBLE_DEVICES=0`.
    - Set `dataset_processes: 1` in your axolotl config or run the training command with `--dataset_processes=1`.
2. **Use a small dataset**: Construct or use a small dataset from HF Hub. When using a small dataset, you will often have to make sure `sample_packing: False` and `eval_sample_packing: False` to avoid errors.  If you are in a pinch and don't have time to construct a small dataset but want to use from the HF Hub, you can shard the data (this will still tokenize the entire dataset, but will only use a fraction of the data for training.  For example, to shard the dataset into 20 pieces, add the following to your axolotl config):
    ```yaml
    dataset:
        ...
        shards: 20
    ```
3. **Use a small model**: A good example of a small model is [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
4. **Minimize iteration time**: Make sure the training loop finishes as fast as possible, with these settings.
    - `micro_batch_size: 1`
    - `max_steps: 1` 
    - `val_set_size: 0`
5. **Clear Caches:** Axolotl caches certain steps and so does the underlying HuggingFace trainer.  You may want to clear some of these caches when debugging.
    - Data preprocessing: When debugging data preprocessing, which includes prompt template formation, you may want to delete the directory set in `dataset_prepared_path:` in your axolotl config.  If you didn't set this value, the default is `last_run_prepared`.
    - HF Hub: If you are debugging data preprocessing, you should clear the relevant HF cache [HuggingFace cache](https://huggingface.co/docs/datasets/cache), by deleting the appropriate `~/.cache/huggingface/datasets/...` folder(s).
    - **The recommended approach is to redirect all outputs and caches to a temporary folder and delete selected subfolders before each run.  This is demonstrated in the example configuration below.**
        

## Debugging with VSCode

### Background

The below example shows how to configure VSCode to debug data preprocessing of the `sharegpt` format.  This is the format used when you have the following in your axolotl config:

```yaml
datasets:
  - path: <path to your sharegpt formatted dataset> # example on HF Hub: philschmid/guanaco-sharegpt-style
    type: sharegpt
```

>[!Important]
> If you are already familiar with advanced VSCode debugging, you can skip the below explanation and look at the files [.vscode/launch.json](../.vscode/launch.json) and [.vscode/tasks.json](../.vscode/tasks.json) for an example configuration.

>[!Tip]
> If you prefer to watch a video, rather than read, you can skip to the [video tutorial](#video-tutorial) below (but doing both is recommended).

### Configuration

The easiest way to get started is to modify the [.vscode/launch.json](../.vscode/launch.json) file in this project.  This is just an example configuration, so you may need to modify or copy it to suit your needs.

For example, to mimic the command `cd devtools && CUDA_VISIBLE_DEVICES=0 accelerate launch -m axolotl.cli.train dev_sharegpt.yml`, you would use the below configuration[^1].  Note that we add additional flags that override the axolotl config and incorporate the tips above (see the comments). We also set the working directory to `devtools` and set the `env` variable `HF_HOME` to a temporary folder that is later partially deleted.  This is because we want to delete the HF dataset cache before each run in this particular

```jsonc
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug axolotl prompt - sharegpt",
            "type": "python",
            "module": "accelerate.commands.launch",
            "request": "launch",
            "args": [
                "-m", "axolotl.cli.train", "dev_sharegpt.yml",
                // The flags below simplify debugging by overriding the axolotl config 
                // with the debugging tips above.  Modify as needed.
                "--dataset_processes=1",      // limits data preprocessing to one process
                "--max_steps=1",              // limits training to just one step
                "--batch_size=1",             // minimizes batch size
                "--micro_batch_size=1",       // minimizes batch size
                "--val_set_size=0",           // disables validation
                "--sample_packing=False",     // disables sample packing which is necessary for small datasets
                "--eval_sample_packing=False",// disables sample packing on eval set
                "--dataset_prepared_path=temp_debug/axolotl_outputs/data", // send data outputs to a temp folder
                "--output_dir=temp_debug/axolotl_outputs/model" // send model outputs to a temp folder
                ],
            "console": "integratedTerminal",      // show output in the integrated terminal
            "cwd": "${workspaceFolder}/devtools", // set working directory to devtools from the root of the project
            "justMyCode": true,                   // step through only axolotl code
            "env": {"CUDA_VISIBLE_DEVICES": "0",  // Since we aren't doing distributed training, we need to limit to one GPU
                    "HF_HOME": "${workspaceFolder}/devtools/temp_debug/.hf-cache"}, // send HF cache to a temp folder
            "preLaunchTask": "cleanup-for-dataprep", // delete temp folders (see below)
        }
    ]
}
```

**Additional notes about this configuration:**

- The argument `justMyCode` is set to `true` such that you step through only the axolotl code.  If you want to step into dependencies, set this to `false`.
- The `preLaunchTask`: `cleanup-for-dataprep` is defined in [.vscode/tasks.json](../.vscode/tasks.json) and is used to delete the following folders before debugging, which is essential to ensure that the data pre-processing code is run from scratch:
    -  `./devtools/temp_debug/axolotl_outputs` 
    - `./devtools/temp_debug/.hf-cache/datasets`

>[!Tip]
> You may not want to delete these folders. For example, if you are debugging model training instead of data pre-processing, you may NOT want to delete the cache or output folders. You may also need to add additional tasks to the `tasks.json` file depending on your use case.

Below is the [./vscode/tasks.json](../.vscode/tasks.json) file that defines the `cleanup-for-dataprep` task.  This task is run before each debugging session when you use the above configuration.  Note how there are two tasks that delete the two folders mentioned above.  The third task `cleanup-for-dataprep` is a composite task that combines the two tasks.  A composite task is necessary because VSCode does not allow you to specify multiple tasks in the `preLaunchTask` argument of the `launch.json` file.

```jsonc
// .vscode/tasks.json
// this file is used by launch.json
{
    "version": "2.0.0",
    "tasks": [
      // this task changes into the devtools directory and deletes the temp_debug/axolotl_outputs folder
      {
        "label": "delete-outputs",
        "type": "shell",
        "command": "rm -rf temp_debug/axolotl_outputs",
        "options":{ "cwd": "${workspaceFolder}/devtools"},
        "problemMatcher": []
      },
      // this task changes into the devtools directory and deletes the `temp_debug/.hf-cache/datasets` folder
      {
        "label": "delete-temp-hf-dataset-cache",
        "type": "shell",
        "command": "rm -rf temp_debug/.hf-cache/datasets",
        "options":{ "cwd": "${workspaceFolder}/devtools"},
        "problemMatcher": []
      },
        // this task combines the two tasks above
      {
       "label": "cleanup-for-dataprep",
       "dependsOn": ["delete-outputs", "delete-temp-hf-dataset-cache"],
      }
    ]
}
```

### Customizing your debugger

Your debugging use case may differ from the example above.  The easiest thing to do is to put your own axolotl config in the `devtools` folder and modify the `launch.json` file to use your config.  You may also want to modify the `preLaunchTask` to delete different folders or not delete anything at all.

### Video Tutorial

The following video tutorial walks through the above configuration and demonstrates how to debug with VSCode, (click the image below to watch):

<div style="text-align: center; line-height: 0;">

<a href="https://youtu.be/xUUB11yeMmc?si=z6Ea1BrRYkq6wsMx" target="_blank"
title="How to debug Axolotl (for fine tuning LLMs)"><img
src="https://i.ytimg.com/vi/xUUB11yeMmc/maxresdefault.jpg"
style="border-radius: 10px; display: block; margin: auto;" width="560" height="315" /></a>

<figcaption style="font-size: smaller;"><a href="https://hamel.dev">Hamel Husain's</a> tutorial: <a href="https://www.youtube.com/watch?v=xUUB11yeMmc">Debugging Axolotl w/VSCode</a></figcaption>

</div>
<br>



[^1]: The config actually mimics the command `CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch -m axolotl.cli.train devtools/sharegpt.yml`, but this is the same thing.