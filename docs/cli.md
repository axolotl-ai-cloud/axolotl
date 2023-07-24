# Axolotl CLI and Configuration Guide

This guide pertains to Axolotl configuration options when using the [click](https://click.palletsprojects.com/) CLI. This guide does not apply to standalone scripts in the ``scripts/`` directory.

## Introducton & Background

Today, Axolotl offers a wide range of functionalities that facilitate the finetuning of Large Language Models (LLMs). These include dataset preparation, creating and saving a merged (q)LoRA model, data sharding, interactive inferencing, and the finetuning process itself.

A typical, albeit not comprehensive, lifecycle sequence for LLM finetuning might be:

```text
Dataset Preparation -> Finetuning -> Quantization & Export -> Validation & Benchmarking -> Inferencing
```

This updated command-line interface (CLI) and configuration significantly simplify the realization of such lifecycle chains, for both end users and developers:

1. In accordance with the [12-factor app best practices](https://12factor.net/config), configuration values can now be overridden by CLI parameters and environment variables. This enables users to orchestrate multiple steps by setting common defaults in a ``config.yaml`` file and adjusting settings in the environment as necessary.
2. Considering that not all configuration options are pertinent to all features—for instance, "evaluation" only necessitates a subset of "finetune" settings—this update establishes a framework that makes it easier for users to identify the settings used by each feature. Moreover, it provides online assistance for each setting (accessible via CLI --help). Additionally, the update ensures full backward compatibility with ``finetune.py``, and capitalizes on existing ``config.yaml`` settings are already integrated into Axolotl.
3. Lastly, this update lays the foundation for a more effortless execution of Axolotl pipeline steps via external orchestration tools, such as [Apache Airflow](https://airflow.apache.org/).

## For Users

This section documents concepts relevant to all users.

### Running

Run the CLI as a Python module, note that Axolotl must be installed in your environment:

```shell
python -m axolotl --help
```

You can also run via the launcher shell script, be sure the path to the ``axolotl`` script is in your ``PATH``:

```shell
axolotl --help
```

You can launch via ``accelerate`` as well:

```shell
accelerate launch --config_file=/path/to/accelerate.yaml -m axolotl inference batch
```

### Enable CLI tab completion

The Axolotl CLI has standard click-based tab completion and can be enabled via:

For bash:

```shell
echo 'eval "$(_AXOLOTL_COMPLETE=bash_source axolotl)"' >> ~/.bashrc
```

For zsh:

```shell
echo 'eval "$(_AXOLOTL_COMPLETE=zsh_source axolotl)"' >> ~/.zshrc
```

Please see the [click shell completion](https://click.palletsprojects.com/en/8.1.x/shell-completion/) documentation for additional details.

### Passing multiple variables via the environment

Sometime it is necessary to pass a list of options, for example, the following configurations are all equivalent:

```shell
DATASETS="/data/GPTeacher/Instruct,gpteacher /data/GPTeacher/Roleplay,gpteacher"
```

and

```shell
--dataset=/data/GPTeacher/Instruct,gpteacher --dataset=/data/GPTeacher/Roleplay,gpteacher
```

For completeness, this is what the ``config.yaml`` setting would look like this:

```yaml
---
datasets:
  - path: /data/GPTeacher/Instruct
    type: gpteacher
  - path: /data/GPTeacher/Roleplay
    type: gpteacher
```

### Override hierarchy

Axolotl configuration implements the following hierarchy of overrides:

1. Configuration yaml
2. Environment variables
3. Explicit CLI options

You can test the override hierarchy and print out an effective configuration via the following command:

```shell
axolotl system config
```

## For Developers

This is a guide for developers who need to make CLI (command line interface) or configuration modifications to Axolotl.

### Does Axolotl need a new option?

Broadly, a new option may be needed when the *meaning* of an existing option doesn't match your use case. To illustrate, you would first need to understand the meaning of existing configuration options. This can be done by understanding the current options in ``tests/fixtures/default_config.`yaml`  and possibly looking at the online help for existing CLI options:

```shell
axolotl system config --help
```

For example, let's say you are adding a new command that calculates the perplexity score on one or more datasets. We need to specify both the model and datasets and you decide that the *meaning* of the existing options covers your use case. Now what about a configurable RNG seed?

* GOOD: Re-use the existing ``seed`` configuration option since the meaning is identical to your use case
* BAD: Create a new option called ``generate_seed``; the meaning of the new option overlaps with the meaning of ``seed``

### Adding a new option

All Axolotl configuration options must be placed in ``axolotl/cli/options.py``. Implementing new configuration options will generally follow this process:

1. Determine if a new option is needed
2. Update ``axolotl/tests/fixtures/default_config.yaml`` with the new default and documentation
3. Write the option decorator in ``axolotl/cli/options.py``
4. Use the option decorator in your command

Consider this example option decorator for ``seed``:

```python
def seed_option(**kwargs: Any) -> Callable:
    """
    Seed is used to control the determinism of operations in the axolotl.
    A value of -1 will randomly select a seed.
    """
    return option_factory(
        "--seed",
        envvar="AXOLOTL_SEED",
        type=click.types.INT,
        help=seed_option.__doc__,
        override_kwargs=kwargs,
    )
```

To use the new option just add the ``seed_option`` decorator:

```python
from axolotl import cfg

@system_group.command(name="seed_test")
@seed_option()
def seed_test(**kwargs: Dict[str, Any]):
    """The docstring will automatically appear in the CLI --help"""

    # Use this helper method to apply the configuration hierarchy logic
    update_config(overrides=kwargs)

    # Once update_config is run, it will be accessable via the cfg singleton
    click.echo(f"Seed = {cfg.seed}")
    ...
```

Run 1 outputs the default seed value from the ``config.yaml``:

```shell
axolotl system seed_test
Seed = 42
```

Run 2 overrides the default with the ``AXOLOTL_SEED`` environment variable:

```shell
AXOLOTL_SEED=43 axolotl system seed_test
Seed = 43
```

Run 3 overrides the default with a CLI parameter:

```shell
axolotl system seed_test --seed=44
Seed = 44
```

Run 4 overrides the default in ``config.yaml`` and ``AXOLOTL_SEED`` via the command line option:

```shell
AXOLOTL_SEED=43 axolotl system seed_test --seed=44
Seed = 44
```

### Conventions

The following conventions are enforced by the CLI unit tests in ``axolotl/tests/test_cli.py``:

* All options must have help documentation with a length of at least 10, the best practice for this is to just use the decorator docstring so you don't end up writing it twice
* The option must be present in ``axolotl/tests/fixtures/default_config.yaml``, this helps keep a "gold copy" of Axolotl configurations up to date as well as catch any copy/paste errors
* Option names must be unique
* Environment variables must be unique
* Environment variables must have a ``AXOLOTL_`` prefix

### Option Groups

Often groups of CLI options will always be provided together. An example of this could be ``base_model``, ``base_model_config``, ``model_type``, and ``tokenizer_type`` for operations that need to load a model. To help keep the code tidy, all four options can be grouped in a single decorator:

```python
def model_option_group(**kwargs) -> Callable:
    """
    Group of options for model configuration
    """

    return option_group_factory(
        options=[
            options.base_model_option,
            options.base_model_config_option,
            options.model_type_option,
            options.tokenizer_type_option,
        ],
        **kwargs,
    )
```

Now the single ``model_option_group`` decorator can add the entire group of options to your command:

```python
@system_group.command(name="example")
@model_option_group()
def example(**kwargs: Dict[str, Any]):
    """Just an example"""

    # Override default configuration
    update_config(overrides=kwargs)
```

The corresponding CLI help would look like this:

```shell
axolotl system example --help
Usage: axolotl system example [OPTIONS]

  Just an example

Options:
  --base_model TEXT         The huggingface model that contains *.pt,
                            *.safetensors, or *.bin files or a path to the
                            model on the disk  [env var: AXOLOTL_BASE_MODEL]
  --base_model_config TEXT  Useful when the base_model repo on HuggingFace hub
                            doesn't include configuration .json files. When
                            empty, defaults to config in base_model  [env var:
                            AXOLOTL_BASE_MODEL_CONFIG]
  --model_type TEXT         Specify the model type to load, ex:
                            AutoModelForCausalLM  [env var:
                            AXOLOTL_MODEL_TYPE]
  --tokenizer_type TEXT     Specify the tokenizer type to load, ex:
                            AutoTokenizer  [env var: AXOLOTL_TOKENIZER_TYPE]
  --help                    Show this message and exit.
```

### CLI Performance

Always avoid time-consuming imports at the top of command groups in the ``axolotl.cli`` module:

Bad:

```python
import torch
...
```

Good:

```python
@click.group(name="eval")
def eval_group():
    """Axolotl evaluation tools"""


@eval_group.command(name="batch_eval")
def batch_eval():
    """Executes a batch evaluation operation"""

    import torch

    ...
```

Click dynamically loads these every time the CLI is invoked and adding too many imports like `torch` will unnecessarily slow down the CLI responsiveness. Moreover, this situation is complicated when CLI tab completion is enabled since this will often invoke the CLI several times.
