"""
This module is used to launch Axolotl with user defined configurations.
"""
import subprocess

import gradio as gr
import yaml


def config(
    base_model,
    dataset,
    dataset_type,
    learn_rate,
    gradient_accumulation_steps,
    micro_batch_size,
    seq_length,
    num_epochs,
    output_dir,
    val_size,
):
    """
    This function generates a configuration dictionary and saves it as a yaml file.
    """
    config_dict = {
        "base_model": base_model,
        "datasets": [{"path": dataset, "type": dataset_type}],
        "learning_rate": learn_rate,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "micro_batch_size": micro_batch_size,
        "sequence_len": seq_length,
        "num_epochs": num_epochs,
        "output_dir": output_dir,
        "val_set_size": val_size,
    }
    with open("config.yml", "w", encoding="utf-8") as file:
        yaml.dump(config_dict, file)
    print(config_dict)
    return yaml.dump(config_dict)


def create_training_job():
    # Start a long-running process
    process = subprocess.Popen(
        ["accelerate launch -m axolotl.cli.train config.yml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Read the output line by line as it becomes available
    while True:
        line = process.stdout.readline()
        if not line:
            break  # No more output
        print(line.strip())


with gr.Blocks(title="Axolotl Launcher") as demo:
    gr.Markdown(
        """
    # Axolotl Launcher
    Fill out the required fields below to create a training run.
    """
    )
    base_model_name = gr.Textbox(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", label="Base model"
    )
    with gr.Row():
        dataset_path = gr.Textbox("mhenrichsen/alpaca_2k_test", label="Dataset")
        dataset_type_name = gr.Dropdown(
            choices=["alpaca", "sharegpt"], label="Dataset type", value="alpaca"
        )
    with gr.Row():
        learning_rate = gr.Number(0.000001, label="Learning rate")
        gradient_accumulation_steps_count = gr.Number(
            1, label="Gradient accumulation steps"
        )
        val_set_size_count = gr.Number(0, label="Validation size")

    with gr.Row():
        micro_batch_size_count = gr.Number(1, label="Micro batch size")
        sequence_length = gr.Number(1024, label="Sequence length")
        num_epochs_count = gr.Number(1, label="Epochs")

    output_dir_path = gr.Textbox("./model-out", label="Output directory")

    mode = gr.Radio(
        choices=["Full finetune", "QLoRA", "LoRA"],
        value="Full finetune",
        label="Training mode",
        info="FFT = 16 bit, Qlora = 4 bit, Lora = 8 bit",
    )

    create_config = gr.Button("Create config")
    output = gr.TextArea(label="Generated config")
    create_config.click(
        config,
        inputs=[
            base_model_name,
            dataset_path,
            dataset_type_name,
            learning_rate,
            gradient_accumulation_steps_count,
            micro_batch_size_count,
            sequence_length,
            num_epochs_count,
            output_dir_path,
            val_set_size_count,
        ],
        outputs=output,
    )

    start_training = gr.Button("Start training")
    start_training.click(create_training_job)

demo.launch(share=True)
