# SLURM Multi-Node Training

This directory contains an example SLURM script for running Axolotl training jobs across multiple nodes in a SLURM cluster.

## Prerequisites

- Access to a SLURM cluster with GPU nodes
- Axolotl installed on all nodes (see [installation docs](https://docs.axolotl.ai/docs/installation.html))

## Usage

### Standard SLURM Clusters

1. Copy [`axolotl.slurm`](./axolotl.slurm) to your working directory.
2. Place your Axolotl config file (`train.yaml`) in the same directory.
3. Set the appropriate environment variables for the job:
    ```bash
    export HF_TOKEN="your-huggingface-token"

    # metric tracking
    # export WANDB_API_KEY="your-wandb-api-key"
    # ...
    ```
4. Submit the job:
   ```bash
   sbatch --export=ALL,NUM_NODES=2,NUM_TRAINERS=8,PRIMARY_ADDR=<master-node>,PRIMARY_PORT=29400 axolotl.slurm
   ```

   Where:
   - `NUM_NODES`: Number of nodes to use
   - `NUM_TRAINERS`: GPUs per node (typically 8)
   - `PRIMARY_ADDR`: Hostname/IP of the master node
   - `PRIMARY_PORT`: Port for distributed training (default: 29400)

5. (Optional) Run other slurm commands:
    ```bash
    # check job info
    scontrol show job axolotl-cli

    # check job queue
    squeue

    # check cluster status
    sinfo
    ```

### RunPod Instant Clusters

Axolotl works with RunPod Instant Clusters. This feature provides managed SLURM clusters with zero configuration.

1. **Deploy a SLURM Cluster**:
   - Go to [RunPod Instant Clusters](https://console.runpod.io/cluster)
   - Click "Create a Cluster"
   - Choose your GPU type, node count, and region
   - Choose an [Axolotl cloud docker image](https://docs.axolotl.ai/docs/docker.html#cloud)
   - Deploy the cluster

2. **Connect to the Controller Node**: Find the controller node in the RunPod console and connect via SSH

3. **Follow the instructions in [Standard SLURM Clusters](#standard-slurm-clusters)**

## Additional Resources

- [Axolotl Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [RunPod SLURM Clusters Guide](https://docs.runpod.io/instant-clusters/slurm-clusters)
