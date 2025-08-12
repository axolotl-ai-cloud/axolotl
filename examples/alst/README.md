# Arctic Long Sequence Training (ALST)

Artic Long Sequence Training (ALST) is a technique for training long context models using a variety of optimization
techniques. It is a combination of:
- TiledMLP: Leverage tiling over the sequence dimension on MLP layers to reduce memory usage
- Tiled Loss: Using optimized loss functions like Liger-Kernel or Cut Cross Entropy to reduce memory usage
- Activation Offloading: Offload activations to CPU RAM to reduce memory usage

For more information, you can check out the ALST paper [here](https://www.arxiv.org/abs/2506.13996).
