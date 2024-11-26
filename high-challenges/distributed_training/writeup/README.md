# Distributed Training - Writeup

`torch.load` will load serialized objects with pickle by default, which may lead to arbitrary code execution.
There is also a warning on the documentation of [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html#torch-load)

In this challenge you need to inject code while keeping the `.pt` file a valid matrix. You can use [fickling](https://github.com/trailofbits/fickling) to construct the payload.
