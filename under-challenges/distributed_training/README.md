# Distributed Training
In the era of large AI models, distributed training has become essential for scaling complex systems across multiple computing nodes.

In this simplified scenario, a server is set up to accept a 4x4 matrix upload in .pt format (e.g., you upload matA). The server will then return the results of the multiplication matA * matB, where matB is a secret matrix stored on the server.

Can you get the flag stored on the server?

```bash
curl $URL -F 'file=@PATH_TO_PT'
```

For example:
```python
import torch
mat = torch.rand(4, 4)
torch.save(mat, "mat.pt")
```
Then upload with `curl`.