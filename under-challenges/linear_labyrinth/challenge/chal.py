import torch 
import numpy as np
import cv2
from model import Model

flag = cv2.imread('flag.png', 0)
tensor = torch.from_numpy(flag).to(torch.float32)
model = Model()
out_tensor = model(tensor)
out_arr = out_tensor.detach().numpy()
cv2.imwrite('out.png', out_arr)
#save model
torch.save(model.state_dict(), 'model.pth')