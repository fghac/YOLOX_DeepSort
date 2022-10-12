import torch
a=torch.tensor([1,2,3,4])
b=torch.unsqueeze(a,0)
print(a)
print(a.size())
print(b)
print(b.size())