import torch

# data

W = torch.tensor([[3,2],[4,3]], requires_grad=True, dtype=torch.float)
b = torch.tensor([-2,-4], requires_grad=True, dtype=torch.float)

#### MATRIX IS LAID OUT DIFFERENTLY NOW

W2 = torch.tensor([5,-5], requires_grad=True, dtype=torch.float)
b2 = torch.tensor([-2], requires_grad=True, dtype=torch.float)

x = torch.tensor([ [0.,0.], [1.,0.], [0.,1.], [1.,1.] ])
t = torch.tensor([ 0.,      1.,      1.,      0. ])

mu = 0.1 

for iteration in range(1000):

  # forward computation

  s = x.mm(W) + b
  ### PRODUCT IS NOW COMPUTED DIFFERENTLY
  h = torch.nn.Sigmoid()(s)

  z = h.mv(W2) + b2
  y = torch.nn.Sigmoid()(z)

  error = 1/2 * (t - y) ** 2
  mean_error = error.mean()

  # backward computation

  mean_error.backward()

  W.data  = W  - mu * W.grad.data
  b.data  = b  - mu * b.grad.data
  W2.data = W2 - mu * W2.grad.data
  b2.data = b2 - mu * b2.grad.data

  W.grad.data.zero_()
  b.grad.data.zero_()
  W2.grad.data.zero_()
  b2.grad.data.zero_()

  print(mean_error)
  
