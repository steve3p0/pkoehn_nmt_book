import torch

# data

W = torch.tensor([[3,4],[2,3]], requires_grad=True, dtype=torch.float)
b = torch.tensor([-2,-4], requires_grad=True, dtype=torch.float)

W2 = torch.tensor([5,-5], requires_grad=True, dtype=torch.float)
b2 = torch.tensor([-2], requires_grad=True, dtype=torch.float)

data = [ [ torch.tensor([0.,0.]), torch.tensor([0.]) ],
         [ torch.tensor([1.,0.]), torch.tensor([1.]) ],
         [ torch.tensor([0.,1.]), torch.tensor([1.]) ],
         [ torch.tensor([1.,1.]), torch.tensor([0.]) ] ]

mu = 0.1 

for iteration in range(1000):

  total_error = 0

  for item in data:
    x = item[0]
    t = item[1]

    # forward computation

    s = W.mv(x) + b
    h = torch.nn.Sigmoid()(s)

    z = torch.dot(W2, h) + b2
    y = torch.nn.Sigmoid()(z)

    error = 1/2 * (t - y) ** 2
    total_error = total_error + error

    # backward computation

    error.backward()

    W.data  = W  - mu * W.grad.data
    b.data  = b  - mu * b.grad.data
    W2.data = W2 - mu * W2.grad.data
    b2.data = b2 - mu * b2.grad.data

    W.grad.data.zero_()
    b.grad.data.zero_()
    W2.grad.data.zero_()
    b2.grad.data.zero_()

  print("error: ",total_error.data/4)
  
