import torch

class ExampleNet(torch.nn.Module):

  def __init__(self):
    super(ExampleNet, self).__init__() # huh?

    self.layer1 = torch.nn.Linear(2,2)
    self.layer2 = torch.nn.Linear(2,1)
    self.layer1.weight = torch.nn.Parameter( torch.tensor([[3.,2.],[4.,3.]]) )
    self.layer1.bias   = torch.nn.Parameter( torch.tensor([-2.,-4.]) )
    self.layer2.weight = torch.nn.Parameter( torch.tensor([[5.,-5.]]) )
    self.layer2.bias   = torch.nn.Parameter( torch.tensor([-2.]) )

  def forward(self, x):
    s = self.layer1(x)
    h = torch.nn.Sigmoid()(s)
    z = self.layer2(h)
    y = torch.nn.Sigmoid()(z)
    return y

x = torch.tensor([ [0.,0.], [1.,0.], [0.,1.], [1.,1.] ])
t = torch.tensor([ [0.],    [1.],    [1.],    [0.] ])

net = ExampleNet()
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for iteration in range(100):
  optimizer.zero_grad()
  out = net.forward( x )
  error = 1/2 * (t - out) ** 2
  mean_error = error.mean()
  print("error: ",mean_error.data)
  mean_error.backward()
  optimizer.step()

