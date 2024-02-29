from typing import Any
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

class Function:
    def __call__(self, input) : 
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

# xの2乗を計算する関数 
class Square(Function):
    def forward(self, x):
        return x ** 2

# exp関数を計算する関数
class Exp(Function):
    def forward(self, x):
        return np.exp(x)

def numerical_diff(f,x,eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2*eps)

x = Variable(np.array(10))
f = Square()
dy = numerical_diff(f,x) 
print(dy)