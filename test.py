import numpy as np
N = 1000
x = np.random.rand(N,2)*2*np.pi

y1 = np.sum(np.conj(np.exp(1j*x[:,0]))*np.exp(1j*x[:,1]))

y2 = np.sum(np.conj(np.exp(1j*x[:,1]))*np.exp(1j*x[:,1]))

y3 = np.sum(np.conj(np.exp(1j* (x[:,1]+ 3)))*np.exp(1j*x[:,1]))


print(np.abs(y1), np.abs(y2), np.abs(y3))