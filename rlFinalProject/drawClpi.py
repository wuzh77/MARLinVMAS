import matplotlib.pyplot as plt
import numpy as np

compress = 2
ippo1 = np.load('E://pycode//rlFinalProject//IPPO//IPPO_wheel_0.1.npy')
ippo1 = np.reshape(ippo1, (-1, compress))
ippo1 = np.mean(ippo1, axis=1)

x = list(range(len(ippo1)))

ippo2 = np.load('E://pycode//rlFinalProject//IPPO//IPPOwheel.npy')
ippo2 = np.reshape(ippo2, (-1, compress * 2))
ippo2 = np.mean(ippo2, axis=1)

ippo5 = np.load('E://pycode//rlFinalProject//IPPO//IPPO_wheel_5.npy')
ippo5 = np.reshape(ippo5, (-1, compress))
ippo5 = np.mean(ippo5,axis=1)

plt.grid()
plt.plot(x, ippo1)
plt.plot(x, ippo2)
plt.plot(x, ippo5)
plt.title('different clip')
plt.ylabel('rewards')
plt.xlabel('iterations')
plt.ylim(-10, -6)
plt.legend(['0.1', '0.2', '0.5'])
plt.show()