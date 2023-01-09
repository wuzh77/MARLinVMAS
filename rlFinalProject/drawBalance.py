from operator import length_hint
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

compress = 2
mappoY = np.load('E://pycode//rlFinalProject//MAPPO//MAPPObalance.npy')
mappoY = np.reshape(mappoY, (-1, compress))
mappoY = np.mean(mappoY, axis=1)

x = list(range(len(mappoY)))

cppoY = np.load('E://pycode//rlFinalProject//CPPO//CPPO_balance.npy')
cppoY = np.reshape(cppoY, (-1, int(compress / 2)))
cppoY = np.mean(cppoY, axis=1)

ippoY = np.load('E://pycode//rlFinalProject//IPPO//IPPO_balance.npy')
ippoY = np.reshape(ippoY, (-1, int(compress / 2)))
ippoY = np.mean(ippoY, axis=1)

plt.subplot(2, 2, 1)
plt.grid()
plt.plot(x, mappoY)
plt.plot(x, cppoY)
plt.plot(x, ippoY)
plt.title('balance')
plt.xlabel('iteration')
plt.ylabel('rewards')
plt.legend(['mappo', 'cppo', 'ippo'])

mappoy2 = np.load('E://pycode//rlFinalProject//MAPPO//,MAPPOwheel.npy')
mappoy2 = np.reshape(mappoy2, (-1, compress))
mappoy2 = np.mean(mappoy2, axis=1)

cppoy2 = np.load('E://pycode//rlFinalProject//CPPO//CPPOwheel.npy')
cppoy2 = np.reshape(cppoy2, (-1, compress))
cppoy2 = np.mean(cppoy2, axis=1)

ippoy2 = np.load('E://pycode//rlFinalProject//IPPO//IPPOwheel.npy')
ippoy2 = np.reshape(ippoy2, (-1, compress))
ippoy2 = np.mean(ippoy2, axis=1)

plt.subplot(2, 2, 2)
plt.grid()
plt.plot(x, mappoy2)
plt.plot(x, cppoy2)
plt.plot(x, ippoy2)
plt.title('wheel')
plt.xlabel('iteration')
plt.ylabel('rewards')
plt.legend(['mappo', 'cppo', 'ippo'])

ippo_balance_cmp = np.load('E://pycode//rlFinalProject//IPPO//IPPO_balance.npy')
ippo_balance_cmp = np.reshape(ippo_balance_cmp, (-1, int(compress / 2)))
ippo_balance_cmp = np.mean(ippo_balance_cmp, axis=1)

ippo_balance_cmp1 = np.load('E://pycode//rlFinalProject//IPPOimprove//IPPO_balance.npy')
ippo_balance_cmp1 = np.reshape(ippo_balance_cmp1, (-1, int(compress / 2)))
ippo_balance_cmp1 = np.max(ippo_balance_cmp1, axis=1)

x_cmp = list(range(len(ippo_balance_cmp1)))

plt.subplot(2, 2, 3)
plt.plot(x_cmp, ippo_balance_cmp)
plt.plot(x_cmp, ippo_balance_cmp1)
plt.ylabel('rewards')
plt.xlabel('iterations')
plt.grid()
plt.legend(['IPPO', 'IPPOimprove'])

ippo_wheel_cmp = np.load('E://pycode//rlFinalProject//IPPO//IPPOwheel.npy')
print('ippowheel = {}'.format(len(ippo_wheel_cmp)))
ippo_wheel_cmp = np.reshape(ippo_wheel_cmp, (-1, compress))
ippo_wheel_cmp = np.mean(ippo_wheel_cmp, axis=1)

ippo_wheel_cmp1 = np.load('E://pycode//rlFinalProject//IPPOimprove//IPPOimp_wheel.npy')
print('ippoimprove = {}'.format(len(ippo_wheel_cmp1)))
ippo_wheel_cmp1 = np.reshape(ippo_wheel_cmp1, (-1, compress))
ippo_wheel_cmp1 = np.max(ippo_wheel_cmp1, axis=1)
x_wheel = list(range(len(ippo_wheel_cmp1)))
plt.subplot(2, 2, 4)
plt.plot(x_wheel, ippo_wheel_cmp)
plt.plot(x_wheel, ippo_wheel_cmp1)
plt.grid()
plt.ylabel('rewards')
plt.xlabel('iterations')
plt.legend(['IPPO', 'IPPOimprove'])
plt.plot()
plt.show()
