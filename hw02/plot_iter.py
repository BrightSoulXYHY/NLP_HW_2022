import matplotlib.pyplot as plt
import numpy as np
import time

data_path = "2022-04-19_22-14-33.txt"

true_alphaL = [0.5, 0.3, 0.2]
true_thetaL = [0.7, 0.9, 0.4]

data_alphaL = []
data_thetaL = []

init_alphaL = [0.3, 0.3, 0.4]
init_thetaL = [0.8, 0.5, 0.5]

with open(data_path,"r",encoding="utf-8") as fp:
    dataL = fp.readlines()





for data in dataL:
    # 0.0838,1,0.333083,0.333083,0.333835,0.802917,0.802917,0.490376,
    data_strL = data.split(",")
    alphaL_vec = np.array([float(i) for i in data_strL[2:5]])
    thetaL_vec = np.array([float(i) for i in data_strL[5:8]])


    data_alphaL.append(alphaL_vec)
    data_thetaL.append(thetaL_vec)
data_alphaL = np.array(data_alphaL)
data_thetaL = np.array(data_thetaL)



plt.rcParams["figure.figsize"] = (12, 6)
plt.subplot(121)
plt.plot(data_alphaL[:,0],label=r'$\alpha_1$')
plt.plot(data_alphaL[:,1],label=r'$\alpha_2$')
plt.plot(data_alphaL[:,2],label=r'$\alpha_3$')
for i in true_alphaL:
    plt.hlines(i,0,len(data_alphaL),linestyles='dashed')

plt.legend()
plt.title('Itration of alpha')
plt.subplot(122)
plt.plot(data_thetaL[:,0],label=r'$\theta_1$')
plt.plot(data_thetaL[:,1],label=r'$\theta_1$')
plt.plot(data_thetaL[:,2],label=r'$\theta_1$')
for i in true_thetaL:
    plt.hlines(i,0,len(data_thetaL),linestyles='dashed')
plt.legend()
plt.title('Itration of theta')



plt.suptitle(r"$\alpha_{init}=$"+f"{init_alphaL}"+r"$\theta_{init}=$"+f"{init_thetaL}")

# plt.show()

plt.savefig(f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.png", dpi=300)
