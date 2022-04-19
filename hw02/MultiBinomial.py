import numpy as np
import time
import random

true_alphaL = [0.5, 0.3, 0.2]
true_thetaL = [0.7, 0.9, 0.4]

init_alphaL = [0.3, 0.3, 0.4]
init_thetaL = [0.8, 0.8, 0.5]




class Binomial_EMSover:
    def __init__(self,alphaL,thetaL,N=1000,size=50):
        '''
            alphaL=[s1,s2,1-s1-s2]
            thetaL=[p,q,r]
            N=50            取N枚硬币
            size=100        一枚硬币抛size次
        '''
        self.alphaL = alphaL
        self.thetaL = thetaL
        self.gamma = None
        self.data = None
        self.N = N
        self.size = size
        self.K = len(alphaL)
    
    def gen_data(self):
        data = []
        for alpha,theta in zip(true_alphaL,true_thetaL):
            for _ in range(int(alpha*self.N)):
                data.append(np.random.binomial(1, theta,size=self.size))
        self.data = data
        
    def f(self, x, p):
        pro = 1
        for xi in x:
            pro *= ( p if xi == 1 else (1-p) )
        return pro

    def Gamma(self):
        gm_n_k = []
        # 遍历data
        for data in self.data:
            sum_k = sum( [self.alphaL[j] * self.f(data, self.thetaL[j]) for j in range(self.K)] )
            gm_i_k = [((self.alphaL[j] * self.f(data, self.thetaL[j])) / sum_k) for j in range(self.K)]
            gm_n_k.append(gm_i_k)
        return gm_n_k

    def P(self, gm_n_k):
        p = [sum([(gm_n_k[j][i]*sum(self.data[j])) for j in range(self.N)]) for i in range(self.K)]
        gm_k = [sum([gm_n_k[j][i]*self.size for j in range(self.N)]) for i in range(self.K)]
        self.thetaL = [np.divide(p[i], gm_k[i]) for i in range(self.K)]

    def Pi(self, gm_n_k):
        gm_k = [sum([gm_n_k[j][i] for j in range(self.N)]) for i in range(self.K)]
        self.alphaL = np.divide(gm_k, self.N)

    def EM(self):
        iter_num = 0
        start_time = time.time()
        fp = open(f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt","w",encoding="utf-8")
        text_templ = "{:.4f},{},"+"{:.6f},"*6+"\n"
        while iter_num < 100:
            last_alphaL_vec = np.array(self.alphaL)
            last_thetaL_vec = np.array(self.thetaL)
            gm_n_k = self.Gamma()
            self.P(gm_n_k)
            self.Pi(gm_n_k)
            iter_num += 1

            print(
                f"[{time.time()-start_time:.4f}] iter={iter_num} " +\
                "alphaL=[{:.6f} {:.6f} {:.6f}] ".format(*self.alphaL) +\
                "thetaL=[{:.6f} {:.6f} {:.6f}] ".format(*self.thetaL)
            )
            fp.write(text_templ.format(time.time()-start_time,iter_num,*self.alphaL,*self.thetaL))
            if np.linalg.norm(last_alphaL_vec-np.array(self.alphaL)) < 1e-6 and np.linalg.norm(last_thetaL_vec-np.array(self.thetaL)) < 1e-6:
            # if np.max(np.abs(last_alphaL_vec-np.array(self.alphaL))) < 1e-6 and np.max(np.abs(last_thetaL_vec-np.array(self.thetaL))) < 1e-6:
            # if iter_num > 20:
                break


        fp.close()
        print("end slove")


if __name__ == "__main__":
    solver = Binomial_EMSover(init_alphaL,init_thetaL)
    solver.gen_data()
    solver.EM()