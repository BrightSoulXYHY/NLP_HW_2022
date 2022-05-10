import re
import matplotlib.pyplot as plt


loss_re = re.compile(r"total_loss=(\d+)\.(\d+)")



log_file = "../log/20220510-164802.log"




with open(log_file,"r",encoding="utf-8") as fp:
    textL = fp.readlines()
total_loss = None
accuracy = None
lossL = []
accuracyL = []
for text in textL:
    loss_str = loss_re.search(text)
    if loss_str is not None:
        total_loss = (loss_str.group()).split("=")[-1]
        lossL.append(float(total_loss))

print(lossL)

plt.plot(lossL,label="loss")
plt.xlabel("iter_nums")
plt.ylabel("loss")

plt.title("loss of CBOW")
plt.legend()
plt.show()