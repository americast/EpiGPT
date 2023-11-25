import pickle
import pudb
import numpy as np
from matplotlib import pyplot as plt

train = [13., 13.,  5., 11.,  9., 12., 10., 17.,  9.,  9., 12., 11., 12.,
         9.,  5., 11., 10., 11.,  6., 13., 11.,  5., 10., 10., 11.,  7.,
        10., 12.,  9., 11.]
test = [ 6., 10.,  9.,  7., 10., 11., 10., 10., 13., 11.]
total = train + test

f = open("gpt4all_hospital.pkl", "rb")
gpt_hosp = pickle.load(f)
f.close()

gpt_pred = gpt_hosp['gpt4all']["samples"][0]
gpt_med = np.median(gpt_hosp['gpt4all']["samples"][0],axis=0)
gpt_med_total = train + gpt_med.tolist()
gpt_std = np.std(gpt_hosp['gpt4all']["samples"][0],axis=0)
gpt_min = gpt_med - 2*gpt_std
gpt_min_total = train + gpt_min.tolist()
gpt_max = gpt_med + 2*gpt_std
gpt_max_total = train + gpt_max.tolist()


f = open("llama-7b_hospital.pkl", "rb")
llama_hosp = pickle.load(f)
f.close()

llama_pred = llama_hosp['llama-7b']["samples"][0]

llama_med = np.median(llama_hosp['llama-7b']["samples"][0],axis=0)
llama_med_total = train + llama_med.tolist()


llama_std = np.std(llama_hosp['llama-7b']["samples"][0],axis=0)
llama_min = llama_med - 2*llama_std
llama_min_total = train + llama_min.tolist()

llama_max = llama_med + 2*llama_std
llama_max_total = train + llama_max.tolist()

plt.plot(llama_med_total, label="LLAMA")
plt.fill_between(x=range(len(llama_med_total)), y1=llama_min_total, y2=llama_max_total, alpha=0.3)
plt.plot(gpt_med_total, label="GPT")
plt.fill_between(x=range(len(gpt_med_total)), y1=gpt_min_total, y2=gpt_max_total, alpha=0.3)

# plt.plot(pred_stat[-40:], label="Stat")
# plt.fill_between(x=range(40), y1=pred_stat_lo[-40:], y2=pred_stat_hi[-40:], alpha=0.3)
plt.plot(total, label="Ground Truth")
plt.legend()
plt.show()
plt.savefig("hospital.png")
