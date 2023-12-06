disease = "symp"

import pickle
import pudb
import numpy as np
from matplotlib import pyplot as plt

# GPT

f = open("gpt4all_"+disease+".pkl", "rb")
gpt_hosp = pickle.load(f)
f.close()

train = gpt_hosp["gpt4all"]["train"][0].tolist()
test = gpt_hosp["gpt4all"]["test"][0].tolist()
total = train + test

gpt_pred = gpt_hosp['gpt4all']["samples"][0]
gpt_med = np.median(gpt_hosp['gpt4all']["samples"][0],axis=0)
gpt_med_total = train + gpt_med.tolist()
gpt_std = np.std(gpt_hosp['gpt4all']["samples"][0],axis=0)
gpt_min = gpt_med - 2*gpt_std
gpt_min_total = train + gpt_min.tolist()
gpt_max = gpt_med + 2*gpt_std
gpt_max_total = train + gpt_max.tolist()

# LLAMA

f = open("llama-7b_"+disease+".pkl", "rb")
llama_hosp = pickle.load(f)
f.close()
train = llama_hosp["llama-7b"]["train"][0].tolist()
test = llama_hosp["llama-7b"]["test"][0].tolist()
total = train + test
llama_pred = llama_hosp['llama-7b']["samples"][0]

llama_med = np.median(llama_hosp['llama-7b']["samples"][0],axis=0)
llama_med_total = train + llama_med.tolist()


llama_std = np.std(llama_hosp['llama-7b']["samples"][0],axis=0)
llama_min = llama_med - 2*llama_std
llama_min_total = train + llama_min.tolist()

llama_max = llama_med + 2*llama_std
llama_max_total = train + llama_max.tolist()

# ARIMA

f = open("arima_"+disease+".pkl", "rb")
arima_hosp = pickle.load(f)
f.close()

train = arima_hosp["arima"]["train"][0]
test = arima_hosp["arima"]["test"][0]
total = train + test

arima_med = arima_hosp['arima']["median"][0]
arima_med_total = train + arima_med

arima_min = arima_hosp['arima']["lower"][0]
arima_min_total = train + arima_min

arima_max = arima_hosp['arima']["upper"][0]
arima_max_total = train + arima_max


plt.plot(llama_med_total, label="LLAMA")
plt.fill_between(x=range(len(llama_med_total)), y1=llama_min_total, y2=llama_max_total, alpha=0.3)
plt.plot(gpt_med_total, label="GPT")
plt.fill_between(x=range(len(gpt_med_total)), y1=gpt_min_total, y2=gpt_max_total, alpha=0.3)
plt.plot(arima_med_total, label="ARIMA")
plt.fill_between(x=range(len(arima_med_total)), y1=arima_min_total, y2=arima_max_total, alpha=0.3)

# plt.plot(pred_stat[-40:], label="Stat")
# plt.fill_between(x=range(40), y1=pred_stat_lo[-40:], y2=pred_stat_hi[-40:], alpha=0.3)
plt.plot(total, label="Ground Truth")
plt.xlabel("Time")
plt.ylabel(disease)
plt.legend()
plt.show()
plt.savefig(disease+".png")
