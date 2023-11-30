import pickle
import pudb
import numpy as np
from matplotlib import pyplot as plt

datasets = ["cdc_flu",
    "cdc_covid",
    "symp",
    "covid_deaths",
    "hospital"]

names = ["(a) CDC Flu", "(b) CDC COVID", "(c) SYMPTOMS", "(d) COVID Deaths", "(e) Hospital"]

for i, disease in enumerate(datasets):
    print(disease)
    # GPT

    f = open("outputs/monash/gpt4all_"+disease+".pkl", "rb")
    gpt_hosp = pickle.load(f)
    f.close()
    idx = 0
    if disease == "covid_deaths":
        idx = 5

    train = gpt_hosp["gpt4all"]["train"][idx].tolist()
    test = gpt_hosp["gpt4all"]["test"][idx].tolist()
    total = train + test

    gpt_pred = gpt_hosp['gpt4all']["samples"][idx]
    gpt_med = np.median(gpt_hosp['gpt4all']["samples"][idx],axis=0)
    gpt_med_total = train + gpt_med.tolist()
    gpt_std = np.std(gpt_hosp['gpt4all']["samples"][idx],axis=0)
    gpt_min = gpt_med - 2*gpt_std
    gpt_min_total = train + gpt_min.tolist()
    gpt_max = gpt_med + 2*gpt_std
    gpt_max_total = train + gpt_max.tolist()

    # GPT Context

    f = open("outputs/monash_explain/gpt4all_"+disease+".pkl", "rb")
    gpt_hosp_explain = pickle.load(f)
    f.close()
    idx = 0
    if disease == "covid_deaths":
        idx = 5

    train = gpt_hosp_explain["gpt4all"]["train"][idx].tolist()
    test = gpt_hosp_explain["gpt4all"]["test"][idx].tolist()
    total = train + test

    gpt_pred = gpt_hosp_explain['gpt4all']["samples"][idx]
    gpt_med = np.median(gpt_hosp_explain['gpt4all']["samples"][idx],axis=0)
    gpt_med_total_explain = train + gpt_med.tolist()
    gpt_std = np.std(gpt_hosp_explain['gpt4all']["samples"][idx],axis=0)
    gpt_min = gpt_med - 2*gpt_std
    gpt_min_total_explain = train + gpt_min.tolist()
    gpt_max = gpt_med + 2*gpt_std
    gpt_max_total_explain = train + gpt_max.tolist()

    # GPT Context less

    f = open("outputs/monash_explain_less/gpt4all_"+disease+".pkl", "rb")
    gpt_hosp_explain_less = pickle.load(f)
    f.close()
    idx = 0
    if disease == "covid_deaths":
        idx = 5

    train = gpt_hosp_explain_less["gpt4all"]["train"][idx].tolist()
    test = gpt_hosp_explain_less["gpt4all"]["test"][idx].tolist()
    total = train + test

    gpt_pred = gpt_hosp_explain_less['gpt4all']["samples"][idx]
    gpt_med = np.median(gpt_hosp_explain_less['gpt4all']["samples"][idx],axis=0)
    gpt_med_total_explain_less = train + gpt_med.tolist()
    gpt_std = np.std(gpt_hosp_explain_less['gpt4all']["samples"][idx],axis=0)
    gpt_min = gpt_med - 2*gpt_std
    gpt_min_total_explain_less = train + gpt_min.tolist()
    gpt_max = gpt_med + 2*gpt_std
    gpt_max_total_explain_less = train + gpt_max.tolist()

    # GPT Context lockdown
    if "covid" in disease:
        f = open("outputs/monash_explain_lockdown/gpt4all_"+disease+".pkl", "rb")
        gpt_hosp_explain_lockdown = pickle.load(f)
        f.close()
        idx = 0
        if disease == "covid_deaths":
            idx = 5

        train = gpt_hosp_explain_lockdown["gpt4all"]["train"][idx].tolist()
        test = gpt_hosp_explain_lockdown["gpt4all"]["test"][idx].tolist()
        total = train + test

        gpt_pred = gpt_hosp_explain_lockdown['gpt4all']["samples"][idx]
        gpt_med = np.median(gpt_hosp_explain_lockdown['gpt4all']["samples"][idx],axis=0)
        gpt_med_total_explain_lockdown = train + gpt_med.tolist()
        gpt_std = np.std(gpt_hosp_explain_lockdown['gpt4all']["samples"][idx],axis=0)
        gpt_min = gpt_med - 2*gpt_std
        gpt_min_total_explain_lockdown = train + gpt_min.tolist()
        gpt_max = gpt_med + 2*gpt_std
        gpt_max_total_explain_lockdown = train + gpt_max.tolist()


    # LLAMA

    f = open("outputs/monash/llama-7b_"+disease+".pkl", "rb")
    llama_hosp = pickle.load(f)
    f.close()
    train = llama_hosp["llama-7b"]["train"][idx].tolist()
    test = llama_hosp["llama-7b"]["test"][idx].tolist()
    total = train + test
    llama_pred = llama_hosp['llama-7b']["samples"][idx]

    llama_med = np.median(llama_hosp['llama-7b']["samples"][idx],axis=0)
    llama_med_total = train + llama_med.tolist()


    llama_std = np.std(llama_hosp['llama-7b']["samples"][idx],axis=0)
    llama_min = llama_med - 2*llama_std
    llama_min_total = train + llama_min.tolist()

    llama_max = llama_med + 2*llama_std
    llama_max_total = train + llama_max.tolist()

    # ARIMA

    f = open("outputs/monash/arima_"+disease+".pkl", "rb")
    arima_hosp = pickle.load(f)
    f.close()

    try:
        train = arima_hosp["arima"]["train"][idx].tolist()
        test = arima_hosp["arima"]["test"][idx].tolist()
    except:
        train = arima_hosp["arima"]["train"][idx]
        test = arima_hosp["arima"]["test"][idx]
    total = train + test

    arima_med = arima_hosp['arima']["median"][idx]
    arima_med_total = train + arima_med

    arima_min = arima_hosp['arima']["lower"][idx]
    arima_min_total = train + arima_min

    arima_max = arima_hosp['arima']["upper"][idx]
    arima_max_total = train + arima_max
    # if disease == "covid_deaths":
    #     import pudb; pu.db

    plt.plot(llama_med_total, label="LLAMA, MAE: "+str(round(llama_hosp['llama-7b']['maes'][idx], 2)))
    plt.fill_between(x=range(len(llama_med_total)), y1=llama_min_total, y2=llama_max_total, alpha=0.3)
    plt.plot(gpt_med_total, label="GPT, MAE: "+str(round(gpt_hosp['gpt4all']['maes'][idx], 2)))
    plt.fill_between(x=range(len(gpt_med_total)), y1=gpt_min_total, y2=gpt_max_total, alpha=0.3)
    plt.plot(gpt_med_total_explain, label="GPT w/ Context, MAE: "+str(round(gpt_hosp_explain['gpt4all']['maes'][idx], 2)))
    plt.fill_between(x=range(len(gpt_med_total_explain)), y1=gpt_min_total_explain, y2=gpt_max_total_explain, alpha=0.3)
    plt.plot(gpt_med_total_explain_less, label="GPT w/ less Context, MAE: "+str(round(gpt_hosp_explain['gpt4all']['maes'][idx], 2)))
    plt.fill_between(x=range(len(gpt_med_total_explain_less)), y1=gpt_min_total_explain_less, y2=gpt_max_total_explain_less, alpha=0.3)
    plt.plot(arima_med_total, label="ARIMA, MAE: "+str(round(arima_hosp['arima']['maes'][idx], 2)))
    plt.fill_between(x=range(len(arima_med_total)), y1=arima_min_total, y2=arima_max_total, alpha=0.3)

    if "covid" in disease:
        plt.plot(gpt_med_total_explain_lockdown, label="GPT w/ less Context, MAE: "+str(round(gpt_hosp_explain['gpt4all']['maes'][idx], 2)))
        plt.fill_between(x=range(len(gpt_med_total_explain_lockdown)), y1=gpt_min_total_explain_lockdown, y2=gpt_max_total_explain_lockdown, alpha=0.3)

    # plt.plot(pred_stat[-40:], label="Stat")
    # plt.fill_between(x=range(40), y1=pred_stat_lo[-40:], y2=pred_stat_hi[-40:], alpha=0.3)
    plt.plot(total, label="Ground Truth")
    plt.xlabel("Time")
    plt.ylabel(disease)
    plt.legend()
    plt.title(names[i])
    plt.show()
    plt.savefig("epi_plots/"+disease+".png")
    plt.clf()
    print("LLAMA MAE: ", np.mean(llama_hosp['llama-7b']['maes']))
    print("GPT MAE: ", np.mean(gpt_hosp['gpt4all']['maes']))
    print("GPT w/ context MAE: ", np.mean(gpt_hosp_explain['gpt4all']['maes']))
    print("GPT w/ less context MAE: ", np.mean(gpt_hosp_explain_less['gpt4all']['maes']))
    if "covid" in disease:
        print("GPT w/ lockdown context MAE: ", np.mean(gpt_hosp_explain_lockdown['gpt4all']['maes']))
    print("ARIMA MAE: ", np.mean(arima_hosp['arima']['maes']))
    print("\n\n")

    # import pudb; pu.db
