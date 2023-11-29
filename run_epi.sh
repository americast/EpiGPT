for data in covid_deaths  hospital cdc_flu  cdc_covid  symp
   for model in  llama-7b gpt4all arima
        CUDA_VISIBLE_DEVICES="0" python -m experiments/run_epi --data $data --model $model
    done
done