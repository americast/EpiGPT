for data in covid_deaths hospital cdc_flu cdc_covid symp
do
    for model in llama-7b gpt4all arima
    do
        CUDA_VISIBLE_DEVICES="0" python -m experiments.run_epi --dataset $data --model $model
    done
done