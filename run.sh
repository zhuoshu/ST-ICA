modelid=STID
python run.py --config_file AQI_12_12.json --modelid $modelid --note 'with ST-ICA' --prompt_type DA --online_adapt 1 --adapt_modules DA_dynamic_prior 
python run.py --config_file EPeMS_12_12.json --modelid $modelid --note 'with ST-ICA' --prompt_type DA --online_adapt 1 --adapt_modules DA_dynamic_prior
python run.py --config_file METRLA_12_12.json --modelid $modelid --note 'with ST-ICA' --prompt_type DA --online_adapt 1 --adapt_modules DA_dynamic_prior
python run.py --config_file UrbanEV_s_12_12.json --modelid $modelid --note 'with ST-ICA' --prompt_type DA --online_adapt 1 --adapt_modules DA_dynamic_prior
python run.py --config_file GBA_12_12.json --modelid $modelid --note 'with ST-ICA' --prompt_type DA --online_adapt 1 --adapt_modules DA_dynamic_prior