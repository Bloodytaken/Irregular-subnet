# Irregular-subnet

 Irregular SUBNET is an identification approach designed for irregularly sampled time series, which is inspired by CT-subnet method. ([https://github.com/MaartenSchoukens/deepSI/tree/legacy](https://github.com/MaartenSchoukens/deepSI/tree/legacy).)


### Information of documents 

- `fitting.py`, `models.py`, `networks.py`, `normalization.py` are fundamental code implentaion. 
`encoder.ipynb` is used to validate the function of time-aware encoder. 

- The simulation study is demonstrated in folder `simulation study`, using the system set up and data generation in folder `MSD`. 

- Additionally, the [EMPS benchmark](https://www.nonlinearbenchmark.org/benchmarks/emps) was modeled by this approach and obtained a competitve result. Simulated irregular sampled data and outcomes can be found in folder `EMPS` and `benchmark_EMPS.ipynb` respectively. Any of these notebooks can serve as an example usage.

