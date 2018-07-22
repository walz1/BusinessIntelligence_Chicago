# BusinessIntelligence_Chicago

## Setup
1. Clone the repository
2. Download the corresbonding datasets from [Kaggle](https://www.kaggle.com/currie32/crimes-in-chicago)
3. Copy the downloaded datas to the /dataset folder of this repository

## How to use

**Important:** We have implemented a parallelization strategy, which is activated by default. If you are using a windows machine search for the *Globals section* (top of code) and set `RUN_PARALLEL = False`

* All sections are surrounded by triple quotes. They allow you to commend / uncommend whole sections: 

```python
# This is a commented section
"""
print("Started section: Data loading")
dataset_04 = pd.read_csv('./dataset/Chicago_Crimes_2001_to_2004.csv',
                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
                      na_values=[''])
#"""

# This is an uncommented section
#"""
print("Started section: Data loading")
dataset_04 = pd.read_csv('./dataset/Chicago_Crimes_2001_to_2004.csv',
                      sep=',', header=0, error_bad_lines=False, low_memory=False, 
                      na_values=[''])
#"""
```

* The section *Data understanding* is commented by default, as the correlation analyses needs quite long. 
* Within the *Modeling* section we have commented all models except for *Decision Tree*. You can add or remove them as you like.

## Nice to know
* We have prepared some great visualizations of the dataset. You can play with them [here](https://walz1.github.io/BusinessIntelligence_Chicago/).
* The dataset is enormous, so bring plenty of time and patience, as well as a powerful notebook.
* *Multinomial Logistic Regression* will need several hours (we did not manage to execute a full blown test), you should think about a nightly run. The correlation analyses are also quite slow. We have added a global constant `CORRELATION_SAMPLE_SIZE = 0.25` to configure a sample size for the calculations.
* We have used a MacBook Pro equipped with an i7 processor and 16GB RAM. You should not use your grandma's PC with this data 
