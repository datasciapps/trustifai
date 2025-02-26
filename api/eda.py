# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric
from api.events import Event, emit



    
async def basicStatistics(data:pd.DataFrame):
    """
    This function returns basic statistics of the dataset
    """
    emit(Event('SummaryStatistics', {'summary statistics': data.describe()}))
    emit(Event('SummaryStatistics', {'missing_values': data.isnull().sum()}))
    emit(Event('SummaryStatistics', {'data_types': data.dtypes}))

    #get distinct values for each column
    distinct_values = {}
    for col in data.columns:
        distinct_values[col] = len(data[col].unique())
    return distinct_values
    