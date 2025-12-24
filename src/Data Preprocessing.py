### DATA PREPROCESSING & CLEANING
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import shap

df.isnull().sum()
df.duplicated().sum()
df = df.drop_duplicates()

df['Amount_log'] = np.log1p(df['Amount'])
