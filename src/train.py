import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model import ChurnModel

df = pd.read_csv("data/processed_churn.csv")

