#!/usr/bin/env python
# coding: utf-8

# In[1]:



get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

from   sklearn.metrics import r2_score
import statsmodels.api as sm


# In[2]:


df_completo = pd.read_csv("..\\Dados\\data_bank.csv", sep=";")
df_completo.head()


# In[3]:


df_completo.info()


# In[4]:


df_completo.isnull().any()


# In[5]:



evaluation = pd.DataFrame(
    {
        "Model": [],
        "Details": [],
        "Root Mean Squared Error (RMSE)": [],
        "R-squared (training)": [],
        "Adjusted R-squared (training)": [],
        "R-squared (test)": [],
        "Adjusted R-squared (test)": [],
        "5-Fold Cross Validation": []
    }
)


# In[6]:


def adjusted_r2(r2, n, k):
    return r2 - (n - 1) * (1 - r2) / (n - k - 1)


# In[37]:


train_data, test_data = train_test_split(df_completo, train_size=0.7, random_state=4)

independent_var = ["Adjusted_savings:_energy_depletion_(%_of_GNI)",
                    #"Age_dependency_ratio_(%_of_working-age_population)",
                    "Agriculture,_forestry,_and_fishing,_value_added_(%_of_GDP)",
                    "GDP_deflator_(base_year_varies_by_country)",
                   "GDP_per_capita_(constant_2010_US$)",
                    "GDP_per_capita_(current_US$)",
                    #"General_government_final_consumption_expenditure_(%_of_GDP)",
                    "GNI_(current_US$)",
                   #"Gross_fixed_capital_formation_(%_of_GDP)",
                    #"Inflation,_GDP_deflator_(annual_%)"
                    ]
lin_reg = LinearRegression()
lin_reg.fit(train_data[independent_var], train_data["Premiums_per_capita_(USD)_-_Total"])

print(f"Intercept: {lin_reg.intercept_}")
print(f"Coefficients: {lin_reg.coef_}")


# In[38]:


pred = lin_reg.predict(test_data[independent_var])


# In[39]:


rmse = metrics.mean_squared_error(test_data["Premiums_per_capita_(USD)_-_Total"], pred)
r2_train = lin_reg.score(train_data[independent_var], train_data["Premiums_per_capita_(USD)_-_Total"])
ar2_train = adjusted_r2(
    r2_train,
    train_data.shape[0],
    len(independent_var)
)

r2_test = lin_reg.score(test_data[independent_var], test_data["Premiums_per_capita_(USD)_-_Total"])
ar2_test = adjusted_r2(
    r2_test,
    test_data.shape[0],
    len(independent_var)
)

cross_val = cross_val_score(lin_reg, df_completo[independent_var], df_completo["Premiums_per_capita_(USD)_-_Total"], cv=5).mean()

r = evaluation.shape[0]
evaluation.loc[r] = ["Multiple Linear Regression-1", "Selected features", rmse, r2_train, ar2_train, r2_test, ar2_test, cross_val]
evaluation.sort_values(by="5-Fold Cross Validation", ascending=False)


# In[40]:



# Retorna um array de zeros com o mesmo shape e tipo do array dado
mask = np.zeros_like(df_completo.corr(), dtype=bool)

# Retorna os índices apenas do triângulo superior do array
mask[np.triu_indices_from(mask)] = True


# In[41]:


plt.subplots(figsize=(12, 8))
plt.title("Pearson Correlation Matrix", fontsize=25)

sns.heatmap(
    df_completo.corr(),
    linewidths=0.25,
    square=True,
    cmap="Blues",
    linecolor="w",
    annot=True,
    annot_kws={"size": 8},
    mask=mask,
    cbar_kws={"shrink": 0.9}
)


# In[42]:


X = np.column_stack((
df_completo["Adjusted_savings:_energy_depletion_(%_of_GNI)"],
#df_completo["Age_dependency_ratio_(%_of_working-age_population)"],
df_completo["Agriculture,_forestry,_and_fishing,_value_added_(%_of_GDP)"],
df_completo["GDP_deflator_(base_year_varies_by_country)"],
df_completo["GDP_per_capita_(constant_2010_US$)"],
df_completo["GDP_per_capita_(current_US$)"],
#df_completo["General_government_final_consumption_expenditure_(%_of_GDP)"],
df_completo["GNI_(current_US$)"],
#df_completo["Gross_fixed_capital_formation_(%_of_GDP)"],
#df_completo["Inflation,_GDP_deflator_(annual_%)"]          
                                        
                     ))
y = df_completo["Premiums_per_capita_(USD)_-_Total"]


X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())





# In[ ]:


cross_val_score(lin_reg, df_completo[independent_var], df_completo["Insurance_penetration"], cv=5)



