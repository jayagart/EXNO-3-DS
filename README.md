## EXNO-3-DS
## Name: Jayagar.T
## Reg.No: 212224220042
## Date:10-03-2026
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE

```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
```


<img width="277" height="411" alt="image" src="https://github.com/user-attachments/assets/c4b54fc4-b941-4938-bcd4-d57e9e74d815" />

```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

pm=['Hot','Warm','Cold']

e1=OrdinalEncoder(categories=[pm])

e1.fit_transform(df[["ord_2"]])
```
<img width="172" height="190" alt="image" src="https://github.com/user-attachments/assets/01718899-f909-4162-a33b-cd02341b4307" />

```
le=LabelEncoder()

dfc=df.copy()

dfc['ord_2']=le.fit_transform(dfc['ord_2'])

dfc
```

<img width="326" height="371" alt="image" src="https://github.com/user-attachments/assets/16225efb-247b-40ae-8423-d48ba88ff489" />

```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

ohe = OneHotEncoder(sparse_output=False)

df2 = df.copy()

enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))

df2 = pd.concat([df2, enc], axis=1)
df2=pd.concat([df2,enc],axis=1)

df2
```

<img width="525" height="381" alt="image" src="https://github.com/user-attachments/assets/058daa97-1ed2-44ff-a5db-b8f064e9d86f" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="745" height="370" alt="image" src="https://github.com/user-attachments/assets/530d37a9-f6b0-460c-9d20-23ab6fc39003" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

<img width="471" height="369" alt="image" src="https://github.com/user-attachments/assets/e21456f6-bcd6-4273-8148-1cbdc7d2b8b8" />

```
be=BinaryEncoder ()

nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)

dfb1=df.copy()

dfb
```

<img width="679" height="362" alt="image" src="https://github.com/user-attachments/assets/aa2d8242-c9c2-422a-bb94-93b89aa46691" />

```
from category_encoders import TargetEncoder

te=TargetEncoder()

cc=df.copy()

new=te.fit_transform(X=cc["City"],y=cc["Target"])

cc=pd.concat([cc,new],axis=1)

cc
```

<img width="540" height="360" alt="image" src="https://github.com/user-attachments/assets/0a13c944-84cb-41da-8548-3405c7dcead5" />

```
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/Data_to_Transform.csv")

df
```

<img width="729" height="437" alt="image" src="https://github.com/user-attachments/assets/4705ad35-39b7-435b-886e-4521d4126dc7" />


```
np.log(df["Highly Positive Skew"])
```

<img width="276" height="448" alt="image" src="https://github.com/user-attachments/assets/6a5deec0-cb18-41ce-a81f-fc9bee449d2f" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="297" height="464" alt="image" src="https://github.com/user-attachments/assets/f38c4dab-ee28-487b-b9b5-5f8926abf9bd" />

```
np.sqrt(df["Highly Positive Skew"])

```

<img width="256" height="477" alt="image" src="https://github.com/user-attachments/assets/c6b65780-f15b-4292-812f-b4637c09d1ed" />

```
np.square(df["Highly Positive Skew"])
```

<img width="284" height="452" alt="image" src="https://github.com/user-attachments/assets/3f0e3501-91a9-42eb-9831-d6b7a50c0754" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])

df
```

<img width="741" height="424" alt="image" src="https://github.com/user-attachments/assets/835c5366-4992-4c30-8586-a91b3f1e2024" />


```
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```

<img width="383" height="265" alt="image" src="https://github.com/user-attachments/assets/57c739c4-8ea4-4d5e-849f-4fde5c54957a" />

```
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])

df.skew()
```

<img width="406" height="304" alt="image" src="https://github.com/user-attachments/assets/1c39fda2-99bc-4f15-8206-95e69d861be3" />

```
from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])

df
```

<img width="748" height="426" alt="image" src="https://github.com/user-attachments/assets/41d7e315-7ed5-4ac1-a4c0-a50edbda5212" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(df["Moderate Negative Skew"], line='45' )
plt.show()
```

<img width="617" height="421" alt="image" src="https://github.com/user-attachments/assets/dae225d7-8c4f-4b4b-b2a1-42c180b5728f" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]), line='45')
plt.show()
```

<img width="606" height="428" alt="image" src="https://github.com/user-attachments/assets/e725db1d-e42f-46ec-84d4-8a06dd6f02c7" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal', n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"], line='45' )
plt. show()
```

<img width="578" height="425" alt="image" src="https://github.com/user-attachments/assets/d47d7804-2917-4dba-a1ed-ef43b03028c3" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```

<img width="563" height="433" alt="image" src="https://github.com/user-attachments/assets/2a981979-eaa5-4307-9129-abb0696b4f64" />

```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
<img width="613" height="441" alt="image" src="https://github.com/user-attachments/assets/f77d6c6b-c667-4fb4-9093-9ce434591da3" />


```
dt=pd.read_csv("/content/titanic_dataset.csv")

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal', n_quantiles=891)

dt["Age_1"]=qt.fit_transform(dt[ ["Age"]])

sm.qqplot(dt['Age'], line='45' )
plt.show()
```

<img width="556" height="324" alt="image" src="https://github.com/user-attachments/assets/d93d3792-a4ca-4010-be94-f69a5ead1439" />

```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```

<img width="574" height="439" alt="image" src="https://github.com/user-attachments/assets/177baf9f-15be-427e-9f7a-0d48f9517c03" />


# RESULT:
Thus, the given data and perform Feature Encoding and Transformation process and save the data to a file is successfully completed.
      
