!pip install Biopython
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import seaborn as sns
from collections import Counter
from sklearn import model_selection, linear_model
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

# data_path = 'https://drive.google.com/uc?id=1f1CtRwSohB7uaAypn8iA4oqdXlD_xXL1'
# cov2_sequences = 'SARS_CoV_2_sequences_global.fasta'
!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20DNA%20Detectives/SARS_CoV_2_sequences_global.fasta'
cov2_sequences = 'SARS_CoV_2_sequences_global.fasta'

sequences = [r for r in SeqIO.parse(cov2_sequences, 'fasta')]

mutation_df = pd.DataFrame()
n_bases_in_seq = len(sequences[0])

print("Creating feature matrix....")

feature_values_dict = {}

for location in (range(n_bases_in_seq)): 
  bases_at_location = np.array([s[location] for s in sequences])

  if len(set(bases_at_location))==1: continue # If
  for base in ['A', 'T', 'G', 'C', '-']:
    feature_values = (bases_at_location==base)

    
    feature_values[bases_at_location=='N'
                   ] = np.nan

    
    feature_values  = feature_values*1

    
    column_name = str(location) + '_' + base
    
    feature_values_dict[column_name] = feature_values
mutation_df = pd.concat([mutation_df, pd.DataFrame(feature_values_dict)], axis = 1)

print("Formatting labels....")
countries = [(s.description).split('|')[-1] for s in sequences]
countries_to_regions_dict = {
         'Australia': 'Oceania',
         'China': 'Asia',
         'Hong Kong': 'Asia',
         'India': 'Asia',
         'Nepal': 'Asia',
         'South Korea': 'Asia',
         'Sri Lanka': 'Asia',
         'Taiwan': 'Asia',
         'Thailand': 'Asia',
         'USA': 'North America',
         'Viet Nam': 'Asia'
}
regions = [countries_to_regions_dict[c] if c in
           countries_to_regions_dict else 'NA' for c in countries]
mutation_df['label'] = regions

print("Balancing data labels....")
balanced_df = mutation_df.copy()
balanced_df['label'] = regions
balanced_df = balanced_df[balanced_df.label!='NA']
balanced_df = balanced_df.drop_duplicates()
samples_north_america = balanced_df[balanced_df.label== 
                                    'North America']
samples_oceania = balanced_df[balanced_df.label== 
                              'Oceania']
samples_asia = balanced_df[balanced_df.label== 
                           'Asia']


n = min(len(samples_north_america),
        len(samples_oceania),
        len(samples_asia))

balanced_df = pd.concat([samples_north_america[:n],
                    samples_asia[:n],
                    samples_oceania[:n]])

X = balanced_df.drop(columns = 'label')
Y = balanced_df.label


lm = linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001, solver='saga', random_state=42)


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, train_size=.8, random_state=42)


lm.fit(X_train, Y_train)

lm.classes_

lm.predict_proba(X_test)



coefficients = lm.coef_[0]
n_possible_features = len(coefficients)
n_features_used = sum(coefficients != 0)
print("The original logistic regression model used %i out of a possible %i features" %
      (n_features_used, n_possible_features))


_1_ = "don't want it to be redundant" #@param {type:"string"}
_2_ = "don't want it to be time consuming" #@param {type:"string"}

print("1. To prevent overfitting. Especially in models with more features than samples, ")
print("we can end up overfitting on the training set.\n")
print("2. In order to determine which features are the most important.")
print("In the case of SARS-CoV-2, we can use the important features as ")
print("biomarkers to determine which lineage came from which region.")



from sklearn.metrics import accuracy_score


Y_pred_train = lm.predict(X_train)
Y_pred_test = lm.predict(X_test)

training_accuracy = accuracy_score(Y_pred_train, Y_train) *100
testing_accuracy = accuracy_score(Y_pred_test, Y_test) *100
print("Training accuracy: %", training_accuracy)
print("Testing accuracy: %", testing_accuracy)


lam = 0.5 #@param {type:"slider", min:0, max:1, step:0.1}
l1m = linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001, C=1/lam,
    penalty='l1', solver='saga', random_state=42)
l1m.fit(X_train, Y_train)
print("Using lambda=", lam)
print("Lasso Training accuracy:", np.mean(Y_train==l1m.predict(X_train)))
print("Lasso Testing accuracy:", np.mean(Y_test==l1m.predict(X_test)))
print("Number of non-zero coefficients in lasso model:", sum(l1m.coef_[0]!=0))



lm_cv = linear_model.LogisticRegressionCV(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001,
    solver='saga', random_state=42, refit=False,
    penalty = 'l1',
    Cs=5
    )

lm_cv.fit(X_train, Y_train)
print("Training accuracy:", np.mean(Y_train==lm_cv.predict(X_train)))
print("Testing accuracy:", np.mean(Y_test==lm_cv.predict(X_test)))
print("Number of non-zero coefficients in lasso model:", sum(lm_cv.coef_[0]!=0))
print("Lambda decided on by cross validation:", 1/lm_cv.C_[0])


data = mutation_df#[variant_df.regions=='USA']
pca = decomposition.PCA(n_components=2)
pca.fit(X)
df = pd.DataFrame()
df['Principal Component 1'] = [pc[0] for pc in pca.transform(X)]
df['Principal Component 2'] = [pc[1] for pc in pca.transform(X)]
plt.figure(figsize=(5,5))
sns.scatterplot(data=df, x='Principal Component 1', y='Principal Component 2')
plt.show()


Response = "" #@param {type:"string"}
print("We are able to make obsevations about the data without ever using any sort of label.")
print("We can see that there are clusters of very similar samples.  ")
print("We should suspect data points near each other are probably viruses ")
print("from similar regions of the world, or that these viruses have similar biological properties.")


df['color'] = Y
plt.figure(figsize=(5,5))
sns.scatterplot(data=df, x='Principal Component 1', y='Principal Component 2', hue='color')