!pip install Biopython
from Bio import SeqIO
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn import model_selection, linear_model

data_path = 'https://drive.google.com/uc?id=1f1CtRwSohB7uaAypn8iA4oqdXlD_xXL1'
cov2_sequences = 'SARS_CoV_2_sequences_global.fasta'

!wget -q --show-progress 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20DNA%20Detectives/SARS_CoV_2_sequences_global.fasta'
cov2_sequences = 'SARS_CoV_2_sequences_global.fasta'



sequences = [r for r in SeqIO.parse(cov2_sequences, 'fasta')]
sequence_num =  0#@param {type:"integer"}
print(sequences[sequence_num])



n_sequences = len(sequences)
print("There are %.0f sequences" % n_sequences)



sequence_1 = np.array(sequences[0])
sequence_10 = np.array(sequences[9])
percent_similarity =  ((np.sum(sequence_1 == sequence_10))/(len(sequence_1))) * 100
print("Sequence 1 and 10 similarity: %", percent_similarity)



reference = np.array(sequences[0])
mutations_per_seq = []
for seq in sequences:
    mutations = ((np.sum(reference != seq)))
    mutations_per_seq.append(mutations)
mutations_per_seq = np.array(mutations_per_seq)
plt.hist(mutations_per_seq)
plt.xlabel('# mutations')
plt.ylabel('# sequences')
plt.show()


min_number_of_mutations  =  300#@param {type:"integer"}
idx = np.random.choice(np.where(mutations_per_seq>min_number_of_mutations)[0])
print("Sequence %i has > %.0f mutations! \n" % (idx, min_number_of_mutations))
print(sequences[idx], '\n')
print("The sequence is composed of: ")
Counter(np.array(sequences[idx]))



n_sequences_with_N = 0
for i in sequences:
    if "N" in i:
        n_sequences_with_N += 1

print('%i sequences have at least 1 "N"!' % n_sequences_with_N)



_1_  =  'training ' #@param {type:"string"}
_2_  =  'testing' #@param {type:"string"}

print('1. We need a set of FEATURES (X).\n',
      '  Our features will be the genomes of the different sequences.')
print('2. We need LABElS (Y).\n',
      '  Our labels will be the country that each sequence came from.')

n_bases_in_seq = len(sequences[0])
columns = {}

for location in tqdm.tqdm(range(n_bases_in_seq)): 
  bases_at_location = np.array([s[location] for s in sequences])
  if len(set(bases_at_location))==1: continue
  for base in ['A', 'T', 'G', 'C', '-']:
    feature_values = (bases_at_location==base)


    feature_values[bases_at_location==['N']] = np.nan


    feature_values  = feature_values*1

    column_name = str(location) + '_' + base


    columns[column_name] = feature_values


mutation_df = pd.DataFrame(columns)


n_rows = np.shape(mutation_df)[0]
n_columns = np.shape(mutation_df)[1]
print("Size of matrix: %i rows x %i columns" %(n_rows, n_columns))


mutation_df.tail()

country = "USA" #@param dict_keys(['China', 'Kazakhstan', 'India', 'Sri Lanka', 'Taiwan', 'Hong Kong', 'Viet Nam', 'Thailand', 'Nepal', 'Israel', 'South Korea', 'Iran', 'Pakistan', 'Turkey', 'Australia', 'USA']
countries = [(s.description).split('|')[-1] for s in sequences]
print("There are %i sequences from %s." %
      (Counter(countries)[country], country))



countries_to_regions_dict = {
         'Australia': 'Oceania',
         'China': 'Asia',
         'Hong Kong': 'Asia' ,
         'India': 'Asia' ,
         'Nepal': 'Asia' ,
         'South Korea': 'Asia' ,
         'Sri Lanka': 'Asia' ,
         'Taiwan': 'Asia' ,
         'Thailand': 'Asia' ,
         'USA': 'North America',
         'Viet Nam': 'Asia'
}

regions = [countries_to_regions_dict[c] if c in
           countries_to_regions_dict else 'NA' for c in countries]
mutation_df['label'] = regions



region = "Asia" #@param ['Oceania', 'North America', 'Asia']
print("There are %i sequences from %s." %
      (Counter(regions)[region], region))



balanced_df = mutation_df.copy()
balanced_df['label'] = regions
balanced_df = balanced_df[balanced_df.label!='NA']
balanced_df = balanced_df.drop_duplicates()
samples_north_america = balanced_df[balanced_df.label== 'North America' ]
samples_oceania = balanced_df[balanced_df.label== 'Oceania' ]
samples_asia = balanced_df[balanced_df.label== 'Asia' ]


n = min(len(samples_north_america),
        len(samples_asia),
        len(samples_oceania))

balanced_df = pd.concat([samples_north_america[:n],
                    samples_asia[:n],
                    samples_oceania[:n]])
print("Number of samples in each region: ", Counter(balanced_df['label']))


X = balanced_df.drop('label', axis=1)
Y = balanced_df.label
data = "X (features)" #@param ['X (features)', 'Y (label)']
start = 1 #@param {type:'integer'}
stop =  10#@param {type:'integer'}

if start>=stop:print("Start must be < stop!")
else:
  if data=='X (features)':
    print(X.iloc[start:stop])
  if data=='Y (label)':
    print(Y[start:stop])


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


lm = linear_model.LogisticRegression(
    multi_class="multinomial", max_iter=1000,
    fit_intercept=False, tol=0.001, solver='saga', random_state=42)

# Split into training/testing set. Use a testing size of 0.2
X_train, X_test, y_train, y_test = train_test_split(
    balanced_df.drop('label', axis=1),  # Features
    balanced_df['label'],               # Labels
    test_size=0.2,                      # 20% of the data for testing
    random_state=42                     # Seed for reproducibility
)

# Train/fit model
lm.fit(X_train, y_train)



# Predict on the test set.
y_pred = lm.predict(X_test)

# Compute accuracy.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %", accuracy *100)

# Compute confusion matrix.
confusion_mat = pd.DataFrame(confusion_matrix(y_test, y_pred))
confusion_mat.columns = [c + ' predicted' for c in lm.classes_]
confusion_mat.index = [c + ' true' for c in lm.classes_]

print(confusion_mat)