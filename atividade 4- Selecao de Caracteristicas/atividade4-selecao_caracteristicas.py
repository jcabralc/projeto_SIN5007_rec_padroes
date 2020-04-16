import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from branch_and_bound import BranchAndBound
from mlxtend.feature_selection import SequentialFeatureSelector
from skrebate import ReliefF
import pymrmr

######## Load dataset
dataset = pd.read_csv('../data/kag_risk_factors_cervical_cancer.csv')

######## Pre-processing
# Replace ? by NAN
dataset.replace('?', np.nan, inplace = True)
# Converting features in numbers
dataset = dataset.apply(pd.to_numeric, errors="ignore")
# Fill with median
continuous_feat = ['Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)', 'STDs (number)','STDs: Time since first diagnosis','STDs: Time since last diagnosis'] 
for cont_col in continuous_feat:
    dataset[cont_col] = dataset[cont_col].fillna(dataset[cont_col].median())
# Fill with most frequently value
dataset['Smokes'] = dataset['Smokes'].fillna(0)
dataset['Hormonal Contraceptives'] = dataset['Hormonal Contraceptives'].fillna(1)
dataset['IUD'] = dataset['IUD'].fillna(0)
dataset['STDs'] = dataset['STDs'].fillna(0)
dataset['STDs:condylomatosis'] = dataset['STDs:condylomatosis'].fillna(0)
dataset['STDs:cervical condylomatosis'] = dataset['STDs:cervical condylomatosis'].fillna(0)
dataset['STDs:vaginal condylomatosis'] = dataset['STDs:vaginal condylomatosis'].fillna(0)
dataset['STDs:vulvo-perineal condylomatosis'] = dataset['STDs:vulvo-perineal condylomatosis'].fillna(0)
dataset['STDs:syphilis'] = dataset['STDs:syphilis'].fillna(0)
dataset['STDs:pelvic inflammatory disease'] = dataset['STDs:pelvic inflammatory disease'].fillna(0)
dataset['STDs:genital herpes'] = dataset['STDs:genital herpes'].fillna(0)
dataset['STDs:molluscum contagiosum'] = dataset['STDs:molluscum contagiosum'].fillna(0)
dataset['STDs:AIDS'] = dataset['STDs:AIDS'].fillna(0)
dataset['STDs:HIV'] = dataset['STDs:HIV'].fillna(0)
dataset['STDs:Hepatitis B'] = dataset['STDs:Hepatitis B'].fillna(0)
dataset['STDs:HPV'] = dataset['STDs:HPV'].fillna(0)
# Split into features and class
features = dataset.iloc[:, 0:35]
classe = dataset.iloc[:, 35]
#########################################

######## Feature selection

#### Branch and Bound
# Biblioteca: branch-and-bound-feature-selection
# Parametros: qtd de caracteristicas a ser selecionada (usar raiz quadrada do valor total do conjunto de caracteristicas)
# Função criterio: medida de inconsistência (um subconjunto de caracteristicas é dado como inconsistente se possui valores iguais de diferentes classes)
# Observacao: Encontra a melhor solução
# Observacao: Tem uma performance melhor utilizando características discretas
branchAndBound = BranchAndBound(features.values, classe.values)
bestFeaturesBnB = branchAndBound.best_subspace(6)
## Resultado
# Caracteristicas selecionadas: Age, Number of sexual partners, First sexual intercourse, Num of pregnancies, Smokes, Smokes (years)
# Tempo de execução: 55s
# Observação: fez sentido a seleção dessas características como mais relevantes

#### Relief F
# Biblioteca: skrebate
# Parametros: qtd decaracteristicas a ser selecionada; numero de neighbors (maior quantidade de neighbors resulta em uma acurácia maior, porém demora mais o processamento)
# Função criterio: distancia euclideana entre neighbors
# Observacao: Essa biblioteca nao utiliza o parametro M (numero de instancias aleatorias), pois utilizando-se todas as instancias obtem-se um resultado melhor
relief = ReliefF(n_features_to_select=10, n_neighbors=200, verbose=True)
bestFeaturesReliefF = relief.fit_transform(features.values, classe.values)
## Resultado
# Caracteristicas selecionadas: Hinselmann, Schiller, Citology, Age, Num of pregnances, Hormonal Contraceptives (years), First sexual intercourse, STDs (number), STDs, STDs: Number of diagnosys
# Tempo de execução: 11.49 segundos
# Observação: também fez sentido a seleção dessas características como mais relevantes

#### Sequential Foward Selection (SFS) | Sequential Backward Selection (SBS) | Foward Backward (Plus L - take away R)
knn = KNeighborsClassifier(n_neighbors=4)
sfs = SequentialFeatureSelector(
    knn,
    k_features=6,
    forward=False,
    floating=False,
    verbose=1, # show logs (0,1,2)
    scoring='accuracy', # accuracy, f1, precision, recall, roc_auc
    cv=0 # we don't perform any cross-validation
)
sfs = sfs.fit(features.values, classe.values)
sfs.k_feature_idx_

#### Minimal Redundance Maximum Redundance (MRMR)
pymrmr.mRMR(features, 'MIQ', 10)