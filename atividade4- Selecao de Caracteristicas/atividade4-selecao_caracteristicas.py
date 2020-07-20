import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from sklearn.neighbors import KNeighborsClassifier
from branch_and_bound import BranchAndBound
from mlxtend.feature_selection import SequentialFeatureSelector
from skrebate import ReliefF
import pymrmr

######## Load dataset
dataset = pd.read_csv('../data/kag_risk_factors_cervical_cancer.csv')

######## Pre-processing

# Substitui ? by NAN
dataset.replace('?', np.nan, inplace = True)
# Transforma as feature em numericas
dataset = dataset.apply(pd.to_numeric, errors="ignore")

# Split em Train e Test
random_state = 5007
X = dataset.drop('Biopsy', axis=1)
y = dataset['Biopsy']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, shuffle=True, test_size = 0.25, random_state=random_state)

# Eliminação (ou não) de instâncias com missing values
numeric_feat = ['Number of sexual partners',
                   'First sexual intercourse',
                   'Num of pregnancies', 
                   'Smokes (years)',
                   'Smokes (packs/year)',
                   'Hormonal Contraceptives (years)',
                   'IUD (years)',
                   'STDs (number)',
                   'STDs: Time since first diagnosis',
                   'STDs: Time since last diagnosis'] 
cathegoric_feat = ['Smokes',
                 'Hormonal Contraceptives',
                 'IUD',
                 'STDs',
                 'STDs:condylomatosis',
                 'STDs:cervical condylomatosis',
                 'STDs:vaginal condylomatosis',
                 'STDs:vulvo-perineal condylomatosis',
                 'STDs:syphilis',
                 'STDs:pelvic inflammatory disease',
                 'STDs:genital herpes',
                 'STDs:molluscum contagiosum',
                 'STDs:AIDS',
                 'STDs:HIV',
                 'STDs:Hepatitis B',
                 'STDs:HPV']
X_train_processed = X_train.copy()
# Preenche com a mediana
imp_median = SimpleImputer(missing_values = np.nan, strategy = 'median')
X_train_processed[numeric_feat] = imp_median.fit_transform(X_train[numeric_feat])
# Preenche com o valor mais frequente
imp_most_freq = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
X_train_processed[cathegoric_feat] = imp_most_freq.fit_transform(X_train[cathegoric_feat])

# Normalizacao
# Aplica MinMax no grupo de treino
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_normalized = minmax_scale.fit_transform(X_train_processed)
# Transforma de volta para dataframe
X_normalized = pd.DataFrame(X_normalized, columns = X_train_processed.columns.tolist())

# Balanceamento de classes
cc = SMOTETomek(random_state=random_state)
X_train_res, Y_train_res = cc.fit_resample(X_normalized, Y_train)
df_train_normalized = X_train_res.join(Y_train_res)

# Voltando para escala original
X_train_escala_orig = pd.DataFrame(minmax_scale.inverse_transform(X_train_res))
X_train_escala_orig.columns = X_train_res.columns.tolist()
df_train_non_normalized = X_train_escala_orig.join(Y_train_res)

# Separando features e classes
features_normalized = df_train_normalized.iloc[:, 0:35]
classe_normalized = df_train_normalized.iloc[:, 35]
features_non_normalized = df_train_non_normalized.iloc[:, 0:35]
classe_non_normalized = df_train_non_normalized.iloc[:, 35]
#########################################

######## Feature selection

#### Branch and Bound
# Biblioteca: branch-and-bound-feature-selection
# Parametros: qtd de caracteristicas a ser selecionada (usar raiz quadrada do valor total do conjunto de caracteristicas)
# Função criterio: medida de inconsistência (um subconjunto de caracteristicas é dado como inconsistente se possuir instancias com valores iguais de diferentes classes)
# Observacao: Encontra a melhor solução
# Observacao: Mais apropriado para características discretas (sera necessário fazer a discretização de algumas caracteristicas)
branchAndBound = BranchAndBound(features_non_normalized.values, classe_non_normalized.values)
bestFeaturesBnB = branchAndBound.best_subspace(6)
## Resultado
# Tempo de execução: 55s
# Observação: fez sentido a seleção dessas características como mais relevantes
#for col in range(len(bestFeaturesBnB)):
#    print(features_non_normalized.columns[col])
features_non_normalized.columns[bestFeaturesBnB]
# Caracteristicas selecionadas:
# Age
# Number of sexual partners
# First sexual intercourse
# Num of pregnancies
# Hormonal Contraceptives (years)
# Hinselmann

#### Relief F
# Biblioteca: skrebate
# Parametros: qtd decaracteristicas a ser selecionada; numero de neighbors (maior quantidade de neighbors resulta em uma acurácia maior, porém demora mais o processamento)
# Função criterio: distancia euclideana entre neighbors
# Observacao: Essa biblioteca nao utiliza o parametro M (numero de instancias aleatorias), pois utilizando-se todas as instancias obtem-se um resultado melhor
relief = ReliefF(n_features_to_select=10, n_neighbors=200, verbose=True)
bestFeaturesReliefF = relief.fit_transform(features_normalized.values, classe_normalized.values)
## Resultado
# Caracteristicas selecionadas: Hinselmann, Schiller, Citology, Age, Num of pregnances, Hormonal Contraceptives (years), First sexual intercourse, STDs (number), STDs, STDs: Number of diagnosys
# Tempo de execução: 30.77 segundos
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
sfs = sfs.fit(features_non_normalized.values, features_non_normalized.values)
sfs.k_feature_idx_

#### Minimal Redundance Maximum Redundance (MRMR)
pymrmr.mRMR(features_non_normalized, 'MIQ', 10)