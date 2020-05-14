import pandas as pd
import numpy as np
# Preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
# Classifier
from sklearn.naive_bayes import GaussianNB
# Stratified Cross Validation
from sklearn.model_selection import StratifiedKFold
# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

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
Y = dataset['Biopsy']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size = 0.25, random_state=random_state)

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
#########################################

X = X_train_processed
Y = Y_train

######## Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train_processed.values, Y_train.values)

######## Stratified Cross Validation
random_state = 5007
k = 4

stratified_k_fold(X, Y, k, random_state)
stratified_k_fold_SMOTE(X, Y, k, random_state)

def stratified_k_fold(X, Y, k, random_state, shuffle=False):
    # Quantidade original de classes
    count_classes = Y.value_counts()
    
    skf = StratifiedKFold(n_splits=k, shuffle=False, random_state=random_state)
    print('k = {}, Dataset {} positivas e {} negativas ({:.2f}% x {:.2f}%)'.format(k, count_classes[0], 
                                                                                      count_classes[1], 
                                                                                      ((count_classes[0]/len(Y))*100), 
                                                                                      ((count_classes[1]/len(Y))*100)))

    for fold in enumerate(skf.split(X, Y)):
        fold_number = fold[0] + 1
        train_index = fold[1][0]
        test_index = fold[1][1]
        # Quantidade de classes dentro da fold
        count_classes_fold = Y.iloc[test_index].value_counts()
        # Proporções
        prop_pos = ((count_classes_fold[0]/count_classes_fold.sum())*100)
        prop_neg = ((count_classes_fold[1]/count_classes_fold.sum())*100)
        print('Fold {}: Pos: {}, Neg: {}, Total: {}, Proporção: {:.2f}% x {:.2f}%'.format(fold_number, 
                                                                            count_classes_fold[0],
                                                                            count_classes_fold[1], 
                                                                            count_classes_fold.sum(),
                                                                            prop_pos, prop_neg))

def stratified_k_fold_SMOTE(X, Y, k, random_state, shuffle=False):
    # Quantidade original de classes
    count_classes = Y.value_counts()
    
    skf = StratifiedKFold(n_splits=k, shuffle=False, random_state=random_state)
    print('k = {}, Dataset {} positivas e {} negativas ({:.2f}% x {:.2f}%)'.format(k, count_classes[0], 
                                                                                      count_classes[1], 
                                                                                      ((count_classes[0]/len(Y))*100), 
                                                                                      ((count_classes[1]/len(Y))*100)))
    cc = SMOTETomek(random_state=random_state)

    for fold, (train_index, test_index) in enumerate(skf.split(X, Y), 1):
        fold_number = fold
        X_train, Y_train = X.iloc[train_index], Y.iloc[train_index]
        X_test, Y_test = X.iloc[test_index], Y.iloc[test_index]

        # SMOTETomek
        X_train, Y_train = cc.fit_resample(X_train, Y_train)

        # Quantidade de classes dentro da fold
        count_classes_fold = Y_test.value_counts()
        
        print('Fold {}, Dataset (balanceado) {} positivas e {} negativas ({:.2f}% x {:.2f}%)'.format(fold_number, 
                                                                                 Y_train.value_counts()[0], 
                                                                                 Y_train.value_counts()[1], 
                                                                                 ((Y_train.value_counts()[0]/len(Y_train))*100), 
                                                                                 ((Y_train.value_counts()[1]/len(Y_train))*100)))
        
        # Proporcoes
        prop_pos = ((count_classes_fold[0]/count_classes_fold.sum())*100)
        prop_neg = ((count_classes_fold[1]/count_classes_fold.sum())*100)
        print('\tPos: {}, Neg: {}, Total: {}, Proporção: {:.2f}% x {:.2f}%'.format(fold_number, 
                                                                            count_classes_fold[0],
                                                                            count_classes_fold[1], 
                                                                            count_classes_fold.sum(),
                                                                            prop_pos, prop_neg))




######################################

c = [0.1, 0.5, 1]
k = 4

# Para cada dataset

# Dataset
X = dataset.iloc[:, 0:35].values
Y = dataset.iloc[:, 35].values

acuracia = [[0 for item in range(k)] for item in range(len(c))]
precisao = [[0 for item in range(k)] for item in range(len(c))]
revocacao = [[0 for item in range(k)] for item in range(len(c))]
acuracia_media = [[0 for item in range(k)] for item in range(len(c))]
precisao_media = [[0 for item in range(k)] for item in range(len(c))]
revocacao_media = [[0 for item in range(k)] for item in range(len(c))]

# Divide o dataset em K folds
skf = StratifiedKFold(n_splits=k, shuffle=True)

# Para cada fold
i_fold = 0
for train_index, test_index in skf.split(X, Y):

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Para cada valor de c
    for c_param in range(len(c)):
        #classifier = GaussianNB(var_smoothing=c_param)
        # Treinar classificador
        #classifier.fit(X_train, Y_train)
        # Testar classificador
        Y_pred = Y_train
        # Calcular metricas
        acuracia[c_param][i_fold] += accuracy_score(Y_train, Y_pred)
        precisao[c_param][i_fold] += precision_score(Y_train, Y_pred)
        revocacao[c_param][i_fold] += recall_score(Y_train, Y_pred)
        
    i_fold += 1

# Para cada valor de c
for c_param in range(len(c)):
    acuracia_media[c_param] = sum(acuracia[c_param]) / k
    precisao_media[c_param] = sum(precisao_media[c_param]) / k
    revocacao_media[c_param] = sum(revocacao_media[c_param]) / k




# Para cada dataset
    # Divide o dataset em K folds
    
    # Para cada fold
        # Acuracia[c], Precisao[c], Revocao[c] = 0
        # Para cada valor de c
            # Treinar classificador com os outros folds
            # Fazer a classificacao de teste com o fold atual
            # Calcular acuracia, revocacao e precisao
        # Acuracia[c], Precisao[c], Revocao[c]
    
    # Para cada valor de c
        # Acuracia[c] = Acuracia[c] / tamanho de K
        # Precisao[c] = Precisao[c] / tamanho de K
        # Revocao[c] = Revocao[c] / tamanho de K