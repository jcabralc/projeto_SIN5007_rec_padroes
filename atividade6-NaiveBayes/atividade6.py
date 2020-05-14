import pandas as pd
import numpy as np
# Preprocessing
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
# Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
# Stratified Cross Validation
from sklearn.model_selection import StratifiedKFold
# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# 1) Listar todas as features categoricas e numericas (está faltando algumas)
numeric_features = ['Number of sexual partners',
                    'First sexual intercourse',
                    'Num of pregnancies', 
                    'Smokes (years)',
                    'Smokes (packs/year)',
                    'Hormonal Contraceptives (years)',
                    'IUD (years)',
                    'STDs (number)',
                    'STDs: Time since first diagnosis',
                    'STDs: Time since last diagnosis'] 
cathegoric_features = ['Smokes',
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

# 2) Salvar os seguintes datasets: df_componentes_principais.csv, df_selecionador1.csv, df_selecionador2.csv

######## Pre-processing
def preprocessing(dataset):
    #### Datatypes
    # Substitui ? by NAN
    dataset.replace('?', np.nan, inplace = True)
    # Transforma as feature em numericas
    dataset = dataset.apply(pd.to_numeric, errors="ignore")

    #### Inicializa variaveis
    X = dataset.drop('Biopsy', axis=1)
    Y = dataset['Biopsy']

    #### Eliminação de instâncias com missing values
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
    X_processed = X.copy()
    # Preenche com a mediana
    imp_median = SimpleImputer(missing_values = np.nan, strategy = 'median')
    X_processed[numeric_feat] = imp_median.fit_transform(X[numeric_feat])
    # Preenche com o valor mais frequente
    imp_most_freq = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    X_processed[cathegoric_feat] = imp_most_freq.fit_transform(X[cathegoric_feat])

    #### Retorna variaveis
    return X_processed.values, Y.values
#########################################


######## Naive Bayes Cross Validation

datasets = [] ## Lista de datasets
alpha = [0.1, 0.5, 1]
k = 4

# Para cada dataset
for dataset in datasets:

    dataset = pd.read_csv('../data/kag_risk_factors_cervical_cancer.csv')

    X, Y = preprocessing(dataset)

    acuracia = [[0 for item in range(k)] for item in range(len(alpha))]
    precisao = [[0 for item in range(k)] for item in range(len(alpha))]
    revocacao = [[0 for item in range(k)] for item in range(len(alpha))]
    acuracia_media = [[0 for item in range(k)] for item in range(len(alpha))]
    precisao_media = [[0 for item in range(k)] for item in range(len(alpha))]
    revocacao_media = [[0 for item in range(k)] for item in range(len(alpha))]

    # Divide o dataset em K folds
    skf = StratifiedKFold(n_splits=k, shuffle=True)

    # Para cada fold
    i_fold = 0
    for train_index, test_index in skf.split(X, Y):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Para cada valor de c
        for alpha_param in range(len(alpha)):
            classifier_gnb = GaussianNB(var_smoothing=alpha_param)
            classifier_bnb = BernoulliNB(alpha=alpha_param)
            
            # Treinar classificador numerico
            Y_pred_gnb = classifier_gnb.fit(X_train[numeric_features], Y_train)
            
            # Treinar classificador categorico
            Y_pred_bnb = classifier_bnb.fit(X_train[cathegoric_features], Y_train)

            # Obter probabilidades
            X_probs = pd.DataFrame(
                            np.hstack((
                                Y_pred_gnb.predict_proba(X_train[numeric_features]), 
                                Y_pred_bnb.predict_proba(X_train[cathegoric_features]))),
                            columns = ['0_G','1_G','0_B','1_B'])    
            X_probs_test = pd.DataFrame(
                            np.hstack((
                                Y_pred_gnb.predict_proba(X_test[numeric_features]), 
                                Y_pred_bnb.predict_proba(X_test[cathegoric_features]))), 
                            columns = ['0_G','1_G','0_B','1_B'])
            
            # Treinar classificador com as probabilidades
            Y_pred = classifier_gnb.fit(X_probs, Y_train).predict(X_probs_test)
            
            # Calcular metricas
            acuracia[alpha_param][i_fold] += accuracy_score(Y_test, Y_pred)
            precisao[alpha_param][i_fold] += precision_score(Y_test, Y_pred)
            revocacao[alpha_param][i_fold] += recall_score(Y_test, Y_pred)
            
        i_fold += 1

    # Para cada valor de c
    for alpha_param in range(len(alpha)):
        acuracia_media[alpha_param] = sum(acuracia[alpha_param]) / k
        precisao_media[alpha_param] = sum(precisao_media[alpha_param]) / k
        revocacao_media[alpha_param] = sum(revocacao_media[alpha_param]) / k

######## Naive Bayes Cross Validation (Com balanceamento)
# ...