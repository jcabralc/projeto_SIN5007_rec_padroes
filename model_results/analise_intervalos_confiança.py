# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:57:24 2020

@author: JessicaCabral
"""

# =============================================================================
# Analise dos Intervalos de Confiaça
# =============================================================================

import pandas as pd
from numpy import mean, inf


# =============================================================================
# separa os ICs e calcula a diferença
# =============================================================================
results = pd.read_excel('C:/git\Mestrado/projeto_SIN5007_rec_padroes/model_results/all_models_results.xlsx')
# 
# ic_up = []
# ic_down = []
# for i in results['Intervalo_Confianca']:
#     ic_up.append(i.split(',')[1].replace(' ', ''))
#     ic_down.append(i.split(',')[2].replace(']', '').replace(' ', ''))
#     
# results['ic_up'] = ic_up
# results['ic_down'] = ic_down
# 
# results['ic_up'] = results['ic_up'].astype(float)
# results['ic_down'] = results['ic_down'].astype(float)
# 
# results['diff_ics'] = results['ic_up'] - results['ic_down'] 
# 
# results.to_excel('C:/git\Mestrado/projeto_SIN5007_rec_padroes/model_results/all_models_results.xlsx', index=False)
# =============================================================================

# =============================================================================
# Transforma as Diferenças dos ICs em Abs
# =============================================================================
results['diff_ics'] = results['diff_ics'].abs()

# =============================================================================
# Função de Analise
# =============================================================================
def analisa_melhor_resultado(df, algoritmo):
    print('----'*20)
    print('Algoritmo: {}'.format(algoritmo))
    print('----'*20)
    df_acc = df[df.Metrica == 'Accuracy']
    df_pre = df[df.Metrica == 'Precision']
    df_rec = df[df.Metrica == 'Recall']
    
    print('\tAcuracia media: {}'.format(df_acc.Valor.mean()))
    print('\t\tICs media: [{},{}]'.format(df_acc.ic_up.mean(), df_acc.ic_down.mean()))
    print('\t\tVariação media entre os ICs: {}'.format(abs(df_acc.ic_up.mean() - df_acc.ic_down.mean())))
    
    print('\tPrecisão media: {}'.format(df_pre .Valor.mean()))
    print('\t\tICs media: [{},{}]'.format(df_pre .ic_up.mean(), df_pre .ic_down.mean()))
    print('\t\tVariação media entre os ICs: {}'.format(abs(df_pre .ic_up.mean() - df_pre .ic_down.mean())))
    
    print('\tRevocação media: {}'.format(df_rec .Valor.mean()))
    print('\t\tICs media: [{},{}]'.format(df_rec .ic_up.mean(), df_rec .ic_down.mean()))
    print('\t\tVariação media entre os ICs: {}'.format(abs(df_rec .ic_up.mean() - df_rec .ic_down.mean())))
    
    # Ordena pela maior Acuracia e pela menor diferenaça entre os Indices de Confiança
    ordena_melhor_acc = df_acc.sort_values('Valor', ascending=False).sort_values('diff_ics', ascending=True)
    ordena_melhor_pre = df_pre .sort_values('Valor', ascending=False).sort_values('diff_ics', ascending=True)
    ordena_melhor_rec = df_rec .sort_values('Valor', ascending=False).sort_values('diff_ics', ascending=True)
    
    # percorre as melhores acuracias ordenadas para achar a melhor precisar e a melhor revocacao
    best_pre = 0
    best_rec = 0
    for cond in ordena_melhor_acc.Condicao:
        precision_value = ordena_melhor_pre[ordena_melhor_pre.Condicao == cond].Valor.values[0]
        recall_value = ordena_melhor_rec[ordena_melhor_rec.Condicao == cond].Valor.values[0]
        # procura o maior valor de precisao e recall
        if  (precision_value > best_pre) and (recall_value > best_rec):
            best_pre = precision_value
            best_rec = recall_value
            best_cond = cond
    
    print('\nMelhores resultados para o {}'.format(algoritmo))
    print('\tMelhor Condição: {}'.format(best_cond))
    print('\t\t Acuracia: {}'.format(df_acc[df_acc.Condicao == best_cond].Valor.values[0]))
    print('\t\t Precisão: {}'.format(best_pre))
    print('\t\t Recall: {}'.format(best_rec))
    

# =============================================================================
# Naive Bayes
# =============================================================================
algoritmo = 'Naive Bayes'
naive_bayes = results[results.Algoritmo == algoritmo]

analisa_melhor_resultado(naive_bayes, algoritmo)


# =============================================================================
# SVM
# =============================================================================
algoritmo = 'SVC'
svm = results[results.Algoritmo == algoritmo]

analisa_melhor_resultado(svm , algoritmo)

# =============================================================================
# MLP
# =============================================================================
algoritmo = 'MLPClassifier'
mlp = results[results.Algoritmo == algoritmo]

analisa_melhor_resultado(mlp , algoritmo)

# =============================================================================
# Random Forest
# =============================================================================
algoritmo = 'RandomForestClassifier'
rf = results[results.Algoritmo == algoritmo]

analisa_melhor_resultado(rf , algoritmo)