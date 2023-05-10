import pandas as pd
import numpy as np
import nltk
# import matplotlib.pyplot as plt
import unidecode
import difflib

from wordcloud import WordCloud
from nltk import tokenize
from nltk.corpus import stopwords 
from string import punctuation

pd.options.display.max_rows = 10
pd.set_option('display.expand_frame_repr', False)

lista_irrelevante = set(stopwords.words('portuguese'))

juntar = ''

palavras_irrelevantes_em_lista = list()
for irrelevante in lista_irrelevante:
    palavras_irrelevantes_em_lista.append(irrelevante)

pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)

palavras_irrelevantes = palavras_irrelevantes_em_lista + pontuacao

#print("Digite a frase : ")

#resenha = [input()]

def processar_string_input(valor_digitado):
    frase_processada = list()
    
    token_espaco = tokenize.WhitespaceTokenizer()
    for opiniao in valor_digitado:
        nova_frase = list()
        opiniao = opiniao.lower()
        palavras_texto = token_espaco.tokenize(opiniao)
        
        for palavra in palavras_texto:
            if palavra not in palavras_irrelevantes:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))
       
    resenha = frase_processada
    
    frase_processada2 = list()
    token_pontuacao = tokenize.WordPunctTokenizer()
    for tratamento1 in resenha:
            nova_frase2 = list()
            palavras_texto2 = token_pontuacao.tokenize(tratamento1)
            for palavra2 in palavras_texto2:
                nova_frase2.append(palavra2)
            frase_processada2.append(' '.join(nova_frase2))
            
    resenha = frase_processada2
    
    frase_processada3 = list()
    for tratamento2 in resenha:
            nova_frase3 = list()
            palavras_texto3 = unidecode.unidecode(tratamento2)
            for palavra3 in palavras_texto3:
                nova_frase3.append(palavra3)
            frase_processada3.append(''.join(nova_frase3))
            
    resenha = frase_processada3
    #
    token_espaco = tokenize.WhitespaceTokenizer()
    stemmer = nltk.RSLPStemmer()
    texto_contraido_palavras = list()
    for texto in resenha:
        novo_texto = list()
        palavra_tokenizada = token_espaco.tokenize(texto)
        for w in palavra_tokenizada:
            novo_texto.append(stemmer.stem(w))
        texto_contraido_palavras.append(' '.join(novo_texto))
                
    resenha = texto_contraido_palavras
    return resenha
#    
# print(processar_string_input(["preciso instalar o mozilla"]))




