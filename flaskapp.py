import nltk
import pandas as pd
from flask import Flask, render_template, request, redirect, session, flash, jsonify, current_app

from nltk.stem.lancaster import LancasterStemmer
nltk.download('stopwords')
nltk.download('rslp')
stemmer = LancasterStemmer()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

from mysql.connector import Error
import mysql.connector

import limpeza_de_string

from datetime import datetime

app = Flask(__name__)

@app.route('/')
def hello_world():
    #return 'Hello from Flask! do /var/www/flaskapp/flaskapp.py teste'
    return render_template('chat.html', titulo='Olá bem vindo ao chatbot da Secretaria de Educação!')

@app.route('/criar', methods=['POST', ])
def criar():
    pergunta = ''

    pergunta_do_usuario = 'A sua pergunta foi : '

    resposta_do_sistema = 'Resposta : '

    pergunta = request.form['pergunta']

    # =============================================================================
    # Pegando o ip de quem pergunta
    # =============================================================================

    endip = request.environ['REMOTE_ADDR']

    # =============================================================================
    #   pegando a data e hora de quem perguntaa
    # =============================================================================
    DateTime = datetime.now()

    dia = DateTime.strftime("%d/%m/%Y")
    hora = DateTime.strftime("%H:%M:%S")

    # =============================================================================
    #   chamada para o chatbot
    # =============================================================================

    resposta = chat(pergunta, endip, dia, hora)

    mycursor = config.cursor()

    print(resposta)

    if resposta.find('Não entendi a pergunta') > -1:
        print('if nas respostas linha 68')
        sql = "SELECT a.link FROM ocs.tbl_chatbot_respostas as a where a.respostas like 'Não entendi a pergunta%'"
    else:
        print('else nas respostas linha 71')
        sql = "SELECT a.link FROM ocs.tbl_chatbot_respostas as a where a.respostas like '" + resposta + "%'"

    #print(sql)

    mycursor.execute(sql)

    myresult = mycursor.fetchone()
    link_do_site = " "
    #Fiz aqui quando nao tem link no site
    try:
        link_do_site = myresult[0]
    except:
        link_do_site = " "
    # print(resposta)

    return uniao(pergunta_do_usuario, pergunta, resposta_do_sistema, resposta, link_do_site)


def uniao(pergunta_do_usuario, pergunta, resposta_do_sistema, resposta, link_do_site):
    return render_template('chat.html', titulo='Olá bem vindo ao chatbot da Secretaria de Educação',
                           pergunta_do_usuario=pergunta_do_usuario,
                           pergunta=pergunta, resposta_do_sistema=resposta_do_sistema, resposta=resposta,
                           link_do_site=link_do_site)
    # return redirect('chat.html',pergunta = pergunta, resposta = resposta)

#dados = '/var/www/flaskapp/json/arquivo.json'
dados = "/home/jardelsewo.seed/Documentos/arquivos_chatbot_seed/json/arquivo.json"

config = mysql.connector.connect(
      host="127.0.0.1",
      user="jards",
      passwd="123",
      database="ocs",
      connection_timeout=3600
    )

words = []
labels = []
docs_x = []
docs_y = []

#--------------------------------------------------------------------------------------------------
#coloquei o encoding que estava dando erro para ler o arquivo
#--------------------------------------------------------------------------------------------------
with open(dados, encoding='utf-8') as file:
    data = json.load(file)

for intent in data["arquivo"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

#--------------------------------------------------------------------------------------------------
#passei o caminho completo da pasta, e dei as permissoes 777 nesse arquivo
#--------------------------------------------------------------------------------------------------
with open("/var/www/flaskapp/data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)
    f.close()

tensorflow.compat.v1.reset_default_graph()


net = tflearn.input_data(shape=[None, len(training[0])])
#net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 16)
#net = tflearn.fully_connected(net, 32)
#net = tflearn.fully_connected(net, 64)
#net = tflearn.fully_connected(net, 128)
#net = tflearn.fully_connected(net, 128)
#net = tflearn.fully_connected(net, 64)
#net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 16)
#net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#model.fit(training, output, n_epoch=980, batch_size=8, show_metric=True)

#try:
#aqui esta o erro no tomcat
#model.load("/var/www/chatbot_flask/modelos_tflearn_salvos/model.tflearn")
model.load("/home/jardelsewo.seed/Documentos/arquivos_chatbot_seed/modelos_tflearn_salvos/model.tflearn")


#app = Flask(__name__)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def log_tabela_pergunta_resposta(val, pergunta_original, resposta, enderecoip, dia, hora):
    mycursor = config.cursor()

    sql = "INSERT INTO `ocs`.`log_tbl_pergunta_resposta` (`pergunta`,`pergunta_sem_o_filtro`, `resposta`, `ip_quem_perguntou`, `dia`, `hora`) VALUES ("+ val +",'" + pergunta_original + "', '" + resposta + "' ,'" + enderecoip + "','" + dia + "','" + hora + "');"

    #print(sql)

    mycursor.execute(sql)

    config.commit()

def insere_pergunta_sem_resposta(pergunta):
    boleano = '0'
    mycursor = config.cursor()

    val = "" + pergunta.lower() + ""

    #sql = "INSERT INTO ocs.tbl_chatbot_para_classificar(texto_para_classificar) VALUES (" + val + ")"
    sql2 = "INSERT INTO ocs.tbl_chatbot_perguntas(perguntas, flag_ja_classificado) VALUES (" + val + "," + boleano + ")"

    #print(sql2)

    mycursor.execute(sql2)

    config.commit()

    #print('Não entendi a pergunta, qualquer duvida abra chamado pelo Seguinte link!!!')
    #resposta = 'Não entendi a pergunta, qualquer duvida abra chamado pelo Seguinte link!!!'

    #log_tabela_pergunta_resposta(val, pergunta_original, resposta, enderecoip, dia, hora)

def chat(pergunta, enderecoip, dia , hora):

    pergunta_original = pergunta
    comando_sair = ('sair', '3', 'quit', 'exit')

    #print("Digite sair para sair do programa!!")
    while True:
        #valor_digitado = limpeza_de_string.processar_string_input([input("Você : ")])
        valor_digitado = limpeza_de_string.processar_string_input([pergunta])

        valida = valor_digitado[0]

        if len(valida) < 4:
            inp = pergunta
        else:
            inp = valida

        #print(inp)

#        if inp.lower() == "sair" or inp.lower() == 3:
        if inp in comando_sair:
            break

        results = model.predict([bag_of_words((inp), words)])[0]

        results_index = numpy.argmax(results)
        tag = labels[results_index]

        print('taxa de confianca {}%'.format(round(results[results_index], 2)*100))
        print('tag {}'.format(tag))
        if results[results_index] > 0.9 :

            for tg in data["arquivo"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print("esta no if linha 275 : {}".format(responses))

            if not random.choice(responses):
                resposta = 'Não entendi a pergunta, qualquer duvida abra chamado pelo Seguinte link!!!'

            else:
                resposta = random.choice(responses)
                # resposta = responses

                val = "'" + inp.lower() + "'"

                if resposta.find('Não entendi a pergunta') == 0:
                    # print('if if')
                    insere_pergunta_sem_resposta(val)
                else:
                    # print('else if')
                    log_tabela_pergunta_resposta(val, pergunta_original, resposta, enderecoip, dia, hora)

            return resposta


        else :

            print('esta no else linha 299')
            boleano = '0'
            mycursor = config.cursor()

            val = "'" + inp.lower() + "'"

            #sql = "INSERT INTO ocs.tbl_chatbot_para_classificar(texto_para_classificar) VALUES (" + val + ")"
            sql2 = "INSERT INTO ocs.tbl_chatbot_perguntas(perguntas, flag_ja_classificado) VALUES (" + val + "," + boleano + ")"

            #print(sql2)

            mycursor.execute(sql2)

            config.commit()

            #print('Não entendi a pergunta, qualquer duvida abra chamado pelo Seguinte link!!!')
            resposta = 'Não entendi a pergunta, qualquer duvida abra chamado pelo Seguinte link!!!'

            log_tabela_pergunta_resposta(val, pergunta_original, resposta, enderecoip, dia, hora)

            return resposta

if __name__ == '__main__':
    app.run()
