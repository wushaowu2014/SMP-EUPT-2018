# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:24:03 2018

@author: wushaowu
"""
import os
import codecs
import jieba
import json
import pandas as pd
import numpy as np
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import  pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding,Input, Convolution1D, MaxPooling1D, Flatten, Dense, Bidirectional,concatenate,GRU,Dropout,LSTM
from keras.models import Model,Sequential
from keras.engine import Layer, InputSpec
from keras.layers import Flatten,BatchNormalization,PReLU,add,average,multiply,maximum
from keras.layers import Conv1D,Activation,MaxPool1D,merge,Lambda
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping,LearningRateScheduler
from sklearn.cross_validation import StratifiedKFold
def f1(y_true,y_pred):
	def recall(y_true,y_pred):
		"""Recall metric.
		Only computes a batch-wise average of recall.
		Computes the recall, a metric for multi-label classification of
		how many relevant items are selected.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
		possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall
	
	def precision(y_true,y_pred):
		"""Precision metric.
		Only computes a batch-wise average of precision.
		Computes the precision, a metric for multi-label classification of
		how many selected items are relevant.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	
	precisions = precision(y_true,y_pred)
	recalls = recall(y_true,y_pred)
	return 2 * ((precisions * recalls) / (precisions + recalls))

def C_RNN_series(vocab_size,max_len,embedding_size):
    model = Sequential()
    model.add(Embedding(vocab_size,embedding_size,input_length=max_len))
    model.add(Convolution1D(32,3,padding='same',strides=1))
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))

    model.add(GRU(32,implementation=2,return_sequences=True, go_backwards=False))
    model.add(GRU(32,implementation=2,return_sequences=True, go_backwards=True))
    model.add(Bidirectional(LSTM(32,return_sequences=True),merge_mode='concat'))
    model.add(Flatten())
    model.add(Dense(4,activation='softmax'))
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",metrics=["accuracy",f1])
    return model

def bi_lstm(vocab_size,max_len,embedding_size):
    main_input = Input(shape=(max_len,))
    embed = Embedding(vocab_size,embedding_size,input_length=max_len)(main_input)
    convs = []
    filter_sizes=[5,3]
    for fsz in filter_sizes:
        conv = Convolution1D(32, kernel_size= fsz, padding='same',activation='relu')(embed)
        pool = MaxPooling1D(4)(conv)
        convs.append(pool)
    # merge1 = merge(convs, mode='concat', concat_axis=1)
    merge1 = concatenate(convs, axis=1)
    
    dw_z_pos = LSTM(32, implementation=2, return_sequences=True, go_backwards=False)(merge1)
    dw_z_neg = LSTM(32, implementation=2, return_sequences=True, go_backwards=True)(merge1)
    dw_z_concat = merge([dw_z_pos, dw_z_neg], mode='concat', concat_axis=-1)
    rnn = Dense(32,activation='relu')(dw_z_concat)
    pool_rnn = Lambda(lambda x: K.max(x, axis=1), output_shape=(32,))(rnn)
    #flt = Flatten()(pool_rnn)
    main_output = Dense(4,activation='softmax')(pool_rnn)
    model = Model(inputs=main_input,outputs=main_output)
    #print(model.summary())
    
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy",f1])    
    return model

def stacking(train_x,train_y,test,model,flag):
    n_folds=5
    print(str(n_folds)+'_folds_stacking')
    stack_train = np.zeros((len(train_y), 4))
    stack_test = np.zeros((test.shape[0], 4))
    score_va = 0
    train_y=train_y.astype(int)
    for i, (tr, va) in enumerate(StratifiedKFold(train_y, n_folds=n_folds, random_state=2018)):
        print('stack:%d/%d' % ((i + 1), n_folds))
        early_stopping = EarlyStopping(monitor="loss", patience=2)
        callbacks_list = [early_stopping]
        model.fit(train_x[tr],to_categorical(train_y[tr]),
                validation_split=0.0,
                batch_size=batch_size,
                epochs=epochs,
                shuffle = True,
                callbacks=callbacks_list)

        score_va = model.predict(train_x[va])
        score_te = model.predict(test)
        stack_train[va] += score_va
        stack_test += score_te
    stack_test /= n_folds
    df_stack_train = pd.DataFrame()
    df_stack_test = pd.DataFrame()
    for i in range(stack_test.shape[1]):
        df_stack_train[flag+str(n_folds)+'_classfiy_{}'.format(i)] = stack_train[:, i]
        df_stack_test[flag+str(n_folds)+'_classfiy_{}'.format(i)] = stack_test[:, i]
    df_stack_train.to_csv(flag+'_stack_'+str(n_folds)+'zhe_train_feat.csv', index=None, encoding='utf8')
    df_stack_test.to_csv(flag+'_stack_'+str(n_folds)+'zhe_test_feat.csv', index=None, encoding='utf8')
    print(str(n_folds)+'_folds_stacking特征已保存\n')

def pre_model(train_x,train_y):
    inputs=Input(shape=(train_x.shape[1],))
    
    x1 = Dense(5, activation='tanh')(inputs)
    x2 = Dense(5, activation='relu')(inputs)
    x=multiply([x1,x2])
    
    x = Dense(20, activation='tanh')(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=#'rmsprop',

                  Adam(lr=0.0001, epsilon=1e-09, decay=0.0),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(train_x, train_y,epochs=10, batch_size=128, validation_split=0.1,)
    return model
def get_keras_data(dataset,max_len):# 序列填充
    X={
        'COMMCONTENT_seq':pad_sequences(dataset.COMMCONTENT_seq,maxlen=max_len)
    }
    return X
def readdata():
    """读入原始数据，转换成中文并保存"""
    train= pd.read_csv('training_new.txt',sep='\t',header=None)
    train=train[0].apply(lambda x:json.loads(x))
    test= pd.read_csv('testing.txt',sep='\t',header=None)
    test=test[0].apply(lambda x:json.loads(x))
    
    traindata=[]
    for i in train:
        traindata.append([i['标签'],str(i['内容']).replace('\\r',' ',10).replace('\r',' ',10),\
                          i['id']])
    traindata=pd.DataFrame(traindata)
    traindata.columns=['标签','内容','id']
    traindata.to_csv('traindata.csv',index=None,encoding='utf8')

    testdata=[]
    for i in test:
        testdata.append([i['id'],str(i['内容']).replace('\\r',' ',10).replace('\r',' ',10)])
    testdata=pd.DataFrame(testdata)
    testdata.columns=['id','内容']
    testdata.to_csv('testdata.csv',index=None,encoding='utf8')
def split_word(text, stopwords):
    """去掉停用词"""
    word_list = jieba.cut(text)
    start = True
    result = ''
    for word in word_list:
        word = word.strip()
        if word not in stopwords:
            if start:
                result = word
                start = False
            else:
                result += ' ' + word
    return result      
def split_word_not_stopwords(text):
    """不取停用词"""
    cut_str = jieba.cut(text)
    return [i for i in cut_str]
def split_word_single(text):
    """字符级分割"""
    result = ' ' + str(list(text))
    return result
def loaddata():
    """分词后并保存"""
    traindata=pd.read_csv('traindata.csv')
    testdata=pd.read_csv('testdata.csv')
    #读入停用词：
    stopwords = []
    for line in codecs.open('stop_words.txt', 'r', 'gbk'):
        stopwords.append(line.split('\r')[0])
    
    ###保存分词后的数据：
    traindata['内容']=traindata['内容'].apply(lambda x: split_word_not_stopwords(str(x)))
    traindata.to_csv('traindata_words_with_stopwords.csv', index=False,encoding='utf8')
    testdata['内容']=testdata['内容'].apply(lambda x: split_word_not_stopwords(str(x)))
    testdata.to_csv('testdata_words_with_stopwords.csv', index=False,encoding='utf8')
if __name__=='__main__':
    if not os.path.exists('traindata_words_with_stopwords.csv'):
        """如果还没分词，就分词"""
        readdata()
        loaddata()
    
    max_len=200 #序列长度
    vocab_size=5000 #最大特征
    embedding_size=64 
    batch_size=128
    epochs=5
    
    #读入分词后的数据
    train= pd.read_csv('traindata_words_with_stopwords.csv')
    test= pd.read_csv('testdata_words_with_stopwords.csv')
    
    #缺失值处理
    train['内容'].fillna('_na_',inplace=True)
    test['内容'].fillna('_na_',inplace=True)
    
    #生成文档词典
    comment_text = np.hstack((list(train['内容'].values)+list(test['内容'].values)))
    tok_raw = Tokenizer(num_words=vocab_size,filters='#$&*+-/<=>@\^_`{|}\t\n',
                                   lower=True)
    tok_raw.fit_on_texts((comment_text))
    train['COMMCONTENT_seq']= tok_raw.texts_to_sequences((train['内容']))
    test['COMMCONTENT_seq'] = tok_raw.texts_to_sequences((test['内容']))

    ##打乱顺序
    from sklearn.utils import shuffle
    train=shuffle(train)
    
    #对标签编码
    y=train['标签']
    y.to_csv('label.csv',index=None,encoding='utf8')
    y=pd.read_csv('label.csv',header=None,encoding='utf8')
    from sklearn import preprocessing
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(y.values))
    yy= lbl.transform(list(y.values))

    
    #第二个模型做5折stacking
    model_C_RNN_series =C_RNN_series(vocab_size,max_len,embedding_size)
    model_C_RNN_series.summary()
    stacking(get_keras_data(train,max_len)['COMMCONTENT_seq'],yy,\
             get_keras_data(test,max_len)['COMMCONTENT_seq'],model_C_RNN_series,'model_C_RNN_series')
    

    

    #第四个模型做5折stacking
    model_bi_lstm =bi_lstm(vocab_size,max_len,embedding_size)
    model_bi_lstm.summary()
    stacking(get_keras_data(train,max_len)['COMMCONTENT_seq'],yy,\
             get_keras_data(test,max_len)['COMMCONTENT_seq'],model_bi_lstm,'model_bi_lstm')
    


    #######################模型预测###########################################
    #读取stacking特征
    new_train_feat1= pd.read_csv('model_C_RNN_series_stack_5zhe_train_feat.csv')
    new_test_feat1= pd.read_csv('model_C_RNN_series_stack_5zhe_test_feat.csv')
    new_train_feat2= pd.read_csv('model_bi_lstm_stack_5zhe_train_feat.csv')
    new_test_feat2= pd.read_csv('model_bi_lstm_stack_5zhe_test_feat.csv')
    #下面进入队友的数据：
    new_train_feat3= pd.read_csv('model_textcnn_stack_5zhe_train_feat.csv')
    new_test_feat3= pd.read_csv('model_textcnn_stack_5zhe_test_feat.csv')
    new_train_feat4= pd.read_csv('model_C_RNN_parallel_stack_5zhe_train_feat.csv')
    new_test_feat4= pd.read_csv('model_C_RNN_parallel_stack_5zhe_test_feat.csv')
    
    #结合以上训练特征，测试特征
    new_tr=pd.concat([new_train_feat1,new_train_feat2,new_train_feat3,new_train_feat4],axis=1)
    new_te=pd.concat([new_test_feat1,new_test_feat2,new_test_feat3,new_test_feat4],axis=1)
    #模型训练

    model=pre_model(new_tr,to_categorical(yy)) #调用pre_model函数
    pre=model.predict(new_te)
    #print(Counter(np.argmax(pre,axis=1)))
    model.save('model.h5') ##保存模型，下次可以直接调用
    
    #保存结果
    preds=np.argmax(pre,axis=1)
    result=list(lbl.inverse_transform(preds)) #转码
    res = pd.DataFrame()
    res['id'] =test['id']
    res['RST'] = result
    res.to_csv('final.csv',index=None,header=None,encoding='utf8')