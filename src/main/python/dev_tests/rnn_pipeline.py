import os
import pandas as pd
import tensorflow as tf
import time

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.preprocessing import sequence, text

from sklearn import metrics
from sklearn.model_selection import train_test_split


class RnnPipeline:
    
    def __init__(self, datasets_path):
        
        self.datasets_path = datasets_path

    #  ------------------------------------------------------------------------
    
    def load_dataset(self):

        train_csv = os.path.join(
            self.datasets_path, 
            "jigsaw-toxic-comment-train.csv")
        
        valida_csv = os.path.join(
            self.datasets_path, 
            "validation.csv")

        test_csv = os.path.join(
            self.datasets_path, 
            "test.csv")

        train_data = pd.read_csv(train_csv)
        valida_data = pd.read_csv(valida_csv)
        test_data = pd.read_csv(test_csv)

        return train_data, valida_data, test_data 

    #  ------------------------------------------------------------------------

    def filter_dataset(self, data, selected_cols, rows_limit):
        data = data.drop(selected_cols, axis=1)
        data = data.iloc[:rows_limit, :]
        return data

    # -------------------------------------------------------------------------

    def calculate_max_comment_text_len(self, data, column_name):
        return data[column_name].apply(lambda x: len(str(x).split())).max()    

    #  ------------------------------------------------------------------------
    
    def data_preparation(self, train_data):
        
        xtrain, xvalid, ytrain, yvalid = train_test_split(
            train_data.comment_text.values, 
            train_data.toxic.values, 
            stratify=train_data.toxic.values, 
            random_state=42,
            test_size=0.2, 
            shuffle=True)
        
        return xtrain, xvalid, ytrain, yvalid
    
    #  ------------------------------------------------------------------------
    
    def tokenizer(self, xtrain, xvalid, max_len):
        
        # using keras tokenizer here
        token = text.Tokenizer(num_words=None)

        token.fit_on_texts(list(xtrain) + list(xvalid))
        xtrain_seq = token.texts_to_sequences(xtrain)
        xvalid_seq = token.texts_to_sequences(xvalid)

        #zero pad the sequences
        xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
        xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

        word_index = token.word_index
        
        return xtrain_pad, xvalid_pad, word_index
        
    #  ------------------------------------------------------------------------
    
    def get_strategy(self):
        return tf.distribute.get_strategy()

    #  ------------------------------------------------------------------------
    
    def build_model(self, word_index, max_len, strategy):
        
        with strategy.scope():
            # A simpleRNN without any pretrained embeddings and one dense layer
            model = Sequential()
            model.add(Embedding(len(word_index)+1, 300, input_length=max_len))
            model.add(SimpleRNN(100))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(  loss='binary_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'])
        return model
        
    #  ------------------------------------------------------------------------
    
    def train(self, model, xtrain_pad, ytrain, strategy, epochs):
        
        model.fit(  xtrain_pad, ytrain, 
                    epochs=epochs, 
                    batch_size=64*strategy.num_replicas_in_sync)
        
        return model
    
    #  ------------------------------------------------------------------------
    
    def roc_auc(self, predictions, target):
        '''
        This methods returns the AUC Score when given the Predictions
        and Labels
        '''
        fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
        roc_auc = metrics.auc(fpr, tpr)
        return roc_auc    

    #  ------------------------------------------------------------------------
    
    def get_scores(self, model, xvalid_pad):
        scores = model.predict(xvalid_pad)
        return self.roc_auc(scores, yvalid)

    #  ------------------------------------------------------------------------
    
    def get_scores_model(self, auc_score):
        
        scores_model = {
            'Model': 'SimpleRNN',
            'AUC_Score': auc_score
        }
        
        return scores_model
        

if __name__ == "__main__":

    # 設置資料集路徑
    datasets_path = "d:\\kaggle\\input\\jigsaw-multilingual-toxic-comment-classification\\"
    pipeline = RnnPipeline(datasets_path)

    # 讀取資料集
    train_data, valida_data, test_data = pipeline.load_dataset()
    print(f"\n訓練資料筆數: {len(train_data)}")
    print(f"驗證資料筆數: {len(valida_data)}")
    print(f"測試資料筆數: {len(test_data)}")

    # 過濾訓練資料集
    selected_cols = [
        'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    ]
    filtered_train_data = pipeline.filter_dataset(
        train_data, 
        selected_cols,
        12000)
    print(f'\nTrain shape: {filtered_train_data.shape}')
    print(filtered_train_data.index)

    # 計算訓練資料集中 comment_text 欄位的最大單字數
    max_comment_text_len = pipeline.calculate_max_comment_text_len(
        filtered_train_data, 
        'comment_text')
    print(f"\n訓練資料集中 'comment_text' 欄位的最大單字數為: {max_comment_text_len}")

    strategy = pipeline.get_strategy()
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    
    start_time = time.time()
    
    xtrain, xvalid, ytrain, yvalid = pipeline.data_preparation(filtered_train_data)
    
    token_max_len = 1500
    xtrain_pad, xvalid_pad, word_index = pipeline.tokenizer(xtrain, xvalid, token_max_len)
    
    base_model = pipeline.build_model(word_index, token_max_len, strategy)
    base_model.summary()
    
    epochs = 5
    trained_model = pipeline.train(base_model, xtrain_pad, ytrain, strategy, epochs)
    
    auc_score = pipeline.get_scores(trained_model, xvalid_pad)
    print("\nAuc: %.2f%%" % (auc_score))

    scores_model = pipeline.get_scores_model(auc_score)
    print("\n"+ str(scores_model))
    
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"\n執行時間: {exec_time:.6f} 秒")
    
    