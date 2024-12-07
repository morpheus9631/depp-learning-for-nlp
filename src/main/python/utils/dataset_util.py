import os
import pandas as pd

class DatasetUtil():
    
    def __init__(self):
        pass
    
    # -------------------------------------------------------------------------
        
    @staticmethod
    def load_dataset(datasets_path: str):
        """
        讀取訓練、驗證、測試資料集
        :return: 三個 pandas DataFrame：訓練、驗證和測試資料
        """
        train_csv = os.path.join(datasets_path, "jigsaw-toxic-comment-train.csv")
        validation_csv = os.path.join(datasets_path, "validation.csv")
        test_csv = os.path.join(datasets_path, "test.csv")

        train_data = pd.read_csv(train_csv)
        validation_data = pd.read_csv(validation_csv)
        test_data = pd.read_csv(test_csv)

        return train_data, validation_data, test_data 
    
    # -------------------------------------------------------------------------
    
    @staticmethod
    def filter_dataset(data, selected_cols, rows_limit):
        """
        過濾資料集：移除指定欄位並限制行數
        :param data: 要過濾的資料集
        :param selected_cols: 要移除的欄位
        :param rows_limit: 限制的最大行數
        :return: 過濾後的資料集
        """
        filtered_data = data.drop(selected_cols, axis=1)
        filtered_data = filtered_data.iloc[:rows_limit, :]
        return filtered_data

    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_max_length(data, column_name):
        """
        計算指定欄位的最大值（例如最長文字的單字數）
        :param data: 資料集
        :param column_name: 欄位名稱
        :return: 欄位資料的最大值
        """
        return data[column_name].apply(lambda x: len(str(x).split())).max()    
    
    
# 使用範例
if __name__ == "__main__":

    # 設置資料集路徑
    datasets_path = "d:\\kaggle\\input\\jigsaw-multilingual-toxic-comment-classification\\"

    # 讀取資料集
    train_data, validation_data, test_data = DatasetUtil.load_dataset(datasets_path)
    print(f"\n訓練資料筆數: {len(train_data)}")
    print(f"驗證資料筆數: {len(validation_data)}")
    print(f"測試資料筆數: {len(test_data)}")

    # 過濾訓練資料集
    selected_cols = [
        'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    ]
    filtered_train_data = DatasetUtil.filter_dataset(
        train_data, 
        selected_cols, 
        rows_limit=12000)
    print(f'\Train shape: {filtered_train_data.shape}')

    # 計算訓練資料集中 comment_text 欄位的最大單字數
    max_length = DatasetUtil.calculate_max_length(
        filtered_train_data, 
        'comment_text')
    print(f"\n訓練資料集中 'comment_text' 欄位的最大單字數為: {max_length}")