import numpy as np
from Dataprocessor import Dataprocessor

class Hopfiled:
    def __init__(self,train_url,test_url) -> None:
        self.train_url = train_url
        self.test_url = test_url

    #Before training, need to calulate w and theta
    #Then train
    def train(self,train_data):
        #train_data,_,_ = Dataprocessor.convert_to_row(url) # get training data

        #calculate w
        dim = len(train_data[0])
        input_size = len(train_data)

        #initial w and theta
        w = np.zeros(dim * dim).reshape(dim,dim)
        theta = np.zeros(dim)

        #cal w and theta
        for single_train in train_data:
            re_sgl_train = np.array(single_train).reshape(dim,1)
            w = w + ((1 / dim) * (re_sgl_train.dot(re_sgl_train.T)))
        w = w - ((input_size / dim) * np.eye(dim))
        for w_pos in range(len(w)):
            theta[w_pos] = np.sum(w[w_pos])

        self.w = w
        self.theta = theta
        self.dim = dim

        

    # test
    def test(self,epochs,test_data,row_num,col_num):
        #這三行應該要從外面傳入，不應該寫在這裡
        #w,theta,dim = self.train(train_url)
        #test_data,row_num,col_num = Dataprocessor.convert_to_row(self.test_url)

        # ans will be stored in result
        predict = []

        # test
        for num in range(len(test_data)):
            for _ in range(epochs):  # every test_data will iterate epoch times
                for pos in range(len(test_data[num])): 
                    re_w = np.array(self.w[pos]).reshape(self.dim,1).transpose()     # w.T
                    re_test_data = np.array(test_data[num]).reshape(self.dim,1) # x
                    sgn = np.sign(re_w.dot(re_test_data) - self.theta[pos])     # w.T * x - theta
                    test_data[num][pos] = sgn

            predict.append(np.array(test_data[num]).reshape(row_num,col_num)) #12 9 還需要改
        return predict