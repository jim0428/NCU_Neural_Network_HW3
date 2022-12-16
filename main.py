import tkinter as tk
from tkinter import filedialog

from Dataprocessor import Dataprocessor
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from Hopfield import Hopfiled
import numpy as np

def train_model(train_finish):
    global model,train_data
    train_data,_,_ = Dataprocessor.convert_to_row(train_file_url) # get training data
    model = Hopfiled(train_file_url,test_file_url)

    model.train(train_data)
    
    #可以改成用try catch看回傳值決定輸出結果
    train_finish.set("訓練完成")
   
def test_predict(interation,test_finish):
    global origin_data,predict,row_num,col_num
    test_data,row_num,col_num = Dataprocessor.convert_to_row(test_file_url)
    origin_data = test_data.copy()

    predict = model.test(interation,test_data,row_num,col_num)

    test_finish.set("共有{}筆資料".format(len(predict)))

def print_result(f,canvas,item_num):
    f.clear()
    axes1 = f.add_subplot(331)
    axes2 = f.add_subplot(332)
    axes3 = f.add_subplot(333)

    origin = np.array(origin_data[item_num])
    origin = origin.reshape(row_num,col_num)
    
    actual = np.array(train_data[item_num])
    actual = actual.reshape(row_num,col_num)

    cmapmine = ListedColormap(['w', 'b'], N=2)

    # Plot matrix
    #print(predict[0])
    #fig, (ax1,ax2) = plt.subplots(1, 2)
    axes1.imshow(origin, cmap=cmapmine, vmin=0, vmax=1)
    axes1.set_title('input')
    axes2.imshow(predict[item_num], cmap=cmapmine, vmin=0, vmax=1)
    axes2.set_title('test')
    axes3.imshow(actual, cmap=cmapmine, vmin=0, vmax=1)
    axes3.set_title('Autual')

    canvas.draw()

def get_train_file_url(file_name):
    global train_file_url 
    train_file_url = filedialog.askopenfilename()
    
    file_name.set(train_file_url.split('/')[-1])
    
    print(train_file_url)

def get_test_file_url(file_name):
    global test_file_url 
    test_file_url = filedialog.askopenfilename()
    
    file_name.set(test_file_url.split('/')[-1])
    
    print(test_file_url)



def main():
    window = tk.Tk()

    f = Figure(figsize=(5, 4), dpi=100)
    plot_view = tk.Frame(window)
    plot_view.place(x=300,y=20)
    canvas = FigureCanvasTkAgg(f, plot_view)
    canvas.get_tk_widget().pack(side=tk.RIGHT, expand=1)
    
    window.geometry("1000x500+200+300")

    window.title('類神經網路-作業二')
    #選擇檔案
    train_file_name = tk.StringVar()   # 設定 text 為文字變數
    train_file_name.set('')            # 設定 text 的內容

    test_file_name = tk.StringVar()   # 設定 text 為文字變數
    test_file_name.set('')            # 設定 text 的內容  

    train_finish = tk.StringVar()   # 設定 text 為文字變數
    train_finish.set('')            # 設定 text 的內容  

    test_finish = tk.StringVar()   # 設定 text 為文變數
    test_finish.set('')            # 設定 text 的內容  

    tk.Label(window, text='輸入訓練資料').place(x = 20,y = 20)

    tk.Button(window, text='選擇檔案',command= lambda: get_train_file_url(train_file_name)).place(x = 120,y = 16)

    tk.Label(window, textvariable=train_file_name).place(x=180, y=20)

    tk.Label(window, text='輸入測試資料').place(x = 20,y = 50)

    tk.Button(window, text='選擇檔案',command= lambda: get_test_file_url(test_file_name)).place(x = 120,y = 46)

    tk.Label(window, textvariable=test_file_name).place(x=180, y=50)

    tk.Button(window, text='開始訓練', command= lambda: train_model(train_finish)).place(x = 120,y = 80)

    tk.Label(window, textvariable=train_finish).place(x=180, y=85)

    #輸入迭代次數
    tk.Label(window, text='Epoch:').place(x = 20,y = 120)
    interation = tk.Entry(window)
    interation.place(x = 120,y = 120)

    tk.Button(window, text='開始預測',command= lambda: test_predict(int(interation.get()),test_finish)).place(x = 120,y = 150)

    tk.Label(window, textvariable=test_finish).place(x=180, y=150)

    #輸入預測第幾個資料
    tk.Label(window, text='test_item:').place(x = 20,y = 180)
    item_num = tk.Entry(window)
    item_num.place(x = 120,y = 185)

    tk.Button(window, text='輸出',command= lambda: print_result(f,canvas,int(item_num.get()) - 1)).place(x = 120,y = 210)
    

    window.mainloop()

if __name__=="__main__":
    main()
    