# DSAI-HW1-2021

### **Team Member**    

| 編輯者       |    暱稱         |                      LinkedIn                                                            |
| :-----------:|:-----------:   |:---------------------------------------------------------------------------------------: |
|  洪志宇      | CHI-YU HONG     | [https://www.linkedin.com/in/chiyuhong/](https://www.linkedin.com/in/chiyuhong/)     
|  施智臏      | ZHI-BIN SHIH     | [https://www.linkedin.com/in/zhibin-shih-9a0a711a9/](https://www.linkedin.com/in/zhibin-shih-9a0a711a9/)     

**Data analysis :**

 下圖為台電提供之108至110年的「瞬時尖峰負載（萬瓩）」、「備轉容量（萬瓩）」及「備轉容量率（％）」，由圖中可以見到110年之「備轉容量（萬瓩）」相對於前兩年皆處於較平穩的趨勢，另外備轉容量率也相較過去較為低緩，吾人猜測是因為恰逢疫情及後疫情時期，故去年之資料較為特別，因此僅採用去年至今年三月的資料來做分析。

 另外，猜測這兩年受經濟的影響（如.台商回流、台積電擴廠及帶動相關產業產能提升）相較天氣的影響來得大，又因無法取得相關資料，因此僅用備轉容量（萬瓩）去做自回歸。 
>>>![image](https://user-images.githubusercontent.com/43928481/111780006-c9dafd00-88f1-11eb-9ccc-9edc17c797f2.png)
>>>![image](https://user-images.githubusercontent.com/43928481/111780025-cf384780-88f1-11eb-8cb8-af7e5d9f30fe.png)
>>>![image](https://user-images.githubusercontent.com/43928481/111780047-d6f7ec00-88f1-11eb-89ac-5a8d13d5c3db.png)



**Model training :**

**自回歸模型**
 <br>由於僅用一個維度的資料去做預測，因此採用利用過去自我觀測值來預測未來自己的自回歸模型。
 <br>自回歸是一種時間序列模型，同時亦是使用滯後變量作為輸入變量的線性回歸模型。它使用前一個時間步長的觀察值作為回歸方程的輸入，以預測下一個時間步長的值。
 <br>我們根據所有先前的觀測結果來預測目標日期的「備轉容量（萬瓩）」。statsmodels庫提供了一個自動回歸模型（在AutoReg 類中提供），我們手動指定要使用的滯後輸入變量後便可訓練一個線性回歸模型。
利用訓練集適配模型後，例用predict（）函數來進行未來7天的預測，結果如下：
>>>![image](https://user-images.githubusercontent.com/43928481/111778272-651ea300-88ef-11eb-9289-608be5512167.png)

 

