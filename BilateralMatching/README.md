# Bilateral Matching
This method refers to the model proposed in "Bilateral Multi-Perspective Matching for Natural Language Sentences". However, my matching layer is much simplified.

## Requirements
All code is written in Python3. You may install the required packages by running the following command.
```
pip3 install -r requirements
```

## Training
Run the following command to train a model for subtask A
```
python3 train_bimpm.py [modelname] preprocess/cleandata/subtaskA/traindata.txt --subtask A --worddim 100
```
Run the following command to train a model for subtask B 
```
python3 train_bimpm.py [modelname] preprocess/cleandata/subtaskB/traindata.txt --subtask B --worddim 100
```
Run the following command to train a model for subtask C 
```
python3 train_bimpm.py [modelname] preprocess/cleandata/subtaskA/traindata.txt --subtask C --worddim 100
```


## Testing
Run the following command to output the predictions of subtask A
```
python3 test.py [modelname] preprocess/cleandata/subtaskA/testdata.txt --subtask A
```
Run the following command to output the predictions of subtask B 
```
python3 test.py [modelname] preprocess/cleandata/subtaskB/testdata.txt --subtask B 
```
Run the following command to output the predictions of subtask C 
```
python3 test.py [modelname] preprocess/cleandata/subtaskA/testdata.txt --subtask C 
```
 

