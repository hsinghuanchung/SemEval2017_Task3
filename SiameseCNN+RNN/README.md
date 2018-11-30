## Requirements
All code is written in Python3. You may install the required packages by running the following command.
```
pip3 install -r requirements.txt
```

## Training
Run the following command to train a model for subtask A
```
python3 crnn.py A cleandata/subtaskA/data.txt [modelname]
```
Run the following command to train a model for subtask B 
```
python3 crnn.py B cleandata/subtaskB/data.txt [modelname]
```
Run the following command to train a model for subtask C 
```
python3 crnn.py C cleandata/subtaskA/data.txt [modelname]
```


## Testing
Run the following command to output the predictions of subtask A
```
python3 testA_crnn.py cleandata/SemEval2017-task3-English-test-input.xml [modelname] [outputfilename]
```
Run the following command to output the predictions of subtask B 
```
python3 testB_crnn.py cleandata/SemEval2017-task3-English-test-input.xml [modelname] [outputfilename]
```
Run the following command to output the predictions of subtask C 
```
python3 testC_crnn.py cleandata/SemEval2017-task3-English-test-input.xmlt [modelname] [outputfilename]
```
 

