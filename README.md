# OLLI-Intern

## TRAIN

File dùng để train: `train.py`
Thay đổi file train: thay đổi `data_file = pd.read_csv('file_train.csv')`
Format file chứa data train:
```
Sentences,Label
câu 1,label 1
câu 2,label2
```
Ví dụ: xem file `num_refined_dataset.csv`

Command để train
```
python train.py
```

## TEST MODEL

File dùng cho test model: `main.py`
Thêm câu và label và biến
Command để test
```
python main.py
```

## CONVERT SENTENCES TO LABELS

File dùng để chuyển câu sang nhãn: `converter.py`
Chọn file chứa câu cần chuyển: `file = pd.read_csv('data.csv')`
Format file data:
```
Sentences
câu 1
câu 2
...
```
Command để convert
```
python converter.py
```