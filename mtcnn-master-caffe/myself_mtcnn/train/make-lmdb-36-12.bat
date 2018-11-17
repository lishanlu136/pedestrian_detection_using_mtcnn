@echo off

if exist train_lmdb_36-12 rd /q /s train_lmdb_36-12

echo create train_lmdb_36-12...
"D:/MTCNN/caffe-buildx64-cpu/convert_imageset.exe" "" 36-12/label-train.txt train_lmdb_36-12 --backend=mtcnn --shuffle=true

echo done.
pause