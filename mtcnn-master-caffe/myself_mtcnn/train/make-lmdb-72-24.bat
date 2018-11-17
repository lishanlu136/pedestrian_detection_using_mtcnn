@echo off

if exist train_lmdb_72-24 rd /q /s train_lmdb_72-24

echo create train_lmdb_72-24...
"D:/MTCNN/caffe-buildx64-cpu/convert_imageset.exe" "" 72-24/label-train.txt train_lmdb_72-24 --backend=mtcnn --shuffle=true

echo done.
pause