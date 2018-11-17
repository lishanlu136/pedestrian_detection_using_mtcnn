@echo off

if exist train_lmdb_144-48 rd /q /s train_lmdb_144-48

echo create train_lmdb_144-48...
"D:/MTCNN/caffe-buildx64-cpu/convert_imageset.exe" "" 144-48/label-train.txt train_lmdb_144-48 --backend=mtcnn --shuffle=true

echo done.
pause