# Custom-Model

## 1. complier command step
- mkdir build
- cd build
- rm -rf ./*
- cmake ..
- make -j16

## 2. run command 
- cd release
- ./first_model --json ../../model-flatbuffer/pzk-metadata.json

## 3. after run command, this will show below info:

### Result: open ../../model-flatbuffer/pzk-metadata.json success
### this model size is 3288

## 4. Now You Can Use Shell Script To Build And Run
- chmod +x run.sh
- ./run.sh