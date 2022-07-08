if [ ! -d build ]; then
    mkdir build
else
    rm -rf build/*
fi
if [ -e test]; then
    rm test
fi
cd build
rm -rf ./*
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j32
./test ../model-flatbuffer/pzk-metadata.json ../test-model/first.pzkm 
cd ..
# ./test