if [ ! -f include/pzk-schema_generated.h ]; then
    flatc -c -o include/  model-flatbuffer/pzk-schema.fbs
fi
if [ ! -d "./build" ]; then
    mkdir build
fi
if [ -f first.json ]; then
    rm first.json
fi
rm -rf build/*
cd build
cmake ..
make -j16
cd release
./first_model --json ../../model-flatbuffer/pzk-metadata.json
cd ../..
flatc --raw-binary -t model-flatbuffer/pzk-schema.fbs -- build/release/first.pzkm