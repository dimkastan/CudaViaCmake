rmdir /Q build
mkdir build
cd build
cmake .. -G "Visual Studio 12 2013 Win64"

echo "building application"

cmake --build .

echo "running application"

Debug\main.exe