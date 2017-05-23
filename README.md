# CudaViaCmake
This is a very simple project using CMake in order to create a project for compiling cuda with cpp code.
I created this repo in order to allow other people have an entry point into creating -cmake- based projects 
supporting both cuda and c++

In the future I plan to add more options, checks etc.
 
# Instructions 
Clone the project and run the following commands:<br />
<br />
    
    mkdir build  
    cd build    
    cmake ..      
    msbuild main.vcxproj
        
<br />
If everything is ok, a main.exe will be built inside Debug/ folder.
Then run Debug/main.exe and the program should return:

Checking Results<br />
Results verified<br />
c[0]=1.000000 (== 1.000000 * 1.000000)<br />
c[1]=4.000000 (== 2.000000 * 2.000000)<br />
<br />
# Options
For release versions run:<br />

    cmake .. -DCMAKE_BUILD_TYPE=RELEASE 
    msbuild main.vcxproj /property:Configuration=Release 

and finally:<br />

    Release/main.exe 
