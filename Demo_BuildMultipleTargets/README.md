# Build Multiple Targets

This subdirectory has a CMake demonstration for multiple target generation.

# Windows Instructions 
Clone the project and run the following commands:<br />
<br />
    
    mkdir build  
    cd build    
    cmake ..      
    msbuild main.vcxproj
    msbuild main2.vcxproj
        
<br />
If everything is ok, a main.exe will be built inside Debug/ folder.
Then run Debug/main.exe and the program should return:

    Checking Results<br />
    Results verified<br />
    c[0]=1.000000 (== 1.000000 * 1.000000)<br />
    c[1]=4.000000 (== 2.000000 * 2.000000)<br />

The same should happer with main2.exe.
    
<br />
# Options
For release versions run:<br />

    cmake .. -DCMAKE_BUILD_TYPE=RELEASE 
    msbuild main.vcxproj /property:Configuration=Release 

and finally:<br />

    Release/main.exe 
    # and then:
    Release/main.exe


## Ubuntu Instructions 
# Tested with cmake 3.9.1 on Ubuntu 16.04 

    mkdir build
    cd build
    cmake ..
    make
    # now run the applications
    ./main
    ./main2

    Which both should return:
    Checking Results 
    Results verified 
    c[0]=1.000000 (== 1.000000 * 1.000000) 
    c[1]=4.000000 (== 2.000000 * 2.000000) 