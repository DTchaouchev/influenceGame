INCLUDE_DIR     = $$HOME/include

testmake: test.cpp influenceGame.cpp
	g++ test.cpp influenceGame.cpp -o test -std=c++11 -larmadillo
