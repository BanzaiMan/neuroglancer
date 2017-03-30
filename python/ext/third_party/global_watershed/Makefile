CPLUS_INCLUDE_PATH :=../zi_lib:./
export CPLUS_INCLUDE_PATH
CFLAGS := -std=c++11 -fno-strict-aliasing
# how to remove -g from CFLAGS??
export CFLAGS
python:
	python setup.py build_ext --inplace
	rm -rf build
#g++ -o bin/stage1 stage1.cpp -I. -I/usr/people/ranl/storage/data/watershed_cleanup/src/zi_lib -Wall -Wextra -lpthread -lrt -O3 -DNDEBUG -fno-omit-frame-pointer
clean:
	rm watershed.so