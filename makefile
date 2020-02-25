# Defined key that should not be confused with existing local files 
.PHONY: test verbose doc cleandoc install export init

init:
	pip install -r requirements.txt

test:
	python -m unittest discover
verbose:
	# Verbose test
	python -m unittest discover -v

install:
	#rm -rf /opt/anaconda3/lib/python3.7/site-packages/pydap*
	#python setup.py bdist_wheel
	#pip install ./dist/pydap-1.0.1.dev0-py3-none-any.whl
	#rm -rf ./build/ ./dist/ ./pydap.egg-info
	python setup.py install
	
env:
	#pip freeze > requirements.txt
clean:
	rm -rf `find | grep __pycache__`
