download_dataset:
	mkdir source-data
	mkdir cache-data
	cd source-data && wget https://s3.amazonaws.com/nist-srd/SD19/by_class.zip && unzip by_class.zip

run:
	python3 ./main.py