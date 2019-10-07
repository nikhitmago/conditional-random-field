# conditional-random-field

## Project Description

- The program produces chunking of descriptions of UCLA courses. The following types of chunks were used: format, requisite, description, grading, and others (which means that they do not belong to any of previous types).
- The extractor component of the program is trained using CRF suite.
- Training is done using the course description of all courses in the Computer Science department (train-ucla.txt).
- Prepared a test dataset using the courses from Chemical Engineering (50 courses)
- A chunk is correct only if it is an exact match of the corresponding chunk in the data. In the example: 

```<format> Lecture, four hours; outside study, eight hours. </format>``` 
is a correct chunk, but 
```<format> Lecture, four hours; outside study, eight hours. Requisite</format>```
is not.
- The project report provides all details about performance of model.

## Dependencies / Requirements:
1) python 2.7
2) numpy
3) nltk
4) pycrfsuite 
5) BeautifulSoup

Command to execute extract.py:
`python2 extract.py ucla.model test-ucla.txt test-ucla-markup,txt`

The above command will create a file called test-ucla-markup.txt, which will contain the markup text.

Command to execute train.py:
`python2 train.py`

The above command will create ucla.model inside the given folder and will print out the train and test performance results. The test set manually created has been fed to this model for performance.
