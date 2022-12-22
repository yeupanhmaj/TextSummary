from Sumarizer import generate_summary
from KeyWordExtractor import KeyWordExtractor

file = open('text.txt', "r")
filedata = file.readlines()[0]

generate_summary(filedata, 2)

extractor = KeyWordExtractor(filedata)

extractor.Extract()
