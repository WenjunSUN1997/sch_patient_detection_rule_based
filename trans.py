import os
import time

from pygoogletranslation import Translator
translator = Translator()
file_list = os.listdir('data_processed_25/text')
for file_name in file_list:
    file_list_en = os.listdir('data_processed_25/text_en')
    if file_name in file_list_en:
        continue
    time.sleep(0.3)
    with open('data_processed_25/text/'+file_name, encoding='utf-8') as file:
        a = file.read()
        print(a)
        print(type(a))
        result = translator.translate(a, src='fr', dest='en')
        print(result.text)
    with open('data_processed_25/text_en/'+file_name, 'w', encoding='utf-8') as file:
        file.write(result.text)
