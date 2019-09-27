# Source : https://gist.github.com/dkohlsdorf/1ccc8534bfaec7a129ac03fa77cd2721
from lxml import html
import requests
import urllib.request
import re
import time
import os.path

'''
Update 'data_tess/' with your own folder name you wish to save data to
'''

HANDLE = '^/handle/[0-9]+/[0-9]+$'
BASE_URL = 'https://tspace.library.utoronto.ca'
page = requests.get(BASE_URL + '/handle/1807/24487')
tree = html.fromstring(page.content)
subset = [ href.attrib['href'] for href in tree.xpath('//a') if re.match(HANDLE, href.attrib['href'])]
for s in subset:
    wav_page = requests.get(BASE_URL + s)
    tree     = html.fromstring(wav_page.content)          
    links = [ href.attrib['href'] for href in tree.xpath('//a') if 'wav' in href.attrib['href']]
    for link in links: 
        local = link.split('/')[-1]
        if not os.path.isfile('data_tess/' + local): 
            try:
                urllib.request.urlretrieve(BASE_URL + link, 'data_tess/' + local)
                print('Download: ', link)
            except IOError:
                print('Err: ', link)                
                time.sleep(1)
            else:
                print('Already Exists: ', link)