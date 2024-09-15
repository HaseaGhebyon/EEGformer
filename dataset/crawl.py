import pprint
import requests
from bs4 import BeautifulSoup

DOWNLOADED_FILES = 10

from os.path import basename
import os

urls = []
with open("test.txt") as file:
    urls = [line.rstrip() for line in file]
    file.close()

download_urls = []

for url in urls:
    # URL halaman yang ingin di-crawl

    # Mendapatkan konten halaman
    response = requests.get(url)
    web_content = response.content

    # Parsing konten halaman menggunakan BeautifulSoup
    soup = BeautifulSoup(web_content, 'html.parser')

    links =  soup.select("a[href]")

    # Menyimpan semua href dari link tersebut
    hrefs = [link.get('href') for link in links]

    # Menampilkan semua link yang ditemukan
    for href in hrefs:
        if "ndownloader" in href:
            download_urls.append(href)
            print(url)
            print(href)
            print()

import requests
import os

j = 0
# URL of the file to be downloaded
for url, base_url in zip(download_urls, urls):
    # Send a GET request to the URL
    if (j < DOWNLOADED_FILES):
        continue
    print(url)
    response = requests.get(url, allow_redirects=True)

    # Try to get the filename from the Content-Disposition header
    file_name = ''
    print("Response : ", response)
    if 'Content-Disposition' in response.headers:
        content_disposition = response.headers['Content-Disposition']
        if 'filename=' in content_disposition:
            file_name = content_disposition.split('filename=')[1].strip('"')

    # Fallback to URL's basename if no filename in headers
    if not file_name:
        file_name = base_url.split("/")[-2].removeprefix("Experiment_")
        file_name = file_name + ".mat"
    print(file_name)

    # Save the file
    with open(file_name, 'wb') as file:
        file.write(response.content)

    print(f"File downloaded and saved as {file_name}")
    j += 1
