import requests
import os
from bs4 import BeautifulSoup
import pandas as pd

# Define the search criteria
search_criteria = 'sport'

min_books = 100
page_num = 1
titles, image_urls = [], []

# Send a GET request to the URL and get the response
url = f'https://www.amazon.com/s?k={search_criteria}&ref=nb_sb_noss_1&node=283155'
# Had problem scraping data from Amazon because of bot detection, 
# this fix was from https://www.digitalocean.com/community/tutorials/scrape-amazon-product-information-beautiful-soup
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (X11; Linux x86_64)'
                    'AppleWebKit/537.36 (KHTML, like Gecko)'
                    'Chrome/44.0.2403.158 Safari/537.36'),
    'Accept-Language': 'en-US, en;q=0.5'
}

# Download the book cover images to a local folder
book_covers_path = 'book_covers/'
os.makedirs(book_covers_path, exist_ok=True)

while len(titles) < min_books:
    response = requests.get(url, headers=HEADERS)
    print("Page: ", page_num)
    print("Response Status: ", response)

    # Use BeautifulSoup to parse the HTML content of the response
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the book titles in the search results
    titles += [title.text for title in soup.find_all('span', class_='a-size-medium')]

    # Find all the book cover image URLs in the search results
    image_urls += [img['src'] for img in soup.select('img.s-image')]

    page_num += 1
    url = f'https://www.amazon.com/s?k={search_criteria}&ref=nb_sb_noss_1&node=283155&page={page_num}'

for i, url in enumerate(image_urls[:100]):
    response = requests.get(url)
    with open(f'{book_covers_path}/book{i+1}.jpg', 'wb') as f:
        f.write(response.content)


# Save the book metadata to a CSV file
try:
    metadata = pd.DataFrame({'title': titles[:100],'image_url': image_urls[:100]})
    metadata.to_csv('book_metadata.csv', index=False)
except ValueError:
    print("number of titles: ", len(titles))
    print("number of images: ", len(image_urls))
    
