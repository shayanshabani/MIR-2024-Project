from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO
        self.crawling_threshold = crawling_threshold
        self.not_crawled = set()
        self.crawled = set()
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        # TODO
        URL_array = URL.split('/')
        return URL_array[URL_array.index('title') + 1]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # TODO
        json_object = json.dumps(self.crawled, indent=4)
        with open('IMDB_crawled.json', 'w') as f:
            f.write(json_object)

        json_object = json.dumps(self.not_crawled, indent=4)
        with open('IMDB_not_crawled.json', 'w') as f:
            f.write(json_object)

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = json.load(f)

        with open('IMDB_not_crawled.json', 'w') as f:
            self.not_crawled = json.load(f)

        self.added_ids = None

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        # TODO
        response = get(URL, headers=self.headers)
        return response

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids
        response = self.crawl(self.top_250_URL)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            movie_links = soup.find_all('a', class_='ipc-title-link-wrapper')[:250]
            for movie_link in movie_links:
                movie_id = self.get_id_from_URL(movie_link['href'])
                self.not_crawled.add(movie_id)
                self.added_ids.add(movie_id)

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        # help variables
        # WHILE_LOOP_CONSTRAINTS = (len(self.crawled) < self.crawling_threshold) and (len(self.not_crawled) > 0)
        # NEW_URL = f'https://www.imdb.com/title/{self.not_crawled.pop()}/'
        # THERE_IS_NOTHING_TO_CRAWL = len(self.not_crawled) > 0

        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while (len(self.crawled) < self.crawling_threshold) and (len(self.not_crawled) > 0):
                URL = f'https://www.imdb.com/title/{self.not_crawled.pop()}/'
                futures.append(executor.submit(self.crawl_page_info, URL))
                if (len(self.not_crawled) == 0) or (len(futures) >= self.crawling_threshold):
                    wait(futures)
                    futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")
        # TODO
        response = self.crawl(URL)
        if response.status_code == 200:
            #soup = BeautifulSoup(response.text, 'html.parser')
            movie_info = self.get_imdb_instance()
            self.extract_movie_info(response, movie_info, URL)
            with self.add_list_lock:
                self.crawled.add(movie_info['id'])


    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        # TODO
        soup = BeautifulSoup(res.text, 'html.parser')
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        mpaa_soup = BeautifulSoup(self.crawl(self.get_parental_link(URL)).text, 'html.parser')
        movie['mpaa'] = self.get_mpaa(mpaa_soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['rating'] = self.get_rating(soup)
        summary_soup = BeautifulSoup(self.crawl(self.get_summary_link(URL)).text, 'html.parser')
        movie['summaries'] = self.get_summary(summary_soup)
        movie['synopsis'] = self.get_synopsis(summary_soup)
        review_soup = BeautifulSoup(self.crawl(self.get_review_link(URL)).text, 'html.parser')
        movie['reviews'] = self.get_reviews_with_scores(review_soup)

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            # TODO
            return f'{url}plotsummary'
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            # TODO
            return f'{url}reviews'
        except:
            print("failed to get review link")

    def get_parental_link(self, url):
        """
        Get the link to the parental page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/parentalguide is the review page
        """
        try:
            # TODO
            return f'{url}parentalguide'
        except:
            print("failed to get parental guide link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            # TODO
            title = str(soup.find_all('span', class_='hero__primary-text')[0].text)
            return title
        except:
            print("failed to get title")

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            # TODO
            first_page_summary = str(soup.find_all('span', class_='sc-466bb6c-0 hlbAws')[0].text)
            return first_page_summary
        except:
            print("failed to get first page summary")

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            # TODO
            casts = soup.find_all('ul', class_='ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')
            directors_soup = BeautifulSoup(str(casts[0]), 'html.parser')
            directors = [str(a.text) for a in directors_soup.find_all('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')]
            return directors
        except:
            print("failed to get director")

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            # TODO
            casts = soup.find_all('ul', class_='ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')
            stars_soup = BeautifulSoup(str(casts[2]), 'html.parser')
            stars = [str(a.text) for a in stars_soup.find_all('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')]
            return stars
        except:
            print("failed to get stars")

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            # TODO
            casts = soup.find_all('ul', class_='ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt')
            writers_soup = BeautifulSoup(str(casts[0]), 'html.parser')
            writers = [str(a.text) for a in writers_soup.find_all('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')]
            return writers
        except:
            print("failed to get writers")

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            # TODO
            related_links = []
            related_objects = soup.find_all('a', class_='ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable')
            for related_object in related_objects:
                related_links.append(f'https://www.imdb.com/title/{self.get_id_from_URL(str(related_object["href"]))}')
            return related_links
        except:
            print("failed to get related links")

    def get_summary(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            # TODO
            summaries_object = soup.find_all('ul', class_='ipc-metadata-list ipc-metadata-list--dividers-after sc-d1777989-0 FVBoi ipc-metadata-list--base')
            summaries_soup = BeautifulSoup(str(summaries_object), 'html.parser')
            summaries = [str(div.text) for div in summaries_soup.find_all('div', class_='ipc-html-content-inner-div')]
            return summaries
        except:
            print("failed to get summary")

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            # TODO
            synopsis = []
            synopsis_object = soup.find_all('div', class_='ipc-html-content ipc-html-content--base sc-8c0e9a24-0 iouSJu')
            synopsis.append(str(synopsis_object[-1].text))
            return synopsis
        except:
            print("failed to get synopsis")

    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            # TODO
            reviews_with_scores = []
            review_object = soup.find_all('div', 'lister-item-content')
            for obj in review_object:
                review_with_score = []
                review_soup = BeautifulSoup(str(obj), 'html.parser')
                title = str(review_soup.find_all('a', 'title')[0].text[1:])
                rating_object = review_soup.find_all('div', 'ipl-ratings-bar')
                rating = ''
                if len(rating_object) != 0:
                    rating = str(rating_object[0].text.replace('\n', ''))
                review = title + '\n' + str(review_soup.find_all('div', 'content')[0].text.split('\n')[1])
                review_with_score.append(review)
                review_with_score.append(rating)
                reviews_with_scores.append(review_with_score)
        except:
            print("failed to get reviews")

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            # TODO
            genres_list = []
            genre_object = soup.find_all('div', 'sc-491663c0-10 rbXFE')
            genre_soup = BeautifulSoup(str(genre_object), 'html.parser')
            genres = genre_soup.find_all('a', 'ipc-chip ipc-chip--on-baseAlt')
            for genre in genres:
                genres_list.append(str(genre.text))
            return genres_list
        except:
            print("Failed to get generes")

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            # TODO
            rate = str(soup.find_all('span', class_='sc-bde20123-1 cMEQkK')[0].text) + '/10'
            return rate
        except:
            print("failed to get rating")

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            # TODO
            mpaa = soup.find_all('tr', id='mpaa-rating')
            return str(mpaa[0].text)[6:-1]
        except:
            print("failed to get mpaa")

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            # TODO
            year_obj = soup.find_all('ul', class_='ipc-inline-list ipc-inline-list--show-dividers sc-d8941411-2 cdJsTz baseAlt')
            year_soup = BeautifulSoup(str(year_obj), 'html.parser')
            year = str(year_soup.find_all('li', 'ipc-inline-list__item')[0].text)
            return year
        except:
            print("failed to get release year")

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            # TODO
            languages_list = []
            languages_object = soup.find('li', {'role': 'presentation', 'class': 'ipc-metadata-list__item', 'data-testid': 'title-details-languages'})
            languages_soup = BeautifulSoup(str(languages_object), 'html.parser')
            languages = languages_soup.find_all('a', 'ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
            for language in languages:
                languages_list.append(str(language.text))
            return languages_list
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            # TODO
            countries_list = []
            countries_object = soup.find('li', {'role': 'presentation', 'class': 'ipc-metadata-list__item', 'data-testid': 'title-details-origin'})
            countries_soup = BeautifulSoup(str(countries_object), 'html.parser')
            countries = countries_soup.find_all('a', 'ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
            for country in countries:
                countries_list.append(str(country.text))
            return countries_list
        except:
            print("failed to get countries of origin")

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            # TODO
            budget_object = soup.find('li', {'role': 'presentation', 'class': 'ipc-metadata-list__item sc-1bec5ca1-2 bGsDqT', 'data-testid': 'title-boxoffice-budget'})
            budget_soup = BeautifulSoup(str(budget_object), 'html.parser')
            budget = str(budget_soup.find('span', 'ipc-metadata-list-item__list-content-item').text)
            return budget
        except:
            print("failed to get budget")

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            # TODO
            gross_object = soup.find('li', {'role': 'presentation', 'class': 'ipc-metadata-list__item sc-1bec5ca1-2 bGsDqT', 'data-testid': 'title-boxoffice-cumulativeworldwidegross'})
            gross_soup = BeautifulSoup(str(gross_object), 'html.parser')
            gross = str(gross_soup.find('span', 'ipc-metadata-list-item__list-content-item').text)
            return gross
        except:
            print("failed to get gross worldwide")


def main():
    print('hellowww')
    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    # }
    # response = get('https://www.imdb.com/title/tt0111161/', headers=headers)
    # if response.status_code == 200:
    #     soup = BeautifulSoup(response.text, 'html.parser')
    #     gross_object = soup.find('li', {'role': 'presentation', 'class': 'ipc-metadata-list__item sc-1bec5ca1-2 bGsDqT', 'data-testid': 'title-boxoffice-cumulativeworldwidegross'})
    #     gross_soup = BeautifulSoup(str(gross_object), 'html.parser')
    #     gross = gross_soup.find('span', 'ipc-metadata-list-item__list-content-item')
    #     print(gross.text)

if __name__ == '__main__':
    main()
