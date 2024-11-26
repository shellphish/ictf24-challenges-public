import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def load_admin_page(admin_cookie, play_token):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        # Load the base URL to set cookies
        base_url = "http://127.0.0.1:5001/"
        driver.get(base_url)

        # Add the admin_cookie and play_token as cookies in the browser
        driver.add_cookie({"name": "admin_cookie", "value": admin_cookie, "domain": "127.0.0.1"})
        driver.add_cookie({"name": "play_token", "value": play_token, "domain": "127.0.0.1"})

        # Now navigate to the admin page
        admin_url = "http://127.0.0.1:5001/admin"
        driver.get(admin_url)

        # Wait for the page to load
        driver.implicitly_wait(10)
        time.sleep(10)

        return 0

    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
