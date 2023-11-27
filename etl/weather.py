from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

class WeatherScraper:
    def __init__(self):
        """
        Initialize the WeatherScraper class.
        """
        self.data_path = os.path.dirname(os.path.abspath(__file__)) + '/data'
        self.url = "https://nowdata.rcc-acis.org/lot/"
        self.stations_to_skip = ["Fairbury Waterwo, IL", "Forreston, IL", "Harvard, IL", "La Grange, IL", "Steward 3 S, IL"]
        self.data_list = []

    def scrape_weather_data(self):
        """
        Scrape weather data from the website and save it to a CSV file.

        Usage:
        scraper = WeatherScraper()
        scraper.scrape_weather_data()
        """
        try:
            # Open the URL in the browser
            driver = webdriver.Firefox()
            driver.get(self.url)

            # Wait for a few seconds to ensure the page is fully loaded
            time.sleep(5)

            # Find and select the radio button for "Monthly summarized data"
            monthly_radio = driver.find_element(By.XPATH, '//input[@name="product_select" and @value="StnProduct_monavg"]')
            monthly_radio.click()

            # Wait for a second (adjust as needed)
            time.sleep(1)

            # Find and select the "Avg temp" option
            variable_select = driver.find_element(By.XPATH, '//select[@class="ui-widget-content ui-corner-all ui-state-default"]')
            variable = Select(variable_select)
            variable.select_by_value("avgt")

            # Wait for a second (adjust as needed)
            time.sleep(1)

            # Find the <select> element by its name attribute
            select_element = driver.find_element(By.NAME, "station")

            # Create a Select object to interact with the <select> element
            select = Select(select_element)

            # Iterate through each option in the <select> element
            for option in select.options:
                # Check if the option should be skipped
                if option.text in self.stations_to_skip:
                    print(f"Skipping {option.text}")
                    continue

                # Click on the option
                option.click()
                time.sleep(1)  # Wait for a second (adjust as needed)

                # Find the "Go" button and scroll it into view
                go_button = driver.find_element(By.ID, "go")
                driver.execute_script("arguments[0].scrollIntoView(true);", go_button)

                # Click the "Go" button
                go_button.click()

                # Wait for the headers to be present
                header_elements = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.XPATH, '//th[@class="sorter-sorttrace string-bottom tablesorter-header"]/div[@class="tablesorter-header-inner"]'))
                )

                # Extract header texts
                header_texts = ['Year'] + [header.text for header in header_elements]
                print(header_texts)
                print(len(header_texts))

                # Find the temperature data in the tbody
                temperature_elements = driver.find_elements(By.XPATH, '//tbody/tr/td')
                temperature_values = [element.text for element in temperature_elements]

                # Reshape the temperature values into a 2D list (rows x columns)
                num_columns = len(header_texts)
                temperature_matrix = [temperature_values[i:i + num_columns] for i in range(0, len(temperature_values), num_columns)]
                print(temperature_matrix)
                print(len(temperature_matrix))

                # Create a list of dictionaries for each row in temperature_matrix
                temp_data_list = []
                for row in temperature_matrix:
                    temp_data_list.extend([{
                        "option": option.text,
                        "type": header_texts[i],
                        "temperature": row[i],
                        "Year": row[0]  # Extracting the year from the first column
                    } for i in range(1, len(header_texts))])

                # Append the data to the list
                self.data_list.extend(temp_data_list)

                # Find the "Close" button and scroll it into view
                close_button = driver.find_element(By.XPATH, '//button[@title="Close"]')
                driver.execute_script("arguments[0].scrollIntoView(true);", close_button)

                # Click the "Close" button
                close_button.click()

                # Wait for a second (adjust as needed)
                time.sleep(1)

            # Close the browser window
            driver.quit()

            # Create a DataFrame from the list
            data = pd.DataFrame(self.data_list)

            # Display the DataFrame
            print(data.head())

            # Save the DataFrame to a CSV file
            data.to_csv(f'{self.data_path}/weather/weather.csv', index=False)

        except Exception as e:
            print("An error occurred:", str(e))
            driver.quit()