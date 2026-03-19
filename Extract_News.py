from bs4 import BeautifulSoup
import pandas as pd
import os

# Put all 3 of your HTML file names here
HTML_FILES = ['ff_historical1.html', 'ff_historical2.html', 'ff_historical3.html'] 
OUTPUT_CSV = 'red_folder_news.csv'

print("Parsing local HTML files for ALL currencies...")

all_events = []

for html_file in HTML_FILES:
    if not os.path.exists(html_file):
        print(f"Warning: Could not find '{html_file}'. Skipping...")
        continue
        
    print(f"Extracting Red Folder news from {html_file}...")
    
    with open(html_file, 'r', encoding='windows-1252', errors='ignore') as f:
        soup = BeautifulSoup(f, 'html.parser')

    current_date = "Unknown"

    # Find all calendar rows
    for row in soup.find_all('tr', class_='calendar__row'):
        
        # Extract Date
        date_col = row.find('td', class_='calendar__date')
        if date_col and date_col.text.strip():
            current_date = date_col.text.strip()
            
        # Extract Time
        time_col = row.find('td', class_='calendar__time')
        time_text = time_col.text.strip() if time_col else ""
        
        # Extract Currency
        currency_col = row.find('td', class_='calendar__currency')
        currency = currency_col.text.strip() if currency_col else ""
        
        # Extract Event Name
        event_col = row.find('td', class_='calendar__event')
        event = event_col.text.strip() if event_col else ""
        
        # NEW LOGIC: Capture everything EXCEPT CNY
        if currency != 'CNY' and time_text != "":
            all_events.append({
                'Date': current_date,
                'Time': time_text,
                'Currency': currency,
                'Event': event
            })

# Convert the massive list into a DataFrame
df = pd.DataFrame(all_events)

# Forward-fill the times for simultaneous events
df['Time'] = df['Time'].replace('', pd.NA).ffill()

# Save everything to one master CSV
df.to_csv(OUTPUT_CSV, index=False)

print("\n🚨 EXTRACTION COMPLETE 🚨")
print(f"Successfully extracted {len(df)} total Red Folder events across all files.")
print(f"Saved master file to: {OUTPUT_CSV}")