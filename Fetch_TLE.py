import requests
import pandas as pd

TLE_URL = "https://celestrak.org/NORAD/elements/weather.txt"

def download_tle_data(url):
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.strip().split("\n")
    # Clean up carriage returns
    lines = [line.replace('\r', '') for line in lines]
    return lines

def parse_tle_to_dataframe(tle_lines, max_objects=100):
    records = []
    
    # Filter out empty lines
    tle_lines = [line for line in tle_lines if line.strip()]
    
    for i in range(0, len(tle_lines), 3):
        # Check if we have enough lines for a complete TLE record
        if i + 2 >= len(tle_lines):
            break
            
        name = tle_lines[i].strip()
        line1 = tle_lines[i+1].strip()
        line2 = tle_lines[i+2].strip()
        
        norad_id = line1[2:7].strip()
        epoch = line1[18:32].strip()
        
        inclination = float(line2[8:16])
        raan = float(line2[17:25])
        eccentricity = float("0." + line2[26:33].strip())
        arg_perigee = float(line2[34:42])
        mean_anomaly = float(line2[43:51])
        mean_motion = float(line2[52:63])
        
        records.append({
            "name": name,
            "norad_id": norad_id,
            "epoch": epoch,
            "inclination_deg": inclination,
            "raan_deg": raan,
            "eccentricity": eccentricity,
            "arg_perigee_deg": arg_perigee,
            "mean_anomaly_deg": mean_anomaly,
            "mean_motion_rev_per_day": mean_motion,
            "line1": line1,
            "line2": line2
        })
        
        if len(records) >= max_objects:
            break
    
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    tle_lines = download_tle_data(TLE_URL)
    print(f"Downloaded {len(tle_lines)} lines")
    print(f"First few lines: {tle_lines[:10]}")
    
    df = parse_tle_to_dataframe(tle_lines, max_objects=100)
    print(f"Parsed {len(df)} records")
    
    df.to_csv("leo_debris_sample.csv", index=False)
    print("Saved leo_debris_sample.csv")