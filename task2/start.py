import src

RAW_DATA_PATH = "data/raw/all_v2.csv"
REGIONAL_DATA_PATH = "data/interim/data_regional.csv"
CLEANED_DATA_PATH = "data/interim/data_cleaned.csv"
FEATURED_DATA_PATH = "data/processed/data_featured.csv"
REGION_ID = 2661


if __name__ == "__main__":
    src.select_region(RAW_DATA_PATH, REGIONAL_DATA_PATH, REGION_ID)
    src.clean_data(REGIONAL_DATA_PATH, CLEANED_DATA_PATH)
    src.add_features(CLEANED_DATA_PATH, FEATURED_DATA_PATH)



