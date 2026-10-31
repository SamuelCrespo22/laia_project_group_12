"""Raw BTS ingestion schema, available for reuse by the API contract work."""
import pandas as pd
import logging

# BTS PREZIP headings -> evaluation contract headings. No sampling or flight filtering.
COLUMN_MAP = {
    'Year': 'YEAR', 'Quarter': 'QUARTER', 'Month': 'MONTH',
    'DayofMonth': 'DAY_OF_MONTH', 'DayOfWeek': 'DAY_OF_WEEK',
    'Reporting_Airline': 'OP_UNIQUE_CARRIER',
    'DOT_ID_Reporting_Airline': 'OP_CARRIER_AIRLINE_ID',
    'Tail_Number': 'TAIL_NUM',
    'Flight_Number_Reporting_Airline': 'OP_CARRIER_FL_NUM',
    'OriginAirportID': 'ORIGIN_AIRPORT_ID',
    'OriginAirportSeqID': 'ORIGIN_AIRPORT_SEQ_ID',
    'OriginCityMarketID': 'ORIGIN_CITY_MARKET_ID',
    'Origin': 'ORIGIN', 'OriginCityName': 'ORIGIN_CITY_NAME',
    'OriginState': 'ORIGIN_STATE_ABR', 'OriginStateName': 'ORIGIN_STATE_NM',
    'OriginWac': 'ORIGIN_WAC',
    'DestAirportID': 'DEST_AIRPORT_ID',
    'DestAirportSeqID': 'DEST_AIRPORT_SEQ_ID',
    'DestCityMarketID': 'DEST_CITY_MARKET_ID',
    'Dest': 'DEST', 'DestCityName': 'DEST_CITY_NAME',
    'DestState': 'DEST_STATE_ABR', 'DestStateName': 'DEST_STATE_NM',
    'DestWac': 'DEST_WAC',
    'CRSDepTime': 'CRS_DEP_TIME', 'DepTime': 'DEP_TIME',
    'DepDelay': 'DEP_DELAY', 'TaxiOut': 'TAXI_OUT',
    'TaxiIn': 'TAXI_IN', 'CRSArrTime': 'CRS_ARR_TIME',
    'ArrTime': 'ARR_TIME', 'ArrDelay': 'ARR_DELAY',
    'Cancelled': 'CANCELLED', 'Diverted': 'DIVERTED',
    'CRSElapsedTime': 'CRS_ELAPSED_TIME', 'Flights': 'FLIGHTS',
    'Distance': 'DISTANCE', 'DistanceGroup': 'DISTANCE_GROUP',
    'FirstDepTime': 'FIRST_DEP_TIME', 'TotalAddGTime': 'TOTAL_ADD_GTIME',
    'LongestAddGTime': 'LONGEST_ADD_GTIME',
}
EXPECTED = list(COLUMN_MAP.values()) + ['FL_DATE']
# Consistent nullable typed schema across months and chunks.
INTEGER_COLUMNS = {
    'YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
    'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID',
    'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN_WAC',
    'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID',
    'DEST_WAC', 'CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME',
    'ARR_TIME', 'DISTANCE_GROUP', 'FIRST_DEP_TIME', 'FLIGHTS',
}
FLOAT_COLUMNS = {
    'DEP_DELAY', 'TAXI_OUT', 'TAXI_IN', 'ARR_DELAY',
    'CRS_ELAPSED_TIME', 'DISTANCE', 'TOTAL_ADD_GTIME',
    'LONGEST_ADD_GTIME', 'CANCELLED', 'DIVERTED',
}

def validate_values(frame):
    """Reject invalid raw values without filtering cancelled/diverted flights."""
    for name in ("YEAR", "MONTH", "DAY_OF_MONTH", "FL_DATE"):
        if frame[name].isna().any():
            raise ValueError(f"Missing required values: {name}")
    for name in ("CANCELLED", "DIVERTED"):
        if not frame[name].dropna().isin([0, 1]).all():
            raise ValueError(f"Invalid binary flag: {name}")
    for name in ("DISTANCE",):
        if (frame[name].dropna() < 0).any():
            raise ValueError(f"Negative scheduled measure: {name}")
    negative_duration = int((frame['CRS_ELAPSED_TIME'].dropna() < 0).sum())
    if negative_duration:
        logging.getLogger('bts.ingestion').warning(
            'Raw contract anomaly: %d negative CRS_ELAPSED_TIME values retained for preprocessing',
            negative_duration)
    for name in ("CRS_DEP_TIME", "CRS_ARR_TIME"):
        values = frame[name].dropna()
        valid = ((values >= 0) & (values <= 2359) & (values % 100 < 60)) | (values == 2400)
        if not valid.all():
            raise ValueError(f"Invalid HHMM time: {name}")
    expected = pd.to_datetime(dict(year=frame["YEAR"], month=frame["MONTH"], day=frame["DAY_OF_MONTH"]), errors="raise")
    if not (frame["FL_DATE"].reset_index(drop=True) == expected.reset_index(drop=True)).all():
        raise ValueError("FL_DATE does not match calendar components")
