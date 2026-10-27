"""Incremental BTS PREZIP ingestion. Run from repository root: python scripts/ingest_bts.py."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml

LOG = logging.getLogger('bts.ingestion')
BASE_URL = 'https://transtats.bts.gov/PREZIP/'
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



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def filenames(year: int, month: int):
    return [
        f'On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip',
        f'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.zip',
    ]


def download_zip(session: requests.Session, year: int, month: int, target: Path):
    """Download candidates, skipping only genuine HTTP 404 responses."""
    for filename in filenames(year, month):
        with session.get(BASE_URL + filename, stream=True, timeout=(30, 180)) as response:
            if response.status_code == 404:
                continue
            response.raise_for_status()
            with target.open('wb') as fh:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        fh.write(block)
            return
    raise FileNotFoundError(f'No BTS archive found for {year}-{month:02d}')


def select_columns(header):
    lookup = {str(name).strip().casefold(): name for name in header}
    missing = [name for name in COLUMN_MAP if name.casefold() not in lookup]
    if missing:
        raise ValueError(f'Missing required BTS columns: {missing}')
    selected = [lookup[name.casefold()] for name in COLUMN_MAP]
    rename = {lookup[name.casefold()]: dest for name, dest in COLUMN_MAP.items()}
    return selected, rename


def normalize_chunk(chunk, rename, year, month):
    chunk = chunk.rename(columns=rename)
    chunk = chunk[list(COLUMN_MAP.values())].copy()
    # The PREZIP archive exposes calendar components, not necessarily FL_DATE.
    # Invalid dates must be rejected rather than silently converted to NaT.
    year_vals = pd.to_numeric(chunk['YEAR'], errors='raise').astype('int32')
    month_vals = pd.to_numeric(chunk['MONTH'], errors='raise').astype('int32')
    day_vals = pd.to_numeric(chunk['DAY_OF_MONTH'], errors='raise').astype('int32')
    if not ((year_vals == year) & (month_vals == month)).all():
        raise ValueError('Unexpected year/month in BTS source')
    chunk['FL_DATE'] = pd.to_datetime({'year': year_vals, 'month': month_vals, 'day': day_vals}, errors='raise')
    return chunk[EXPECTED]


def validate_parquet(path: Path, year: int, month: int, expected_count: int | None = None):
    """Local structural checks. Integrate team's Issue #1.6 contract as an additional check."""
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    if columns != EXPECTED:
        raise ValueError(f'Unexpected schema: {columns}')
    if parquet.metadata.num_rows == 0:
        raise ValueError('Empty monthly dataset')
    if expected_count is not None and parquet.metadata.num_rows != expected_count:
        raise ValueError('Mismatch between processed and stored row counts')
    for batch in parquet.iter_batches(batch_size=100_000, columns=['YEAR', 'MONTH', 'FL_DATE']):
        frame = batch.to_pandas()
        if not ((frame['YEAR'] == year) & (frame['MONTH'] == month)).all():
            raise ValueError('Wrong year/month values in output')
        if frame['FL_DATE'].isna().any():
            raise ValueError('Missing FL_DATE in output')
    return parquet.metadata.num_rows


def convert_zip(zip_path: Path, output: Path, year: int, month: int, chunksize: int):
    count = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith('.csv')]
        if len(members) != 1:
            raise ValueError(f'Expected one CSV in archive; found {len(members)}')
        csv_name = members[0]
        with archive.open(csv_name) as fh:
            header = pd.read_csv(fh, nrows=0, encoding='latin-1').columns
        selected, rename = select_columns(header)
        writer = None
        try:
            with archive.open(csv_name) as fh:
                reader = pd.read_csv(fh, usecols=selected, chunksize=chunksize, encoding='latin-1', low_memory=False)
                for chunk in reader:
                    frame = normalize_chunk(chunk, rename, year, month)
                    # Nullable explicit dtypes prevent schema drift across chunks.
                    for column in frame.columns:
                        if column in INTEGER_COLUMNS:
                            frame[column] = pd.to_numeric(frame[column], errors='raise').astype('Int64')
                        elif column in FLOAT_COLUMNS:
                            frame[column] = pd.to_numeric(frame[column], errors='raise').astype('Float64')
                        elif column != 'FL_DATE':
                            frame[column] = frame[column].astype('string')
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(output, table.schema, compression='zstd')
                    writer.write_table(table)
                    count += len(frame)
            if writer is None:
                raise ValueError('CSV contained no records')
        finally:
            if writer is not None:
                writer.close()
    return count


def load_config(path: Path):
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    settings = config['ingestion']
    years = [int(y) for y in settings['years']]
    if not years:
        raise ValueError('ingestion.years must contain at least one year')
    months = [int(m) for m in settings.get('months', range(1, 13))]
    if not months or len(set(months)) != len(months) or any(m < 1 or m > 12 for m in months):
        raise ValueError('ingestion.months must contain distinct months from 1 to 12')
    settings['months'] = months
    return settings, years


def run(config_path: Path):
    settings, years = load_config(config_path)
    # Output paths are interpreted relative to the repository root.
    root = config_path.resolve().parent.parent
    output_dir = root / settings['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    chunksize = int(settings['chunk_size'])
    manifest_path = output_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    with requests.Session() as session:
        for year in years:
            for month in settings['months']:
                key = f'{year}-{month:02d}'
                target = output_dir / f'flights_{year}_{month:02d}.parquet'
                existing = manifest.get(key)
                if target.exists() and existing and existing.get('sha256') == sha256(target):
                    try:
                        num_rows = validate_parquet(target, year, month, existing.get('rows'))
                    except (ValueError, OSError, pa.ArrowException) as exc:
                        LOG.warning('%s fails validation: %s; rebuilding', key, exc)
                    else:
                        LOG.info('%s skipped (verified %d rows)', key, num_rows)
                        continue
                LOG.info('%s downloading / converting', key)
                with tempfile.TemporaryDirectory(dir=output_dir) as temp:
                    temp = Path(temp)
                    archive = temp / 'source.zip'
                    pending = temp / 'output.parquet'
                    download_zip(session, year, month, archive)
                    count = convert_zip(archive, pending, year, month, chunksize)
                    validate_parquet(pending, year, month, count)
                    digest = sha256(pending)
                    os.replace(pending, target)
                manifest[key] = {'file': target.name, 'rows': count, 'sha256': digest}
                temp_manifest = manifest_path.with_suffix('.json.tmp')
                temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
                os.replace(temp_manifest, manifest_path)
                LOG.info('%s complete rows=%d sha256=%s', key, count, digest)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/config.yaml'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(args.config)
