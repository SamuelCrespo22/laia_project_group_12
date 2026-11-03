"""Incremental BTS PREZIP ingestion. Run from repository root: python scripts/ingest_bts.py."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
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
# Support direct execution from the repository root and module imports.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.contract import (COLUMN_MAP, EXPECTED, INTEGER_COLUMNS,
                               FLOAT_COLUMNS, validate_values)


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


def download_zip(session: requests.Session, year: int, month: int, target: Path,
                 max_bytes: int = 1024**3, min_free_bytes: int = 512 * 1024**2):
    """Download candidates, skipping only genuine HTTP 404 responses."""
    for filename in filenames(year, month):
        with session.get(BASE_URL + filename, stream=True, timeout=(30, 180)) as response:
            if response.status_code == 404:
                continue
            response.raise_for_status()
            size = 0
            with target.open('wb') as fh:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        size += len(block)
                        if size > max_bytes:
                            raise ValueError('BTS archive exceeds configured download limit')
                        if shutil.disk_usage(target.parent).free < len(block) + min_free_bytes:
                            raise OSError('Insufficient free disk space for BTS archive')
                        fh.write(block)
            return
    raise FileNotFoundError(f'No BTS archive found for {year}-{month:02d}')


def select_columns(header):
    lookup = {str(name).strip().casefold(): name for name in header}
    missing = [name for name in COLUMN_MAP if name.casefold() not in lookup]
    if missing:
        raise ValueError(f'Missing required BTS columns: {missing}')
    selected = [lookup[name.casefold()] for name in COLUMN_MAP]
    LOG.info('Source schema verified: %d required columns', len(selected))
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
    """Validate raw column names, types, calendar values and record counts."""
    with pq.ParquetFile(path) as parquet:
        columns = parquet.schema_arrow.names
        if columns != EXPECTED:
            raise ValueError(f'Unexpected schema: {columns}')
        for field in parquet.schema_arrow:
            if field.name in INTEGER_COLUMNS:
                valid = pa.types.is_int64(field.type)
            elif field.name in FLOAT_COLUMNS:
                valid = pa.types.is_float64(field.type)
            elif field.name == 'FL_DATE':
                valid = pa.types.is_timestamp(field.type)
            else:
                valid = pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
            if not valid:
                raise ValueError(f'Unexpected type for {field.name}: {field.type}')
        if parquet.metadata.num_rows == 0:
            raise ValueError('Empty monthly dataset')
        if expected_count is not None and parquet.metadata.num_rows != expected_count:
            raise ValueError('Mismatch between processed and stored row counts')
        for batch in parquet.iter_batches(batch_size=100_000, columns=['YEAR', 'MONTH', 'DAY_OF_MONTH', 'FL_DATE', 'CANCELLED', 'DIVERTED',
                                                           'DISTANCE', 'CRS_ELAPSED_TIME', 'CRS_DEP_TIME', 'CRS_ARR_TIME']):
            frame = batch.to_pandas()
            if not ((frame['YEAR'] == year) & (frame['MONTH'] == month)).all():
                raise ValueError('Wrong year/month values in output')
            if frame['FL_DATE'].isna().any():
                raise ValueError('Missing FL_DATE in output')
            validate_values(frame)
            dates = frame['FL_DATE'].dt
            if not ((dates.year == year) & (dates.month == month)).all():
                raise ValueError('FL_DATE does not match output year/month')
        LOG.info('%04d-%02d schema verified: columns=%d rows=%d',
                 year, month, len(columns), parquet.metadata.num_rows)
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
    if int(settings['chunk_size']) <= 0:
        raise ValueError('ingestion.chunk_size must be positive')
    if int(settings.get('max_download_bytes', 1024**3)) <= 0:
        raise ValueError('ingestion.max_download_bytes must be positive')
    if int(settings.get('min_free_disk_bytes', 512 * 1024**2)) < 0:
        raise ValueError('ingestion.min_free_disk_bytes must be nonnegative')
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
                        LOG.info('%s skipped (verified %d rows) sha256=%s', key, num_rows, existing['sha256'])
                        continue
                LOG.info('%s downloading / converting', key)
                with tempfile.TemporaryDirectory(dir=output_dir) as temp:
                    temp = Path(temp)
                    archive = temp / 'source.zip'
                    pending = temp / 'output.parquet'
                    download_zip(session, year, month, archive,
                                 int(settings.get('max_download_bytes', 1024**3)),
                                 int(settings.get('min_free_disk_bytes', 512 * 1024**2)))
                    LOG.info('%s source sha256=%s', key, sha256(archive))
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
