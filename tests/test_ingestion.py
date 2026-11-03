"""Synthetic fixtures: no BTS downloads or real datasets needed in CI."""
import json
import logging
import zipfile

import pandas as pd
import pytest
import yaml

from scripts import ingest_bts as ingestion


def source_zip(path, missing=None):
    values = {}
    for source, dest in ingestion.COLUMN_MAP.items():
        values[source] = [1, 1, 1] if dest in ingestion.INTEGER_COLUMNS | ingestion.FLOAT_COLUMNS else ['AA'] * 3
    values.update(Year=[2022] * 3, Month=[1] * 3, DayofMonth=[1, 2, 3],
                  ArrDelay=[None, -10, 70], Cancelled=[1, 0, 0], Diverted=[0, 0, 0])
    if missing:
        del values[missing]
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('flights.csv', pd.DataFrame(values).to_csv(index=False))


def test_conversion_preserves_all_flights_and_nullable_types(tmp_path):
    archive = tmp_path / 'source.zip'
    source_zip(archive)
    output = tmp_path / 'out.parquet'
    assert ingestion.convert_zip(archive, output, 2022, 1, 1) == 3
    assert ingestion.validate_parquet(output, 2022, 1, 3) == 3
    frame = pd.read_parquet(output)
    assert list(frame.columns) == ingestion.EXPECTED
    assert frame['CANCELLED'].sum() == 1
    assert frame['ARR_DELAY'].isna().sum() == 1
    assert str(frame['YEAR'].dtype) == 'Int64'


def test_missing_columns_rejected(tmp_path):
    archive = tmp_path / 'source.zip'
    source_zip(archive, missing='ArrDelay')
    with pytest.raises(ValueError, match='Missing required BTS columns'):
        ingestion.convert_zip(archive, tmp_path / 'out.parquet', 2022, 1, 2)


@pytest.mark.parametrize('column,value', [('CANCELLED', 2), ('DISTANCE', -1),
                                         ('CRS_DEP_TIME', 1260), ('FL_DATE', pd.Timestamp('2022-02-01'))])
def test_contract_rejects_invalid_values(tmp_path, column, value):
    archive = tmp_path / 'source.zip'
    source_zip(archive)
    output = tmp_path / 'out.parquet'
    ingestion.convert_zip(archive, output, 2022, 1, 2)
    frame = pd.read_parquet(output)
    frame.loc[0, column] = value
    frame.to_parquet(output, index=False)
    with pytest.raises(ValueError):
        ingestion.validate_parquet(output, 2022, 1, 3)


def test_incremental_skip_and_corruption_recovery(tmp_path, monkeypatch, caplog):
    configs = tmp_path / 'configs'
    configs.mkdir()
    config = configs / 'config.yaml'
    config.write_text(yaml.safe_dump({'ingestion': dict(years=[2022], months=[1],
                                                      output_dir='data/raw', chunk_size=1)}))
    calls = []

    def download(session, year, month, target, *limits):
        calls.append((year, month))
        source_zip(target)

    monkeypatch.setattr(ingestion, 'download_zip', download)
    caplog.set_level(logging.INFO, logger='bts.ingestion')
    ingestion.run(config)
    ingestion.run(config)
    assert len(calls) == 1
    assert 'schema verified' in caplog.text
    assert 'sha256=' in caplog.text
    assert 'skipped' in caplog.text
    raw = tmp_path / 'data/raw'
    manifest = json.loads((raw / 'manifest.json').read_text())
    target = raw / manifest['2022-01']['file']
    assert ingestion.sha256(target) == manifest['2022-01']['sha256']
    target.write_bytes(b'corrupt')
    ingestion.run(config)
    assert len(calls) == 2
    assert ingestion.validate_parquet(target, 2022, 1, 3) == 3
    assert not list(raw.glob('tmp*'))


def test_download_limit(tmp_path):
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b'123456'

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    with pytest.raises(ValueError, match='download limit'):
        ingestion.download_zip(Session(), 2022, 1, tmp_path / 'source.zip', 5)
