"""Focused smoke tests for the safe, workspace-confined report layer."""

from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook
import pandas as pd
import pytest

from icapa.analytics import BrinsonInput, analyze_backtest
from icapa.backtesting.backtester import BacktestResult
from icapa.backtesting.simulator import IndexSimulationResult
from icapa.reporting import (
    ExcelReportRenderer,
    ReportDataError,
    ReportPayload,
    SHEET_COLUMNS,
    validate_report_filename,
    write_index_research_report,
)
from icapa.tools.container import DataContext


FIRST_EFFECTIVE_DATE = pd.Timestamp("2026-03-23")
SECOND_EFFECTIVE_DATE = pd.Timestamp("2026-06-22")


def _context(
    effective_date: pd.Timestamp,
    rows: list[dict[str, object]],
) -> DataContext:
    context = DataContext(
        reference_date=effective_date - pd.Timedelta("14D"),
        effective_date=effective_date,
        index_id="GENERIC_DEMO",
        provider_name="FactSet",
        provider_parameters={
            "password": "must-not-appear",
            "connection_string": "must-not-appear",
        },
    )
    context.set_dataframe(pd.DataFrame.from_records(rows))
    return context


def _backtest_result() -> BacktestResult:
    first = _context(
        FIRST_EFFECTIVE_DATE,
        [
            {
                "instrument_id": "A",
                "name": "=unsafe formula text",
                "country": "US",
                "industry": "Technology",
                "benchmark_weight": 0.50,
                "index_weight": 0.60,
            },
            {
                "instrument_id": "B",
                "name": "Example B",
                "country": "GB",
                "industry": "Health Care",
                "benchmark_weight": 0.50,
                "index_weight": 0.40,
            },
        ],
    )
    second = _context(
        SECOND_EFFECTIVE_DATE,
        [
            {
                "instrument_id": "A",
                "name": "=unsafe formula text",
                "country": "US",
                "industry": "Technology",
                "benchmark_weight": 0.45,
                "index_weight": 0.55,
            },
            {
                "instrument_id": "C",
                "name": "Example C",
                "country": "JP",
                "industry": "Industrials",
                "benchmark_weight": 0.55,
                "index_weight": 0.45,
            },
        ],
    )
    reviews = {
        FIRST_EFFECTIVE_DATE: first,
        SECOND_EFFECTIVE_DATE: second,
    }
    rows: list[pd.DataFrame] = []
    for effective_date, context in reviews.items():
        frame = context.cons[["index_weight"]].reset_index()
        frame.insert(0, "effective_date", effective_date)
        rows.append(frame)
    weights = pd.concat(rows, ignore_index=True).set_index(
        ["effective_date", "instrument_id"]
    )
    return BacktestResult(weights=weights, reviews=reviews)


def _simulation_result() -> IndexSimulationResult:
    business_dates = pd.date_range("2026-06-22", periods=3, freq="B")
    daily = pd.DataFrame(
        {
            "index_price_return": [0.01, -0.005, 0.003],
            "index_gross_total_return": [0.011, -0.004, 0.004],
            "index_net_total_return": [0.0105, -0.0045, 0.0035],
            "benchmark_price_return": [0.008, -0.004, 0.002],
            "benchmark_gross_total_return": [0.009, -0.003, 0.003],
            "benchmark_net_total_return": [0.0085, -0.0035, 0.0025],
        },
        index=pd.DatetimeIndex(business_dates, name="business_date"),
    )
    for prefix in ("index", "benchmark"):
        for variant in ("price", "gross_total", "net_total"):
            return_column = f"{prefix}_{variant}_return"
            level_column = f"{prefix}_{variant}_level"
            daily[level_column] = 100.0 * (1.0 + daily[return_column]).cumprod()
    rebalances = pd.DataFrame(
        {
            "scheduled_effective_date": [
                FIRST_EFFECTIVE_DATE,
                SECOND_EFFECTIVE_DATE,
            ],
            "applied_business_date": [
                FIRST_EFFECTIVE_DATE,
                SECOND_EFFECTIVE_DATE,
            ],
            "index_turnover": [float("nan"), 0.25],
            "benchmark_turnover": [float("nan"), 0.10],
            "review_source": ["computed", "computed"],
        }
    )
    return IndexSimulationResult(
        daily=daily,
        holdings=pd.DataFrame(),
        rebalances=rebalances,
        asset_returns=pd.DataFrame(),
        metadata={"credentials": "must-not-appear"},
    )


def _analytics_result(
    backtest: BacktestResult,
    simulation: IndexSimulationResult,
):
    attribution = BrinsonInput(
        pd.DataFrame(
            {
                "period": ["2026-06-30", "2026-06-30"],
                "instrument_id": ["A", "C"],
                "industry": ["Technology", "Industrials"],
                "index_weight": [0.55, 0.45],
                "benchmark_weight": [0.45, 0.55],
                "asset_return": [0.04, 0.01],
            }
        )
    )
    return analyze_backtest(
        backtest,
        simulation,
        brinson_input=attribution,
    )


def _payload(*, include_optional: bool = True) -> ReportPayload:
    backtest = _backtest_result()
    simulation = _simulation_result() if include_optional else None
    analytics = (
        _analytics_result(backtest, simulation)
        if simulation is not None
        else None
    )
    return ReportPayload.from_backtest_result(
        backtest,
        simulation=simulation,
        analytics=analytics,
        index_name="Generic Demonstration Index",
        methodology_name="External Weight Producer",
        methodology_parameters={
            "maximum_weight": 0.05,
            "calculation_label": "configured",
        },
        data_sources=[
            {
                "capability": "universe",
                "provider_name": "FactSet",
                "data_type": "canonical_universe",
                "fields": ["instrument_id", "benchmark_weight"],
            },
            {
                "capability": "factor_data",
                "provider_name": "Snowflake",
                "data_type": "ThirdPartyFactorData",
                "fields": ["quality", "value"],
            },
            {
                "capability": "ad_hoc_upload",
                "provider_name": "CSV/Excel",
                "data_type": "",
                "fields": ["instrument_id"],
            },
        ],
        workspace_name="reporting-smoke",
        generated_at="2026-07-26T12:00:00Z",
    )


def test_payload_whitelists_data_and_matches_analytics_contract():
    payload = _payload()

    assert payload.index_id == "GENERIC_DEMO"
    assert payload.latest_holdings.columns.tolist() == list(
        SHEET_COLUMNS["Latest Holdings"]
    )
    assert "provider_parameters" not in payload.latest_holdings
    assert payload.performance["active_net_total_return"].notna().all()
    assert set(payload.exposures["exposure_type"]) == {"country", "industry"}
    assert payload.turnover.iloc[-1]["one_way_turnover"] == pytest.approx(0.25)
    assert not payload.attribution.empty
    assert set(payload.data_sources["provider_name"]) == {
        "FactSet",
        "Snowflake",
        "CSV/Excel",
    }
    report_text = " ".join(
        frame.astype(str).to_csv(index=False)
        for frame in payload.sheet_frames().values()
    )
    assert "must-not-appear" not in report_text

    with pytest.raises(ReportDataError, match="sensitive field"):
        ReportPayload.from_backtest_result(
            _backtest_result(),
            methodology_parameters={"api_token": "secret"},
        )
    with pytest.raises(ReportDataError, match="unsupported fields"):
        ReportPayload.from_backtest_result(
            _backtest_result(),
            data_sources=[
                {
                    "capability": "universe",
                    "provider_name": "FactSet",
                    "provider_parameters": {"password": "secret"},
                }
            ],
        )


def test_renderer_validates_template_and_marks_unavailable_sections(tmp_path):
    payload = _payload(include_optional=False)
    destination = tmp_path / "research-report.xlsx"

    ExcelReportRenderer().render(payload, destination)

    workbook = load_workbook(destination, data_only=False)
    assert workbook["_metadata"].sheet_state == "hidden"
    assert workbook["_metadata"]["B1"].value == "1.0"
    assert workbook["_metadata"]["B3"].value == "GENERIC_DEMO"
    assert workbook["Performance"]["A4"].value == "Not available"
    assert workbook["Exposures"]["A4"].value == "Not available"
    assert workbook["Latest Holdings"]["B4"].value == "'=unsafe formula text"
    assert not getattr(workbook, "_external_links", ())


@dataclass
class _Workspace:
    name: str
    path: Path

    @property
    def reports_path(self) -> Path:
        return self.path / "reports"

    def report_path(self, filename: str) -> Path:
        return self.reports_path / filename


def test_workspace_helper_confines_report_output(tmp_path):
    workspace = _Workspace(name="reporting-smoke", path=tmp_path / "workspace")

    destination = write_index_research_report(
        workspace,
        _backtest_result(),
        filename="generic research",
        generated_at="2026-07-26T12:00:00Z",
    )

    assert destination == workspace.reports_path / "generic research.xlsx"
    assert destination.is_file()
    assert validate_report_filename("example") == "example.xlsx"
    with pytest.raises(ValueError, match="directory"):
        validate_report_filename("../outside.xlsx")
