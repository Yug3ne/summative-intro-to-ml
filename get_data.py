"""
get_data.py
===========
Builds the project dataset by downloading and merging FIVE real, open data sources.
The project is about Kenya; the sources sit at three levels.

KENYA, COUNTY LEVEL (47 counties; attached to Kenyan respondents only)
    Source 1 - International Budget Partnership Kenya, County Budget Transparency
               Survey (CBTS) 2023. For every county: which key budget documents were
               published (incl. public-participation reports, Citizens Budget), and a
               question-by-question comprehensiveness score (94 questions).
               https://internationalbudget.org/publications/kenyas-county-budget-transparency-survey-2023/

    Source 2 - Office of the Controller of Budget (OCoB), County Governments Budget
               Implementation Review Report FY 2020/21 (July 2020 - June 2021, the
               12 months before Afrobarometer interviewed Kenyans in November 2021).
               Budget absorption, own-source revenue, MCA sitting allowances, pending bills.
               https://cob.go.ke/reports/consolidated-county-budget-implementation-review-reports/

INDIVIDUAL LEVEL (one row per citizen)
    Source 3 - Afrobarometer Round 9 (2021-2023) merged survey, 39 African countries,
               including 2,400 Kenyans in all 47 counties. Gives us the TARGET (did this
               person contact their local councillor / MCA in the past year?) plus the
               barriers each person faces: rural location, distance to government
               offices, internet use, poverty, language of interview, education.
               Kenya alone has too few respondents for the 20,000-row requirement and
               for deep learning, so the other countries are used as extra training data.
               https://www.afrobarometer.org/data/merged-data/

COUNTRY LEVEL (lets the model compare Kenya with the other countries it learns from)
    Source 4 - IBP Open Budget Survey 2023: national budget transparency, public
               participation and oversight scores (0-100).
               https://internationalbudget.org/open-budget-survey

    Source 5 - World Bank, World Development Indicators (WDI) API: internet
               penetration, mobile subscriptions, % rural population, GDP per capita.
               https://data.worldbank.org

The merge is a multi-level join: every respondent inherits the scores of their country
(join key: ISO-3 code), and every Kenyan respondent also inherits the scores of their
county (join key: normalised county name). Non-Kenyan rows have blank county columns.

Nothing here is simulated. Every value in the output comes from one of the five
published sources above.

Usage:
    python get_data.py              # download (if needed) + merge
    python get_data.py --offline    # only use files already in data/raw/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUT_CSV = PROCESSED / "citizen_voice_merged.csv"

AFROBAROMETER_URL = (
    "https://www.afrobarometer.org/wp-content/uploads/2025/06/"
    "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav"
)
AFROBAROMETER_FILE = RAW / "afrobarometer_r9.sav"

IBP_URL = "https://internationalbudget.org/sites/default/files/2024-05/IBP-OBS-Data-Tables-2023.pdf"
IBP_FILE = RAW / "ibp" / "IBP-OBS-Data-Tables-2023.pdf"

CBTS_URL = ("https://internationalbudget.org/publications/kenyas-county-budget-transparency-survey-2023/"
            "county-budget-transparency-survey-2023-data/")   # serves the .xlsx directly
CBTS_FILE = RAW / "kenya" / "ibp_cbts_2023.xlsx"

COB_URL = ("https://cob.go.ke/download/annual-county-governments-budget-implementation-review-report-fy-2020-21/"
           "?wpdmdl=15580")
COB_FILE = RAW / "kenya" / "cob_cbirr_fy2020_21.pdf"

WDI_FILE = RAW / "worldbank_wdi.csv"
WDI_INDICATORS = {
    "IT.NET.USER.ZS": "wb_internet_users_pct",
    "IT.CEL.SETS.P2": "wb_mobile_subs_per100",
    "SP.RUR.TOTL.ZS": "wb_rural_pop_pct",
    "NY.GDP.PCAP.PP.KD": "wb_gdp_pc_ppp",
}
WDI_YEARS = "2019:2023"  # Afrobarometer R9 fieldwork ran 2021-2023

# ---------------------------------------------------------------------------
# Afrobarometer columns we keep, renamed to readable snake_case.
# Codes such as -1, 7, 8, 9 (missing / not applicable / refused / don't know) are
# kept untouched here. Cleaning them is a learning step in the notebook.
# ---------------------------------------------------------------------------
AFRO_COLUMNS = {
    # identifiers & design
    "RESPNO": "respondent_id",
    "COUNTRY": "country_code_ab",
    "URBRUR": "urban_rural",
    "REGION": "region_code",
    "withinwt_hh": "survey_weight",
    # TARGET: Q11B contact local government councillor (MCA in Kenya)
    "Q11B": "q11b_contact_councillor",
    # enumeration-area (EA) observations: proxies for distance to services/government
    "EA_SVC_A": "ea_electricity_grid",
    "EA_SVC_B": "ea_piped_water",
    "EA_SVC_C": "ea_sewage",
    "EA_SVC_D": "ea_mobile_service",
    "EA_FAC_A": "ea_post_office",
    "EA_FAC_B": "ea_school",
    "EA_FAC_C": "ea_police_station",
    "EA_FAC_D": "ea_health_clinic",
    "EA_FAC_E": "ea_market",
    "EA_FAC_F": "ea_bank",
    "EA_FAC_F2": "ea_govt_office",
    "EA_FAC_G": "ea_paid_transport",
    "EA_ROAD_A": "ea_road_tarred",
    # demographics
    "Q1": "age",
    "Q100": "gender",
    "Q94": "education",
    "Q93A": "employment",
    # lived poverty: how often gone without...
    "Q6A": "pov_food",
    "Q6B": "pov_water",
    "Q6C": "pov_medical",
    "Q6D": "pov_fuel",
    "Q6E": "pov_cash",
    # information & digital access (the data-cost barrier)
    "Q74A": "news_radio",
    "Q74B": "news_tv",
    "Q74C": "news_newspaper",
    "Q74D": "news_internet",
    "Q74E": "news_social_media",
    "Q90A": "own_radio",
    "Q90B": "own_tv",
    "Q90C": "own_vehicle",
    "Q90D": "own_computer",
    "Q90E": "own_bank_account",
    "Q90F": "own_mobile_phone",
    "Q90G": "phone_has_internet",
    "Q90H": "freq_mobile_use",
    "Q90I": "freq_internet_use",
    "Q92A": "home_electricity_connected",
    # language barrier: language the interview was conducted in
    "Q102": "interview_language",
    # civic attitudes (used in a separate feature-group experiment)
    "Q8": "discuss_politics",
    "Q11A": "efficacy_make_councillor_listen",
    "Q34B": "councillors_listen",
    "Q35B": "info_access_local_budget",
    "Q37D": "trust_local_council",
    "Q38D": "corruption_councillors",
    "Q76A": "want_local_budgets_public",
}

# Afrobarometer country label -> ISO-3 code. The ISO code is the join key that lets us
# link survey respondents to IBP and World Bank country statistics.
COUNTRY_ISO3 = {
    "Angola": "AGO", "Benin": "BEN", "Botswana": "BWA", "Burkina Faso": "BFA",
    "Cabo Verde": "CPV", "Cameroon": "CMR", "Congo-Brazzaville": "COG",
    "Côte d'Ivoire": "CIV", "Eswatini": "SWZ", "Ethiopia": "ETH", "Gabon": "GAB",
    "Gambia": "GMB", "Ghana": "GHA", "Guinea": "GIN", "Kenya": "KEN",
    "Lesotho": "LSO", "Liberia": "LBR", "Madagascar": "MDG", "Malawi": "MWI",
    "Mali": "MLI", "Mauritania": "MRT", "Mauritius": "MUS", "Morocco": "MAR",
    "Mozambique": "MOZ", "Namibia": "NAM", "Niger": "NER", "Nigeria": "NGA",
    "São Tomé and Príncipe": "STP", "Senegal": "SEN", "Seychelles": "SYC", "Sierra Leone": "SLE",
    "South Africa": "ZAF", "Sudan": "SDN", "Tanzania": "TZA", "Togo": "TGO",
    "Tunisia": "TUN", "Uganda": "UGA", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}


def _normalise(name: str) -> str:
    """Lower-case, strip accents/punctuation so 'Côte d’Ivoire' == "Cote d'Ivoire"."""
    import unicodedata

    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", name.lower())


NORMALISED_ISO3 = {_normalise(k): v for k, v in COUNTRY_ISO3.items()}
NORMALISED_ISO3.update({_normalise("Republic of Congo"): "COG", _normalise("Cape Verde"): "CPV",
                        _normalise("Swaziland"): "SWZ", _normalise("The Gambia"): "GMB"})


def download(url: str, dest: Path) -> None:
    """Stream a file to disk (skips if it already exists)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {url}")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


# ---------------------------------------------------------------------------
# Source 1: Afrobarometer
# ---------------------------------------------------------------------------
def load_afrobarometer() -> pd.DataFrame:
    import pyreadstat

    df, meta = pyreadstat.read_sav(
        AFROBAROMETER_FILE, usecols=list(AFRO_COLUMNS), apply_value_formats=False
    )
    missing_cols = set(AFRO_COLUMNS) - set(df.columns)
    if missing_cols:
        sys.exit(f"Afrobarometer file is missing expected columns: {sorted(missing_cols)}")

    country_labels = meta.variable_value_labels["COUNTRY"]
    df = df.rename(columns=AFRO_COLUMNS)
    df["country"] = df["country_code_ab"].map(country_labels).str.strip()
    df["iso3"] = df["country"].map(lambda c: NORMALISED_ISO3.get(_normalise(c)))

    unmatched = sorted(df.loc[df["iso3"].isna(), "country"].unique())
    if unmatched:
        sys.exit(f"No ISO-3 code for Afrobarometer countries: {unmatched}")

    # Keep the human-readable region name too (for Kenya these are counties/regions).
    region_labels = meta.variable_value_labels.get("REGION", {})
    df["region"] = df["region_code"].map(region_labels)
    return df.drop(columns=["country_code_ab"])


# ---------------------------------------------------------------------------
# Source 2: IBP Open Budget Survey 2023 (published as a PDF table)
# ---------------------------------------------------------------------------
ROW_PATTERN = re.compile(r"^\s*(?P<name>[^\d]*?)\s*(?P<nums>(?:\d+\s+){4}\d+)\s*$")


def load_ibp() -> pd.DataFrame:
    """Parse the 'Transparency, Public Participation, and Oversight Scores' table.

    That table spans pages 3-5 of the PDF. Each row is: country name followed by five
    numbers (transparency, participation, oversight, legislative oversight, audit
    oversight). Some long names wrap onto two lines, e.g.
        'São Tomé and'
        '      32   2   46   42   56'
        'Príncipe'
    so when a numeric row has no name we glue together the lines either side of it.
    """
    import pdfplumber

    with pdfplumber.open(IBP_FILE) as pdf:
        lines = []
        for page in pdf.pages[2:5]:
            lines.extend((page.extract_text() or "").splitlines())

    rows, pending_prefix = [], ""
    for i, line in enumerate(lines):
        m = ROW_PATTERN.match(line)
        if not m:
            continue
        name = m["name"].strip()
        nums = [int(x) for x in m["nums"].split()]
        if not name:  # wrapped name: previous line + next line
            name = f"{lines[i - 1].strip()} {lines[i + 1].strip()}"
        rows.append([name, *nums])

    ibp = pd.DataFrame(
        rows,
        columns=["ibp_country", "ibp_transparency", "ibp_participation",
                 "ibp_oversight", "ibp_legislative_oversight", "ibp_audit_oversight"],
    )
    ibp["iso3"] = ibp["ibp_country"].map(lambda c: NORMALISED_ISO3.get(_normalise(c)))
    return ibp.dropna(subset=["iso3"]).drop(columns="ibp_country")


# ---------------------------------------------------------------------------
# Source 3: World Bank WDI (JSON API)
# ---------------------------------------------------------------------------
def fetch_wdi(offline: bool) -> pd.DataFrame:
    if offline or WDI_FILE.exists():
        if not WDI_FILE.exists():
            sys.exit("Offline mode but data/raw/worldbank_wdi.csv is missing.")
        print(f"  [cached] {WDI_FILE.relative_to(ROOT)}")
        return pd.read_csv(WDI_FILE)

    iso_list = ";".join(sorted(set(COUNTRY_ISO3.values())))
    records = []
    for code, col in WDI_INDICATORS.items():
        url = (f"https://api.worldbank.org/v2/country/{iso_list}/indicator/{code}"
               f"?format=json&date={WDI_YEARS}&per_page=2000")
        print(f"  [api] World Bank {code}")
        payload = requests.get(url, timeout=60).json()
        for obs in payload[1] or []:
            if obs["value"] is not None:
                records.append({"iso3": obs["countryiso3code"], "indicator": col,
                                "year": int(obs["date"]), "value": obs["value"]})

    long = pd.DataFrame(records)
    # Most recent available year per country & indicator (WDI has gaps).
    latest = long.sort_values("year").groupby(["iso3", "indicator"]).tail(1)
    wdi = latest.pivot(index="iso3", columns="indicator", values="value").reset_index()
    wdi.columns.name = None
    WDI_FILE.parent.mkdir(parents=True, exist_ok=True)
    wdi.to_csv(WDI_FILE, index=False)
    return wdi


# ---------------------------------------------------------------------------
# Kenya county level: shared helpers
# ---------------------------------------------------------------------------
def county_key(name: str) -> str:
    """'Nairobi City' / 'Elgeyo/Marakwet' / "Murang'a" / 'Tharaka-Nithi' -> one spelling."""
    key = _normalise(str(name))
    return {"nairobicity": "nairobi"}.get(key, key)


# ---------------------------------------------------------------------------
# Source 1: IBP Kenya County Budget Transparency Survey 2023 (Excel)
# ---------------------------------------------------------------------------
def load_cbts() -> pd.DataFrame:
    """One row per county.

    cbts_transparency_score   mean of the 94 question scores (0-100); a document that
                              was not published scores 0 on all of its questions
    cbts_docs_published       key budget documents published online (of 16)
    cbts_pp_reports           public-participation reports published (of 4)
    cbts_citizens_budget      1 if a simplified 'Citizens / Mwananchi Budget' was published
    """
    scores = pd.read_excel(CBTS_FILE, sheet_name="CBTS 2023 Comprehensive Results", header=1)
    scores = scores.dropna(subset=["County", "Score"])
    per_county = scores.groupby(scores["County"].map(county_key))["Score"].mean()

    avail = pd.read_excel(CBTS_FILE, sheet_name="CBTS 2023 Availability Results", header=1)
    avail.columns = [str(c).strip() for c in avail.columns]
    avail = avail[pd.to_numeric(avail.iloc[:, 1], errors="coerce").notna()]   # the 47 county rows
    doc_cols = avail.columns[5:21]                                           # the 16 key documents
    published = avail[doc_cols].apply(lambda s: s.astype(str).str.strip().str.lower().eq("available"))
    pp_cols = [c for c in doc_cols if c.startswith("Public Participation")]
    cb_col = next(c for c in doc_cols if c.startswith("Citizen"))

    cbts = pd.DataFrame({
        "county_key": avail["County"].map(county_key).values,
        "cbts_docs_published": published.sum(axis=1).values,
        "cbts_pp_reports": published[pp_cols].sum(axis=1).values,
        "cbts_citizens_budget": published[cb_col].astype(int).values,
    })
    cbts["cbts_transparency_score"] = cbts["county_key"].map(per_county).round(1)
    return cbts


# ---------------------------------------------------------------------------
# Source 2: Controller of Budget, County Budget Implementation Review FY 2020/21 (PDF)
# ---------------------------------------------------------------------------
NUM = r"(?:-|[\d,]+(?:\.\d+)?)"


def _cob_table(lines: list[str], title: str, n_nums: int, min_nums: int | None = None) -> pd.DataFrame:
    """Rows of one cross-county table: '<County name> <n_nums numbers>'.

    Columns are numbered 1..n_nums. If a row may have fewer numbers (min_nums), it is
    right-aligned, so the last column is always the table's last column.

    The table runs from its title to the 'Total' row. Long names wrap, e.g.
        'Elgeyo'
        '3,502.64 2,449.10 ...'
        'Marakwet'
    so a numeric row without a name takes the lines above and below it.
    """
    lo = (min_nums or n_nums) - 1
    row = re.compile(rf"^(?P<name>[A-Za-z' /’.-]*?)\s*(?P<nums>(?:{NUM}\s+){{{lo},{n_nums - 1}}}{NUM})\s*$")
    start = next(i for i, l in enumerate(lines) if l.startswith(title))
    out = []
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("Total"):
            break
        m = row.match(lines[i].strip())
        if not m:
            continue
        name = m["name"].strip() or f"{lines[i - 1].strip()} {lines[i + 1].strip()}"
        nums = [float("nan") if x == "-" else float(x.replace(",", "")) for x in m["nums"].split()]
        out.append([county_key(name), *[float("nan")] * (n_nums - len(nums)), *nums])
    return pd.DataFrame(out).drop_duplicates(0).set_index(0)


def load_cob() -> pd.DataFrame:
    """One row per county, FY 2020/21.

    cob_dev_absorption_pct     development spending / development budget (Table 2.3)
    cob_total_absorption_pct   total spending / total budget (Table 2.3)
    cob_dev_budget_share_pct   development budget / total budget (Table 2.3)
    cob_osr_target_met_pct     own-source revenue collected / target (Table 2.2)
    cob_mca_sitting_allowance  average monthly sitting allowance per MCA, Kshs (Table 2.4);
                               MCAs are paid per sitting, so this tracks assembly activity
    cob_mcas                   number of MCAs (Table 2.4)
    cob_pending_bills_pct      pending bills / total budget (Table 2.5)
    """
    import pdfplumber

    with pdfplumber.open(COB_FILE) as pdf:   # chapter 2 holds the 47-county comparison tables
        lines = [l for p in pdf.pages[20:34] for l in (p.extract_text() or "").splitlines()]

    osr = _cob_table(lines, "Table 2.2: Own Source Revenue", 3)
    absorb = _cob_table(lines, "Table 2.3: County Budget Allocation", 9)
    mca = _cob_table(lines, "Table 2.4: MCAs Budget Allocation", 5)
    pending = _cob_table(lines, "Table 2.5: Pending Bills", 3, min_nums=1)  # rec, dev, total

    cob = pd.DataFrame({
        "cob_dev_absorption_pct": absorb[8],
        "cob_total_absorption_pct": absorb[9],
        "cob_dev_budget_share_pct": (absorb[2] / absorb[3] * 100).round(1),
        "cob_osr_target_met_pct": osr[3],
        "cob_mca_sitting_allowance": mca[5],
        "cob_mcas": mca[4],
    })
    # Pending bills are in Kshs; the budget in Table 2.3 is in Kshs million. A '-' total = none reported.
    cob["cob_pending_bills_pct"] = (pending[3].fillna(0) / (absorb[3] * 1e6) * 100).round(2)
    return cob.rename_axis("county_key").reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="do not hit the network")
    args = parser.parse_args()

    print("1/5 Getting raw files")
    if not args.offline:
        download(AFROBAROMETER_URL, AFROBAROMETER_FILE)
        download(IBP_URL, IBP_FILE)
        download(CBTS_URL, CBTS_FILE)
        download(COB_URL, COB_FILE)

    print("2/5 Loading Afrobarometer (individual level)")
    afro = load_afrobarometer()
    print(f"     {len(afro):,} respondents x {afro.shape[1]} columns, {afro['iso3'].nunique()} countries "
          f"({(afro['iso3'] == 'KEN').sum():,} in Kenya)")

    print("3/5 Loading Kenya county data: IBP CBTS 2023 + Controller of Budget FY 2020/21")
    county = load_cbts().merge(load_cob(), on="county_key", how="outer", validate="one_to_one")
    if len(county) != 47 or county.isna().any().any():
        sys.exit(f"Expected 47 complete counties, got {len(county)}:\n{county[county.isna().any(axis=1)]}")
    print(f"     {len(county)} counties x {county.shape[1] - 1} indicators")

    print("4/5 Loading IBP Open Budget Survey + World Bank WDI (country level)")
    ibp = load_ibp()
    wdi = fetch_wdi(args.offline)
    print(f"     IBP: {len(ibp)} matched countries | WDI: {len(wdi)} countries")

    print("5/5 Merging: country scores on ISO-3 code, county scores on county name (Kenya only)")
    merged = afro.merge(ibp, on="iso3", how="left", validate="many_to_one")
    merged = merged.merge(wdi, on="iso3", how="left", validate="many_to_one")
    # Only Kenyan regions are counties; other countries' regions get no county key.
    merged["county_key"] = merged["region"].map(county_key).where(merged["iso3"] == "KEN")
    merged = merged.merge(county, on="county_key", how="left", validate="many_to_one")
    unmatched = merged.loc[(merged["iso3"] == "KEN") & merged["cbts_transparency_score"].isna(), "region"].unique()
    if len(unmatched):
        sys.exit(f"Kenyan regions with no county match: {sorted(unmatched)}")
    merged = merged.drop(columns="county_key")

    no_ibp = sorted(merged.loc[merged["ibp_participation"].isna(), "country"].unique())
    if no_ibp:
        print(f"     Note: not covered by OBS 2023 (country scores left blank): {no_ibp}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV.relative_to(ROOT)}  ->  {merged.shape[0]:,} rows x {merged.shape[1]} columns")


if __name__ == "__main__":
    main()
