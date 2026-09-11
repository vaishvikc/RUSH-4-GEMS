import clifpy
import polars as pl

TABLES = {
    "clif_medication_admin_continuous_converted": "MED-CTS",
    "clif_medication_admin_intermittent_converted": "MED-INT",
    "clif_respiratory_support_processed": "RESP",
    "clif_sofa": "SOFA",
}

# These dose targets are the units GEM's tokenizer binned against; changing one
# silently reassigns that medication's quantile tokens.
CONTINUOUS_UNITS = {
    "acetaminophen": "mg/min",
    "albumin_infusion": "ml/hr",
    "alteplase": "mg/hr",
    "aminocaproic": "g/hr",
    "aminophylline": "mg/kg/hr",
    "amiodarone": "mg/min",
    "angiotensin": "ng/kg/min",
    "argatroban": "mcg/kg/min",
    "bivalirudin": "mg/kg/hr",
    "bumetanide": "mg/hr",
    "cisatracurium": "mg/kg/hr",
    "clevidipine": "mg/hr",
    "dexmedetomidine": "mcg/kg/hr",
    "dextrose": "ml/hr",
    "dextrose_in_water_d5w": "ml/hr",
    "diltiazem": "mg/hr",
    "dobutamine": "mcg/kg/min",
    "dopamine": "mcg/kg/min",
    "epinephrine": "mcg/kg/min",
    "epoprostenol": "ng/kg/min",
    "eptifibatide": "mcg/kg/min",
    "esmolol": "mcg/kg/min",
    "fentanyl": "mcg/hr",
    "furosemide": "mg/hr",
    "heparin": "u/hr",
    "hydromorphone": "mg/hr",
    "insulin": "u/hr",
    "ketamine": "mg/kg/hr",
    "labetalol": "mg/min",
    "lidocaine": "mg/min",
    "lorazepam": "mg/hr",
    "magnesium": "g/hr",
    "midazolam": "mg/hr",
    "milrinone": "mcg/kg/min",
    "morphine": "mg/hr",
    "naloxone": "mg/hr",
    "nicardipine": "mcg/kg/min",
    "nitroglycerin": "mcg/kg/min",
    "nitroprusside": "mcg/kg/min",
    "norepinephrine": "mcg/kg/min",
    "octreotide": "mcg/hr",
    "pantoprazole": "mg/hr",
    "pentobarbital": "mg/kg/hr",
    "phenylephrine": "mcg/kg/min",
    "procainamide": "mg/min",
    "propofol": "mcg/kg/min",
    "rocuronium": "mcg/kg/min",
    "sodium chloride": "ml/hr",
    "tpn": "ml/hr",
    "treprostinil": "ng/kg/min",
    "vasopressin": "u/hr",
    "vecuronium": "mg/kg/hr",
}

INTERMITTENT_UNITS = {
    "acetaminophen": "mg",
    "acyclovir": "dose",
    "adenosine": "mg",
    "amikacin": "dose",
    "amiodarone": "mg",
    "ampicillin": "dose",
    "ampicillin_sulbactam": "dose",
    "azithromycin": "dose",
    "aztreonam": "dose",
    "bumetanide": "mg",
    "caspofungin": "dose",
    "cefazolin": "dose",
    "cefepime": "dose",
    "ceftaroline": "dose",
    "ceftazidime": "dose",
    "ceftriaxone": "dose",
    "ciprofloxacin": "dose",
    "cisatracurium": "mg",
    "clindamycin": "dose",
    "colistin": "dose",
    "daptomycin": "dose",
    "dextrose": "ml",
    "dextrose_in_water_d5w": "ml",
    "diazepam": "mg",
    "diltiazem": "mg",
    "doxycycline": "dose",
    "epinephrine": "mg",
    "ertapenem": "dose",
    "erythromycin": "dose",
    "esomeprazole": "dose",
    "fentanyl": "mcg",
    "fluconazole": "dose",
    "foscarnet": "dose",
    "furosemide": "mg",
    "gentamicin": "dose",
    "heparin": "dose",
    "hydromorphone": "mg",
    "imipenem": "dose",
    "insulin": "units",
    "ketamine": "mcg",
    "labetalol": "mg",
    "levofloxacin": "dose",
    "lidocaine": "mg",
    "linezolid": "dose",
    "lorazepam": "mg",
    "magnesium": "grams",
    "meropenem": "dose",
    "metronidazole": "dose",
    "micafungin": "dose",
    "midazolam": "mg",
    "morphine": "mg",
    "moxifloxacin": "dose",
    "nafcillin": "dose",
    "naloxone": "mg",
    "pantoprazole": "dose",
    "penicillin": "dose",
    "piperacillin_tazobactam": "dose",
    "propofol": "mg",
    "rifampin": "dose",
    "rocuronium": "mg",
    "sodium bicarbonate": "ml",
    "sodium chloride": "ml",
    "tigecycline": "dose",
    "tobramycin": "dose",
    "vancomycin": "dose",
    "vecuronium": "mg",
    "voriconazole": "dose",
}


def fail_missing(name):
    raise RuntimeError(
        f"clifpy could not build {name}; tokens prefixed {TABLES[name]}// will be lost")


def build(cfg, force=False):
    out_dir = cfg["data"]["derived_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    co = clifpy.ClifOrchestrator(
        data_directory=str(cfg["data"]["clif_dir"]),
        filetype="parquet", timezone=cfg["data"]["timezone"])
    written = []
    for name in TABLES:
        path = out_dir / f"{name}.parquet"
        if path.exists() and not force:
            written.append(path)
            continue
        frame = _build_one(co, name)
        if frame is None:
            fail_missing(name)
        frame.write_parquet(path)
        written.append(path)
    return written


def _build_one(co, name):
    if name == "clif_medication_admin_continuous_converted":
        co.convert_dose_units_for_continuous_meds(
            preferred_units=CONTINUOUS_UNITS, override=True)
        return pl.from_pandas(co.medication_admin_continuous.df_converted)
    if name == "clif_medication_admin_intermittent_converted":
        co.convert_dose_units_for_intermittent_meds(
            preferred_units=INTERMITTENT_UNITS, override=True)
        return pl.from_pandas(co.medication_admin_intermittent.df_converted)
    if name == "clif_respiratory_support_processed":
        co.load_table("respiratory_support")
        return pl.from_pandas(
            co.respiratory_support.waterfall(return_dataframe=True, verbose=False))
    if name == "clif_sofa":
        stays = pl.from_pandas(co.load_table("hospitalization").df).select(
            pl.col("hospitalization_id").cast(pl.Utf8),
            pl.col("admission_dttm").alias("start_dttm"),
            pl.col("discharge_dttm").alias("end_dttm"))
        scores = clifpy.compute_sofa_polars(
            co.data_directory, stays, filetype=co.filetype, timezone=co.timezone)
        return scores.join(
            stays.select("hospitalization_id", pl.col("end_dttm").alias("event_time")),
            on="hospitalization_id", how="inner")
