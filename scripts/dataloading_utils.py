import os
import ast
import pandas as pd
from collections import Counter
import re
import pandas as pd
import numpy as np


## no snv calls on github - if you want to override with your snvs, change this to 'snv'
MUT_PATHWAY_SOURCE = "import"

def make_unique(names):
    counts = Counter()
    result = []
    for name in names:
        counts[name] += 1
        if counts[name] == 1:
            result.append(name)
        else:
            result.append(f"{name}_{counts[name]}")
    return result

# def deduplicate_pathway_dict(pathway_dict):
#     unique_keys = make_unique(list(pathway_dict.keys()))
#     return dict(zip(unique_keys, pathway_dict.values()))


def deduplicate_pathway_dict(pathway_dict):
    """
    Make pathway names unique. If duplicate gene sets have the same name,
    suffix them with an index to make them unique.
    """
    from collections import defaultdict

    name_counts = defaultdict(int)
    unique_dict = {}

    for name, genes in pathway_dict.items():
        key = (name, tuple(sorted(genes)))  # hashable representation
        name_counts[key] += 1
        if name_counts[key] == 1:
            unique_name = name
        else:
            unique_name = f"{name}_{name_counts[key]}"

        unique_dict[unique_name] = genes

    return unique_dict


import os
import ast
import pandas as pd
from collections import Counter

def make_unique(names):
    counts = Counter()
    result = []
    for name in names:
        counts[name] += 1
        if counts[name] == 1:
            result.append(name)
        else:
            result.append(f"{name}_{counts[name]}")
    return result

def deduplicate_pathway_dict(pathway_dict):
    unique_keys = make_unique(list(pathway_dict.keys()))
    return dict(zip(unique_keys, pathway_dict.values()))

import os
import pandas as pd
import ast


## new loader than can do ALL or LUMINALS


import os
import ast
import pandas as pd



def load_all_data(data_path: str, subtype: str, pathway_options: list):
    """
    Alternative loader that supports combined cohorts:
      - subtype="Luminal" => LumA + LumB
      - subtype="ALL"     => LumA + LumB + Basal (excludes any other PAM50)
      - otherwise behaves like single-subtype loader

    Returns:
        data: dict of DataFrames keyed like:
            {
              meta, mut, cnv, rna, rna_raw, tf, tmb, pathogenic_count, cibersort, hoadley,
              cnv_pct_altered, clinical,
              tide,
              ecotyper_cell_states, ecotyper_ecotypes,
              estimate, band_binary, arm_binary
            }

        all_pathway_dicts: {'mutation': {pw: {GeneSet: [genes]}}, 'copynumber': {...}, 'RNA': {...}}
    """
    import os
    import re
    import ast
    import numpy as np
    import pandas as pd

    # ---------------- helpers ----------------
    def _norm_ids(idx):
        """
        Standardize IDs across sources:
          - cast to str, strip
          - remove trailing .0
          - remove leading 'X' ONLY when pattern is X + digits (EcoTyper / ESTIMATE style)
        """
        s = pd.Index(idx).astype(str).str.strip()
        s = s.str.replace(r"\.0$", "", regex=True)
        s = s.str.replace(r"^X(?=\d+$)", "", regex=True)
        return s

    def _slug(s: str) -> str:
        s = str(s).strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    def _resolve_subtypes(meta_df: pd.DataFrame, st: str):
        st = str(st).strip()
        if st.lower() == "luminal":
            return "Luminal", ["LumA", "LumB"]
        if st.upper() == "ALL":
            return "ALL", ["LumA", "LumB", "Basal"]
        return st, [st]

    def _read_csv_norm(path, index_col=None, allow_missing=False):
        if not os.path.exists(path):
            if allow_missing:
                print(f"[WARN] Missing file: {path}")
                return None
            raise FileNotFoundError(path)
        df = pd.read_csv(path, index_col=index_col)
        df.index = _norm_ids(df.index)
        return df

    def _read_table_any(path, allow_missing=False):
        if not path or not os.path.exists(path):
            if allow_missing:
                print(f"[WARN] Missing file: {path}")
                return None
            raise FileNotFoundError(path)
        try:
            return pd.read_csv(path, sep=None, engine="python", low_memory=False)
        except Exception:
            return pd.read_csv(path, low_memory=False)

    def _find_first_file(search_roots, name_includes=(), exts=(".tsv", ".txt", ".csv")):
        hits = []
        toks = [str(x).lower() for x in name_includes]
        for root in search_roots:
            if not root or not os.path.exists(root):
                continue
            if os.path.isfile(root):
                fn = os.path.basename(root).lower()
                if fn.endswith(exts) and all(tok in fn for tok in toks):
                    hits.append(root)
                continue
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    flo = fn.lower()
                    if not flo.endswith(exts):
                        continue
                    if toks and not all(tok in flo for tok in toks):
                        continue
                    hits.append(os.path.join(dirpath, fn))
        hits = sorted(set(hits))
        return hits[0] if hits else None

    def _find_subtype_files(search_roots, subtype_token, required_tokens=("band",), exts=(".csv", ".tsv", ".txt")):
        """
        Recursively find files whose filename contains the subtype token
        and all required tokens.
        """
        subtype_token = str(subtype_token).lower()
        toks = [subtype_token] + [str(x).lower() for x in required_tokens]
        hits = []
        for root in search_roots:
            if not root or not os.path.exists(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    flo = fn.lower()
                    if not flo.endswith(exts):
                        continue
                    if all(tok in flo for tok in toks):
                        hits.append(os.path.join(dirpath, fn))
        return sorted(set(hits))

    def _load_cibersort(path):
        if not os.path.exists(path):
            print(f"[WARN] Missing CIBERSORT file: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path, sep=None, engine="python")
        if "Mixture" in df.columns:
            df["Mixture"] = _norm_ids(df["Mixture"])
            df = df.set_index("Mixture")
        else:
            df = df.set_index(df.columns[0])
        drop_cols = ["P-value", "Correlation", "RMSE", "Absolute score (sig.score)"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        df = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        df.index = _norm_ids(df.index)
        print("[CIB LOAD] shape:", df.shape, "| nonzero:", int((df.fillna(0) != 0).sum().sum()))
        return df

    def _concat_mut_by_subtypes(base_dir, subtype_list, target_ids):
        parts = []
        for st in subtype_list:
            p = os.path.join(base_dir, f"mutations/mutationmatrix_{st}.csv")
            if not os.path.exists(p):
                print(f"[WARN] missing mutation file for {st}: {p}")
                continue
            df = pd.read_csv(p, index_col=0)
            if "patient_id" in df.columns:
                df["patient_id"] = _norm_ids(df["patient_id"])
                df = df.set_index("patient_id")
            else:
                df.index = _norm_ids(df.index)
            df = df.drop(columns=["pam50_subtype", "Disease Status", "group"], errors="ignore")
            parts.append(df)
        if not parts:
            raise FileNotFoundError(f"No mutation matrices found for requested subtypes: {subtype_list}")
        mut = pd.concat(parts, axis=0, sort=False)
        mut = mut[~mut.index.duplicated(keep="first")]
        mut = mut.loc[mut.index.isin(target_ids)]
        mut = mut.reindex(target_ids).fillna(0)
        return mut

    def _union_pathway_sets(file_paths):
        union_map = {}
        found_any = False
        for p in file_paths:
            if not os.path.exists(p):
                print(f"[WARN] Missing pathway file: {p}")
                continue
            df = pd.read_csv(p)
            if "Genes" not in df.columns or "GeneSet" not in df.columns:
                print(f"[WARN] Bad pathway file columns (expect GeneSet, Genes): {p}")
                continue
            df["Genes"] = df["Genes"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else (list(x) if isinstance(x, (list, tuple)) else [])
            )
            for _, row in df.iterrows():
                gs = str(row["GeneSet"])
                genes = set(map(str, row["Genes"]))
                union_map.setdefault(gs, set()).update(genes)
            found_any = True
        if not found_any:
            return {}
        return {k: sorted(list(v)) for k, v in union_map.items()}


    def _load_arm_binary_matrix(path: str) -> pd.DataFrame:
        """
        Load a subtype-specific arm binary matrix.

        Expected format:
          - rows = patient/sample IDs
          - columns = arm event features like:
                1p_gain, 1p_loss, 8q_gain, 16q_loss
          - values = 0/1

        Accepts either:
          - first column = patient_id-like column
          - or already indexed CSV
        """
        if not path or not os.path.exists(path):
            print(f"[WARN] Missing arm binary file: {path}")
            return pd.DataFrame()

        df = _read_table_any(path, allow_missing=False)
        if df is None or df.empty:
            return pd.DataFrame()

        # choose index column
        if "patient_id" in df.columns and df["patient_id"].notna().any():
            df["patient_id"] = _norm_ids(df["patient_id"])
            df = df.set_index("patient_id")
        else:
            # if saved with index_label="patient_id", first col often becomes patient_id anyway,
            # but this fallback handles generic first-column index files
            c0 = df.columns[0]
            df[c0] = _norm_ids(df[c0])
            df = df.set_index(c0)

        df.index = _norm_ids(df.index)

        # numeric 0/1 only
        out = df.apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0)
        out = (out > 0).astype(int)

        # collapse duplicate rows safely
        if out.index.duplicated().any():
            out = out.groupby(out.index).max()

        # remove duplicate columns if present
        out = out.loc[:, ~out.columns.duplicated()]

        print("[ARM LOAD] path:", path, "| shape:", out.shape)
        return out


    def _concat_arm_binary_by_subtypes(base_dir, subtype_list, target_ids):
        """
        Load one arm-binary matrix per subtype and combine them.

        For combined cohorts:
          - row-bind subtype-specific matrices
          - union columns
          - collapse duplicate sample IDs by max
          - align to target_ids
        """
        parts = []

        search_roots = [
            os.path.join(base_dir, "copynumber"),
            os.path.join(base_dir, "copynumber", "arms"),
            os.path.join(base_dir, "copynumber", "arm_level"),
            base_dir,
        ]

        for st in subtype_list:
            hits = []

            # prefer specific filenames containing both subtype + arm + binary
            for root in search_roots:
                if not root or not os.path.exists(root):
                    continue
                for dirpath, _, filenames in os.walk(root):
                    for fn in filenames:
                        flo = fn.lower()
                        if not flo.endswith((".csv", ".tsv", ".txt")):
                            continue
                        st_lo = str(st).lower()
                        if st_lo not in flo:
                            continue

                        # flexible match for files like:
                        #   Basal.arm_binary_matrix.csv
                        #   arm_binary_Basal.csv
                        #   Basal_arm_binary_matrix.txt
                        if ("arm_binary" in flo) or (("arm" in flo) and ("binary" in flo)):
                            hits.append(os.path.join(dirpath, fn))

            hits = sorted(set(hits))

            if not hits:
                print(f"[WARN] No arm binary matrix found for subtype={st}")
                continue

            p = hits[0]
            arm_df = _load_arm_binary_matrix(p)
            if arm_df is not None and not arm_df.empty:
                parts.append(arm_df)

        if not parts:
            return pd.DataFrame(index=target_ids)

        out = pd.concat(parts, axis=0, sort=False)
        out.index = _norm_ids(out.index)

        if out.index.duplicated().any():
            out = out.groupby(out.index).max()

        out = out.reindex(target_ids).fillna(0).astype(int)
        return out


    def _load_estimate_table(path: str) -> pd.DataFrame:
        """
        Expected wide-ish ESTIMATE file with one row per sample and columns like:
        Sample, StromalScore, ImmuneScore, ESTIMATEScore, patient_id, ...
        Output:
        index = patient_id
        columns = numeric ESTIMATE score columns only
        """
        if not path or not os.path.exists(path):
            print(f"[WARN] Missing ESTIMATE file: {path}")
            return pd.DataFrame()

        df = _read_table_any(path, allow_missing=False)
        if df is None or df.empty:
            return pd.DataFrame()

        # choose ID column
        if "patient_id" in df.columns and df["patient_id"].notna().any():
            df["patient_id"] = _norm_ids(df["patient_id"])
            df = df.set_index("patient_id")
        elif "_merge_sample_id_" in df.columns and df["_merge_sample_id_"].notna().any():
            df["_merge_sample_id_"] = _norm_ids(df["_merge_sample_id_"])
            df = df.set_index("_merge_sample_id_")
        elif "Sample" in df.columns and df["Sample"].notna().any():
            df["Sample"] = _norm_ids(df["Sample"])
            df = df.set_index("Sample")
        else:
            c0 = df.columns[0]
            df[c0] = _norm_ids(df[c0])
            df = df.set_index(c0)

        preferred = ["StromalScore", "ImmuneScore", "ESTIMATEScore", "TumorPurity"]
        keep_cols = [c for c in preferred if c in df.columns]

        if not keep_cols:
            meta_like = {
                "patient_id", "_merge_sample_id_", "Sample",
                "pam50_subtype", "Disease_Status_std", "Disease Status",
                "RFS_Time", "RFS_Status", "age_dx", "chemo_med", "hormtxt_evr",
                "_subtype_std_", "_recurrence_bin_", "_event_bin_", "_time_num_", "_age_num_",
                "__txbin__chemo_med", "__txbin__hormtxt_evr"
            }
            cand = [c for c in df.columns if c not in meta_like]
            tmp = df[cand].apply(lambda s: pd.to_numeric(s, errors="coerce"))
            keep_cols = tmp.columns[tmp.notna().any(axis=0)].tolist()

        out = df[keep_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out.index = _norm_ids(out.index)

        # NEW: collapse duplicate samples here
        if out.index.duplicated().any():
            dup_ids = out.index[out.index.duplicated()].unique().tolist()
            print(f"[ESTIMATE LOAD] duplicate sample IDs found; collapsing by mean. Example: {dup_ids[:10]}")
            out = out.groupby(out.index).mean()

        out = out.loc[:, ~out.columns.duplicated()]
        print("[ESTIMATE LOAD] path:", path, "| shape:", out.shape, "| cols:", list(out.columns))
        return out


    # ---------------- NEW: band long -> binary matrix ----------------
    def _load_band_long_to_binary(path: str) -> pd.DataFrame:
        """
        Expected long format with columns like:
          patient_id, band_label, amp_call, del_call, ...
        Output:
          index = patient_id
          columns = <band_label>_gain, <band_label>_loss
          values = 0/1
        """
        if not path or not os.path.exists(path):
            print(f"[WARN] Missing band file: {path}")
            return pd.DataFrame()

        df = _read_table_any(path, allow_missing=False)
        if df is None or df.empty:
            return pd.DataFrame()

        need = {"patient_id", "band_label"}
        missing = need - set(df.columns)
        if missing:
            print(f"[WARN] Band file missing required columns {missing}: {path}")
            return pd.DataFrame()

        df = df.copy()
        df["patient_id"] = _norm_ids(df["patient_id"])
        df["band_label"] = df["band_label"].astype(str).str.strip()

        parts = []

        if "amp_call" in df.columns:
            amp = df[["patient_id", "band_label", "amp_call"]].copy()
            amp["amp_call"] = pd.to_numeric(amp["amp_call"], errors="coerce").fillna(0)
            amp["amp_call"] = (amp["amp_call"] > 0).astype(int)
            amp_wide = amp.pivot_table(
                index="patient_id",
                columns="band_label",
                values="amp_call",
                aggfunc="max",
                fill_value=0,
            )
            amp_wide.columns = [f"{str(c).strip()}_gain" for c in amp_wide.columns]
            parts.append(amp_wide)

        if "del_call" in df.columns:
            dele = df[["patient_id", "band_label", "del_call"]].copy()
            dele["del_call"] = pd.to_numeric(dele["del_call"], errors="coerce").fillna(0)
            dele["del_call"] = (dele["del_call"] > 0).astype(int)
            del_wide = dele.pivot_table(
                index="patient_id",
                columns="band_label",
                values="del_call",
                aggfunc="max",
                fill_value=0,
            )
            del_wide.columns = [f"{str(c).strip()}_loss" for c in del_wide.columns]
            parts.append(del_wide)

        if not parts:
            print(f"[WARN] Band file has neither amp_call nor del_call: {path}")
            return pd.DataFrame()

        out = pd.concat(parts, axis=1).fillna(0)
        out.index = _norm_ids(out.index)
        out = out.apply(lambda s: pd.to_numeric(s, errors="coerce")).fillna(0)
        out = out.loc[:, ~out.columns.duplicated()]
        out = out[~out.index.duplicated(keep="first")]
        print("[BAND LOAD] path:", path, "| shape:", out.shape)
        return out

    def _concat_band_binary_by_subtypes(base_dir, subtype_list, target_ids):
        """
        Loads one long-format band file per subtype and returns one wide binary matrix.
        Searches recursively for filenames containing both the subtype token and 'band'.
        """
        parts = []
        search_roots = [
            os.path.join(base_dir, "copynumber"),
            os.path.join(base_dir, "copynumber", "bands"),
            os.path.join(base_dir, "copynumber", "band_level"),
            base_dir,
        ]

        for st in subtype_list:
            hits = _find_subtype_files(
                search_roots=search_roots,
                subtype_token=st,
                required_tokens=("band",),
                exts=(".csv", ".tsv", ".txt"),
            )
            if not hits:
                print(f"[WARN] No band long-format file found for subtype={st}")
                continue

            # take first match; adjust if you want a stricter filename rule
            p = hits[0]
            band_df = _load_band_long_to_binary(p)
            if band_df is not None and not band_df.empty:
                parts.append(band_df)

        if not parts:
            return pd.DataFrame(index=target_ids)

        out = pd.concat(parts, axis=0, sort=False)
        out.index = _norm_ids(out.index)
        out = out.groupby(out.index).max()
        out = out.reindex(target_ids).fillna(0)
        return out

    # ---------------- NEW: TIDE loader ----------------
    def _load_tide_table(tide_path: str) -> pd.DataFrame:
        if not tide_path or not os.path.exists(tide_path):
            print(f"[WARN] Missing TIDE file: {tide_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(tide_path, sep="\t", dtype=str)
        except Exception:
            df = pd.read_csv(tide_path, sep=None, engine="python", dtype=str)

        if df.empty:
            return pd.DataFrame()

        if "_patient_id" in df.columns and df["_patient_id"].notna().any():
            df["_patient_id"] = _norm_ids(df["_patient_id"])
            df = df.set_index("_patient_id")
        elif "Patient" in df.columns:
            df["Patient"] = _norm_ids(df["Patient"])
            df = df.set_index("Patient")
        else:
            c0 = df.columns[0]
            df[c0] = _norm_ids(df[c0])
            df = df.set_index(c0)

        drop_cols = set([
            "Patient", "No benefits", "Responder",
            "_status_std", "_subtype",
            "Surv_Time", "Surv_Event",
            "Chemo", "Hormone", "age_dx"
        ])
        drop_cols |= {c for c in df.columns if str(c).startswith("_")}

        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        out = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        out.index = _norm_ids(out.index)
        print("[TIDE LOAD] path:", tide_path, "| shape:", out.shape)
        return out

    # ---------------- NEW: EcoTyper loader 
    def _read_ecotyper_abundance(path: str, label: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_csv(path, sep="\t", index_col=0)
        if df.empty:
            return pd.DataFrame()

        df.columns = _norm_ids(df.columns)
        states = df.index.astype(str).str.strip()
        feat_cols = [f"{label}__{_slug(st)}" for st in states]

        out = df.T
        out.columns = feat_cols
        out.index = _norm_ids(out.index)
        out = out.apply(lambda s: pd.to_numeric(s, errors="coerce"))
        return out
    


    def _read_ecotyper_assignment(path: str, label: str) -> pd.DataFrame:
        """
        Read an EcoTyper assignment file and return a sample x binary one-hot matrix.

        Handles files like:
        states
        "ID"   "State"   "InitialState"

        or:
        types
        "ID"   "MaxEcotype" ... "Ecotype" ...
        """
        if not os.path.exists(path):
            return pd.DataFrame()

        # --- first try: detect and skip a leading banner line like 'states' or 'types'
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if not lines:
            return pd.DataFrame()

        first = lines[0].strip().strip('"').strip("'").lower()
        skiprows = 1 if first in {"states", "state", "types", "type", "ecotypes", "ecotype"} else 0

        try:
            df = pd.read_csv(path, sep="\t", dtype=str, skiprows=skiprows)
        except Exception:
            try:
                df = pd.read_csv(path, sep=None, engine="python", dtype=str, skiprows=skiprows)
            except Exception:
                return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).astype(str).str.lower().str.startswith("unnamed")].copy()
        if df.empty:
            return pd.DataFrame()

        # -------------------------
        # Identify sample column
        # -------------------------
        sample_candidates = ["ID", "sample", "sample_id", "patient_id", "Patient", "_patient_id", "Mixture"]
        sample_col = next((c for c in sample_candidates if c in df.columns), None)
        if sample_col is None:
            sample_col = df.columns[0]

        # -------------------------
        # Identify assignment column
        # -------------------------
        cols = [c for c in df.columns if c != sample_col]
        assign_col = None

        # prefer exact label columns first
        preferred_exact = ["State", "Ecotype", "MaxEcotype"]
        for c in preferred_exact:
            if c in cols:
                assign_col = c
                break

        if assign_col is None:
            bad_like = {
                "AssignmentP", "AssignmentQ", "AssignedToEcotypeStates",
                "Tissue", "Histology", "Type", "RFS_Time", "RFS_Status"
            }
            cand = [c for c in cols if c not in bad_like]
            for key in ["ecotype", "state", "cluster", "label", "class"]:
                hit = next((c for c in cand if key in str(c).lower()), None)
                if hit is not None:
                    assign_col = hit
                    break

        if assign_col is None:
            print(f"[ECOTYPER ASSIGN] No assignment column found in {path}. Columns={list(df.columns)}")
            return pd.DataFrame()

        sub = df[[sample_col, assign_col]].copy()
        sub[sample_col] = _norm_ids(sub[sample_col])
        sub[assign_col] = sub[assign_col].astype(str).str.strip().str.replace('"', "", regex=False)

        sub = sub[
            sub[sample_col].notna()
            & (sub[sample_col] != "")
            & sub[assign_col].notna()
            & (sub[assign_col] != "")
            & (~sub[assign_col].str.lower().isin({"nan", "na", "none", "missing", "unassigned", "false"}))
        ].copy()

        if sub.empty:
            print(f"[ECOTYPER ASSIGN] Parsed empty assignment table from {path}")
            return pd.DataFrame()

        sub["feature"] = sub[assign_col].map(lambda x: f"{label}__{_slug(x)}")
        out = pd.crosstab(sub[sample_col], sub["feature"])
        out[out > 0] = 1
        out.index = _norm_ids(out.index)
        out = out.groupby(out.index).max().astype(float)

        print(f"[ECOTYPER ASSIGN] Loaded {path} -> {out.shape}")
        return out

    def _load_ecotyper_tree(ecotyper_base_dir: str):
        out = {
            "cell_states_abundance": pd.DataFrame(),
            "cell_states_assignment": pd.DataFrame(),
            "ecotypes_abundance": pd.DataFrame(),
            "ecotypes_assignment": pd.DataFrame(),
        }

        if not ecotyper_base_dir or not os.path.exists(ecotyper_base_dir):
            print(f"[WARN] EcoTyper base dir missing: {ecotyper_base_dir}")
            return out

        cell_abunds = []
        cell_assigns = []
        eco_abunds = []
        eco_assigns = []

        for name in sorted(os.listdir(ecotyper_base_dir)):
            d = os.path.join(ecotyper_base_dir, name)
            if not os.path.isdir(d):
                continue

            lo = name.lower()
            is_ecotypes_dir = ("ecotype" in lo) or ("ecotypes" in lo)

            abund_path = None
            states_path = None
            types_path = None

            for fn in sorted(os.listdir(d)):
                flo = fn.lower()
                if not (flo.endswith(".txt") or flo.endswith(".tsv") or flo.endswith(".csv")):
                    continue

                full = os.path.join(d, fn)

                # abundance files
                if "abund" in flo:
                    abund_path = full

                # cell-state assignment files
                if (
                    "state_assign" in flo
                    or "state_assignment" in flo
                    or flo == "state_assignments.txt"
                ):
                    states_path = full

                # ecotype assignment files
                if ("type" in flo or "ecotype" in flo) and ("abund" not in flo):
                    types_path = full

            print(f"[ECOTYPER DEBUG] dir={name} | abund={abund_path} | states={states_path} | types={types_path}")

            if is_ecotypes_dir:
                abund_label = "ECOTYPES"
                assign_label = "ECOTYPE_ASSIGN"

                if abund_path is not None:
                    eco_abunds.append(_read_ecotyper_abundance(abund_path, label=abund_label))

                if types_path is not None:
                    eco_assigns.append(_read_ecotyper_assignment(types_path, label=assign_label))

            else:
                safe_name = _slug(name)
                abund_label = f"CELLSTATE_{safe_name}"
                assign_label = f"CELLSTATE_ASSIGN__{safe_name}"

                if abund_path is not None:
                    cell_abunds.append(_read_ecotyper_abundance(abund_path, label=abund_label))

                if states_path is not None:
                    cell_assigns.append(_read_ecotyper_assignment(states_path, label=assign_label))

        if cell_abunds:
            out["cell_states_abundance"] = pd.concat(cell_abunds, axis=1)
        if cell_assigns:
            out["cell_states_assignment"] = pd.concat(cell_assigns, axis=1)
        if eco_abunds:
            out["ecotypes_abundance"] = pd.concat(eco_abunds, axis=1)
        if eco_assigns:
            out["ecotypes_assignment"] = pd.concat(eco_assigns, axis=1)

        for k, df in out.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.index = _norm_ids(df.index)
                df = df.groupby(df.index).max()
                df = df.loc[:, ~df.columns.duplicated()]
                out[k] = df

        print("[ECOTYPER LOAD] base:", ecotyper_base_dir)
        print("[ECOTYPER LOAD] cell_states_abundance:", out["cell_states_abundance"].shape)
        print("[ECOTYPER LOAD] cell_states_assignment:", out["cell_states_assignment"].shape)
        print("[ECOTYPER LOAD] ecotypes_abundance:", out["ecotypes_abundance"].shape)
        print("[ECOTYPER LOAD] ecotypes_assignment:", out["ecotypes_assignment"].shape)
        return out



    # ================================
    # Clinical block (numeric only)
    # ================================
    def _fillna_preserving_categories(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        out = df.copy()

        num_cols = out.select_dtypes(include=[np.number, "Float64", "Int64"]).columns
        if len(num_cols):
            out[num_cols] = out[num_cols].fillna(0)

        bool_cols = out.select_dtypes(include=["boolean", "bool"]).columns
        if len(bool_cols):
            out[bool_cols] = out[bool_cols].fillna(False)

        cat_cols = out.select_dtypes(include=["category"]).columns
        for c in cat_cols:
            out[c] = out[c].cat.add_categories(["missing"]).fillna("missing")

        obj_cols = out.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols):
            out[obj_cols] = out[obj_cols].fillna("missing")

        return out

    # ---------------- 1) Metadata & subtype resolution ----------------
    meta = pd.read_csv(os.path.join(data_path, "metadata_withsurvival.csv"))
    meta["Disease Status"] = meta["Disease_Status_std"]
    meta = meta.set_index("patient_id")
    meta.index = _norm_ids(meta.index)

    subtype_label, subtype_list = _resolve_subtypes(meta, subtype)

    if len(subtype_list) == 1:
        meta_filt = meta[meta["pam50_subtype"] == subtype_list[0]]
    else:
        meta_filt = meta[meta["pam50_subtype"].isin(subtype_list)]

    subtype_meta = meta_filt.copy()
    subtype_meta_short = subtype_meta[
        [
            "tumor_filename",
            "RNA_filename",
            "pam50_subtype",
            "Disease Status",
            "RFS_Time",
            "RFS_Status",
            "chemo_med",
            "hormtxt_evr",
            "herceptin_med",
        ]
    ].copy()
    subtype_meta_short.index = _norm_ids(subtype_meta_short.index)
    subtype_meta_short["time_to_event"] = subtype_meta_short["RFS_Time"]

    subtype_ids = _norm_ids(subtype_meta_short.index)

    # ---------------- 2) Load matrices ----------------
    # RNA TPM
    rna_wide = _read_csv_norm(
        os.path.join(data_path, "RNAseq/TPM_annotated_matrix.csv"), index_col=0
    )
    print("Loading TPMS")

    rna_wide.columns = rna_wide.columns.astype(str).str.strip()
    rna = rna_wide.loc[:, rna_wide.columns.isin(subtype_ids)].T
    rna.index = _norm_ids(rna.index)
    rna.index.name = "patient_id"

    # Raw counts
    try:
        rna_raw = pd.read_csv(
            os.path.join(data_path, "RNAseq/RNAseq_rawcts/RNAseq_count_matrix.csv"),
            sep=None,
            engine="python",
            index_col=0,
        )
    except Exception:
        rna_raw = _read_csv_norm(
            os.path.join(data_path, "RNAseq/RNAseq_rawcts/RNAseq_count_matrix.csv"),
            index_col=0,
            allow_missing=False,
        )
    rna_raw.index = _norm_ids(rna_raw.index)

    # CNV

    cnv_path = os.path.join(data_path, "copynumber/gene_level_matrix.tsv")

    # EXPLICIT TSV READ
    cnv = pd.read_csv(cnv_path, sep="\t", low_memory=False)
    cnv.columns = pd.Index(cnv.columns).astype(str).str.replace(r"^\ufeff", "", regex=True).str.strip()

    print("[CNV LOAD raw cols first 10]:", list(cnv.columns[:10]))

    if "patient_id" not in cnv.columns:
        raise KeyError(f"[CNV] patient_id not found after TSV parse. First columns: {list(cnv.columns[:10])}")

    cnv["patient_id"] = _norm_ids(cnv["patient_id"])
    cnv = cnv.set_index("patient_id")
    cnv.index = _norm_ids(cnv.index)

    if "pam50_subtype" in cnv.columns:
        cnv = cnv[cnv["pam50_subtype"].isin(subtype_list)]

    cnv = cnv.drop(columns=["pam50_subtype", "Disease Status", "Unnamed: 0"], errors="ignore")

    cnv = cnv.apply(pd.to_numeric, errors="coerce")

    if cnv.index.duplicated().any():
        dup_ids = cnv.index[cnv.index.duplicated()].unique().tolist()
        print(f"[CNV LOAD] duplicate patient IDs found; collapsing by mean. Example: {dup_ids[:10]}")
        cnv = cnv.groupby(cnv.index).mean()

    cnv = cnv.reindex(subtype_ids)

    print("[CNV LOAD] final shape:", cnv.shape)
    print("[CNV LOAD] first genes:", list(cnv.columns[:10]))
    print("[CNV LOAD] nonzero entries:", int((cnv.fillna(0) != 0).sum().sum()))

    # TF / TMB / pathogenic_count / Hoadley / % altered / CIBERSORT
    tf = _read_csv_norm(os.path.join(data_path, "copynumber/TF_ploidy.csv"), index_col=0, allow_missing=True)
    if tf is None:
        tf = pd.DataFrame()

    tmb = _read_csv_norm(os.path.join(data_path, "mutations/TMB.csv"), index_col=0, allow_missing=True)
    if tmb is None:
        tmb = pd.DataFrame()

    path_count = _read_csv_norm(
        os.path.join(data_path, "mutations/pathogenic_count.csv"), index_col=0, allow_missing=True
    )
    if path_count is None:
        path_count = pd.DataFrame()

    hoadley = _read_csv_norm(
        os.path.join(data_path, "RNAseq/genescores/Hoadley_scores.csv"), index_col=0, allow_missing=True
    )
    if hoadley is None:
        hoadley = pd.DataFrame()

    cnv_pct_altered = _read_csv_norm(
        os.path.join(data_path, "copynumber/percentaltered_copy.csv"), index_col=0, allow_missing=True
    )
    if cnv_pct_altered is None:
        cnv_pct_altered = pd.DataFrame()

    cibersort = _load_cibersort(os.path.join(data_path, "RNAseq/cibersort/CIBERSORTx_Job31_Results_LIMMA_CPM.txt"))

    if isinstance(cnv_pct_altered, pd.DataFrame) and cnv_pct_altered.shape[1] > 0:
        before_cols = cnv_pct_altered.shape[1]
        keep_mask = ~cnv_pct_altered.columns.astype(str).str.contains("weighted", case=False, na=False)
        cnv_pct_altered = cnv_pct_altered.loc[:, keep_mask]
        after_cols = cnv_pct_altered.shape[1]
        print(f"[CNV %ALTERED] Dropped {before_cols - after_cols} 'weighted' columns; kept {after_cols}.")

    # ---------------- NEW: ESTIMATE load ----------------
    estimate_path = _find_first_file(
        search_roots=[
            os.path.join(data_path, "RNAseq", "estimate"),
            os.path.join(data_path, "RNAseq", "ESTIMATE"),
            os.path.join(data_path, "estimate"),
            os.path.join(data_path, "ESTIMATE"),
            data_path,
        ],
        name_includes=("estimate",),
        exts=(".csv", ".tsv", ".txt"),
    )
    estimate_path = os.path.join(data_path, "RNAseq", "ESTIMATE", "estimate.csv")
    estimate = _load_estimate_table(estimate_path) if os.path.exists(estimate_path) else pd.DataFrame()

    # ---------------- NEW: Band binary load ----------------
    band_binary = _concat_band_binary_by_subtypes(
        base_dir=data_path,
        subtype_list=subtype_list,
        target_ids=subtype_ids,
    )

    # ---------------- NEW: Arm binary load ----------------
    arm_binary = _concat_arm_binary_by_subtypes(
        base_dir=data_path,
        subtype_list=subtype_list,
        target_ids=subtype_ids,
    )

    # ---------------- NEW: TIDE load ----------------
    tide_path = _find_first_file(
        search_roots=[
            os.path.join(data_path, "RNAseq/tide"),
            os.path.join(data_path, "RNAseq/TIDE"),
            os.path.join(data_path, "tide"),
            data_path,
        ],
        name_includes=("tide",),
        exts=(".tsv", ".txt", ".csv"),
    )
    tide = _load_tide_table(tide_path) if tide_path else pd.DataFrame()

    # ---------------- NEW: EcoTyper load ----------------
    ecotyper_base = None
    for cand in [
        os.path.join(data_path, "RNAseq/ecotyper")
    ]:
        if os.path.exists(cand) and os.path.isdir(cand):
            ecotyper_base = cand
            break

    eco = _load_ecotyper_tree(ecotyper_base) if ecotyper_base else {
        "cell_states_abundance": pd.DataFrame(),
        "cell_states_assignment": pd.DataFrame(),
        "ecotypes_abundance": pd.DataFrame(),
        "ecotypes_assignment": pd.DataFrame(),
    }

    ecotyper_cell_states = eco["cell_states_abundance"]
    ecotyper_cell_assign = eco["cell_states_assignment"]
    ecotyper_ecotypes = eco["ecotypes_abundance"]
    ecotyper_ecotype_assign = eco["ecotypes_assignment"]

    # ================================
    # Clinical block (numeric only)
    # ================================
    # clinical_raw = subtype_meta.drop(
    #     columns=[
    #         "pam50_subtype",
    #         "Disease Status",
    #         "tumor_filename",
    #         "RNA_filename",
    #         "matchid",
    #         "Cancer Subtype",
    #         "condition",
    #         "tumor_seqid",
    #         "normal_seqid",
    #         "normal_filename",
    #         "normals_material_source",
    #         "transcript_seqid",
    #         "RNA_QC_pass",
    #         "RNA_flowcell_id",
    #         "targeted_panel_version",
    #         "normal_mean_target_coverage",
    #         "tumor_mean_target_coverage",
    #         "hist_subtype",
    #         "PAM50_condition",
    #         "PAM50_SWITCH",
    #         "MATCH_PAM50_SWITCH",
    #         "time_recur",
    #         "time_norecur",
    #         "local_dist",
    #         "firstfh",
    #         "surg_rad_med",
    #         "meno_med_int_gr",
    #         "race_eth_gr",
    #         "chemo_med",
    #         "hormtxt_evr",
    #         "herceptin_med",
    #         "bmidx_medint_who",
    #     ],
    #     errors="ignore",
    # ).copy()

    clinical_drop_cols = [
        "pam50_subtype",
        "Disease Status",
        "tumor_filename",
        "RNA_filename",
        "matchid",
        "Cancer Subtype",
        "condition",
        "tumor_seqid",
        "normal_seqid",
        "normal_filename",
        "normals_material_source",
        "transcript_seqid",
        "RNA_QC_pass",
        "RNA_flowcell_id",
        "targeted_panel_version",
        "normal_mean_target_coverage",
        "tumor_mean_target_coverage",
        "hist_subtype",
        "PAM50_condition",
        "PAM50_SWITCH",
        "MATCH_PAM50_SWITCH",
        "time_recur",
        "time_norecur",
        "local_dist",
        "firstfh",
        "surg_rad_med",
        "meno_med_int_gr",
        "race_eth_gr",
        "chemo_med",
        "hormtxt_evr",
        "herceptin_med",
        "bmidx_medint_who",
    ]

    clinical_raw = subtype_meta.drop(columns=clinical_drop_cols, errors="ignore").copy()

    def _norm_txt(x) -> str:
        if pd.isna(x):
            return ""
        return (
            str(x).strip().lower()
            .replace("–", "-")
            .replace("—", "-")
            .replace("≥", ">=")
            .replace("≤", "<=")
        )

    def _map_binary_text(series: pd.Series, pos_kw, neg_kw) -> pd.Series:
        def _m(v):
            t = _norm_txt(v)
            if not t:
                return pd.NA
            if any(k in t for k in pos_kw):
                return 1
            if any(k in t for k in neg_kw):
                return 0
            return pd.NA
        return series.map(_m).astype("Int64")

    _ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}

    def _map_stage_to_124(series: pd.Series) -> pd.Series:
        def _m(v):
            t = _norm_txt(v)
            if not t or t == "stage x":
                return pd.NA
            t = re.sub(r"^stage\s*", "", t)
            t = re.sub(r"\s+", "", t)
            m = re.match(r"^(i{1,3}v?|[1-4])", t)
            if not m:
                return pd.NA
            tok = m.group(1)
            if tok in _ROMAN:
                return float(_ROMAN[tok])
            try:
                n = int(tok)
                return float(n) if 1 <= n <= 4 else pd.NA
            except Exception:
                return pd.NA
        return series.map(_m).astype("Float64")

    def _map_ki67_0_1(series: pd.Series, cutoff=14.0) -> pd.Series:
        def _m(v):
            t = _norm_txt(v)
            if not t:
                return pd.NA
            if "positive" in t:
                return 1
            if "negative" in t:
                return 0
            m = re.search(r"(\d+(\.\d+)?)\s*%?", t)
            if m:
                return int(float(m.group(1)) >= float(cutoff))
            return pd.NA
        return series.map(_m).astype("Int64")

    def _map_tumor_size_t_to_num(series: pd.Series) -> pd.Series:
        def _m(v):
            t = _norm_txt(v).replace(" ", "")
            m = re.match(r"^t(?:(is|0)|([1-4]))", t)
            if not m:
                return pd.NA
            if m.group(1) is not None:
                return 0.0
            return float(m.group(2))
        return series.map(_m).astype("Float64")

    def _map_lymphocyte_resp(series: pd.Series) -> pd.Series:
        POS = ("brisk", "high", "present", "positive", "pos")
        NEG = ("nonbrisk", "non-brisk", "low", "absent", "negative", "neg")

        def _m(v):
            t = _norm_txt(v)
            if not t:
                return pd.NA
            if any(k in t for k in POS):
                return 1
            if any(k in t for k in NEG):
                return 0
            return pd.NA
        return series.map(_m).astype("Int64")

    def _map_her2_012(series: pd.Series) -> pd.Series:
        def _m(v):
            t = _norm_txt(v).replace("indeterminate", "equivocal")
            if not t:
                return pd.NA
            if any(k in t for k in ("positive", "amplified", "3+")):
                return 2.0
            if "equivocal" in t:
                return 1.0
            if any(k in t for k in ("negative", "non-amplified", "0-2", "1+", "2+")):
                return 0.0
            return pd.NA
        return series.map(_m).astype("Float64")

    def _map_lymphnode_invol(series: pd.Series) -> pd.Series:
        def _m(v):
            t = _norm_txt(v)
            if not t:
                return pd.NA
            m = re.search(r"(\d+)", t)
            if m:
                return 1 if int(m.group(1)) >= 1 else 0
            if any(k in t for k in ("yes", "positive", "present", "pos")):
                return 1
            if any(k in t for k in ("no", "negative", "absent", "neg")):
                return 0
            return pd.NA
        return series.map(_m).astype("Int64")

    clinical_num = pd.DataFrame(index=clinical_raw.index)

    if "AGE_AT_DIAGNOSIS" in clinical_raw.columns:
        clinical_num["CLINICALblock__age_dx"] = pd.to_numeric(
            clinical_raw["AGE_AT_DIAGNOSIS"], errors="coerce"
        ).astype("Float64")
    elif "age_dx" in clinical_raw.columns:
        clinical_num["CLINICALblock__age_dx"] = pd.to_numeric(clinical_raw["age_dx"], errors="coerce").astype("Float64")

    for col in ("ER_STATUS", "er_lab"):
        if col in clinical_raw.columns:
            clinical_num["CLINICALblock__er_status"] = _map_binary_text(
                clinical_raw[col], pos_kw=("positive", "pos"), neg_kw=("negative", "neg")
            )
            break

    for col in ("PR_STATUS", "pr_lab"):
        if col in clinical_raw.columns:
            clinical_num["CLINICALblock__pr_status"] = _map_binary_text(
                clinical_raw[col], pos_kw=("positive", "pos"), neg_kw=("negative", "neg")
            )
            break

    if "HER2_STATUS" in clinical_raw.columns:
        clinical_num["CLINICALblock__her2_status"] = _map_her2_012(clinical_raw["HER2_STATUS"])
    elif "her2_lab" in clinical_raw.columns:
        clinical_num["CLINICALblock__her2_lab"] = _map_her2_012(clinical_raw["her2_lab"])

    if "TUMOR_STAGE" in clinical_raw.columns:
        clinical_num["CLINICALblock__tumor_stage"] = _map_stage_to_124(clinical_raw["TUMOR_STAGE"])
    elif "ajcc_css" in clinical_raw.columns:
        clinical_num["CLINICALblock__ajcc_css"] = _map_stage_to_124(clinical_raw["ajcc_css"])

    if "ki67_lab" in clinical_raw.columns:
        clinical_num["CLINICALblock__ki67_lab"] = _map_ki67_0_1(clinical_raw["ki67_lab"])

    if "LYMPH_NODES_EXAMINED_POSITIVE" in clinical_raw.columns:
        ln = pd.to_numeric(clinical_raw["LYMPH_NODES_EXAMINED_POSITIVE"], errors="coerce")
        clinical_num["CLINICALblock__lymph_nodes_pos"] = ln.astype("Float64")
        clinical_num["CLINICALblock__lymph_nodes_pos_binary"] = (ln.fillna(0) >= 1).astype("Int64")
    elif "lymphnode_invol" in clinical_raw.columns:
        clinical_num["CLINICALblock__lymphnode_invol"] = _map_lymphnode_invol(clinical_raw["lymphnode_invol"])

    for m in ("ck56_lab", "egfr_lab"):
        if m in clinical_raw.columns:
            clinical_num[f"CLINICALblock__{m}"] = _map_binary_text(
                clinical_raw[m], pos_kw=("positive", "present", "pos"), neg_kw=("negative", "absent", "neg")
            )

    if "tumor_size" in clinical_raw.columns:
        clinical_num["CLINICALblock__tumor_size"] = _map_tumor_size_t_to_num(clinical_raw["tumor_size"])

    if "lymphocyteresponse" in clinical_raw.columns:
        clinical_num["CLINICALblock__lymphocyteresponse"] = _map_lymphocyte_resp(clinical_raw["lymphocyteresponse"])

    clinical_num = clinical_num.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    clinical = clinical_num

    # ---------------- Mutations ----------------
    # if len(subtype_list) == 1:
    #     mut_path = os.path.join(data_path, f"mutations/mutationmatrix_{subtype_list[0]}.csv")
    #     mut_df = pd.read_csv(mut_path, index_col=0)
    #     if "patient_id" in mut_df.columns:
    #         mut_df["patient_id"] = _norm_ids(mut_df["patient_id"])
    #         mut_df = mut_df.set_index("patient_id")
    #     else:
    #         mut_df.index = _norm_ids(mut_df.index)
    #     mut_df = mut_df.drop(columns=["pam50_subtype", "Disease Status", "group"], errors="ignore")
    #     mut_df = mut_df.loc[mut_df.index.isin(subtype_ids)]
    #     mut_df = mut_df.reindex(subtype_ids).fillna(0)
    # else:
    #     mut_df = _concat_mut_by_subtypes(data_path, subtype_list, subtype_ids)
    


    # ---------------- Mutations ----------------
    if MUT_PATHWAY_SOURCE == "snv":

        if len(subtype_list) == 1:
            mut_path = os.path.join(
                data_path,
                f"mutations/mutationmatrix_{subtype_list[0]}.csv"
            )

            mut_df = pd.read_csv(mut_path, index_col=0)

            if "patient_id" in mut_df.columns:
                mut_df["patient_id"] = _norm_ids(mut_df["patient_id"])
                mut_df = mut_df.set_index("patient_id")
            else:
                mut_df.index = _norm_ids(mut_df.index)

            mut_df = mut_df.drop(
                columns=["pam50_subtype", "Disease Status", "group"],
                errors="ignore"
            )

            mut_df = mut_df.loc[mut_df.index.isin(subtype_ids)]
            mut_df = mut_df.reindex(subtype_ids).fillna(0)

        else:
            mut_df = _concat_mut_by_subtypes(data_path, subtype_list, subtype_ids)

    else:
        # Do not load SNV mutation matrix unless mut_pathway_source == "snv"
        mut_df = pd.DataFrame(index=subtype_ids)

    # ---------------- 3) Pathway gene sets ----------------
    all_pathway_dicts = {"mutation": {}, "copynumber": {}, "RNA": {}}
    for pw in pathway_options:
        if len(subtype_list) == 1:
            paths = {
                "mutation": os.path.join(
                    data_path,
                    f"pathway_genesets/mutation-filtered/{pw}_mutation_variancefiltered_{subtype_list[0]}.csv",
                ),
                "copynumber": os.path.join(
                    data_path,
                    f"pathway_genesets/copynumber-filtered/{pw}_copynumber_variancefiltered_{subtype_list[0]}.csv",
                ),
                "RNA": os.path.join(
                    data_path,
                    f"pathway_genesets/RNA-filtered/{pw}_RNA_variancefiltered_{subtype_list[0]}.csv",
                ),
            }
            mut_tbl = _read_csv_norm(paths["mutation"], allow_missing=False)
            cnv_tbl = _read_csv_norm(paths["copynumber"], allow_missing=False)
            rna_tbl = _read_csv_norm(paths["RNA"], allow_missing=False)
            for df_pw in [mut_tbl, cnv_tbl, rna_tbl]:
                df_pw["Genes"] = df_pw["Genes"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else (list(x) if isinstance(x, (list, tuple)) else [])
                )
            all_pathway_dicts["mutation"][pw] = deduplicate_pathway_dict(dict(zip(mut_tbl["GeneSet"], mut_tbl["Genes"])))
            all_pathway_dicts["copynumber"][pw] = deduplicate_pathway_dict(dict(zip(cnv_tbl["GeneSet"], cnv_tbl["Genes"])))
            all_pathway_dicts["RNA"][pw] = deduplicate_pathway_dict(dict(zip(rna_tbl["GeneSet"], rna_tbl["Genes"])))
        else:
            mut_files = [
                os.path.join(
                    data_path,
                    f"pathway_genesets/mutation-filtered/{pw}_mutation_variancefiltered_{st}.csv",
                )
                for st in subtype_list
            ]
            cnv_files = [
                os.path.join(
                    data_path,
                    f"pathway_genesets/copynumber-filtered/{pw}_copynumber_variancefiltered_{st}.csv",
                )
                for st in subtype_list
            ]
            rna_files = [
                os.path.join(
                    data_path,
                    f"pathway_genesets/RNA-filtered/{pw}_RNA_variancefiltered_{st}.csv",
                )
                for st in subtype_list
            ]

            mut_union = _union_pathway_sets(mut_files)
            cnv_union = _union_pathway_sets(cnv_files)
            rna_union = _union_pathway_sets(rna_files)

            if not mut_union or not cnv_union or not rna_union:
                missing = [k for k, v in [("mutation", mut_union), ("copynumber", cnv_union), ("RNA", rna_union)] if not v]
                raise FileNotFoundError(
                    f"No pathway files found for {pw} in layers: {missing} (subtypes={subtype_list})"
                )

            all_pathway_dicts["mutation"][pw] = deduplicate_pathway_dict(mut_union)
            all_pathway_dicts["copynumber"][pw] = deduplicate_pathway_dict(cnv_union)
            all_pathway_dicts["RNA"][pw] = deduplicate_pathway_dict(rna_union)

    # ---------------- 4) Final alignment ----------------
    data = {
        "meta": subtype_meta_short,
        "mut": mut_df,
        "cnv": cnv,
        "rna": rna,
        "rna_raw": rna_raw,
        "tf": tf,
        "tmb": tmb,
        "pathogenic_count": path_count,
        "cibersort": cibersort,
        "hoadley": hoadley,
        "cnv_pct_altered": cnv_pct_altered,
        "clinical": clinical,

        # new additions
        "estimate": estimate,
        "band_binary": band_binary,
        "arm_binary": arm_binary,

        "tide": tide,
        "ecotyper_cell_states": ecotyper_cell_states,
        "ecotyper_cell_assign": ecotyper_cell_assign,
        "ecotyper_ecotypes": ecotyper_ecotypes,
        "ecotyper_ecotype_assign": ecotyper_ecotype_assign,
    }

    for k, df in data.items():
        if k == "meta" or not isinstance(df, pd.DataFrame):
            continue
        df.index = _norm_ids(df.index)
        if df.index.duplicated().any():
            # for binary/event style tables, max is usually safer than first
            if k in {"band_binary", "arm_binary"}:
                df = df.groupby(df.index).max()
            else:
                df = df[~df.index.duplicated(keep="first")]
        aligned = df.reindex(subtype_ids)

        if k == "cibersort":
            print(
                "[CIB ALIGN] overlap:",
                aligned.shape[0],
                "/",
                len(subtype_ids),
                "| nonzero before fill:",
                int((aligned.fillna(0) != 0).sum().sum()),
            )
            data[k] = aligned
        else:
            data[k] = _fillna_preserving_categories(aligned)

    return data, all_pathway_dicts


