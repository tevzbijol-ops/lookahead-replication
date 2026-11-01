"""
Step 1 of the Build: turn the LookAhead SQLite databases into one small CSV.

Uses only the Python standard library, so it runs with the python3 that
ships with macOS. Run it from the folder where you unzipped
LookAhead_Dataset.zip:

    python3 extract_features.py

Inputs (same names as in the LookAhead release):
    features.db, benign_evm_data.db, attack_timestamp.csv
Output:
    lookahead_features.csv  (one row per contract, label "tag": 1 = attack)

Mirrors data_loader.py from the LookAhead repo: drops token contracts and
ERC1967 proxies, keeps a time key so the split can be chronological.
The large decompiled-code text column (only used by their transformer)
is left out.
"""
import csv
import sqlite3
import sys
from pathlib import Path

HERE = Path(".")
FEATURES_DB = HERE / "features.db"
BENIGN_EVM_DB = HERE / "benign_evm_data.db"
ATTACK_TS_CSV = HERE / "attack_timestamp.csv"
OUT = HERE / "lookahead_features.csv"

SKIP = {"output_text_replaced"}


def find(name):
    hits = list(HERE.rglob(name))
    if not hits:
        sys.exit(f"Missing {name} - is it in this folder?")
    return hits[0]


def table_rows(conn, table):
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})")]
    keep = [c for c in cols if c not in SKIP]
    cur = conn.execute(f"SELECT {', '.join(keep)} FROM {table}")
    return keep, cur


def main():
    fdb = sqlite3.connect(find("features.db"))
    tables = [r[0] for r in fdb.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print("tables in features.db:", tables)

    # attack time order
    attack_time = {}
    with open(find("attack_timestamp.csv"), newline="") as f:
        for row in csv.DictReader(f):
            attack_time[row["contract_address"].lower()] = int(row["blockNumber"]) if row.get("blockNumber") else 0
            attack_time[row["contract_address"].lower() + "#ts"] = int(row["timeStamp"])

    # benign block numbers
    edb = sqlite3.connect(find("benign_evm_data.db"))
    benign_block = {}
    for addr, bn in edb.execute("SELECT contract_address, blockNumber FROM tx_info_table"):
        benign_block[addr.lower()] = int(bn, 16)

    header = None
    n_written = {0: 0, 1: 0}
    with open(OUT, "w", newline="") as out:
        w = csv.writer(out)
        for table, tag in [("attack_bsc_table", 1), ("attack_ether_table", 1), ("benign_table", 0)]:
            cols, cur = table_rows(fdb, table)
            if header is None:
                header = cols + ["chain", "order_key", "tag"]
                w.writerow(header)
            elif cols + ["chain", "order_key", "tag"] != header:
                sys.exit(f"column mismatch in {table}")
            ia, it, ip = cols.index("contract_address"), cols.index("isTokenContract"), cols.index("isERC1967Proxy")
            for r in cur:
                if str(r[it]) != "0" or str(r[ip]) != "0":
                    continue
                addr = r[ia].lower()
                if tag == 1:
                    if addr + "#ts" not in attack_time:
                        continue  # paper inner-joins on the timestamp file
                    key = attack_time[addr + "#ts"]
                else:
                    if addr not in benign_block:
                        continue
                    key = benign_block[addr]
                chain = "bsc" if table == "attack_bsc_table" else "eth"
                w.writerow(list(r) + [chain, key, tag])
                n_written[tag] += 1
    print(f"wrote {OUT}: {n_written[1]} attack, {n_written[0]} benign")


if __name__ == "__main__":
    main()
