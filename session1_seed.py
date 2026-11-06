
import argparse
import json
import os
import random
from datetime import datetime, timedelta

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

SEED = 26

CITIES = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Rawalpindi",
    "Faisalabad",
    "Multan",
    "Peshawar",
    "Quetta",
    "Sialkot",
    "Hyderabad",
]

CITY_WEIGHTS = [22, 25, 12, 8, 9, 7, 6, 3, 4, 4]

FIRST = [
    "Ali", "Ahmed", "Fatima", "Ayesha", "Hassan",
    "Zainab", "Usman", "Maryam", "Bilal", "Hira",
    "Omar", "Sana", "Hamza", "Amna", "Saad",
    "Iqra", "Talha", "Noor", "Imran", "Mahnoor",
]

LAST = [
    "Khan", "Malik", "Butt", "Sheikh", "Chaudhry",
    "Qureshi", "Raza", "Siddiqui", "Mirza", "Abbasi",
]

SEGMENTS = ["Consumer", "Corporate", "Small Business"]

CATEGORIES = {
    "Electronics": (2500, 90000),
    "Grocery": (300, 6000),
    "Fashion": (900, 15000),
    "Home": (1500, 40000),
    "Beauty": (400, 8000),
    "Books": (300, 3500),
}

STATUSES = ["delivered", "shipped", "pending", "cancelled", "returned"]
STATUS_WEIGHTS = [55, 20, 12, 8, 5]

PAYMENTS = ["card", "cash_on_delivery", "bank_transfer", "wallet"]

START = datetime(2025, 1, 1)
END = datetime(2026, 10, 24, 23, 59)


def make_customers(rng, n=2000):
    """
    Customers as they are today. About 5% moved city in 2026:
    Postgres holds the new city (overwritten, SCD Type 1),
    while older app documents still carry the old one.
    """
    rows, movers = [], {}

    for cid in range(1, n + 1):
        signup = START - timedelta(days=rng.randint(0, 900))

        rows.append({
            "customer_id": cid,
            "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "city": rng.choices(CITIES, CITY_WEIGHTS)[0],
            "segment": rng.choices(
                SEGMENTS, [70, 15, 15]
            )[0],
            "signup_date": signup.date(),
        })

        if rng.random() < 0.05:
            old = rng.choices(
                [c for c in CITIES if c != rows[-1]["city"]]
            )[0]

            movers[cid] = (
                old,
                datetime(2026, 1, 1) + timedelta(
                    days=rng.randint(0, 240)
                ),
            )

    return pd.DataFrame(rows), movers


def make_orders(rng, customers, n=200_000):
    span = int((END - START).total_seconds())
    cust_ids = customers["customer_id"].tolist()

    # A few heavy buyers, with a long tail of occasional buyers.
    weights = [
        min(rng.paretovariate(1.5), 15)
        for _ in cust_ids
    ]

    picks = rng.choices(cust_ids, weights, k=n)
    rows = []

    for oid, cid in enumerate(picks, start=100001):
        order_ts = START + timedelta(
            seconds=rng.randint(0, span)
        )

        cat = rng.choice(list(CATEGORIES))
        lo, hi = CATEGORIES[cat]

        qty = rng.choices(
            [1, 2, 3, 4, 5],
            [60, 22, 10, 5, 3],
        )[0]

        amount = round(rng.uniform(lo, hi) * qty, 2)

        status = rng.choices(
            STATUSES, STATUS_WEIGHTS
        )[0]

        # updated_at records when the row last changed.
        days_after = {
            "pending": 0,
            "shipped": rng.randint(1, 3),
            "delivered": rng.randint(2, 8),
            "cancelled": rng.randint(0, 2),
            "returned": rng.randint(10, 25),
        }[status]

        updated = min(
            order_ts + timedelta(
                days=days_after,
                minutes=rng.randint(0, 600),
            ),
            END,
        )

        rows.append({
            "order_id": oid,
            "customer_id": cid,
            "order_date": order_ts.date(),
            "order_ts": order_ts,
            "category": cat,
            "quantity": qty,
            "amount": amount,
            "status": status,
            "payment_method": rng.choice(PAYMENTS),
            "updated_at": updated,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("order_ts")
        .reset_index(drop=True)
    )


def make_documents(rng, orders, customers, movers, n=5000):
    """
    Mongo-style documents: same orders, but messy on purpose.
    """
    cust = customers.set_index("customer_id")

    sample = (
        orders.sample(n=n, random_state=SEED)
        .sort_values("order_ts")
    )

    docs = []

    for r in sample.itertuples(index=False):
        doc = {
            "order_id": int(r.order_id),
            "order_date": r.order_ts.isoformat(),
            "status": r.status,
            "amount": float(r.amount),
            "items": [{
                "category": r.category,
                "quantity": int(r.quantity),
            }],
            "payment": {
                "method": r.payment_method,
            },
            "updated_at": r.updated_at.isoformat(),
        }

        if rng.random() < 0.7:
            # Nested customer sub-document.
            c = cust.loc[r.customer_id]
            city = c["city"]

            if (
                r.customer_id in movers
                and r.order_ts < movers[r.customer_id][1]
            ):
                city = movers[r.customer_id][0]

            doc["customer"] = {
                "id": int(r.customer_id),
                "name": c["name"],
                "city": city,
            }

            if rng.random() < 0.3:
                doc["customer"]["segment"] = c["segment"]

        else:
            # Older app version: only the customer ID.
            doc["customer_id"] = int(r.customer_id)

        if rng.random() < 0.2:
            doc["coupon"] = rng.choice([
                "WELCOME10",
                "ELEVEN11",
                "EID25",
                "FREESHIP",
            ])

        if rng.random() < 0.1:
            doc["delivery_notes"] = rng.choice([
                "Call before arriving",
                "Leave with guard",
                "Office hours only",
            ])

        if rng.random() < 0.15:
            doc["payment"]["installments"] = rng.choice([
                3, 6, 12
            ])

        docs.append(doc)

    return docs


def write_files(orders, customers, docs, out="data"):
    os.makedirs(out, exist_ok=True)

    orders.drop(columns=["order_ts"]).to_csv(
        f"{out}/orders.csv",
        index=False,
    )

    customers.to_csv(
        f"{out}/customers.csv",
        index=False,
    )

    with open(f"{out}/orders.json", "w") as f:
        json.dump(docs, f, indent=1)

    print(
        f"Wrote {out}/orders.csv ({len(orders):,} rows), "
        f"{out}/customers.csv, "
        f"{out}/orders.json ({len(docs):,} docs)"
    )


def load_postgres(url, orders, customers, ro_password):
    from sqlalchemy import create_engine, text

    # Pin the driver to psycopg2.
    url = (
        url.replace("postgres://", "postgresql://", 1)
        .replace(
            "postgresql://",
            "postgresql+psycopg2://",
            1,
        )
    )

    engine = create_engine(url)

    with engine.begin() as con:
        # WARNING: These commands replace existing tables.
        con.execute(text("DROP TABLE IF EXISTS orders"))
        con.execute(text("DROP TABLE IF EXISTS customers"))

        con.execute(text("""
            CREATE TABLE customers (
                customer_id  INTEGER PRIMARY KEY,
                name         TEXT NOT NULL,
                city         TEXT NOT NULL,
                segment      TEXT NOT NULL,
                signup_date  DATE NOT NULL
            )
        """))

        con.execute(text("""
            CREATE TABLE orders (
                order_id        INTEGER PRIMARY KEY,
                customer_id     INTEGER NOT NULL
                                REFERENCES customers(customer_id),
                order_date      DATE NOT NULL,
                category        TEXT NOT NULL,
                quantity        INTEGER NOT NULL,
                amount          NUMERIC(12, 2) NOT NULL,
                status          TEXT NOT NULL,
                payment_method  TEXT NOT NULL,
                updated_at      TIMESTAMP NOT NULL
            )
        """))

    customers.to_sql(
        "customers",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    print("Loading 200k orders into Postgres, about 1 to 3 minutes...")

    orders.drop(columns=["order_ts"]).to_sql(
        "orders",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    with engine.begin() as con:
        con.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_orders_updated_at
            ON orders(updated_at)
        """))

        con.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_orders_order_date
            ON orders(order_date)
        """))

        n = con.execute(
            text("SELECT COUNT(*) FROM orders")
        ).scalar()

        print(
            f"Postgres: customers={len(customers):,}, "
            f"orders={n:,}"
        )

        if ro_password:
            db = con.execute(
                text("SELECT current_database()")
            ).scalar()

            exists = con.execute(
                text("""
                    SELECT 1
                    FROM pg_roles
                    WHERE rolname = 'student_ro'
                """)
            ).scalar()

            verb = "ALTER" if exists else "CREATE"
            pw = ro_password.replace("'", "''")

            con.execute(text(
                f"{verb} ROLE student_ro "
                f"WITH LOGIN PASSWORD '{pw}'"
            ))

            con.execute(text(
                f'GRANT CONNECT ON DATABASE "{db}" TO student_ro'
            ))

            con.execute(text(
                "GRANT USAGE ON SCHEMA public TO student_ro"
            ))

            con.execute(text(
                "GRANT SELECT ON customers, orders TO student_ro"
            ))

            print(
                "Read-only role student_ro is ready. "
                "Student string = your URL with student_ro "
                "and that password."
            )


def load_mongo(url, docs):
    from pymongo import MongoClient

    client = MongoClient(
        url,
        serverSelectionTimeoutMS=15000,
    )

    col = client["shop"]["orders"]

    # WARNING: This replaces the existing collection.
    col.drop()

    col.insert_many([dict(d) for d in docs])

    col.create_index("status")
    col.create_index("updated_at")

    print(
        f"MongoDB: shop.orders has "
        f"{col.count_documents({}):,} documents "
        f"({col.count_documents({'status': 'shipped'}):,} shipped)"
    )

    client.close()


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--pg", default=os.getenv("PG_URL"))
    p.add_argument("--mongo", default=os.getenv("MONGO_URL"))
    p.add_argument(
        "--readonly-password",
        default=os.getenv("RO_PASSWORD"),
    )

    p.add_argument("--orders", type=int, default=200_000)
    p.add_argument("--docs", type=int, default=5000)
    p.add_argument("--out", default="data")

    p.add_argument("--skip-pg", action="store_true")
    p.add_argument("--skip-mongo", action="store_true")
    p.add_argument("--files-only", action="store_true")

    a = p.parse_args()

    rng = random.Random(SEED)

    customers, movers = make_customers(rng)
    orders = make_orders(rng, customers, a.orders)
    docs = make_documents(
        rng,
        orders,
        customers,
        movers,
        a.docs,
    )

    write_files(
        orders,
        customers,
        docs,
        a.out,
    )

    if not (a.files_only or a.skip_pg):
        if not a.pg:
            raise SystemExit(
                "Pass --pg (or set PG_URL), or use --skip-pg"
            )

        load_postgres(
            a.pg,
            orders,
            customers,
            a.readonly_password,
        )

    if not (a.files_only or a.skip_mongo):
        if not a.mongo:
            raise SystemExit(
                "Pass --mongo (or set MONGO_URL), or use --skip-mongo"
            )

        load_mongo(a.mongo, docs)


if __name__ == "__main__":
    main()
