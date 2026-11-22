"""Generate the small deterministic retail dataset used by the demo and tests."""
import csv
import random
from pathlib import Path

random.seed(7)
out = Path(__file__).parent

countries = ["Germany", "United States", "Pakistan", "United Kingdom", "Japan"]
segments = ["consumer", "business"]
first = ["Aisha", "Ben", "Chen", "Dara", "Elif", "Farid", "Grace", "Hiro", "Ines", "Jamal",
         "Kira", "Liam", "Mina", "Noor", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara"]
customers = [
    {"customer_id": f"C{i:03d}", "name": f"{n} {random.choice(['Khan','Weber','Smith','Sato','Ali'])}",
     "country": countries[i % len(countries)], "segment": random.choice(segments),
     "signup_date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"}
    for i, n in enumerate(first, start=1)
]
catalog = [("Laptop", "electronics", 1200.0), ("Headphones", "electronics", 150.0),
           ("Monitor", "electronics", 320.0), ("Desk", "furniture", 410.0),
           ("Chair", "furniture", 230.0), ("Lamp", "furniture", 45.0),
           ("Notebook", "stationery", 4.5), ("Pen set", "stationery", 12.0),
           ("Backpack", "accessories", 60.0), ("Water bottle", "accessories", 18.0),
           ("Keyboard", "electronics", 85.0), ("Mouse", "electronics", 35.0)]
products = [{"product_id": f"P{i:03d}", "name": n, "category": c, "price": p}
            for i, (n, c, p) in enumerate(catalog, start=1)]
price = {p["product_id"]: p["price"] for p in products}

orders, items = [], []
item_no = 1
for o in range(1, 61):
    status = random.choices(["completed", "cancelled", "pending"], [0.7, 0.15, 0.15])[0]
    lines = random.sample(products, random.randint(1, 4))
    total = 0.0
    for p in lines:
        qty = random.randint(1, 3)
        total += qty * price[p["product_id"]]
        items.append({"item_id": f"I{item_no:04d}", "order_id": f"O{o:04d}",
                      "product_id": p["product_id"], "quantity": qty})
        item_no += 1
    orders.append({"order_id": f"O{o:04d}", "customer_id": random.choice(customers)["customer_id"],
                   "order_date": f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                   "status": status, "total": round(total, 2)})


def write(name, rows):
    with open(out / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


write("customers.csv", customers)
write("products.csv", products)
write("orders.csv", orders)
write("order_items.csv", items)
print(len(customers), len(products), len(orders), len(items))
