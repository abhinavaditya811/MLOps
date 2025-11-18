import time
import requests
import random
import json
from faker import Faker
from datetime import datetime

fake = Faker()

# Products and Categories
PRODUCTS = [
    {"name": "Gaming Laptop", "price": 1200, "category": "Electronics"},
    {"name": "Wireless Mouse", "price": 25, "category": "Electronics"},
    {"name": "Mechanical Keyboard", "price": 85, "category": "Electronics"},
    {"name": "Running Shoes", "price": 60, "category": "Apparel"},
    {"name": "Graphic T-Shirt", "price": 20, "category": "Apparel"},
    {"name": "Coffee Maker", "price": 45, "category": "Home"},
    {"name": "Desk Chair", "price": 150, "category": "Home"}
]

url = 'http://localhost:8080'

print("Starting fake E-Commerce stream... Press Ctrl+C to stop.")

while True:
    item = random.choice(PRODUCTS)
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "user": fake.name(),
        "country": fake.country(),
        "product": item["name"],
        "category": item["category"],
        "price": item["price"],
        "quantity": random.randint(1, 3),
        "payment_method": random.choice(["Credit Card", "PayPal", "Apple Pay"])
    }

    try:
        response = requests.post(url, json=data)
        print(f"✅ Sold: {data['product']} for ${data['price']} to {data['country']}")
    except Exception as e:
        print(f"❌ Error: Is Logstash running? {e}")

    time.sleep(random.uniform(0.5, 3.0))