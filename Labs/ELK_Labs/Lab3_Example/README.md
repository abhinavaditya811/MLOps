# Real-Time Telemetry & Analytics Pipeline using the Elastic Stack

## 1. System Architecture

This project demonstrates the implementation of a scalable data ingestion and analytics pipeline designed to monitor e-commerce transaction events in real-time. The architecture follows a microservices pattern orchestrated via Docker Compose, utilizing the ELK Stack (Elasticsearch, Logstash, Kibana) for data processing, storage, and visualization.

### Data Flow Topology

```
Synthetic Data Generator → HTTP/REST → Logstash (Ingestion) → JSON Parse → Elasticsearch (Indexing) → Aggregations → Kibana (Visualization)
```

## 2. Component Configuration & Logic

### A. Ingestion Layer (Logstash)

Logstash functions as the server-side data processing pipeline. It was configured to decouple the data source from the storage layer using an Input-Filter-Output model.

- **Input**: Configured an `http` input plugin listening on port `8080`. This exposes a REST endpoint to accept incoming `POST` requests containing transaction payloads.
- **Filter**: Applied a `json` filter to deserialize the incoming message body, transforming raw string data into structured fields for indexing.
- **Output**: Configured the `elasticsearch` output plugin to route processed documents to the cluster container, targeting the specific index `ecommerce-live`.
- **Monitoring**: Enabled X-Pack monitoring to track pipeline health and throughput via the internal Elasticsearch connection.

### B. Storage & Indexing Layer (Elasticsearch)

Elasticsearch serves as the distributed search and analytics engine.

- **Deployment**: Deployed as a single-node cluster for development purposes (`discovery.type=single-node`).
- **Security**: Basic security (X-Pack) was disabled to streamline the development pipeline and reduce SSL overhead for internal container communication.
- **Indexing Strategy**: Data is stored in the `ecommerce-live` index. Upon ingestion, Elasticsearch automatically maps fields (Dynamic Mapping), assigning `keyword` types to categorical strings (e.g., `category`, `country`) and numerical types to metrics (e.g., `price`), enabling efficient aggregation.

### C. Synthetic Data Generation (Client Side)

A Python-based stochastic simulation script acts as the data producer.

- **Mechanism**: The script executes an infinite loop to generate synthetic transaction events.
- **Payload Delivery**: Events are serialized to JSON and transmitted via HTTP POST requests to the Logstash container's exposed port.
- **Data Characteristics**: The generator utilizes the `Faker` library to produce high-cardinality data (randomized users, geolocations) and controlled categorical data (product inventory) to simulate realistic traffic patterns.

## 3. Data Schema Definition

The ingestion pipeline processes JSON objects with the following schema structure:

| Field | Data Type | Description |
|-------|-----------|-------------|
| `@timestamp` | `date` | ISO 8601 formatted timestamp of the transaction event |
| `product` | `text` / `keyword` | Name of the item sold |
| `category` | `keyword` | Product classification (e.g., Electronics, Home) |
| `price` | `float` | Unit cost of the item |
| `quantity` | `integer` | Number of units purchased |
| `country` | `keyword` | Geolocation of the user (used for spatial analysis) |
| `payment_method` | `keyword` | Transaction type (e.g., PayPal, Credit Card) |


![alt text](<Screenshot 2025-11-17 at 9.56.50 PM.png>)

![alt text](<Screenshot 2025-11-17 at 9.57.41 PM.png>)