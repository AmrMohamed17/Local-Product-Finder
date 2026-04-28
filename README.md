# Local Product Finder

Real-time product recognition app using MobileNet (99% accuracy) with a hybrid recommendation engine combining Sentence-BERT and LightFM.

## Overview

Local Product Finder is a graduation project that enables users to identify products in real time using their mobile camera and receive personalized recommendations from local stores. It combines computer vision, NLP-based semantic similarity, and collaborative filtering into a single mobile-integrated pipeline.

## Features

- Real-time product recognition via MobileNet (achieving 99% accuracy over baseline models)
- Hybrid recommendation system combining Sentence-BERT (semantic similarity) and LightFM (collaborative filtering)
- Mobile app integration for end-to-end inference
- ~20% projected improvement in user engagement from hybrid recommendations vs single-model baseline

## Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | MobileNet (TensorFlow/Keras) |
| Semantic Similarity | Sentence-BERT |
| Collaborative Filtering | LightFM |
| Backend | Python |
| Mobile Integration | REST API |

## Architecture

```
User Camera Input
       ↓
MobileNet (Real-time Recognition)
       ↓
Product Identified
       ↓
 ┌─────────────────────────┐
 │  Hybrid Recommender     │
 │  Sentence-BERT + LightFM│
 └─────────────────────────┘
       ↓
Personalized Local Product Results
```

## Results

- 99% recognition accuracy over MobileNet baseline
- Hybrid recommendation system outperformed single-model approach by ~20% in projected user engagement
- Grade: A+
