"""Adapter di source: producono Prediction da un dominio specifico.

Ogni adapter è una funzione pura `build_*_predictions(...) -> list[Prediction]`,
così è testabile senza rete e riusabile sia dai cron sia dall'API.
"""
