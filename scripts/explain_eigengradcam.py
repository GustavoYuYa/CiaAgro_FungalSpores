"""Run EigenGradCAM separately from EigenCAM outputs."""

from src.explainability.pipeline import main


if __name__ == "__main__":
    main("EigenGradCAM")
