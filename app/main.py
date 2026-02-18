from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.controller import Controller


def main():
    controller = Controller()
    result = controller.run_pipeline(
        {
            "problem_statement": "Predict customer churn from customer usage and plan details."
        }
    )
    print(result)


if __name__ == "__main__":
    main()
