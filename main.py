import argparse
import subprocess

from src.ingest import ingest_folder
from src.preprocess import load_and_process
from src.train import train
from src.evaluate import evaluate


def run_server():
    """Start the Flask UI."""
    subprocess.Popen(["python", "src/app.py"])
    print("UI server started at: http://127.0.0.1:8787")


def main():
    parser = argparse.ArgumentParser(description="Transaction Categorisation Pipeline")
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "ingest",
            "preprocess",
            "train",
            "evaluate",
            "predict",
            "serve",
        ],
        default="all",
    )
    parser.add_argument("--text", type=str, help="Text for prediction (predict mode)")
    parser.add_argument(
        "--run-server", action="store_true", help="Launch UI after pipeline"
    )

    args = parser.parse_args()

    if args.mode == "ingest":
        ingest_folder()
        return

    if args.mode == "preprocess":
        load_and_process()
        return

    if args.mode == "train":
        train()
        if args.run_server:
            run_server()
        return

    if args.mode == "evaluate":
        evaluate()
        return

    if args.mode == "predict":
        if not args.text:
            print('ERROR: use: python main.py --mode predict --text "Your text"')
            return
        from src.infer import predict

        pred, conf = predict([args.text])[0]
        print(f"Prediction: {pred} | Confidence: {conf:.3f}")
        return

    if args.mode == "serve":
        run_server()
        return

    if args.mode == "all":
        print("=== INGEST ===")
        ingest_folder()

        print("=== PREPROCESS ===")
        load_and_process()

        print("=== TRAIN ===")
        train()

        print("=== EVALUATE ===")
        evaluate()

        if args.run_server:
            run_server()

        print("Pipeline finished.")
        return


if __name__ == "__main__":
    main()
