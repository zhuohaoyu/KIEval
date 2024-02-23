import subprocess
import argparse


def run_docker_container(model, data_path, gpus, port, follow_logs=False):
    # Prepare the docker run command
    num_shards = len(gpus.split(","))
    container_name = f"tgi_{''.join(gpus.split(','))}"
    if num_shards > 1:
        docker_command = [
            "docker",
            "run",
            "--rm",
            "-d",
            f"--gpus",
            f'"device={gpus}"',
            "--shm-size",
            "32g",
            "-p",
            f"{port}:80",
            "-v",
            f"{data_path}:/data",
            "--name",
            container_name,
            "ghcr.io/huggingface/text-generation-inference:1.1.0",
            "--model-id",
            f"{model}",
            "--sharded",
            "true",
            "--num-shard",
            f"{num_shards}",
            "--max-input-length=3000",
            "--max-total-tokens=4096",
            "--max-best-of=8",
            "--max-stop-sequences=20",
            "--max-batch-prefill-tokens=4096",
        ]
    else:
        docker_command = [
            "docker",
            "run",
            "--rm",
            "-d",
            f"--gpus",
            f'"device={gpus}"',
            "--shm-size",
            "32g",
            "-p",
            f"{port}:80",
            "-v",
            f"{data_path}:/data",
            "--name",
            container_name,
            "ghcr.io/huggingface/text-generation-inference:1.1.0",
            "--model-id",
            f"{model}",
            "--sharded",
            "false",
            "--max-input-length=3000",
            "--max-total-tokens=4096",
            "--max-best-of=8",
            "--max-stop-sequences=20",
            "--max-batch-prefill-tokens=4096",
        ]

    # Execute the docker run command
    subprocess.run(docker_command)

    if follow_logs:
        log_command = ["docker", "logs", "-f", container_name]
        subprocess.run(log_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a Docker container with specified GPUs and port."
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        help='Specify the model to use, e.g., "gpt2"',
        default="/data",
    )
    parser.add_argument(
        "-d",
        "--data-path",
        type=str,
        required=False,
        help='Specify the path to the checkpoints, e.g., "/your-model-path"',
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=str,
        required=True,
        help='Specify the GPUs to use, e.g., "0,1"',
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        required=True,
        help="Specify the port to use, e.g., 8081",
    )
    parser.add_argument(
        "-f",
        "--follow-logs",
        action="store_true",
        help="Follow logs immediately after starting the container",
    )

    args = parser.parse_args()

    run_docker_container(
        args.model, args.data_path, args.gpus, args.port, args.follow_logs
    )
