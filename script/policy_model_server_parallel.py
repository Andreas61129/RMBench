"""Batched variant of policy_model_server.py -- accepts multiple concurrent client connections
(multiple SAPIEN sim worker processes, each running its own episode) and batches their get_action
requests into single forward passes, instead of one full model call per request.

Modeled on sir_main_non_temporal/sir's envs/robocasa/parallel/policy_server.py, adapted for
RMBench's transport: that reference runs sim workers and the policy server in one process tree
(torch.multiprocessing queues), which doesn't work here since RMBench's SIR sim client and model
server live in different conda envs (see rmbench_adapter.py's module docstring) and are launched
as independent OS processes talking over the existing length-prefixed JSON+base64-numpy TCP
protocol. So batching happens across socket connections instead of across multiprocessing queues:
each client connection gets its own handler thread (as policy_model_server.py already does) and
its own _EpisodeState (see rmbench_adapter.py); get_action requests are handed to a shared
in-process queue.Queue instead of being answered immediately, and a single background thread
drains that queue with a short collection window and calls RMBenchModelAdapter.get_actions_batched
once per window. reset_model/set_instruction/update_obs stay per-connection synchronous calls --
they're cheap (no GPU forward pass), so batching them would only add latency for no benefit.

Usage: identical CLI to policy_model_server.py (same config file, same --overrides), plus two
optional overrides:
    --batch_timeout_ms 20     # how long to wait for more requests once the first arrives
    --max_batch 8             # safety cap on batch size independent of connection count

    python script/policy_model_server_parallel.py --config policy/SIR/deploy_policy.yml \\
        --port 9999 --overrides --policy_name SIR --checkpoint <dir> --batch_timeout_ms 20
"""
import os
import queue as _queue
import sys
import threading
import time

# Same convention as policy_model_server.py and every other RMBench script: invoked with cwd at
# the RMBench root (model_server.sh does `cd ../..` before launching), and this dir's own script/
# directory is auto-added to sys.path[0] by Python -- so a plain module-name import works.
sys.path.append("./")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy_model_server import (  # noqa: E402
    ModelServer,
    eval_function_decorator,
    json_to_numpy,
    numpy_to_json,
    parse_args_and_config,
)


class BatchedModelServer(ModelServer):
    """Overrides only _handle_client's dispatch logic; connection acceptance / socket framing /
    error handling are inherited unchanged from ModelServer."""

    def __init__(self, model, host="localhost", port=None,
                 batch_timeout_ms: float = 20.0, max_batch: int = 8):
        super().__init__(model, host=host, port=port)
        self.batch_timeout_s = batch_timeout_ms / 1000.0
        self.max_batch = max(1, int(max_batch))
        self._batch_queue = _queue.Queue()
        self._batch_stop = threading.Event()
        self._batch_thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._stats_batches = 0
        self._stats_items = 0
        self._stats_last_emit = time.monotonic()

    def start(self):
        self._batch_thread.start()
        super().start()

    def stop(self):
        self._batch_stop.set()
        super().stop()
        self._batch_thread.join(timeout=5.0)

    # ----- batching thread -------------------------------------------------
    def _batch_loop(self):
        print("[BatchedModelServer] batch thread started "
              f"(timeout_ms={self.batch_timeout_s*1000:.0f}, max_batch={self.max_batch})")
        while not self._batch_stop.is_set():
            try:
                first = self._batch_queue.get(timeout=0.5)
            except _queue.Empty:
                continue
            items = [first]
            deadline = time.monotonic() + self.batch_timeout_s
            while len(items) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    items.append(self._batch_queue.get(timeout=remaining))
                except _queue.Empty:
                    break
            self._process_batch(items)

    def _process_batch(self, items):
        # items: list of (obs_payload, episode_state, result_box, event)
        try:
            model_items = [(obs, state) for obs, state, _, _ in items]
            results = self.model.get_actions_batched(model_items)
            for (_, _, box, event), result in zip(items, results):
                box["result"] = result
                event.set()
        except Exception as e:
            for _, _, box, event in items:
                box["error"] = e
                event.set()
        self._stats_batches += 1
        self._stats_items += len(items)
        if self._stats_batches % 50 == 0:
            wall = time.monotonic() - self._stats_last_emit
            avg_b = self._stats_items / max(1, self._stats_batches)
            rate = self._stats_batches / max(1e-6, wall)
            print(f"[BatchedModelServer] {self._stats_batches} batches | "
                  f"avg_size={avg_b:.2f} | {rate:.1f} batch/s", flush=True)
            self._stats_batches = 0
            self._stats_items = 0
            self._stats_last_emit = time.monotonic()

    # ----- per-connection handling (overrides ModelServer._handle_client) --
    def _handle_client(self, client_socket):
        conn_state = self.model.new_episode_state()
        with client_socket:
            while self.running:
                try:
                    len_bytes = client_socket.recv(4)
                    if not len_bytes:
                        print("🔌 Client disconnected")
                        break
                    msg_length = int.from_bytes(len_bytes, "big")

                    chunks = []
                    remaining = msg_length
                    while remaining > 0:
                        chunk = client_socket.recv(min(remaining, 4096))
                        if not chunk:
                            raise ConnectionError("Incomplete data received")
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    raw_msg = b"".join(chunks).decode("utf-8")

                    data = json_to_numpy(raw_msg)
                    cmd = data.get("cmd")
                    obs = data.get("obs")

                    if cmd == "get_action":
                        box, event = {}, threading.Event()
                        self._batch_queue.put((obs, conn_state, box, event))
                        event.wait()
                        if "error" in box:
                            raise box["error"]
                        result = box["result"]
                    elif cmd == "update_obs":
                        result = self.model.update_obs(obs, state=conn_state)
                    elif cmd == "reset_model":
                        result = self.model.reset_model(state=conn_state)
                    elif cmd == "set_instruction":
                        result = self.model.set_instruction(obs, state=conn_state)
                    else:
                        method = getattr(self.model, cmd, None)
                        if not callable(method):
                            raise AttributeError(f"No model method named '{cmd}'")
                        result = method(obs) if obs is not None else method()

                    response = {"res": result}
                    resp_bytes = numpy_to_json(response).encode("utf-8")
                    client_socket.sendall(len(resp_bytes).to_bytes(4, "big"))
                    client_socket.sendall(resp_bytes)

                except (ConnectionResetError, BrokenPipeError):
                    print("🔌 Client connection lost")
                    break
                except Exception as e:
                    err = f"Error handling request: {e}"
                    print(f"⚠️ {err}")
                    import traceback
                    tb = traceback.format_exc()
                    error_resp = numpy_to_json({"error": err, "traceback": tb}).encode("utf-8")
                    client_socket.sendall(len(error_resp).to_bytes(4, "big"))
                    client_socket.sendall(error_resp)


def main(usr_args):
    policy_name = usr_args["policy_name"]
    port = usr_args.get("port")
    batch_timeout_ms = float(usr_args.get("batch_timeout_ms", 20.0))
    max_batch = int(usr_args.get("max_batch", 8))

    get_model = eval_function_decorator(policy_name, "get_model")
    model = get_model(usr_args)
    if not hasattr(model, "get_actions_batched") or not hasattr(model, "new_episode_state"):
        raise RuntimeError(
            f"{type(model).__name__} doesn't support batched parallel serving "
            f"(needs get_actions_batched() and new_episode_state() -- see rmbench_adapter.py)."
        )

    server = BatchedModelServer(model, port=port, batch_timeout_ms=batch_timeout_ms, max_batch=max_batch)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.stop()
        thread.join()


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    main(usr_args)
