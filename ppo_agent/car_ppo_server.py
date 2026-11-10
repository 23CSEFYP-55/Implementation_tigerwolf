"""
CAR Pointer-Network trajectory server.

Serves the CAR trajectory planner to the iFogSim Java simulation:
inputs per UAV -> {id, origin:[x,y], regions:[[x,y],...]}
outputs       -> {tours: [{id, order:[r0,r1,...]}]}

Uses the pointer network from car_pointer_net.py, trained with the paper's
actor-critic procedure on small cluster instances (converges ~2000 steps, Fig. 5)
and serves greedy (argmax) tours at inference. The trained weights are cached in
car_ptrnet.pt so repeat runs load instantly; the first run trains the model
(steps configurable via the CAR_TRAIN_STEPS env var, default 2000).
"""

import json
import os
import socket

import torch

from car_pointer_net import Critic, PointerNetwork

HOST, PORT = "localhost", 5510
MODEL_PATH = os.path.join(os.path.dirname(__file__), "car_ptrnet.pt")
TRAIN_STEPS = int(os.environ.get("CAR_TRAIN_STEPS", "2000"))
TRAIN_BATCH = int(os.environ.get("CAR_TRAIN_BATCH", "64"))
TRAIN_SEED = int(os.environ.get("CAR_TRAIN_SEED", "0"))


def build_policy(seed=0):
    torch.manual_seed(seed)
    model = PointerNetwork(input_dim=2, emb_dim=128, hidden_dim=128)
    critic = Critic(input_dim=2, emb_dim=128, hidden_dim=128)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        print(f"[CAR-PtrNet] Loaded trained model from {MODEL_PATH}.")
        return model
    print("[CAR-PtrNet] No checkpoint found - training from scratch...")
    from car_pointer_net import CostModel, train
    cost = CostModel(speed=16.0, scan_width=130.0, power=1.0, sigma=1.0)
    train(model, critic,
          torch.optim.Adam(model.parameters(), lr=1e-3),
          torch.optim.Adam(critic.parameters(), lr=1e-3),
          cost, steps=TRAIN_STEPS, batch_size=TRAIN_BATCH, seed=seed)
    torch.save({"actor": model.state_dict(), "critic": critic.state_dict()}, MODEL_PATH)
    print(f"[CAR-PtrNet] Saved trained model to {MODEL_PATH}.")
    return model


def handle_plan(model, req):
    plan = req.get("plan", [])
    response = {"tours": []}
    for entry in plan:
        uid = entry.get("id", "")
        origin = entry.get("origin", [0.0, 0.0])
        regions = entry.get("regions", [])
        if not regions:
            response["tours"].append({"id": uid, "order": []})
            continue
        order = decode_from_policy(model, origin, regions)
        response["tours"].append({"id": uid, "order": order})
    return response


def decode_from_policy(model, origin, regions):
    from car_pointer_net import decode
    order = decode(model, origin, regions)
    return [int(i) for i in order]


def run_server():
    model = build_policy()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"\n[CAR Server] Listening on {HOST}:{PORT}")
    print("[CAR Server] Waiting for Java iFogSim to connect...\n")

    while True:
        try:
            conn, addr = server.accept()
            print(f"[CAR Server] Connected to {addr}")
        except KeyboardInterrupt:
            break

        buffer = ""
        while True:
            try:
                chunk = conn.recv(8192).decode("utf-8")
                if not chunk:
                    break
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line == "CLOSE":
                        print("[CAR Server] Simulation ended.")
                        break
                    data = json.loads(line)
                    resp = handle_plan(model, data)
                    conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            except Exception as e:
                import traceback
                print(f"[CAR Server] Error: {e}")
                traceback.print_exc()
                break

        conn.close()


if __name__ == "__main__":
    run_server()