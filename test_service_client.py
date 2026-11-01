import socket
import subprocess
import time
import json
import sys

def run_test():
    exe = r"target\release\gpu_extrema_classifier.exe"
    port = 8769
    print(f"Starter test-service på port {port}...")
    proc = subprocess.Popen([exe, "service", "--port", str(port), "--bind", "127.0.0.1", "--max-k", "32"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(0.8)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s_file = s.makefile("r", encoding="utf-8")

        # 1. Send 1 verdi
        print("Test 1: Send 1 verdi...")
        s.sendall(b"0.5, 1.0\n")
        resp1 = json.loads(s_file.readline().strip())
        print("Svar 1:", resp1)
        assert resp1["status"] == "ok"
        assert resp1["count"] == 1
        assert "fx" in resp1

        # 2. Send 2. verdi
        print("Test 2: Send 2. verdi...")
        s.sendall(b"1.0, 2.0\n")
        resp2 = json.loads(s_file.readline().strip())
        print("Svar 2:", resp2)
        assert resp2["status"] == "ok"
        assert resp2["count"] == 2

        # 3. Send 3. verdi
        print("Test 3: Send 3. verdi...")
        s.sendall(b"1.5, 1.5\n")
        resp3 = json.loads(s_file.readline().strip())
        print("Svar 3:", resp3)
        assert resp3["status"] == "ok"
        assert resp3["count"] == 3

        # 4. Send BATCH
        print("Test 4: Send BATCH...")
        s.sendall(b"BATCH\n2.0, 0.5\n2.5, -0.5\n3.0, -1.0\nENDBATCH\n")
        resp4 = json.loads(s_file.readline().strip())
        print("Svar 4 (Batch):", resp4)
        assert resp4["status"] == "ok"
        assert resp4["count"] == 6
        assert resp4["added"] == 3

        # 5. Sikkerhet: Injeksjonsforsøk (SQL, shell, NaN)
        print("Test 5: Sikkerhetstest (Injeksjonsforsøk)...")
        s.sendall(b"DROP TABLE users; 1, 2\n")
        resp_inj = json.loads(s_file.readline().strip())
        print("Svar Injeksjon:", resp_inj)
        assert resp_inj["status"] == "error"

        s.sendall(b"NaN, 1.0\n")
        resp_nan = json.loads(s_file.readline().strip())
        print("Svar NaN:", resp_nan)
        assert resp_nan["status"] == "error"

        # 6. Test QUIT
        print("Test 6: Send QUIT...")
        s.sendall(b"QUIT\n")
        resp_quit = json.loads(s_file.readline().strip())
        print("Svar QUIT:", resp_quit)
        assert resp_quit["status"] == "bye"
        assert resp_quit["count"] == 6

        # Verifiser at sokkelen lukkes av serveren
        remaining = s.recv(1024)
        assert len(remaining) == 0, f"Forventet EOF etter QUIT, men fikk {remaining}"
        print("Socket lukket rent av server som forventet!")
        s.close()
        print("=== ALLE TESTER BESTÅTT MED GLANS! ===")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    run_test()
