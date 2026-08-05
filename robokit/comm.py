"""客户端-服务端双向通讯：长度前缀 + pickle 的 TCP 协议（与旧 Utils/bisocket.py 协议兼容）。"""
import pickle
import socket
from threading import Event, Thread

from robokit.utils import log


class RequestClient:
    """把 BiSocket 的异步回包封装成部署客户端使用的同步 request/response。"""

    def __init__(self, host, port, timeout=30.0):
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((host, port))
        self.timeout = timeout
        self._event = Event()
        self._response = None
        self.bisocket = BiSocket(conn, self._on_message)
        log("client", f"connected to {host}:{port}", "INFO")

    def _on_message(self, message):
        self._response = message
        self._event.set()

    def request(self, payload):
        self._event.clear()
        self.bisocket.send(payload)
        if not self._event.wait(self.timeout):
            raise TimeoutError("no response from inference server")
        return self._response

    def close(self):
        self.bisocket.close()


class BiSocket:
    """在已连接的 socket 上收发 pickle 消息。

    handler(message) 在接收线程中被调用；send_back=True 时把 handler 的返回值回发给对端
    （服务端模式），False 时只调用不回发（客户端模式）。
    """

    def __init__(self, conn: socket.socket, handler, send_back=False):
        self.conn = conn
        self.handler = handler
        self.send_back = send_back
        self.running = Event()
        self.running.set()
        Thread(target=self._recv_loop, daemon=True).start()

    def _recv_exact(self, n):
        data = b""
        while len(data) < n:
            try:
                packet = self.conn.recv(n - len(data))
            except OSError:
                return None
            if not packet:
                return None
            data += packet
        return data

    def _recv_loop(self):
        try:
            while self.running.is_set():
                header = self._recv_exact(4)
                if header is None:
                    break
                body = self._recv_exact(int.from_bytes(header, "big"))
                if body is None:
                    break
                try:
                    message = pickle.loads(body)
                except Exception as e:
                    log("bisocket", f"unpickle error: {e}", "WARNING")
                    continue
                try:
                    result = self.handler(message)
                    if self.send_back:
                        self.send(result)
                except Exception as e:
                    log("bisocket", f"handler error: {e}", "ERROR")
        finally:
            self.close()

    def send(self, data):
        try:
            payload = pickle.dumps(data)
            self.conn.sendall(len(payload).to_bytes(4, "big") + payload)
        except OSError as e:
            log("bisocket", f"send failed: {e}", "ERROR")
            self.close()

    def close(self):
        if self.running.is_set():
            self.running.clear()
            try:
                self.conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.conn.close()
            log("bisocket", "connection closed", "INFO")
