import threading
import time
import webbrowser

import uvicorn
from app.main import app


def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8012")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8012,
        reload=False,
        workers=1,
        log_level="info",
    )